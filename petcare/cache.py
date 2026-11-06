"""On-disk caching for archive queries, and the CDS rate limit.

Two hard-won constraints live here.

**Never re-download what is already on disk.**  At 30k targets a re-run that
re-queries every cone is not merely slow, it gets the machine's IP filtered.

**CDS is rate-limited per IP.**  A minimum of 2 s between requests to any
``*.cds.unistra.fr`` / ``*.u-strasbg.fr`` service, one CDS-hitting job at a
time, and back off on the first timeout.  An IP filtered by CDS breaks
ordinary interactive use for days.  Prefer the archive that owns the data --
Gaia at ESA, 2MASS/WISE at IRSA, TESS at MAST -- and reach for CDS only when
nothing else has the table.

The cache lives in ``$XDG_CACHE_HOME/petcare`` (``~/.cache/petcare`` by default).
Override it with ``$PETCARE_CACHE``, and keep it off file-syncing services, which
do not cope with this much write churn.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable

ENV_CACHE = "PETCARE_CACHE"
DEFAULT_CACHE = (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                 / "petcare")

#: Seconds between successive requests to a CDS-hosted service.
CDS_MIN_INTERVAL_S = 2.0

_cds_lock = threading.Lock()
_cds_last = 0.0


def cache_dir(name: str | None = None) -> Path:
    root = Path(os.environ.get(ENV_CACHE, DEFAULT_CACHE)).expanduser()
    d = root / name if name else root
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def cached(namespace: str, *key_parts, max_age_days: float | None = None):
    """Decorator-free cache: ``value = cached(ns, key)(fn)``.

    ``max_age_days`` re-runs ``fn`` when the cached entry is older than that.
    Leave it ``None`` for catalogue queries, which do not change under us.
    """
    path = cache_dir(namespace) / f"{_key(*key_parts)}.pkl"

    def run(fn: Callable[[], Any]) -> Any:
        if path.exists():
            if max_age_days is None or (
                    time.time() - path.stat().st_mtime < max_age_days * 86400):
                try:
                    with open(path, "rb") as fh:
                        return pickle.load(fh)
                except Exception:  # noqa: BLE001 - a corrupt entry is not fatal
                    path.unlink(missing_ok=True)
        value = fn()
        tmp = path.with_suffix(".pkl.tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(value, fh)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - an uncacheable value is still a value
            tmp.unlink(missing_ok=True)
        return value

    return run


def cds_throttle() -> None:
    """Block until at least :data:`CDS_MIN_INTERVAL_S` has passed since the last call."""
    global _cds_last
    with _cds_lock:
        wait = CDS_MIN_INTERVAL_S - (time.monotonic() - _cds_last)
        if wait > 0:
            time.sleep(wait)
        _cds_last = time.monotonic()


def with_retries(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 3.0,
                 what: str = "query"):
    """Run ``fn``, retrying with exponential backoff.

    IRSA's VO endpoints in particular return 502s while the main site is fine,
    so an index query needs its own retries rather than inheriting the
    caller's assumption that the service is up.
    """
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last


def ensure_certificates() -> None:
    """Point requests/urllib at certifi's CA bundle.

    python.org builds may ship without a system certificate bundle, which
    kills Gaia TAP with ``CERTIFICATE_VERIFY_FAILED`` and nothing else.
    Cheap to set unconditionally; harmless when already correct.
    """
    try:
        import certifi
    except ImportError:  # pragma: no cover
        return
    where = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", where)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", where)

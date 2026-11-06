"""The PetCare result object: one JSON per test, per candidate.

Every test in this package returns a :class:`Result` and writes it to
``<out>/<target>/<test_name>.json``.  The driver merges those JSONs; it never
re-runs logic.  That means this schema is the only contract between the tests
and everything downstream, so it is deliberately explicit about the things
that were easy to lose track of in earlier pipelines:

* which light curve the numbers came from (``provenance.lightcurve``),
* the dilution correction and where it came from (``dilution``), kept beside
  the depth rather than baked into it,
* the other signals already known on this star (``known_signals``), so a test
  can say whether it modelled them,
* a human disposition block that a rerun must never overwrite (``human``).

See ``README.md`` for the full rationale.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import subprocess
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# status


class Status:
    """Allowed values of ``Result.status``.

    Deliberately not an ``enum.Enum``: these are written to and read back from
    JSON as plain strings, and a bare string compares correctly against a
    value loaded from disk without an import.
    """

    PASS = "PASS"
    FLAG = "FLAG"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

    ALL = (PASS, FLAG, FAIL, INCONCLUSIVE, ERROR)

    #: Worst-first, for summarising several tests side by side.  Note this is
    #: an ordering for *display*, not a score: the driver does not collapse
    #: these into a number (README, "Do not compute an overall score").
    SEVERITY = {ERROR: 0, FAIL: 1, FLAG: 2, INCONCLUSIVE: 3, PASS: 4}


# --------------------------------------------------------------------------
# JSON sanitising


def _jsonable(obj: Any) -> Any:
    """Convert ``obj`` into something ``json.dump`` can write losslessly.

    NaN and infinity become ``None``.  Python's ``json`` writes them as the
    bare tokens ``NaN``/``Infinity``, which are not valid JSON and which every
    non-Python reader rejects -- and a vetting record that a collaborator's
    tooling cannot parse is worse than one that says ``null``.
    """
    # numpy is imported lazily so that result.py stays importable in a bare
    # interpreter (the driver's merge step needs nothing else).
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        np = None

    if np is not None:
        if isinstance(obj, np.generic):
            return _jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return [_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, (Sequence, set, frozenset)) or isinstance(obj, Iterable):
        return [_jsonable(v) for v in obj]
    return str(obj)


# --------------------------------------------------------------------------
# provenance helpers


def _git_sha(path: Path | None = None) -> str | None:
    """Short SHA of the PetCare checkout, or None outside a repo."""
    path = path or Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# sub-records


@dataclass
class Target:
    """The star.  At least one identifier must be set."""

    tic: int | None = None
    gaia_dr3: int | None = None
    name: str | None = None
    ra: float | None = None
    dec: float | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            if self.tic is not None:
                self.name = f"TIC {self.tic}"
            elif self.gaia_dr3 is not None:
                self.name = f"Gaia DR3 {self.gaia_dr3}"

    @property
    def slug(self) -> str:
        """Filesystem-safe directory name, e.g. ``TIC166527623``."""
        if self.tic is not None:
            return f"TIC{int(self.tic)}"
        if self.gaia_dr3 is not None:
            return f"GDR3{int(self.gaia_dr3)}"
        return "".join(c if c.isalnum() else "_" for c in (self.name or "target"))


@dataclass
class Candidate:
    """The signal under test.

    ``duration_h`` and ``depth_ppm`` are the values *as handed to the test*;
    a test that refines them reports the refined value in ``metrics``, never
    by mutating this.
    """

    period: float | None = None
    t0_btjd: float | None = None
    duration_h: float | None = None
    depth_ppm: float | None = None
    index: int = 0
    source: str | None = None  # where the ephemeris came from

    def duration_days(self, default: float | None = None) -> float | None:
        if self.duration_h is None:
            return default
        return self.duration_h / 24.0


@dataclass
class Dilution:
    """The contamination correction, recorded rather than silently applied.

    ``factor`` is the fraction of the flux in the aperture that belongs to the
    target (so 1.0 is uncontaminated).  This matches the ``dilution`` field
    written by tessquicklook, where higher is better.

    **Never divide a depth by it.**  The light curve owns the correction:
    tessquicklook de-dilutes as ``(f - (1 - c)) / c`` and PDCSAP is crowding
    corrected, so ``applied`` is True for both.  Treat the factor as a measure
    of how far to trust the depth (1/c amplifies errors in c).
    """

    factor: float | None = None
    source: str | None = None  # 'tessquicklook', 'CROWDSAP', 'prf_scene', 'none'
    applied: bool = False
    note: str | None = None


@dataclass
class KnownSignal:
    """Another signal already established on this star.

    Matching against these must be done by *transit overlap*, never by period
    proximity -- a period-proximity match fails at harmonics and subtracts a
    candidate from its own light curve (this cost 23% of known planets in the
    2026 search).  See :func:`petcare.ephem.overlaps_known`.
    """

    label: str
    period: float
    t0_btjd: float
    duration_h: float
    depth_ppm: float | None = None
    kind: str = "planet"  # planet | candidate | eb


@dataclass
class Human:
    """A human's disposition.  Written by hand, preserved across reruns."""

    disposition: str | None = None
    note: str | None = None
    by: str | None = None
    date: str | None = None


# --------------------------------------------------------------------------
# the result


@dataclass
class Result:
    """One test's answer about one candidate.

    Construct it, fill in ``metrics``/``noise``/``self_check``, set the status
    via :meth:`conclude` (or one of the shortcuts), and :meth:`write` it.
    """

    test: str
    version: str
    target: Target
    candidate: Candidate = field(default_factory=Candidate)

    status: str = Status.INCONCLUSIVE
    reason: str = "not run"

    metrics: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    self_check: dict = field(default_factory=lambda: {"passed": None, "detail": ""})
    plots: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    dilution: Dilution = field(default_factory=Dilution)
    known_signals: list = field(default_factory=list)
    human: Human = field(default_factory=Human)

    schema_version: str = SCHEMA_VERSION
    traceback: str | None = None

    # -- construction helpers ------------------------------------------------

    def set_noise(self, ppm: float | None, estimator: str, **extra) -> "Result":
        """Record the noise level a statistic was built on.

        Standing rule on this project: print the noise level beside any
        statistic built on an estimated noise level.  Calling this is how a
        test satisfies that rule, and several statuses below refuse to be set
        without it.
        """
        self.noise = {"white_ppm": _f(ppm), "estimator": estimator}
        self.noise.update({k: _jsonable(v) for k, v in extra.items()})
        return self

    def set_self_check(self, passed: bool | None, detail: str) -> "Result":
        self.self_check = {"passed": None if passed is None else bool(passed),
                           "detail": detail}
        return self

    def add_plot(self, path) -> "Result":
        if path is not None:
            self.plots.append(str(path))
        return self

    def set_provenance(self, lightcurve=None, n_points=None, **extra) -> "Result":
        """``lightcurve`` is mandatory for any test that touches photometry.

        A candidate can pass on one extraction and fail on another; a number
        with no light curve attached cannot be compared with anything.
        """
        self.provenance.update({
            "lightcurve": None if lightcurve is None else str(lightcurve),
            "n_points": None if n_points is None else int(n_points),
            "run_utc": utc_now(),
            "git_sha": _git_sha(),
        })
        self.provenance.update({k: _jsonable(v) for k, v in extra.items()})
        return self

    def set_dilution(self, factor=None, source=None, applied=False, note=None) -> "Result":
        self.dilution = Dilution(factor=_f(factor), source=source,
                                 applied=bool(applied), note=note)
        return self

    def set_known_signals(self, signals) -> "Result":
        self.known_signals = [
            s if isinstance(s, KnownSignal) else KnownSignal(**s) for s in (signals or [])
        ]
        return self

    # -- status shortcuts ----------------------------------------------------

    def conclude(self, status: str, reason: str) -> "Result":
        if status not in Status.ALL:
            raise ValueError(f"unknown status {status!r}")
        self.status = status
        self.reason = reason
        return self

    def passed(self, reason: str) -> "Result":
        return self.conclude(Status.PASS, reason)

    def flag(self, reason: str) -> "Result":
        return self.conclude(Status.FLAG, reason)

    def fail(self, reason: str) -> "Result":
        return self.conclude(Status.FAIL, reason)

    def inconclusive(self, reason: str) -> "Result":
        """The test could not be applied.  Not a pass and not a fail.

        This is the single most important state in the package.  Missing
        sectors, too few epochs, no pixels and a failed self-check are all
        *normal* at 30k targets and must land here rather than crashing or
        quietly passing.
        """
        return self.conclude(Status.INCONCLUSIVE, reason)

    def error(self, exc: BaseException) -> "Result":
        """Record a crash, with the traceback.  Never swallow it."""
        self.traceback = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return self.conclude(Status.ERROR, f"{type(exc).__name__}: {exc}")

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "test": self.test,
            "version": self.version,
            "schema_version": self.schema_version,
            "target": asdict(self.target),
            "candidate": asdict(self.candidate),
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
            "noise": self.noise,
            "self_check": self.self_check,
            "plots": self.plots,
            "provenance": self.provenance,
            "dilution": asdict(self.dilution),
            "known_signals": [asdict(s) for s in self.known_signals],
            "human": asdict(self.human),
        }
        if self.traceback:
            d["traceback"] = self.traceback
        return _jsonable(d)

    def filename(self) -> str:
        """``<test>.json``, or ``<test>_c<index>.json`` for multi-candidate stars."""
        if self.candidate.index:
            return f"{self.test}_c{self.candidate.index}.json"
        return f"{self.test}.json"

    def write(self, outdir) -> Path:
        """Write to ``<outdir>/<target.slug>/<filename>``, preserving ``human``.

        A rerun must not destroy a human's written disposition -- most of the
        value in the previous vetting round came from those notes sitting
        beside the automated flags.  If a record already exists at this path
        and carries a non-empty ``human`` block, it is carried forward unless
        this Result sets one of its own.
        """
        outdir = Path(outdir)
        target_dir = outdir / self.target.slug
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / self.filename()

        payload = self.to_dict()
        if path.exists():
            try:
                old = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                old = {}
            old_human = old.get("human") or {}
            if any(v is not None for v in old_human.values()) and not any(
                v is not None for v in payload["human"].values()
            ):
                payload["human"] = old_human
            # keep a shallow trail of what the previous run concluded
            payload["previous"] = {
                "status": old.get("status"),
                "reason": old.get("reason"),
                "run_utc": (old.get("provenance") or {}).get("run_utc"),
            }

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
        os.replace(tmp, path)  # atomic: a killed batch job leaves no half file
        return path

    # -- reading back --------------------------------------------------------

    @staticmethod
    def load(path) -> dict:
        return json.loads(Path(path).read_text())

    def summary_line(self) -> str:
        noise = self.noise.get("white_ppm")
        noise_s = f"  [noise {noise:.0f} ppm]" if isinstance(noise, (int, float)) else ""
        return f"{self.test:<16s} {self.status:<13s} {self.reason}{noise_s}"


def _f(x):
    """float() that maps None/NaN/non-numeric to None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# running a test safely


def run_guarded(fn, result: Result, *args, **kwargs) -> Result:
    """Call ``fn(result, *args, **kwargs)`` and turn any exception into ERROR.

    Rule 8: at 30k targets a crash in one test must not take down the run, and
    must not be mistaken for a pass.
    """
    try:
        out = fn(result, *args, **kwargs)
        return out if isinstance(out, Result) else result
    except Exception as exc:  # noqa: BLE001 - deliberately broad, recorded
        return result.error(exc)

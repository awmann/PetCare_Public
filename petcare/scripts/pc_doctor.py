"""pc_doctor -- is this installation able to do what you are about to ask of it?

Checks packages, paths, network services and credentials, and says what each
missing piece costs.  A problem found here is otherwise found mid-run, where an
unreachable archive or an absent light curve is easy to mistake for a result.

Secrets are never printed.  By default credentials are only checked for
presence; ``--check-credentials`` also tries them against the service.

    pc-doctor
    pc-doctor --offline
    pc-doctor --check-credentials
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .. import cache, io as pcio, models, stellar
from .pc_archival import (ENV_DEATHSTAR, ENV_DEATHSTAR_PY, _atlas_token,
                          _missing_credentials)
from .pc_rotation import TARS_ENV

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"

REQUIRED_PACKAGES = {"numpy": "numpy", "scipy": "scipy", "astropy": "astropy",
                     "matplotlib": "matplotlib", "pandas": "pandas",
                     "batman": "batman-package", "celerite2": "celerite2",
                     "astroquery": "astroquery", "certifi": "certifi",
                     "requests": "requests"}
OPTIONAL_PACKAGES = {"lightkurve": ("lightkurve", "pc-crossmatch cannot read MAST products"),
                     "pyarrow": ("pyarrow", "parquet catalogues and the TARS table cannot be read")}

#: What each service is for.  Any HTTP answer below 500 counts as reachable.
SERVICES = {
    "MAST": ("https://mast.stsci.edu/", "TIC stellar parameters"),
    "TESScut": ("https://mast.stsci.edu/tesscut/api/v0.1/sector?ra=0&dec=0",
                "pixel cutouts for tier 2"),
    "Gaia TAP": ("https://gea.esac.esa.int/tap-server/tap/capabilities",
                 "neighbours and astrometry"),
    "ExoFOP": ("https://exofop.ipac.caltech.edu/tess/", "TOI table and follow-up notes"),
    "LCO archive": ("https://archive-api.lco.global/", "public frames for pc-followup"),
    "Zenodo": (f"https://zenodo.org/api/records/{models.ZENODO_RECORD}",
               "downloading the Sonora grid"),
}


def _report(level, name, detail="") -> str:
    print(f"  [{level:^4s}] {name}" + (f": {detail}" if detail else ""))
    return level


def check_packages() -> list:
    out = []
    for module, package in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(module) is not None
        out.append(_report(OK if found else FAIL, package,
                           "" if found else f"pip install {package}"))
    for module, (package, cost) in OPTIONAL_PACKAGES.items():
        found = importlib.util.find_spec(module) is not None
        out.append(_report(OK if found else WARN, package,
                           "" if found else f"optional: {cost}"))
    return out


def check_paths() -> list:
    out = []
    try:
        root = cache.cache_dir()
        with tempfile.NamedTemporaryFile(dir=root):
            pass
        out.append(_report(OK, "cache", f"{root} (set ${cache.ENV_CACHE} to move it)"))
    except OSError as exc:
        out.append(_report(FAIL, "cache", f"not writable: {exc}"))

    roots = [r for r in pcio.lc_roots(None)]
    if not roots:
        out.append(_report(WARN, "light curves",
                           f"${pcio.ENV_LC_ROOTS} is unset; pass --lc or --lc-root per run"))
    for r in roots:
        out.append(_report(OK if r.exists() else FAIL, "light-curve root",
                           f"{r}" if r.exists() else f"{r} does not exist"))

    cat = os.environ.get(stellar.ENV_CATALOG)
    if not cat:
        out.append(_report(OK, "stellar parameters",
                           f"from the TIC; set ${stellar.ENV_CATALOG} to use your own"))
    else:
        out.append(_report(OK if Path(cat).expanduser().exists() else FAIL,
                           "stellar catalogue", cat))

    grid = models.grid_path()
    out.append(_report(OK if grid else WARN, "Sonora grid",
                       str(grid) if grid else "absent; pc-companion --fetch-models "
                       f"downloads it (0.8 MB), or set ${models.ENV_PATH}"))

    tars = os.environ.get(TARS_ENV)
    if tars:
        out.append(_report(OK if Path(tars).expanduser().exists() else FAIL, "TARS table", tars))
    return out


def check_network() -> list:
    out = []
    cache.ensure_certificates()
    for name, (url, purpose) in SERVICES.items():
        try:
            urllib.request.urlopen(url, timeout=15).close()
            out.append(_report(OK, name))
        except urllib.error.HTTPError as exc:
            level = WARN if exc.code >= 500 else OK
            detail = f"HTTP {exc.code}: {purpose} may fail" if exc.code >= 500 else ""
            out.append(_report(level, name, detail))
        except Exception as exc:  # noqa: BLE001 - DNS, timeout, certificates
            out.append(_report(WARN, name, f"{type(exc).__name__}: {purpose} will fail"))
    return out


def check_archival(verify: bool) -> list:
    """The optional ground-based tier: DEATHSTAR, ATLAS and ZTF."""
    out = []
    root = os.environ.get(ENV_DEATHSTAR)
    if not root:
        out.append(_report(SKIP, "DEATHSTAR", f"${ENV_DEATHSTAR} is unset; only "
                           "pc-archival needs it"))
    else:
        p = Path(root).expanduser()
        source = "".join(f.read_text(errors="ignore") for f in p.glob("*.py")) if p.is_dir() else ""
        if not p.is_dir():
            out.append(_report(FAIL, "DEATHSTAR", f"{p} does not exist"))
        elif "is_ephemeris_scan" not in source:
            out.append(_report(FAIL, "DEATHSTAR", "this checkout lacks the ephemeris "
                               "scan; pc-archival needs the awmann/DEATHSTAR fork"))
        else:
            out.append(_report(OK, "DEATHSTAR", str(p)))
        py = os.environ.get(ENV_DEATHSTAR_PY)
        if not py:
            out.append(_report(WARN, "DEATHSTAR python",
                               f"${ENV_DEATHSTAR_PY} is unset; it needs its own environment"))
        else:
            out.append(_report(OK if Path(py).expanduser().exists() else FAIL,
                               "DEATHSTAR python", py))

    has_atlas = bool(_atlas_token() or (os.environ.get("ATLAS_USERNAME")
                                        and os.environ.get("ATLAS_PASSWORD")))
    if not has_atlas:
        out.append(_report(SKIP, "ATLAS login", "not set; needed by pc-archival in the south"))
    elif verify:
        problem = _missing_credentials(True)
        out.append(_report(WARN if problem else OK, "ATLAS login", problem or "accepted"))
    else:
        out.append(_report(OK, "ATLAS login", "set, not tried; --check-credentials tries it"))

    problem = _missing_credentials(False)
    out.append(_report(SKIP if problem else OK, "ZTF (IRSA) login",
                       "not set; needed by pc-archival in the north" if problem else "found"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pc-doctor", description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="skip the network checks")
    ap.add_argument("--check-credentials", action="store_true",
                    help="try credentials against the service, not just their presence")
    args = ap.parse_args(argv)

    levels = []
    print("packages")
    levels += check_packages()
    print("paths")
    levels += check_paths()
    print("network")
    levels += [_report(SKIP, "all services", "--offline")] if args.offline else check_network()
    print("ground-based archives (optional)")
    levels += check_archival(args.check_credentials)

    n_fail, n_warn = levels.count(FAIL), levels.count(WARN)
    print(f"\n{n_fail} problem(s), {n_warn} warning(s)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

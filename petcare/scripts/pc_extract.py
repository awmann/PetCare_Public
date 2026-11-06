"""pc_extract -- make the light curves PetCare expects, with tessquicklook.

Runs `tessquicklook <https://github.com/awmann/tessquicklook>`_, which fits the
systematics and the stellar variability simultaneously, and writes one ``.npz``
per sector.  Prefer this to tessquicklook's own CSV: the CSV has no quality mask,
so cadences the systematics fit rejected come back, which can turn a planet's
joint GP fit from a clear detection into a refutation.

    pc-extract --tic 410214986 --out lightcurves
    petcare --candidates candidates.csv --lc-root lightcurves

Downloads can run to several GB per star (5 GB for a star with ten 20 s sectors);
set ``$TESSQUICKLOOK_CACHE`` to move them from ``~/.tessquicklook``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

#: The settings our validation used, written out so a change of tessquicklook
#: default cannot quietly change the light curves.
EXTRACTION = dict(order=2, torder=5, variability_basis="spline",
                  allowscatteredlight=False, usecbv=True, contamination=True,
                  discard_quaternion_fits=True)


def sector_payload(sec: dict, tic: int) -> dict:
    """The arrays PetCare reads, from one entry of tessquicklook's ``sectors``."""
    from tessquicklook.idlcompat import point_to_point_scatter

    if "best" in sec:                           # FFI: our own aperture photometry
        best, kind = sec["best"], "fcirc" if sec["usecirc"] else "fpsf"
        short = "circ" if sec["usecirc"] else "psf"
        data = sec["data"]
        t, fcor = data["t"], data[f"{kind}cor"][:, best]
        err = sec["raw"][f"err_{short}_photon"][:, best]
        mask = data[f"{kind}_fit_good_mask"][best]
        dilution = float(sec[f"{short}contam"][best])
        aperture_id = f"{short}_{best}"
        aperture_mask = sec[f"{short}masks"][best]
        path = "ffi"
    else:                                       # SPOC short cadence
        t, fcor, err = sec["t"], sec["fcor"], sec["err_photon"]
        mask = sec["fit_good_mask"]
        dilution = float(sec["crowdsap"])
        aperture_id, aperture_mask = "spoc_sap", np.array([])
        path = "20s" if sec["exptime"] < 60 else "120s"
    return {
        "t": t, "fcor": fcor, "err_photon": err,
        "err_empirical": np.float64(point_to_point_scatter(fcor)),
        "cadence_s": np.full(np.size(t), float(sec["exptime"])),
        "fit_good_mask": mask, "dilution": dilution,
        "aperture_id": aperture_id, "aperture_mask": aperture_mask,
        "camera": sec["camera"], "ccd": sec["ccd"], "cadence_path": path,
        "tic_id": tic, "sector": sec["sector"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pc-extract", description=__doc__.splitlines()[0])
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--out", required=True, help="directory; use it as --lc-root afterwards")
    ap.add_argument("--sectors", type=int, nargs="+", default=None)
    ap.add_argument("--cadence", default="auto",
                    help="tessquicklook's choice of data: auto, 20s, 120s, ffi, spoc")
    args = ap.parse_args(argv)

    try:
        from tessquicklook import quicklooktess
    except ImportError:
        print('tessquicklook is not installed: pip install "petcare[extract]"')
        return 2

    result = quicklooktess(args.tic, cadence=args.cadence, only_sectors=args.sectors,
                           verbose=True, **EXTRACTION)
    outdir = Path(args.out) / f"tic{args.tic:010d}"
    outdir.mkdir(parents=True, exist_ok=True)
    for sec in result["sectors"]:
        path = outdir / f"tic{args.tic:010d}_s{sec['sector']:04d}.npz"
        np.savez_compressed(path, **sector_payload(sec, args.tic))
        print(f"wrote {path}")
    for failure in result.get("failures") or []:
        print(f"could not extract {failure}")
    return 0 if result["sectors"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""pc_archival -- ground-based archival photometry, via DEATHSTAR.

A thin driver for DEATHSTAR (Ross et al. 2024), which stays where it is and is
**called as a subprocess, never vendored**.  It is a separate project with its
own dependencies -- python 3.9 and numpy < 2, which this package cannot
use -- and its own repository.  The only PetCare file that knows its API is
:mod:`petcare.deathstar_runner`, which DEATHSTAR's interpreter runs, not ours.

**What this test is for.**  TESS cannot resolve a star 30 arcseconds away; a
30-cm telescope can.  So for a signal that may belong to a neighbour, archival
ground-based photometry measures the neighbour *directly* and settles it.  In
an earlier search this was the main tool for killing false positives, and the
thing that validated the difference images.

**The depth that matters is the neighbour's, not the target's.**  This is the
point of the whole test and the first version of this file got it backwards.
A 2000 ppm dip in a TESS aperture is undetectable from the ground -- but if
its source is a neighbour contributing 2% of the flux in that aperture, the
neighbour is eclipsing by ``2000 / 0.02`` = 10%, which any survey sees easily.
So the gate is the **largest depth any viable source would have to show**,
taken from ``pc_neighbours``'s ``required_eclipse_depth``, and a candidate is
worth running whenever *either* the target or some neighbour clears the floor.
Gating on the candidate's own depth threw away exactly the cases this test
exists for.

**Point it at the suspect.**  When ``pc_centroid`` has already measured an
offset toward a particular star -- which is the usual situation, and the usual
reason to reach for this test -- that star is passed to DEATHSTAR as
``signal_tic_ID`` so the comparison plot is made against the right neighbour.
``--neighbour`` overrides it.

Both inputs are read from the output directory the way ``--use-fit`` reads the
fit, so the ordinary invocation is::

    python -m petcare.scripts.pc_archival --tic 166527623 --use-fit \\
        --deathstar ~/DEATHSTAR --deathstar-python ~/.venvs/deathstar/bin/python

Installation this expects: the fork ``awmann/DEATHSTAR`` (the ATLAS /
southern-declination path was non-functional upstream and is fixed there), in
its own environment.  ATLAS needs ``$ATLAS_USERNAME``/``$ATLAS_PASSWORD`` (or
``$ATLAS_TOKEN``); ZTF needs an IRSA login for ``ztfquery``.  When any of that
is missing the test returns INCONCLUSIVE and says precisely what -- it does
not quietly pass, and it does not try to do the job itself with something
else.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import cli
from ..result import Result

TEST = "archival"
VERSION = "0.2.1"

ENV_DEATHSTAR = "PETCARE_DEATHSTAR"
ENV_DEATHSTAR_PY = "PETCARE_DEATHSTAR_PYTHON"

#: Depth below which ground-based archival photometry cannot say anything.
#: ATLAS and ZTF reach a few mmag per epoch at best, and correlated
#: systematics at the few-hundred-ppm level do not bin down.  Applied to the
#: depth the *source* would show, not to the candidate's TESS depth.
#: PROVISIONAL: 3000 ppm is the conventional few-mmag figure; it should be
#: replaced with a measured one, per-survey and per-magnitude.
GROUND_DEPTH_FLOOR_PPM = 3000.0

#: Declination below which ATLAS is used instead of ZTF.  ZTF is a northern
#: survey; the fork exists largely because the ATLAS path did not work.
ATLAS_DECLINATION_DEG = -28.0

#: Above this many field stars flagged at once, the ephemeris scan is seeing a
#: bad night rather than a real eclipse -- see the warning in
#: ``Plotting.ephemeris_scan``'s own docstring.
MAX_PLAUSIBLE_FLAGS = 4

#: When more than that many stars flag, one of them is still believable if its
#: photometry is this much better than the others' and it is this much more
#: significant.  PROVISIONAL, set from the one measured case (a neighbour with
#: 8.4% scatter against a median 21%, 8.6 sigma against a best-other 7.3).
DOMINANCE_SCATTER_RATIO = 0.6
DOMINANCE_SIGMA_RATIO = 1.1

#: Beyond this a star is not in the TESS aperture and cannot be the source of
#: the signal, whatever it does photometrically.  Four TESS pixels; the
#: ground-based field of view is far wider, so an eclipsing star at 128 arcsec
#: is a coincidence, not an explanation.
MAX_CONTAMINATION_ARCSEC = 84.0

#: Default ATLAS baseline, in years back from today.  ATLAS has been running
#: since 2015, but every frame is a download and a short-period candidate has
#: hundreds of transits in three years.  Widen with --atlas-years when the
#: period is long or the epoch coverage comes back thin.
ATLAS_DEFAULT_YEARS = 3.0

#: The runner, shipped with PetCare and executed by DEATHSTAR's interpreter.
RUNNER = Path(__file__).resolve().parent.parent / "deathstar_runner.py"


def add_arguments(ap):
    g = ap.add_argument_group("archival")
    g.add_argument("--deathstar", default=None,
                   help=f"path to the DEATHSTAR checkout; else ${ENV_DEATHSTAR}")
    g.add_argument("--deathstar-python", default=None,
                   help=f"interpreter to run it with (its environment is "
                        f"separate from this one: python 3.9, numpy < 2); "
                        f"else ${ENV_DEATHSTAR_PY}, else the current one")
    g.add_argument("--entry", default=None,
                   help="override the runner script (default: the adapter "
                        "shipped with PetCare)")
    g.add_argument("--neighbour", type=int, default=None,
                   help="TIC of the star to compare the target against; by "
                        "default the one pc_centroid's offset points at")
    g.add_argument("--survey", choices=("auto", "atlas", "ztf"), default="auto",
                   help=f"auto picks ATLAS below dec {ATLAS_DECLINATION_DEG:g}")
    g.add_argument("--atlas-years", type=float, default=ATLAS_DEFAULT_YEARS,
                   help="how far back to fetch ATLAS frames, in years. The "
                        "download is the whole cost of this test, and a few "
                        "years already contains hundreds of transits at a "
                        "short period.")
    g.add_argument("--mjd-min", type=float, default=None,
                   help="explicit ATLAS window start (overrides --atlas-years)")
    g.add_argument("--mjd-max", type=float, default=None)
    g.add_argument("--reference-image", action="store_true",
                   help="also draw DEATHSTAR's reference-field image; needs "
                        "the raw survey FITS still on disk, and takes the "
                        "whole run down if any of them is truncated")
    g.add_argument("--overwrite", action="store_true",
                   help="re-extract light curves even if DEATHSTAR has them "
                        "cached for this target (the download is the cost)")
    g.add_argument("--depth-floor", type=float, default=GROUND_DEPTH_FLOOR_PPM,
                   help="minimum source depth in ppm worth running on")
    g.add_argument("--timeout", type=float, default=7200.0)
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r

    floor = float(getattr(a, "depth_floor", GROUND_DEPTH_FLOOR_PPM))
    r.set_noise(None, "not applicable: this test reports DEATHSTAR's own output",
                ground_depth_floor_ppm=floor)

    # ---- what depth would the source have to show? ------------------------
    target_depth = ctx.candidate.depth_ppm
    suspect = getattr(a, "neighbour", None)
    reach = _reachable_depth(ctx, target_depth, suspect)
    r.metrics.update(reach)
    r.metrics["ground_depth_floor_ppm"] = floor
    suspect = suspect or reach.get("suspect_tic")

    best = reach.get("best_source_depth_ppm")
    if best is None:
        r.set_self_check(False, "no depth for the target or any neighbour")
        return r.inconclusive(
            "no depth to work from: run pc_fit (for the target's depth) and "
            "pc_neighbours (for the depth each neighbour would need), then "
            "re-run with --use-fit. Without one of them there is no way to "
            "know whether a ground-based survey could see this at all.")

    if best < floor:
        r.set_self_check(
            True, f"deepest plausible source is {best:.0f} ppm, below the "
                  f"{floor:.0f} ppm floor for ground-based archival photometry")
        return r.inconclusive(
            f"nothing here is deep enough for ATLAS or ZTF: the target would "
            f"show {_ppm(target_depth)} and the most diluted viable neighbour "
            f"{_ppm(reach.get('max_neighbour_depth_ppm'))}, against a "
            f"{floor:.0f} ppm floor. Not a statement about the candidate -- "
            f"this test cannot be applied.")

    # ---- is DEATHSTAR installed and usable? -------------------------------
    root = getattr(a, "deathstar", None) or os.environ.get(ENV_DEATHSTAR)
    if not root:
        r.set_self_check(False, "DEATHSTAR path not configured")
        return r.inconclusive(
            f"DEATHSTAR is not configured: pass --deathstar or set "
            f"${ENV_DEATHSTAR}. Expected the fork awmann/DEATHSTAR, which "
            f"fixes the non-functional ATLAS / southern-declination path.")
    root = Path(root).expanduser()
    if not root.exists():
        r.set_self_check(False, f"{root} does not exist")
        return r.inconclusive(
            f"DEATHSTAR is configured as {root}, which does not exist. Clone "
            f"awmann/DEATHSTAR there, or point --deathstar somewhere else.")
    if not (root / "DEATHSTAR.py").exists():
        r.set_self_check(False, "no DEATHSTAR.py in the checkout")
        return r.inconclusive(
            f"{root} exists but has no DEATHSTAR.py in it, so it is not a "
            f"DEATHSTAR checkout.")

    python = (getattr(a, "deathstar_python", None)
              or os.environ.get(ENV_DEATHSTAR_PY) or sys.executable)
    if not shutil.which(python) and not Path(python).exists():
        r.set_self_check(False, f"interpreter {python} not found")
        return r.inconclusive(
            f"the interpreter {python} does not exist. DEATHSTAR needs its "
            f"own environment (python 3.9, numpy < 2); set "
            f"${ENV_DEATHSTAR_PY} to it.")

    entry = Path(getattr(a, "entry", None) or RUNNER).expanduser()
    if not entry.exists():
        r.set_self_check(False, f"runner {entry} not found")
        return r.inconclusive(f"the runner script {entry} does not exist")

    # ---- which survey, and do we have credentials for it? -----------------
    star = ctx.star
    ra = ctx.target.ra if ctx.target.ra is not None else star.ra
    dec = ctx.target.dec if ctx.target.dec is not None else star.dec
    survey = getattr(a, "survey", "auto")
    if survey == "auto":
        if dec is None:
            r.set_self_check(False, "no declination")
            return r.inconclusive(
                "no declination for this target, so there is no way to choose "
                "between ATLAS and ZTF; pass --survey")
        survey = "atlas" if dec < ATLAS_DECLINATION_DEG else "ztf"
    is_atlas = survey == "atlas"
    r.metrics["survey"] = survey
    r.metrics["declination"] = dec

    # Credentials are only needed to *download*.  A target DEATHSTAR has
    # already extracted can be re-run -- re-plotted, re-scanned at a new
    # ephemeris -- with no network at all, and refusing to do that because an
    # API login is unset would be silly.
    cached = (root / "TICs" / str(ctx.target.tic)
              / f"All_{ctx.target.tic}_Star_Data.csv").exists()
    r.metrics["deathstar_cached"] = cached
    missing = None if (cached and not getattr(a, "overwrite", False)) \
        else _missing_credentials(is_atlas)
    if missing:
        r.set_self_check(False, f"{survey.upper()} credentials missing")
        return r.inconclusive(
            f"this target is a {survey.upper()} one (dec = "
            f"{dec:.2f}) and {missing}. The archival check cannot run without "
            f"it; this is a configuration gap, not a verdict.")

    # ---- run it -----------------------------------------------------------
    # ATLAS window.  MJD = JD - 2400000.5, and BTJD = BJD - 2457000, so a
    # BTJD epoch is MJD 56999.5 + BTJD.
    import time as _time
    mjd_now = _time.time() / 86400.0 + 40587.0
    mjd_max = getattr(a, "mjd_max", None) or mjd_now
    mjd_min = getattr(a, "mjd_min", None)
    if mjd_min is None:
        mjd_min = mjd_max - 365.25 * float(
            getattr(a, "atlas_years", ATLAS_DEFAULT_YEARS))
    r.metrics["atlas_window_mjd"] = [mjd_min, mjd_max]

    outdir = ctx.out / ctx.target.slug / "deathstar"
    outdir.mkdir(parents=True, exist_ok=True)
    spec_path = outdir / "spec.json"
    result_path = outdir / "deathstar_result.json"

    spec = {
        "tic": int(ctx.target.tic),
        "deathstar_root": str(root),
        "is_ATLAS": is_atlas,
        "is_overwrite": bool(getattr(a, "overwrite", False)),
        "is_ephemeris_scan": True,
        # Off by default: this plot is the one DEATHSTAR step that needs the
        # raw FITS frames still on disk, and they are routinely deleted after
        # extraction to save space (the fork added is_reference_image for
        # exactly that). A truncated or missing frame takes the whole run
        # down with "No SIMPLE card found", which is a plot failing, not an
        # answer changing.
        "is_reference_image": bool(getattr(a, "reference_image", False)),
        "signal_tic_ID": int(suspect) if suspect else 0,
        "result_json": str(result_path),
        "mjd_min": mjd_min,
        "mjd_max": mjd_max,
        "manual_ephemeris": {
            "ra": ra,
            "dec": dec,
            "period": float(ctx.candidate.period),
            # DEATHSTAR works in BJD; everything in PetCare is BTJD.
            "epoch": float(ctx.candidate.t0_btjd) + 2457000.0,
            "depth": float(target_depth) if target_depth else float(best),
            "Tmag": reach.get("target_tmag") or getattr(star, "tmag", None),
            "duration": ctx.candidate.duration_h,
        },
    }
    spec["manual_ephemeris"] = {k: v for k, v in spec["manual_ephemeris"].items()
                                if v is not None}
    # the ATLAS fetch must give up before this process kills it: a SIGKILLed
    # fetch cannot release its image task, and the account has only 5
    spec["fetch_timeout_s"] = 0.75 * float(getattr(a, "timeout", 7200.0))
    spec_path.write_text(json.dumps(spec, indent=1))

    cmd = [str(python), str(entry), str(spec_path)]
    r.set_provenance(lightcurve=None, command=" ".join(cmd),
                     deathstar_root=str(root),
                     deathstar_git_sha=_git_sha(root),
                     deathstar_python=str(python))

    # DEATHSTAR's atlas_fetch reads $ATLAS_TOKEN specifically; normalise
    # whichever variable the token actually arrived in.
    env = dict(os.environ)
    token = _atlas_token()
    if token:
        env["ATLAS_TOKEN"] = token

    try:
        run_started = _time.time() - 1
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=float(getattr(a, "timeout", 7200.0)),
                              cwd=str(root), env=env)
    except subprocess.TimeoutExpired:
        r.set_self_check(False, "DEATHSTAR timed out")
        return r.inconclusive(
            f"DEATHSTAR did not finish within "
            f"{getattr(a, 'timeout', 7200.0):.0f} s. Image downloads are the "
            f"slow part; raise --timeout or pre-fetch them.")
    except OSError as exc:
        return r.inconclusive(f"could not start DEATHSTAR: {exc}")

    r.metrics["returncode"] = proc.returncode
    r.metrics["stdout_tail"] = proc.stdout[-4000:]
    r.metrics["stderr_tail"] = proc.stderr[-4000:]

    payload = {}
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    r.metrics["deathstar"] = {k: v for k, v in payload.items()
                              if k != "traceback"}

    if not payload.get("ok"):
        r.set_self_check(False, "DEATHSTAR did not complete")
        return r.inconclusive(
            f"DEATHSTAR did not complete: "
            f"{payload.get('error') or f'exit code {proc.returncode}'}. Its "
            f"output is in metrics.stdout_tail / stderr_tail; this is a tool "
            f"failure, not a verdict on the candidate.")

    # The folder is per star and outlives the run: anything older is another candidate's.
    for p in sorted(Path(root, "TICs", str(ctx.target.tic)).glob("*.png")):
        if p.stat().st_mtime >= run_started and len(r.plots) < 8:
            r.add_plot(str(p))

    # PetCare's own scan, which is the one the verdict is read from; see
    # _robust_scan for why DEATHSTAR's is kept only as a cross-check.
    try:
        scan = _robust_scan(root, ctx.target.tic, float(ctx.candidate.period),
                            float(ctx.candidate.t0_btjd) + 2457000.0,
                            (ctx.candidate.duration_h or 3.0) / 24.0)
    except Exception as exc:  # noqa: BLE001
        scan = None
        r.provenance.setdefault("notes", []).append(
            f"robust scan failed: {exc!r}; fell back to DEATHSTAR's own")
    r.metrics["deathstar_own_scan"] = payload.get("ephemeris_scan")
    return _verdict(r, payload, ctx, suspect, reach, survey, scan)


# --------------------------------------------------------------------------


#: Robust-scan thresholds.  PROVISIONAL, but chosen against a measured case:
#: in one field the true source came in at 20.5% and 8.6 sigma while three
#: field stars with 2-3x worse photometry reached 4-7 sigma on noise, so the
#: significance bar alone does not separate them and the scatter has to be
#: part of the judgement.
SCAN_SIGMA = 5.0
SCAN_MIN_DEPTH_FRAC = 0.03
SCAN_MIN_IN_TRANSIT = 5
SCAN_MIN_OUT = 50

#: Per-point scatter above which a star's photometry cannot support an
#: eclipse claim, however significant the dip looks.  Measured by injection:
#: eclipses were put on field stars of known scatter in a real ATLAS
#: extraction and both this scan and an independent one were asked to rank
#: them.  On a star with 10% scatter a 20% eclipse ranks first and a 5% one
#: ranks sixth; on a star with 47% scatter nothing below 40% is found at all.
#: Above this ceiling the ranking is noise, which is why two runs over
#: different frame sets of the same field name different "sources".
SCAN_MAX_SCATTER_PCT = 15.0

#: And the depth below which even good photometry does not rank the right
#: star first.  From the same injection: 20% recovers, 10% ranks third, 5%
#: ranks sixth.  A detection under this is reported but not leant on.
SCAN_RELIABLE_DEPTH_PCT = 20.0

#: A non-detection on the star pc_centroid named counts against the centroid
#: only if that star's own photometry would have shown the eclipse it needs at
#: this many sigma.
SUSPECT_REACH_SIGMA = 5.0


def _robust_scan(root, tic, period, t0_bjd, dur_days):
    """Fold every extracted field star at the ephemeris, robustly.

    PetCare computes its own scan rather than taking DEATHSTAR's, for two
    reasons measured on one field:

    * **Robust statistics.**  DEATHSTAR's ``ephemeris_scan`` uses the mean and
      the standard deviation.  On ground-based photometry with outliers that
      inflates the noise by a factor of three: the true source -- a 20.5%
      eclipse -- came out at 0.4 sigma and was not reported at all, where
      medians and a MAD give 8.6 sigma.
    * **Filters.**  ATLAS observes in two bands and DEATHSTAR's extraction
      pools them.  A star's flux differs between them, so mixing o and c
      turned 9% per-point scatter into 23%.  This scans each band separately
      and reports the better one.

    Photometry is differential against the field: every frame is divided by
    the median of the stars present in most frames, which removes transparency
    and airmass without choosing comparison stars by hand.
    """
    import csv as _csv
    path = Path(root) / "TICs" / str(tic) / f"All_{tic}_Star_Data.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(_csv.DictReader(fh))
    if len(rows) < SCAN_MIN_OUT:
        return None

    cols = [c for c in (rows[0] or {}) if c and c.endswith("_brightness")]
    if not cols:
        return None
    # ATLAS extractions call it "time", ZTF ones "jd_time"; without the second
    # every ZTF scan came back empty and the verdict fell to DEATHSTAR's own
    tcol = "time" if "time" in (rows[0] or {}) else "jd_time"
    t = np.array([_num(r.get(tcol)) for r in rows], float)
    band = np.array([str(r.get("image_filter") or "") for r in rows])
    flux = np.array([[_num(r.get(c)) for r in rows] for c in cols], float)
    flux[~np.isfinite(flux)] = np.nan
    flux[flux <= 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        logf = np.log10(flux)

    present = np.isfinite(logf).mean(axis=1) > 0.8
    if present.sum() < 3:
        return None
    ens = np.nanmedian(logf[present], axis=0)

    # separations, so a flagged star can be placed on the sky without
    # opening another file
    seps = {}
    cone_path = Path(root) / "TICs" / str(tic) / f"{tic}_Cone.csv"
    if cone_path.exists():
        with cone_path.open() as fh:
            for row in _csv.DictReader(fh):
                key = f"{row.get('ID')}_distance_to_real_target"
                if key in (rows[0] or {}):
                    vals = [_num(r.get(key)) for r in rows]
                    vals = [v for v in vals if np.isfinite(v)]
                    if vals:
                        seps[str(row.get("ID"))] = round(float(np.median(vals)), 1)

    ph = ((t - t0_bjd + 0.5 * period) % period) - 0.5 * period
    out = []
    for i, col in enumerate(cols):
        star = col[:-len("_brightness")]
        best = None
        for b in sorted(set(band)):
            y = logf[i] - ens
            sel = np.isfinite(y) & (band == b)
            if sel.sum() < SCAN_MIN_OUT:
                continue
            intr = sel & (np.abs(ph) < 0.4 * dur_days)
            oot = sel & (np.abs(ph) > 1.0 * dur_days)
            if intr.sum() < SCAN_MIN_IN_TRANSIT or oot.sum() < SCAN_MIN_OUT:
                continue
            base = float(np.median(y[oot]))
            mad = float(1.4826 * np.median(np.abs(y[oot] - base)))
            if not np.isfinite(mad) or mad <= 0:
                continue
            drop = base - float(np.median(y[intr]))
            rec = {"tic_ID": star,
                   "is_target": str(star) == str(tic),
                   "separation_arcsec": seps.get(str(star)),
                   "filter": b,
                   "n_in": int(intr.sum()), "n_out": int(oot.sum()),
                   "frac_depth_pct": round(float(1 - 10 ** (-drop)) * 100, 1),
                   "scatter_pct": round(float(1 - 10 ** (-mad)) * 100, 1),
                   "sig": round(float(drop / (mad / np.sqrt(intr.sum()))), 1)}
            if best is None or rec["sig"] > best["sig"]:
                best = rec
        if best is not None:
            out.append(best)
    out.sort(key=lambda x: -x["sig"])
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _verdict(r, payload, ctx, suspect, reach, survey, scan=None) -> Result:
    """Read the ephemeris scan: who in the field eclipsed at this ephemeris?"""
    if scan is not None:
        r.metrics["scan"] = scan[:30]
        r.metrics["scan_source"] = "petcare robust scan"
        flags = [f for f in scan
                 if f["sig"] >= SCAN_SIGMA
                 and f["frac_depth_pct"] >= SCAN_MIN_DEPTH_FRAC * 100]
        # Drop stars whose photometry cannot support the claim.  Significance
        # alone does not separate a real eclipse from a faint star wandering:
        # a 90% "eclipse" at 30% per-point scatter is the latter every time.
        noisy = [f for f in flags
                 if (f.get("scatter_pct") or 0) > SCAN_MAX_SCATTER_PCT]
        flags = [f for f in flags if f not in noisy]
        if noisy:
            r.metrics["flagged_but_too_noisy"] = noisy[:10]
            r.provenance.setdefault("notes", []).append(
                f"{len(noisy)} star(s) cleared the significance bar on "
                f"photometry too noisy to support it (scatter above "
                f"{SCAN_MAX_SCATTER_PCT:g}%); see flagged_but_too_noisy")
    else:
        r.metrics["scan_source"] = "DEATHSTAR ephemeris_scan"
        flags = payload.get("ephemeris_scan")
    if flags is None:
        r.set_self_check(False, "no ephemeris scan was produced")
        return r.inconclusive(
            "DEATHSTAR ran but wrote no ephemeris scan, so there is nothing "
            "to read a verdict from. Its plots are attached; read them.")

    on_target = [f for f in flags if f.get("is_target")]
    others = [f for f in flags if not f.get("is_target")]
    r.metrics["ephemeris_scan_flags"] = flags
    r.metrics["n_field_stars_flagged"] = len(others)
    r.metrics["target_flagged"] = bool(on_target)

    checks = [f"{survey.upper()}; {len(flags)} star(s) flagged by the "
              f"ephemeris scan"]

    # A whole field "eclipsing" at once is one bad night, not a discovery --
    # unless one star stands clearly above the rest.  In one field four
    # stars cleared the significance bar and only the true source had good
    # photometry: 20.5% at 8.6 sigma with 8.4% scatter, against 18-41% at
    # 4-7 sigma with 15-24% scatter.  Counting flags alone throws that away,
    # so the scatter decides.
    dominant = None
    if len(others) > MAX_PLAUSIBLE_FLAGS and others:
        ranked = sorted(others, key=lambda f: -(f.get("sig") or 0))
        top, rest = ranked[0], ranked[1:]
        top_scat = top.get("scatter_pct")
        rest_scat = [f.get("scatter_pct") for f in rest
                     if f.get("scatter_pct") is not None]
        if (top_scat is not None and rest_scat
                and top_scat <= DOMINANCE_SCATTER_RATIO * float(np.median(rest_scat))
                and (top.get("sig") or 0) >= DOMINANCE_SIGMA_RATIO
                * max((f.get("sig") or 0) for f in rest)):
            dominant = top

    if len(others) > MAX_PLAUSIBLE_FLAGS and dominant is None:
        checks.append(f"{len(others)} field stars flagged together with no "
                      f"one of them standing out, which is the signature of a "
                      f"common-mode systematic")
        r.set_self_check(False, "; ".join(checks))
        return r.inconclusive(
            f"{len(others)} field stars show an in-transit drop at this "
            f"ephemeris simultaneously and none is cleaner than the rest. "
            f"That is a bad night's frames hitting every star at once, not "
            f"{len(others)} eclipses; DEATHSTAR's own scan documents this "
            f"failure mode. Read the plots.")
    if dominant is not None:
        checks.append(f"{len(others)} stars flagged, but TIC "
                      f"{dominant['tic_ID']} dominates "
                      f"({dominant.get('sig')} sigma at "
                      f"{dominant.get('scatter_pct')}% scatter)")
        others = [dominant]
        named = [f for f in others
                 if suspect and str(f.get("tic_ID")) == str(int(suspect))]
    r.set_self_check(True, "; ".join(checks))

    if dominant is None:
        named = [f for f in others
                 if suspect and str(f.get("tic_ID")) == str(int(suspect))]
    where = ""
    if suspect:
        where = (f" pc_centroid's offset pointed at TIC {int(suspect)}"
                 if not getattr(ctx.args, "neighbour", None)
                 else f" TIC {int(suspect)} was named with --neighbour")

    # A FAIL names the star that eclipsed, so the scan has to be able to rank it:
    # under SCAN_RELIABLE_DEPTH_PCT the right star does not come first even on good
    # photometry, and above SCAN_MAX_SCATTER_PCT it is a faint star wandering.
    # Anything less falls through to the FLAG below.
    named = [f for f in named if (f.get("frac_depth_pct") or 0) >= SCAN_RELIABLE_DEPTH_PCT
             and (f.get("scatter_pct") or 0) <= SCAN_MAX_SCATTER_PCT]
    if named and not on_target:
        f = named[0]
        req = reach.get("suspect_required_depth_ppm")
        agrees = ""
        if req and np.isfinite(req):
            agrees = (f" pc_neighbours predicted this star would need a "
                      f"{req / 1e4:.0f}% eclipse to produce the TESS signal, "
                      f"and it shows {f.get('frac_depth_pct')}%.")
        return r.fail(
            f"the eclipse is on TIC {f['tic_ID']}, not the target: "
            f"{survey.upper()} photometry that resolves the two shows a "
            f"{f.get('frac_depth_pct')}% drop on the neighbour at "
            f"{f.get('sig')} sigma at this ephemeris "
            f"({f.get('n_in')} in-transit points, {f.get('scatter_pct')}% "
            f"per-point scatter in {f.get('filter')}), and nothing on the "
            f"target.{where}.{agrees} The candidate belongs to that star.")

    # A star too far away to be in the TESS aperture cannot be the source,
    # however convincingly it eclipses -- it is a coincidental variable in a
    # wide ground-based field of view.  One scan found an 87%
    # eclipse 128 arcsec out, six TESS pixels away, which contributes nothing
    # to the aperture.
    too_far = [f for f in others
               if (f.get("separation_arcsec") or 0) > MAX_CONTAMINATION_ARCSEC]
    others = [f for f in others if f not in too_far]
    if too_far:
        r.metrics["flagged_but_too_far"] = too_far
        checks_note = (f"{len(too_far)} flagged star(s) are beyond "
                       f"{MAX_CONTAMINATION_ARCSEC:.0f} arcsec and cannot be "
                       f"in the TESS aperture")
        r.provenance.setdefault("notes", []).append(checks_note)

    if others and not on_target:
        f = max(others, key=lambda x: x.get("sig") or 0)
        clash = ""
        if suspect and str(f.get("tic_ID")) != str(int(suspect)):
            clash = (f" This is NOT the star pc_centroid pointed at "
                     f"(TIC {int(suspect)}), so the difference image and the "
                     f"ground-based photometry disagree about which "
                     f"neighbour it is -- which is why this flags rather than "
                     f"fails.")
        return r.flag(
            f"a field star shows the eclipse and the target does not: TIC "
            f"{f['tic_ID']} at {f.get('sig')} sigma, {f.get('frac_depth_pct')}% "
            f"deep, {f.get('separation_arcsec')} arcsec away.{where}.{clash} "
            f"Confirm it against the target in the attached comparison plot "
            f"before concluding.")

    if on_target and not others:
        f = on_target[0]
        return r.passed(
            f"the eclipse is on the target: {survey.upper()} shows a "
            f"{f.get('frac_depth_pct')}% drop at {f.get('sig')} sigma at this "
            f"ephemeris, and no field star does. Ground-based photometry that "
            f"resolves the neighbours puts the signal on this star.")

    if on_target and others:
        return r.flag(
            f"both the target and {len(others)} field star(s) show a drop at "
            f"this ephemeris, so the scan does not separate them; read the "
            f"comparison plots. Target at {on_target[0].get('sig')} sigma.")

    # Nothing flagged at all.  When pc_centroid named a star, what matters is
    # whether THIS survey could have seen that star's eclipse -- its own noise
    # floor and whether the survey separates it from the target -- not the
    # deepest eclipse any neighbour might need (one verdict quoted a 52%
    # neighbour while the named star, 7.6 arcsec from a target 4.5 mag
    # brighter, is not separable by ATLAS at all).
    if suspect:
        s = next((f for f in (scan or []) if str(f.get("tic_ID")) == str(int(suspect))), None)
        req = reach.get("suspect_required_depth_ppm")
        row = _row_for(ctx, suspect) or {}
        sep = row.get("sep_arcsec")
        dmag = (row.get("tmag") - reach["target_tmag"]) if (
            row.get("tmag") is not None and reach.get("target_tmag") is not None) else None
        sepable = row.get("resolvable_by_survey")
        if sepable is None and sep is not None and dmag is not None:
            from . import pc_neighbours     # neighbours.json written before 0.1.1
            sepable = pc_neighbours._survey_separable(sep, dmag)
        if sepable is False:
            return r.inconclusive(
                f"no star shows an eclipse at this ephemeris in {survey.upper()}, but the "
                f"star pc_centroid pointed at (TIC {int(suspect)}"
                + (f", {sep:.1f} arcsec, {dmag:+.1f} mag" if sep is not None and dmag is not None
                   else "") + f") is not separable from the target in a survey with "
                f"{survey.upper()}'s pixels and seeing, so its photometry is blended with the "
                f"target's and a non-detection says nothing about it.")
        if s is None:
            return r.inconclusive(
                f"no star shows an eclipse at this ephemeris in {survey.upper()}, but the "
                f"star pc_centroid pointed at (TIC {int(suspect)}) has no usable "
                f"{survey.upper()} light curve (not extracted, or too few points in "
                f"transit), so it was not tested.")
        floor = (s.get("scatter_pct") or np.nan) / np.sqrt(max(1, s.get("n_in") or 1))
        need = (req or 0) / 1e4
        if not (np.isfinite(floor) and need >= SUSPECT_REACH_SIGMA * floor):
            return r.inconclusive(
                f"no star shows an eclipse at this ephemeris in {survey.upper()}. The star "
                f"pc_centroid pointed at (TIC {int(suspect)}) would need a {need:.1f}% "
                f"eclipse; its photometry ({s.get('scatter_pct')}% per point, "
                f"{s.get('n_in')} points in transit, {s.get('filter')}) reaches "
                f"{floor:.1f}% at 1 sigma, so {SUSPECT_REACH_SIGMA:g} sigma needs "
                f"{SUSPECT_REACH_SIGMA * floor:.1f}%: it could not have been seen. It "
                f"shows {s.get('frac_depth_pct')}% ({s.get('sig')} sigma).")
        return r.flag(
            f"no star shows an eclipse at this ephemeris in {survey.upper()}, including the "
            f"star pc_centroid pointed at (TIC {int(suspect)}), which would need a "
            f"{need:.1f}% eclipse and shows {s.get('frac_depth_pct')}% ({s.get('sig')} "
            f"sigma) on photometry that reaches {floor:.1f}% at 1 sigma "
            f"({s.get('scatter_pct')}% per point, {s.get('n_in')} points in transit). "
            f"Either the ephemeris has drifted at these epochs, the source is another "
            f"star, or the centroid is wrong; check the epoch coverage in the plots.")
    best = reach.get("best_source_depth_ppm")
    if best is None or (scan is not None and not any(
            f.get("is_target") and (f.get("n_in") or 0) > 0 for f in scan)):
        r.set_self_check(False, "the scan has no in-transit photometry of the target")
        return r.inconclusive(
            f"no star shows an eclipse at this ephemeris in {survey.upper()}, but the "
            f"scan has no in-transit photometry of the target, so that says nothing.")
    return r.flag(
        f"no star in the field -- target included -- shows an eclipse at this "
        f"ephemeris in {survey.upper()}. The deepest plausible source would "
        f"be {best:.0f} ppm ({best / 1e4:.2f}%), which this survey should "
        f"reach, so a clean non-detection is itself informative: either the "
        f"ephemeris is wrong at these epochs or the signal is not "
        f"astrophysical. Check the epoch coverage in the attached plots "
        f"before leaning on it.")


def _reachable_depth(ctx, target_depth, suspect) -> dict:
    """The deepest eclipse any plausible source would have to show, in ppm.

    Reads ``neighbours.json`` beside the fit.  ``required_eclipse_depth`` is a
    *fraction* there, so a neighbour needing a 21% eclipse comes back as 0.21.
    Anything above 1 was excluded outright by that test and is ignored here.
    """
    explicit = suspect is not None
    if suspect is None:
        suspect = _suspect_from_centroid(ctx)
    out = {"target_depth_ppm": target_depth, "neighbours_used": False,
           "suspect_tic": int(suspect) if suspect else None}
    path = ctx.out / ctx.target.slug / (
        f"neighbours_c{ctx.candidate.index}.json" if ctx.candidate.index
        else "neighbours.json")
    best = float(target_depth) if target_depth else None
    best_from = "target" if best else None

    if path.exists():
        try:
            m = (json.loads(path.read_text()).get("metrics") or {})
        except (OSError, json.JSONDecodeError):
            m = {}
        out["target_tmag"] = m.get("target_tmag")
        viable = [n for n in (m.get("neighbours") or [])
                  if n.get("verdict") != "excluded"
                  and n.get("required_eclipse_depth") is not None
                  and 0 < n["required_eclipse_depth"] <= 1.0]
        if viable:
            out["neighbours_used"] = True
            out["n_viable_neighbours"] = len(viable)
            deepest = max(viable, key=lambda n: n["required_eclipse_depth"])
            out["max_neighbour_depth_ppm"] = deepest["required_eclipse_depth"] * 1e6
            out["max_neighbour_tic"] = deepest.get("tic") or deepest.get("gaia_dr3")
            if best is None or out["max_neighbour_depth_ppm"] > best:
                best = out["max_neighbour_depth_ppm"]
                best_from = "neighbour"
    # The suspect pc_centroid named overrides pc_neighbours' verdict on it.
    # Those two tests currently disagree, and the centroid is the one with
    # evidence: in one field the centroid puts the signal on a star 49.7
    # arcsec away at 70.8 sigma, while pc_neighbours "excluded" that same star
    # by requiring a 397% eclipse -- because its Gaussian PRF (sigma = 0.65 px)
    # gives a source 2.4 pixels out a flux fraction of 0.0005, and the real
    # TESS PRF has far broader wings than that.  An eclipse cannot exceed
    # 100%, so a required depth above 1 means the dilution model is wrong, not
    # that the star is innocent.  For the purpose of deciding whether a
    # ground-based survey could see it, take a total eclipse as the bound.
    if suspect is not None:
        out["suspect_from"] = "pc_centroid" if not explicit else "--neighbour"
        named = _row_for(ctx, suspect)
        if named is not None:
            req = named.get("required_eclipse_depth")
            capped = min(float(req), 1.0) if req else 1.0
            out["suspect_required_depth_ppm"] = capped * 1e6
            out["suspect_required_uncapped"] = req
            out["suspect_tmag"] = named.get("tmag")
            if req and req > 1.0:
                out["suspect_note"] = (
                    f"pc_neighbours excluded this star, requiring a "
                    f"{req:.0%} eclipse from a Gaussian-PRF flux share of "
                    f"{named.get('flux_share_in_aperture', float('nan')):.4f}. "
                    f"An eclipse cannot exceed 100%; capped at a total one "
                    f"for this gate. See the PRF note in pc_neighbours.")
            if capped * 1e6 > (best or 0):
                best = capped * 1e6
                best_from = f"the star pc_centroid pointed at (TIC {int(suspect)})"

    out["best_source_depth_ppm"] = best
    out["best_source_is"] = best_from
    return out


def _row_for(ctx, tic):
    """The pc_neighbours row for a TIC, matched through the TIC cone.

    pc_neighbours keys its rows by Gaia DR3 and does not carry a TIC, so a
    star named by pc_centroid (which works in TIC) has to be matched on sky
    position.  Within 2 arcsec is unambiguous at these separations.
    """
    path = ctx.out / ctx.target.slug / (
        f"neighbours_c{ctx.candidate.index}.json" if ctx.candidate.index
        else "neighbours.json")
    if not path.exists():
        return None
    try:
        m = (json.loads(path.read_text()).get("metrics") or {})
    except (OSError, json.JSONDecodeError):
        return None
    rows = m.get("neighbours") or []
    for n in rows:
        if str(n.get("tic")) == str(int(tic)):
            return n
    # fall back to the centroid's own record of where that star is
    cpath = ctx.out / ctx.target.slug / (
        f"centroid_c{ctx.candidate.index}.json" if ctx.candidate.index
        else "centroid.json")
    if not cpath.exists():
        return None
    try:
        cm = (json.loads(cpath.read_text()).get("metrics") or {})
    except (OSError, json.JSONDecodeError):
        return None
    gaia = cm.get("best_source")
    if gaia and str(gaia) != "target" and _tic_for_gaia(ctx, gaia) == int(tic):
        for n in rows:
            if str(n.get("gaia_dr3")) == str(gaia):
                hyp = {h.get("id"): h for h in cm.get("hypotheses") or []}
                implied = (hyp.get(str(gaia)) or {}).get("implied_depth")
                # the centroid's PRF scene fit is the better dilution model;
                # prefer its implied depth over pc_neighbours' when it has one
                return dict(n, required_eclipse_depth=implied) if implied else n
    best = cm.get("best_aligned_neighbour") or {}
    if not best or str(best.get("id")) != str(int(tic)):
        return None
    dra, ddec = best.get("dra"), best.get("ddec")
    if dra is None or ddec is None:
        return None
    for n in rows:
        if (abs((n.get("dra_arcsec") or 1e9) - dra) < 2.0
                and abs((n.get("ddec_arcsec") or 1e9) - ddec) < 2.0):
            return n
    return None


def _suspect_from_centroid(ctx):
    """The star pc_centroid's offset points at, if it measured one."""
    path = ctx.out / ctx.target.slug / (
        f"centroid_c{ctx.candidate.index}.json" if ctx.candidate.index
        else "centroid.json")
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("status") not in ("FAIL", "FLAG"):
        return None
    m = d.get("metrics") or {}
    # centroid >= 0.2 names the best-fitting scene star by Gaia DR3 id
    # (``best_source``); DEATHSTAR works in TIC, so translate through the TIC
    # cone.  Without this the suspect was silently dropped for every v0.2 run.
    gaia = m.get("best_source")
    if gaia and str(gaia) != "target":
        tic = _tic_for_gaia(ctx, gaia)
        if tic:
            return tic
    best = m.get("best_aligned_neighbour") or {}
    cos = m.get("best_alignment_cos")
    if best.get("id") and cos is not None and cos > 0.5:
        return int(best["id"])
    return None


def _tic_for_gaia(ctx, gaia):
    """The TIC id of a Gaia DR3 source near the target, or None."""
    from .. import catalogs
    star = ctx.star
    ra = ctx.target.ra if ctx.target.ra is not None else star.ra
    dec = ctx.target.dec if ctx.target.dec is not None else star.dec
    if ra is None or dec is None:
        return None
    try:
        rows = catalogs.tic_cone(float(ra), float(dec), radius_arcsec=150.0,
                                 tmag_limit=21.0)
    except Exception:  # noqa: BLE001 -- a failed lookup just means no suspect
        return None
    for row in rows:
        if str(row.get("GAIA") or "").split(".")[0] == str(gaia):
            return int(row["ID"])
    # The TIC's GAIA column is DR2, and DR2 and DR3 ids differ for some stars.
    # Fall back to position: pc_neighbours records each DR3 star's offset.
    path = ctx.out / ctx.target.slug / (
        f"neighbours_c{ctx.candidate.index}.json" if ctx.candidate.index
        else "neighbours.json")
    try:
        nb = (json.loads(path.read_text()).get("metrics") or {}).get("neighbours") or []
    except (OSError, json.JSONDecodeError):
        return None
    me = next((n for n in nb if str(n.get("gaia_dr3")) == str(gaia)), None)
    if not me or me.get("dra_arcsec") is None:
        return None
    cosd = np.cos(np.radians(float(dec)))
    best, best_d = None, 2.0
    for row in rows:
        try:
            dra = (float(row["ra"]) - float(ra)) * cosd * 3600
            ddec = (float(row["dec"]) - float(dec)) * 3600
        except (KeyError, TypeError, ValueError):
            continue
        d = float(np.hypot(dra - me["dra_arcsec"], ddec - me["ddec_arcsec"]))
        if d < best_d:
            best, best_d = int(row["ID"]), d
    return best


#: ATLAS forced-photometry API, for the credential pre-flight.
ATLAS_AUTH_URL = "https://fallingstar-data.com/forcedphot/api-token-auth/"


#: Environment variables an ATLAS API token may arrive in.  DEATHSTAR's
#: atlas_fetch reads $ATLAS_TOKEN; $ATLAS_API_TOKEN is the name the ATLAS site
#: itself uses, so both are accepted here and normalised on the way out.
ATLAS_TOKEN_VARS = ("ATLAS_TOKEN", "ATLAS_API_TOKEN")


def _atlas_token():
    for var in ATLAS_TOKEN_VARS:
        tok = (os.environ.get(var) or "").strip()
        if tok:
            return tok
    return None


def _missing_credentials(is_atlas):
    if is_atlas:
        token = _atlas_token()
        if token:
            return _atlas_token_error(token)
        user = os.environ.get("ATLAS_USERNAME")
        pwd = os.environ.get("ATLAS_PASSWORD")
        if not (user and pwd):
            return ("no ATLAS credentials are set ($ATLAS_API_TOKEN or "
                    "$ATLAS_TOKEN, or $ATLAS_USERNAME and $ATLAS_PASSWORD). "
                    "Note that zsh only "
                    "reads ~/.zshrc for interactive shells, so exports there "
                    "are invisible to a batch run -- put them in ~/.zshenv, "
                    "or source it explicitly")
        # Check them before handing the job to DEATHSTAR.  A rejected login
        # otherwise surfaces as a traceback inside a subprocess, which reads
        # like the tool is broken rather than like a password being wrong.
        return _atlas_auth_error(user, pwd)
    for p in (Path.home() / ".ztfquery", Path.home() / ".config" / "ztfquery"):
        if p.exists():
            return None
    return ("ztfquery has no stored IRSA login (~/.ztfquery); run "
            "ztfquery.io.set_account('irsa') once")


ATLAS_QUEUE_URL = "https://fallingstar-data.com/forcedphot/queue/"


def _atlas_token_error(token):
    """None if ATLAS accepts this token, else a sentence saying why not."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        ATLAS_QUEUE_URL, headers={"Authorization": f"Token {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return None if resp.status == 200 else (
                f"the ATLAS API returned HTTP {resp.status} for this token")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return ("ATLAS rejected this API token. Get a current one from "
                    "https://fallingstar-data.com/forcedphot/apiguide/ and "
                    "set $ATLAS_API_TOKEN")
        return f"the ATLAS API returned HTTP {exc.code}; it may be down"
    except Exception as exc:  # noqa: BLE001 - network, DNS, anything
        return (f"could not reach the ATLAS API to check the token "
                f"({type(exc).__name__}); treating that as unavailable "
                f"rather than guessing")


def _atlas_auth_error(user, pwd):
    """None if ATLAS accepts these credentials, else a sentence saying so."""
    try:
        import urllib.error
        import urllib.parse
        import urllib.request
        data = urllib.parse.urlencode(
            {"username": user.strip(), "password": pwd.strip()}).encode()
        with urllib.request.urlopen(
                urllib.request.Request(ATLAS_AUTH_URL, data=data),
                timeout=30) as resp:
            body = json.loads(resp.read().decode())
        return None if body.get("token") else (
            "the ATLAS API accepted the request but returned no token")
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403):
            return ("ATLAS rejected these credentials ($ATLAS_USERNAME / "
                    "$ATLAS_PASSWORD). The forced-photometry service at "
                    "fallingstar-data.com needs its own account, separate "
                    "from any other ATLAS or IfA login -- register at "
                    "https://fallingstar-data.com/forcedphot/ and use that "
                    "username and password")
        return f"the ATLAS API returned HTTP {exc.code}; it may be down"
    except Exception as exc:  # noqa: BLE001 - network, DNS, anything
        return (f"could not reach the ATLAS API to check the credentials "
                f"({type(exc).__name__}); treating that as unavailable "
                f"rather than guessing")


def _ppm(x):
    return f"{x:.0f} ppm" if isinstance(x, (int, float)) else "an unknown depth"


def _git_sha(root: Path):
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "--short",
                            "HEAD"], capture_output=True, text=True, timeout=5)
        return p.stdout.strip() or None if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Ground-based archival photometry via DEATHSTAR, run as a "
              "subprocess. Gates on the depth the source would show, not the "
              "target's."))
    args = ap.parse_args(argv)
    ctx = cli.Context.from_args(args)
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # noqa: BLE001
        r.error(exc)
    return cli.finish(r, ctx)


if __name__ == "__main__":
    sys.exit(main())

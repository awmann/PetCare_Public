"""Adapter that drives DEATHSTAR from a JSON spec.

DEATHSTAR has no command-line interface -- it is a library whose entry point
is ``DEATHSTAR.setup(tic_ID, ...)``, and it writes its products into
``TICs/<tic>/`` relative to its own checkout.  This file is the small piece of
glue that turns that into something :mod:`petcare.scripts.pc_archival` can
call as a subprocess and read the answer from.

It is deliberately the *only* PetCare file that knows DEATHSTAR's API, and it
is run by **DEATHSTAR's own interpreter**, not PetCare's: the two have
incompatible dependency sets (DEATHSTAR wants python 3.9 with numpy < 2).  So
it imports nothing from PetCare, and nothing here may.

    <deathstar-python> deathstar_runner.py spec.json

``spec.json``::

    {"tic": <TIC>,
     "manual_ephemeris": {"ra": ..., "dec": ..., "period": ...,
                          "epoch": <BJD>, "depth": <ppm>, "Tmag": ...,
                          "duration": <hours>},
     "is_ATLAS": true,
     "signal_tic_ID": <TIC of the suspected source>,
     "is_overwrite": false,
     "is_ephemeris_scan": true,
     "result_json": "/path/to/write/result.json"}

The result records what DEATHSTAR wrote, the ephemeris-scan flags parsed out
of ``Ephemeris_Scan_Flagged.csv``, and any exception -- never a verdict.
Deciding what the flags mean is PetCare's job, on PetCare's side.
"""

import csv
import glob
import json
import os
import sys
import time
import traceback

#: Re-runs of setup() after a failure that is the archive's, not the data's.
#: 2026-10-27 audit of 1052 runs: 184 crashed, 68 on an empty ZTF metadata
#: table (ztfquery's KeyError on 'filtercode'/'imgtypecode' -- IRSA answering
#: with nothing) and 33 on a ZTF frame its pooled download never wrote
#: (FileNotFoundError under Data/sci), plus connection resets.  A re-run keeps
#: the frames that did arrive and fetches the rest.
RETRIES = 2
RETRY_WAIT_S = 90


def _transient(exc):
    msg = str(exc)
    if isinstance(exc, FileNotFoundError) and "Data/sci" in msg:
        return True
    if isinstance(exc, KeyError) and ("filtercode" in msg or "imgtypecode" in msg):
        return True
    name = type(exc).__name__
    return name in ("ConnectionError", "ConnectionResetError", "ReadTimeout",
                    "ChunkedEncodingError", "JSONDecodeError")


def _read_flags(root, tic):
    path = os.path.join(root, "TICs", str(tic), "Ephemeris_Scan_Flagged.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        rec = {}
        for k, v in row.items():
            if v in ("", None):
                rec[k] = None
                continue
            if k in ("is_target",):
                rec[k] = str(v).strip().lower() in ("true", "1", "yes")
                continue
            try:
                rec[k] = float(v) if "." in str(v) else int(v)
            except (TypeError, ValueError):
                rec[k] = v
        out.append(rec)
    return out


def main(argv):
    if len(argv) < 2:
        print("usage: deathstar_runner.py spec.json", file=sys.stderr)
        return 2
    with open(argv[1]) as fh:
        spec = json.load(fh)

    root = spec.get("deathstar_root") or os.getcwd()
    tic = int(spec["tic"])
    result = {"tic": tic, "deathstar_root": root, "ok": False}

    # DEATHSTAR resolves TICs/ and Data/ against the working directory.
    os.chdir(root)
    sys.path.insert(0, root)

    # Headless: several paths call plt.show(), which blocks forever without a
    # display and is how a batch run silently hangs.
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:  # noqa: BLE001 - not fatal, just riskier
        pass

    before = set(glob.glob(os.path.join(root, "TICs", str(tic), "*")))

    # ATLAS images have to be fetched before DEATHSTAR runs.  The fork ships
    # atlas_fetch (the upstream repo has no ATLAS downloader at all), but
    # Extracting_Lightcurves only ever *lists* Data/Atlas/<tic>/*.fits -- it
    # prints "Automatically downloading ATLAS images..." and then downloads
    # nothing.  With an empty directory the run reaches the BJD conversion
    # with zero frames and dies on a missing Star_Data.csv, which reads like a
    # bug three stages later than the actual problem.
    if spec.get("is_ATLAS") and spec.get("atlas_fetch", True):
        out_dir = os.path.join(root, "Data", "Atlas", str(tic))
        have = glob.glob(os.path.join(out_dir, "*.fits"))
        result["atlas_frames_cached"] = len(have)
        # DEATHSTAR re-plots from its extracted light curves alone (setup()
        # skips extraction when All_<tic>_Star_Data.csv exists), and the
        # vetter purges raw frames after each run -- so without this every
        # re-run of a star, e.g. its next alias period, re-downloaded ~1 GB.
        lc_cached = os.path.isfile(os.path.join(
            root, "TICs", str(tic), "All_%s_Star_Data.csv" % tic))
        result["atlas_lightcurves_cached"] = lc_cached
        if lc_cached and not spec.get("is_overwrite") \
                and not spec.get("is_reference_image"):
            have = have or ["(light curves cached)"]
        if not have or spec.get("is_overwrite"):
            try:
                import atlas_fetch
                me = spec["manual_ephemeris"]
                state = os.path.join(root, "Data", "Atlas",
                                     "fetch_state_%s.json" % tic)
                fetched = atlas_fetch.fetch_atlas_images_batch(
                    {str(tic): (float(me["ra"]), float(me["dec"]))},
                    mjd_min=float(spec["mjd_min"]),
                    mjd_max=float(spec["mjd_max"]),
                    state_path=state,
                    timeout_s=float(spec.get("fetch_timeout_s") or 3 * 3600))
                result["atlas_fetch"] = fetched.get(str(tic), fetched)
            except Exception as exc:  # noqa: BLE001
                result["error"] = "ATLAS fetch failed: {}: {}".format(
                    type(exc).__name__, exc)
                result["traceback"] = traceback.format_exc()
                _finish(result, root, tic, spec, before)
                return 1
        result["atlas_frames"] = len(
            glob.glob(os.path.join(out_dir, "*.fits")))
        if not result["atlas_frames"] and not (
                lc_cached and not spec.get("is_overwrite")):
            result["error"] = (
                "ATLAS returned no images for this field in MJD "
                "%s-%s" % (spec["mjd_min"], spec["mjd_max"]))
            _finish(result, root, tic, spec, before)
            return 1

    # DEATHSTAR plots signal_tic_ID against the target, but it only extracts
    # cone stars that pass its own magnitude cut -- so a star pc_centroid
    # named can be absent, and asking for it dies with
    # KeyError: '<tic>_brightness' after the whole extraction has run.
    signal = int(spec.get("signal_tic_ID") or 0)
    if signal:
        cone = os.path.join(root, "TICs", str(tic), f"{tic}_Cone.csv")
        if os.path.exists(cone):
            with open(cone) as fh:
                ids = {row.get("ID") for row in csv.DictReader(fh)}
            if str(signal) not in ids:
                result["signal_tic_dropped"] = signal
                result["signal_tic_reason"] = (
                    "not in DEATHSTAR's extracted cone (its magnitude cut "
                    "removed it); the comparison plot is skipped, the "
                    "ephemeris scan still covers every extracted star")
                signal = 0

    from DEATHSTAR import setup
    for attempt in range(1 + RETRIES):
        try:
            setup(tic,
                  is_ATLAS=bool(spec.get("is_ATLAS", False)),
                  is_overwrite=bool(spec.get("is_overwrite", False)),
                  manual_ephemeris=spec.get("manual_ephemeris"),
                  signal_tic_ID=signal,
                  is_ephemeris_scan=bool(spec.get("is_ephemeris_scan", True)),
                  is_reference_image=bool(spec.get("is_reference_image", True)),
                  is_plotting=bool(spec.get("is_plotting", True)),
                  is_saving=True,
                  is_plotting_MAD_vs_MAG=bool(spec.get("is_plotting_MAD_vs_MAG", True)))
            result["ok"] = True
            result.pop("error", None)
            result.pop("traceback", None)
            break
        except Exception as exc:  # noqa: BLE001 - report it, never raise through
            result["error"] = "{}: {}".format(type(exc).__name__, exc)
            result["traceback"] = traceback.format_exc()
            result["attempts"] = attempt + 1
            if not _transient(exc) or attempt == RETRIES:
                break
            time.sleep(RETRY_WAIT_S * (attempt + 1))

    _finish(result, root, tic, spec, before)
    return 0 if result["ok"] else 1


def _finish(result, root, tic, spec, before):
    after = set(glob.glob(os.path.join(root, "TICs", str(tic), "*")))
    result["products"] = sorted(after)
    result["new_products"] = sorted(after - before)
    result["ephemeris_scan"] = _read_flags(root, tic)

    out = spec.get("result_json")
    if out:
        with open(out, "w") as fh:
            json.dump(result, fh, indent=1, default=str)
    else:
        json.dump(result, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

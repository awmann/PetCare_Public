"""Calibration: what does each test cost on known planets, and what does it catch?

A test with an uncalibrated threshold is not finished, and a test that has
never been run on known planets and known false positives has no defensible
threshold at all.  This script measures both sides.

It reads the ``summary.csv`` files a driver run produced for each sample,
joins them to the truth labels, and answers three questions per test:

1. **What does it cost?**  The fraction of *known planets* it flags or fails.
   This is the number that matters most.  Plausible-looking cuts can be
   expensive: in our own search a rotation-proximity veto removed a published
   planet, an odd/even cut at 3 sigma would have removed 18% of known planets,
   and a global anti-transit null 25%.
2. **What does it catch?**  The fraction of known false positives it flags or
   fails.
3. **Can it separate them at all?**  For every test that produces a
   continuous statistic, the distributions are compared and the threshold that
   best separates them is reported next to the one the code uses now.

    petcare --candidates calibration/samples/known_young_planets.csv \\
        --tiers 0 1 --out runs/knowns
    petcare --candidates my_false_positives.csv --tiers 0 1 \\
        --out runs/fps
    python calibration/calibrate.py --knowns runs/knowns/summary.csv \\
        --false-positives runs/fps/summary.csv \\
        --fp-sample my_false_positives.csv --out calibration/out

The false-positive table is yours: any candidate table with a ``truth``
column.  Rows whose truth is in ``--fp-truth`` are scored; anything else (a
``candidate``, say) is carried but not counted, because a candidate is not
evidence either way.  Both samples are usually small, so the report prints
the sample size beside every number.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

#: Statistics worth scanning for a better threshold, as
#: ``test -> (key in the per-test JSON metrics, direction, threshold)``.
#: ``direction`` is +1 when large values are suspicious.  ``threshold`` names
#: the module constant the test uses (``"module:NAME"``), so the scan always
#: compares against the code as it is, or is None where the test has no cut.
STATISTICS = {
    "depths": [("jackknife_survival", -1, "pc_depths:JACKKNIFE_FAIL"),
               ("chi2_p_value", -1, "pc_depths:CHI2_P_FLAG"),
               ("scatter_over_errors", +1, None)],
    "odd_even": [("difference_sigma", +1, "pc_oddeven:ODD_EVEN_FLAG_SIGMA")],
    "shape": [("ingress_fraction", +1, "pc_shape:INGRESS_FRACTION_FLAG")],
    "secondary": [("best_sigma", +1, "pc_secondary:SECONDARY_FLAG_SIGMA"),
                  ("ellipsoidal.sigma", +1, "pc_secondary:ELLIPSOIDAL_FLAG_SIGMA")],
    "fit": [("gp_sigma_over_depth", +1, "pc_fit:GP_SIGMA_OVER_DEPTH_FLAG"),
            ("delta_logl_vs_no_transit", -1, None)],
    "duration": [("reconcile_ecc_required", +1, "pc_duration:ECC_FLAG"),
                 ("implied_b", +1, None)],
    "instrument": [("depth_by_detector.sector.chi2_p_value", -1,
                    "pc_instrument:SECTOR_CHI2_P_FLAG")],
    "centroid": [("combined_snr", +1, "pc_centroid:SNR_MIN")],
    "neighbours": [("n_viable", +1, None)],
    "flares": [("depth_fractional_change", +1, "pc_flares:DEPTH_CHANGE_FLAG")],
    "crossmatch": [("depth_spread_fraction", +1, "pc_crossmatch:DISAGREE_FRACTION")],
    "interp": [("z_corrected", -1, "pc_interp:FAIL_SIGMA")],
    "uniqueness": [("MS2", -1, "pc_uniqueness:MS2_FAIL"),
                   ("MS3", -1, "pc_uniqueness:MS3_FAIL")],
}


def current_threshold(ref):
    """The value of a ``"module:NAME"`` constant in ``petcare.scripts``."""
    if ref is None:
        return None
    module, name = ref.split(":")
    return float(getattr(importlib.import_module(f"petcare.scripts.{module}"), name))


#: Statuses counted as "this test objected".
OBJECTED = ("FLAG", "FAIL")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowns", required=True,
                    help="summary.csv from a run over the known-planet sample")
    ap.add_argument("--false-positives", required=True,
                    help="summary.csv from a run over the false-positive sample")
    ap.add_argument("--knowns-out", default=None,
                    help="the run's output directory, for reading per-test "
                         "JSONs; default: alongside --knowns")
    ap.add_argument("--fp-out", default=None)
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--fp-truth", nargs="+",
                    default=["false_positive", "likely_false_positive"],
                    help="which truth labels count as false positives")
    ap.add_argument("--fp-sample", default=None,
                    help="the candidate table behind --false-positives; its truth "
                         "column selects the rows that --fp-truth names")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import pandas as pd

    knowns = pd.read_csv(args.knowns)
    fps = pd.read_csv(args.false_positives)
    if "truth" not in fps.columns and args.fp_sample:
        sample = pd.read_csv(args.fp_sample)
        if "label" in sample.columns and "name" in fps.columns:
            # per candidate: two signals on one star can have different truths
            lab = sample[["tic", "label", "truth"]].rename(columns={"label": "name"})
            fps = fps.merge(lab.drop_duplicates(["tic", "name"]), on=["tic", "name"], how="left")
        else:
            fps = fps.merge(sample[["tic", "truth"]].drop_duplicates("tic"), on="tic", how="left")
    if "truth" in fps.columns:
        fps = fps[fps["truth"].isin(args.fp_truth)]
    else:
        print("WARNING: no truth column and no --fp-sample; every row counts as a false positive")
    k_root = Path(args.knowns_out or Path(args.knowns).parent)
    f_root = Path(args.fp_out or Path(args.false_positives).parent)

    tests = sorted({c[:-7] for c in knowns.columns if c.endswith("_status")}
                   | {c[:-7] for c in fps.columns if c.endswith("_status")})

    rows = []
    for test in tests:
        k = _counts(knowns, test)
        f = _counts(fps, test)
        cost = _frac(k["objected"], k["decided"])
        catch = _frac(f["objected"], f["decided"])
        rows.append({
            "test": test,
            "n_knowns": k["total"], "n_knowns_decided": k["decided"],
            "knowns_objected": k["objected"],
            "cost_on_known_planets": cost,
            "knowns_inconclusive": k["inconclusive"],
            "knowns_error": k["error"],
            "n_fp": f["total"], "n_fp_decided": f["decided"],
            "fp_objected": f["objected"],
            "catch_on_false_positives": catch,
            "fp_inconclusive": f["inconclusive"],
            "fp_error": f["error"],
            "separation": (None if cost is None or catch is None
                           else round(catch - cost, 3)),
        })

    stats = _scan_statistics(knowns, fps, k_root, f_root)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conf = pd.DataFrame(rows)
    conf.to_csv(out / "confusion_matrix.csv", index=False)
    (out / "threshold_scan.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False, default=str))
    report = _report(rows, stats, len(knowns), len(fps))
    (out / "CALIBRATION.md").write_text(report)

    print(report)
    print(f"-> {out / 'confusion_matrix.csv'}")
    print(f"-> {out / 'threshold_scan.json'}")
    print(f"-> {out / 'CALIBRATION.md'}")
    return 0


# --------------------------------------------------------------------------


def _counts(df, test) -> dict:
    col = f"{test}_status"
    if col not in df.columns:
        return {"total": 0, "decided": 0, "objected": 0,
                "inconclusive": 0, "error": 0}
    s = df[col].fillna("MISSING")
    decided = int((s.isin(["PASS", "FLAG", "FAIL"])).sum())
    return {"total": int(len(s)), "decided": decided,
            "objected": int(s.isin(OBJECTED).sum()),
            "inconclusive": int((s == "INCONCLUSIVE").sum()),
            "error": int((s == "ERROR").sum())}


def _frac(n, d):
    return None if not d else round(n / d, 3)


def _scan_statistics(knowns, fps, k_root, f_root) -> dict:
    """For each continuous statistic, where would the best cut be?"""
    out = {}
    for test, specs in STATISTICS.items():
        for key, direction, ref in specs:
            current = current_threshold(ref)
            k_vals = _collect(knowns, k_root, test, key)
            f_vals = _collect(fps, f_root, test, key)
            if len(k_vals) < 4 or len(f_vals) < 3:
                out[f"{test}.{key}"] = {
                    "status": "not enough data",
                    "n_knowns": len(k_vals), "n_false_positives": len(f_vals),
                    "current_threshold": current,
                    "note": "at least 4 known planets and 3 false positives "
                            "are needed before a threshold means anything"}
                continue

            best = _best_threshold(k_vals, f_vals, direction)
            entry = {
                "status": "scanned",
                "n_knowns": len(k_vals), "n_false_positives": len(f_vals),
                "direction": ("large values are suspicious" if direction > 0
                              else "small values are suspicious"),
                "current_threshold": current,
                "knowns": _describe(k_vals),
                "false_positives": _describe(f_vals),
                "best_threshold": best["threshold"],
                "best_cost_on_knowns": best["cost"],
                "best_catch_on_fps": best["catch"],
                "best_separation": best["separation"],
                "separable": bool(best["separation"] is not None
                                  and best["separation"] > 0.3),
            }
            if current is not None:
                entry["current_cost_on_knowns"] = _cost_at(k_vals, current, direction)
                entry["current_catch_on_fps"] = _cost_at(f_vals, current, direction)
            out[f"{test}.{key}"] = entry
    return out


def _collect(df, root, test, key) -> list:
    """Pull one metric out of every per-target JSON in a run."""
    vals = []
    for _, row in df.iterrows():
        tic = row.get("tic")
        if tic is None or (isinstance(tic, float) and not np.isfinite(tic)):
            continue
        idx = int(row.get("candidate_index") or 0)
        name = f"{test}_c{idx}.json" if idx else f"{test}.json"
        path = Path(root) / f"TIC{int(tic)}" / name
        if not path.exists():
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") not in ("PASS", "FLAG", "FAIL"):
            continue
        v = rec.get("metrics") or {}
        for part in key.split("."):
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(part)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return vals


def _describe(vals) -> dict:
    a = np.asarray(vals, float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def _best_threshold(k_vals, f_vals, direction) -> dict:
    """The cut maximising (catch on false positives) - (cost on real planets).

    Not an accuracy: accuracy on unequal samples is dominated by whichever
    class is larger.  Maximising the *difference* is Youden's J, and it treats
    losing a real planet and passing a false positive as equally bad -- which
    on this project they are not, but the cost column is printed beside it so
    a human can move the cut toward whichever they prefer.
    """
    k = np.asarray(k_vals, float)
    f = np.asarray(f_vals, float)
    grid = np.unique(np.concatenate([k, f]))
    if grid.size < 3:
        return {"threshold": None, "cost": None, "catch": None,
                "separation": None}
    best = None
    for thr in grid:
        cost = float(np.mean(k > thr if direction > 0 else k < thr))
        catch = float(np.mean(f > thr if direction > 0 else f < thr))
        sep = catch - cost
        if best is None or sep > best["separation"]:
            best = {"threshold": float(thr), "cost": round(cost, 3),
                    "catch": round(catch, 3), "separation": round(sep, 3)}
    return best


def _cost_at(vals, threshold, direction) -> float:
    a = np.asarray(vals, float)
    return round(float(np.mean(a > threshold if direction > 0 else a < threshold)), 3)


def _report(rows, stats, n_k, n_f) -> str:
    lines = [
        "# PetCare calibration",
        "",
        f"Known planets: **{n_k}**.  False positives: **{n_f}**.",
        "",
        "A test with an uncalibrated threshold is not finished.  This file is",
        "generated by `calibration/calibrate.py`; do not edit it by hand.",
        "",
        "## What each test costs and catches",
        "",
        "`cost` is the fraction of **known planets** the test flagged or failed.",
        "`catch` is the same fraction among **known false positives**.",
        "Both are computed over the candidates where the test actually reached a",
        "verdict, so a test that is mostly INCONCLUSIVE shows that in its own column.",
        "",
        "| test | n knowns | cost | n FP | catch | separation | knowns INCON | FP INCON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: -(x["separation"] or -9)):
        lines.append(
            f"| {r['test']} | {r['n_knowns_decided']} | "
            f"{_pct(r['cost_on_known_planets'])} | {r['n_fp_decided']} | "
            f"{_pct(r['catch_on_false_positives'])} | "
            f"{r['separation'] if r['separation'] is not None else '-'} | "
            f"{r['knowns_inconclusive']} | {r['fp_inconclusive']} |")

    lines += ["", "## Threshold scan", "",
              "For each continuous statistic: the distribution in each sample,",
              "the threshold currently hardcoded, and the one that best separates",
              "them.  `separable` is true only where the best achievable",
              "separation exceeds 0.3 -- below that the statistic is not doing",
              "useful work on these samples and its threshold should be treated",
              "as a placeholder however it is set.", ""]
    for name, s in sorted(stats.items()):
        if s.get("status") != "scanned":
            lines.append(f"- **{name}**: {s['status']} "
                         f"({s['n_knowns']} knowns, {s['n_false_positives']} FP)")
            continue
        k, f = s["knowns"], s["false_positives"]
        lines.append(
            f"- **{name}** ({s['direction']}): "
            f"knowns median {k['median']:.3g} [{k['p10']:.3g}, {k['p90']:.3g}], "
            f"FP median {f['median']:.3g} [{f['p10']:.3g}, {f['p90']:.3g}]. "
            f"Current threshold {s['current_threshold']} costs "
            f"{_pct(s.get('current_cost_on_knowns'))} of knowns and catches "
            f"{_pct(s.get('current_catch_on_fps'))} of FPs. "
            + (f"Best available: {s['best_threshold']:.3g} "
               f"(cost {_pct(s['best_cost_on_knowns'])}, "
               f"catch {_pct(s['best_catch_on_fps'])}). "
               if s["best_threshold"] is not None else "No threshold can be scanned. ")
            + ("**Separable.**" if s["separable"] else
               "*Not separable on this sample.*"))

    lines += [
        "", "## Reading this honestly", "",
        "The samples are small.  A cost of 0.05 on 20 known planets means one",
        "planet; the difference between 0.05 and 0.10 is one object, not a",
        "measurement.  Use this to catch a test that deletes half the real",
        "planets, not to set a threshold to two significant figures.",
        "",
        "The false-positive sample is usually weaker still: dispositions are",
        "often by eye, some are 'probable' rather than certain, and a candidate",
        "is *not* evidence that it is a planet.  A high `catch` is encouraging;",
        "a low one is not proof of anything.",
        "",
        "Finally, `cost` counts FLAG as an objection.  Most of these tests flag",
        "rather than fail on purpose, and a flag is a request for a human, not",
        "a deletion -- so a test with a high cost and a high catch may still be",
        "doing exactly its job.",
    ]
    return "\n".join(lines) + "\n"


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


if __name__ == "__main__":
    raise SystemExit(main())

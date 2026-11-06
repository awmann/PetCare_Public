"""calibration/calibrate.py on a tiny synthetic run: it must count, filter by
truth per candidate, and read thresholds from the code rather than a copy."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "calibration"))
import calibrate  # noqa: E402


def _run(root, rows):
    root.mkdir()
    for r in rows:
        d = root / f"TIC{r['tic']}"
        d.mkdir(exist_ok=True)
        (d / "odd_even.json").write_text(json.dumps(
            {"status": r["odd_even_status"], "metrics": {"difference_sigma": r["sig"]}}))
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    return root / "summary.csv"


def test_calibrate_counts_and_filters(tmp_path):
    knowns = _run(tmp_path / "k", [
        {"tic": t, "name": f"K{t} b", "candidate_index": 0,
         "odd_even_status": s, "sig": g}
        for t, s, g in [(1, "PASS", 0.5), (2, "PASS", 1.0), (3, "PASS", 1.5),
                        (4, "FLAG", 3.5), (5, "INCONCLUSIVE", 0.2)]])
    fps = _run(tmp_path / "f", [
        {"tic": t, "name": n, "candidate_index": 0, "odd_even_status": s, "sig": g}
        for t, n, s, g in [(11, "F11 b", "FLAG", 6.0), (12, "F12 b", "FAIL", 9.0),
                           (13, "F13 b", "FLAG", 4.5), (14, "F14 b", "PASS", 0.1)]])
    sample = tmp_path / "fp.csv"
    pd.DataFrame({"tic": [11, 12, 13, 14], "label": ["F11 b", "F12 b", "F13 b", "F14 b"],
                  "truth": ["false_positive", "false_positive", "false_positive",
                            "candidate"]}).to_csv(sample, index=False)
    out = tmp_path / "out"
    assert calibrate.main(["--knowns", str(knowns), "--false-positives", str(fps),
                           "--fp-sample", str(sample), "--out", str(out)]) == 0
    conf = pd.read_csv(out / "confusion_matrix.csv").set_index("test")
    row = conf.loc["odd_even"]
    assert row["n_knowns_decided"] == 4 and row["knowns_objected"] == 1
    assert row["n_fp_decided"] == 3 and row["fp_objected"] == 3   # the candidate is not scored
    scan = json.loads((out / "threshold_scan.json").read_text())["odd_even.difference_sigma"]
    from petcare.scripts.pc_oddeven import ODD_EVEN_FLAG_SIGMA
    assert scan["current_threshold"] == ODD_EVEN_FLAG_SIGMA


def test_every_threshold_reference_resolves():
    for specs in calibrate.STATISTICS.values():
        for _, _, ref in specs:
            v = calibrate.current_threshold(ref)
            assert ref is None or isinstance(v, float)

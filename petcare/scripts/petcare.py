"""petcare -- run the tests over a candidate file, or over one candidate.

The generic driver CLI.  It hardcodes nothing: light-curve roots, the stellar
catalogue and the candidate list are all arguments (or environment variables).
The script that knows where those live on a particular machine belongs outside
this repository -- see ``docs/batch_runs.md``.

    # one candidate
    python -m petcare.scripts.petcare --tic 166527623 \\
        --period 6.959473 --t0 2458604.02376 --duration 4.85 --tiers 0 1

    # a whole table
    python -m petcare.scripts.petcare --candidates candidates.csv \\
        --tiers 0 1 --out out

Outputs, all under ``--out``:

* ``<TARGET>/<test>.json`` -- one record per test, written by the test itself;
* ``<TARGET>/<test>.png`` -- the plots, which are what a vetter reads;
* ``<TARGET>/petcare.json`` -- every record for that candidate, merged, with
  its ``vetting_metrics``: one flat set of numbers, reported and never graded;
* ``summary.csv`` -- one row per candidate across every target: the statuses,
  the reasons, and the vetting metrics as ``m_*`` columns.

**No overall score is computed.**  The summary table carries the per-test
statuses side by side, because which test objected is the information.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .. import driver
from ..cli import _read_table, _to_btjd


def build_parser():
    ap = argparse.ArgumentParser(
        prog="petcare",
        description="Run the PetCare vetting tests and merge the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("what to vet")
    g.add_argument("--candidates", help="candidate table (CSV/parquet)")
    g.add_argument("--tic", type=int)
    g.add_argument("--gaia", type=int)
    g.add_argument("--name")
    g.add_argument("--ra", type=float)
    g.add_argument("--dec", type=float)
    g.add_argument("--age", type=float, help="stellar age in Myr; in a table, "
                                             "an age_myr column")
    g.add_argument("--prot", type=float,
                   help="rotation period in days; in a table, a prot column. "
                        "Without it, measured with a Lomb-Scargle if the light "
                        "curve carries none")
    g.add_argument("--period", type=float)
    g.add_argument("--t0", type=float, help="BTJD or BJD")
    g.add_argument("--duration", type=float, help="T14 in hours")
    g.add_argument("--depth", type=float, help="ppm")
    g.add_argument("--index", type=int, default=0)
    g.add_argument("--limit", type=int, default=None,
                   help="stop after this many candidates")
    g.add_argument("--start", type=int, default=0,
                   help="skip this many rows of the candidate table")

    g = ap.add_argument_group("which tests")
    g.add_argument("--tiers", type=int, nargs="+", default=[0, 1],
                   help="0 catalogue, 1 light curve, 2 pixels, 3 archives")
    g.add_argument("--only", nargs="+", default=[],
                   help="run only these tests, whatever their tier")
    g.add_argument("--skip", nargs="+", default=[])
    g.add_argument("--list-tests", action="store_true")

    g = ap.add_argument_group("where things are")
    g.add_argument("--catalog", help="stellar parameter catalogue; else "
                                     "$PETCARE_STELLAR_CATALOG")
    g.add_argument("--lc-root", nargs="+", default=[],
                   help="light-curve directories; else $PETCARE_LC_ROOTS")
    g.add_argument("--lc", nargs="+", default=[],
                   help="light-curve files for a single --tic; a table finds "
                        "its files through --lc-root")
    g.add_argument("--known", nargs="+", default=[],
                   help="known signals: a JSON file, or 'label:P:t0:dur'")
    g.add_argument("--out", default="out")

    g = ap.add_argument_group("output")
    g.add_argument("--no-plot", action="store_true")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--append-summary", action="store_true",
                   help="merge into an existing summary.csv instead of "
                        "replacing it, so several invocations accumulate. "
                        "Rows are keyed on (tic, candidate_index).")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tests:
        print(f"{'test':<12s} {'tier':<5s} uses pc_fit")
        for name, _, tier, uses_fit in driver.TESTS:
            print(f"{name:<12s} {tier:<5d} {'yes' if uses_fit else 'no'}")
        return 0

    opts = driver.DriverOptions(
        out=args.out, tiers=tuple(args.tiers), only=tuple(args.only),
        skip=tuple(args.skip), catalog=args.catalog,
        lc_root=tuple(args.lc_root), lc=tuple(args.lc), known=tuple(args.known),
        no_plot=args.no_plot, quiet=args.quiet)

    if args.candidates and (args.lc or args.prot or args.age):
        raise SystemExit("--lc, --prot and --age describe one star; with "
                         "--candidates use --lc-root and prot/age_myr columns")
    jobs = _jobs(args)
    if not jobs:
        raise SystemExit("nothing to do: give --candidates, or --tic with "
                         "--period and --t0")

    rows, t0 = [], time.time()
    for i, (target, candidate) in enumerate(jobs, start=1):
        label = target.get("name") or f"TIC {target.get('tic')}"
        if not args.quiet:
            print(f"[{i}/{len(jobs)}] {label}  "
                  f"P={candidate.get('period')} d")
        merged = driver.run_candidate(target, candidate, opts)
        rows.append(driver.summary_row(merged))
        if not args.quiet:
            s = merged["summary"]["counts"]
            print(f"    -> {merged['path']}  "
                  + " ".join(f"{k}:{v}" for k, v in s.items() if v))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    _write_summary(rows, summary_path, append=args.append_summary)

    if not args.quiet:
        elapsed = time.time() - t0
        print(f"\n{len(rows)} candidate(s) in {elapsed:.0f} s "
              f"({elapsed / max(len(rows), 1):.1f} s each)")
        print(f"-> {summary_path}")
        _print_tally(rows)
    return 0


PERIOD_COLUMNS = ("period", "per", "pl_orbper", "period_days", "period_d")
T0_COLUMNS = ("t0_btjd", "t0", "T0", "pl_tranmid", "epoch")


def _jobs(args) -> list:
    """(target, candidate) pairs, from the table or from the command line."""
    if args.candidates:
        df = _read_table(args.candidates)
        jobs = []
        rows = df.iloc[args.start:]
        if args.limit:
            rows = rows.iloc[:args.limit]
        # per-star candidate index, so several candidates on one star land in
        # separate files rather than overwriting each other
        seen = {}
        # every other row on the same star is a known signal for this one:
        # each modelling test has to mask them, and matching is by transit
        # overlap inside PetCare so a harmonic is still recognised
        siblings = {}
        for _, row in df.iterrows():
            d = row.to_dict()
            tv = _first(d, ("tic", "tic_id", "TIC", "TIC ID", "ticid"))
            if tv is None:
                continue
            key = _tic(tv)
            per = _float(_first(d, PERIOD_COLUMNS))
            ep = _to_btjd(_float(_first(d, T0_COLUMNS)))
            du = _float(_first(d, ("duration_h", "duration", "T14", "pl_trandur")))
            lb = _first(d, ("label", "pl_name", "name")) or f"row{len(siblings)}"
            if per and ep is not None and du:
                siblings.setdefault(key, []).append(
                    (str(lb).replace(" ", "_"), per, ep, du))

        # rows skipped by --start still count, or their results get overwritten
        for _, row in df.iloc[:args.start].iterrows():
            tic = _tic(_first(row.to_dict(), ("tic", "tic_id", "TIC", "TIC ID", "ticid")))
            seen[tic] = seen.get(tic, 0) + 1

        for _, row in rows.iterrows():
            d = row.to_dict()
            tic = _tic(_first(d, ("tic", "tic_id", "TIC", "TIC ID", "ticid")))
            idx = seen.get(tic, 0)
            seen[tic] = idx + 1
            target = {
                "tic": tic,
                "gaia": _int(_first(d, ("gaia_dr3", "gaia_dr3_source_id",
                                        "source_id"))),
                "name": _first(d, ("label", "pl_name", "name", "candidate")),
                "ra": _float(_first(d, ("ra", "RA", "ra_deg"))),
                "dec": _float(_first(d, ("dec", "Dec", "DEC", "dec_deg"))),
                # A candidate table often knows the cluster age when the
                # stellar catalogue does not, and the radius ceiling in
                # pc_duration is age-dependent -- without it every target
                # falls back to the no-age default.
                "age_myr": _float(_first(d, ("age_myr", "age", "cluster_age_myr"))),
                "prot": _float(_first(d, ("prot", "prot_d", "prot_days"))),
            }
            candidate = {
                "period": _float(_first(d, PERIOD_COLUMNS)),
                "t0": _to_btjd(_float(_first(d, T0_COLUMNS))),
                "duration": _float(_first(d, ("duration_h", "duration", "T14",
                                              "pl_trandur"))),
                "depth": _float(_first(d, ("depth_ppm", "depth", "Depth",
                                         "depth_gp_ppm"))),
                "index": idx,
                "known": [f"{lb}:{per}:{ep}:{du}"
                          for (lb, per, ep, du) in siblings.get(tic, [])
                          if abs(per - (_float(_first(d, PERIOD_COLUMNS)) or -1))
                          > 1e-9],
            }
            jobs.append((target, candidate))
        return jobs

    if args.tic is None and args.gaia is None:
        return []
    return [({"tic": args.tic, "gaia": args.gaia, "name": args.name,
              "ra": args.ra, "dec": args.dec, "age_myr": args.age,
              "prot": args.prot},
             {"period": args.period, "t0": _to_btjd(args.t0),
              "duration": args.duration, "depth": args.depth,
              "index": args.index})]


def _write_summary(rows, path, append: bool = False) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    if append and Path(path).exists():
        try:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
            # last write wins for a repeated candidate, so a re-run of one
            # target updates its row rather than duplicating it
            df = df.drop_duplicates(subset=["tic", "candidate_index"],
                                    keep="last")
        except Exception:  # noqa: BLE001 - a corrupt old file must not lose
            # the rows just computed
            pass
    # keep the identifying columns first; the rest in a stable order
    lead = [c for c in ("tic", "name", "candidate_index", "period", "t0_btjd",
                        "duration_h", "depth_ppm", "n_fail", "n_flag",
                        "n_inconclusive", "n_error", "objections")
            if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(path, index=False)


def _print_tally(rows) -> None:
    tests = sorted({c[:-7] for r in rows for c in r if c.endswith("_status")})
    if not tests:
        return
    print(f"\n{'test':<12s} {'PASS':>5s} {'FLAG':>5s} {'FAIL':>5s} "
          f"{'INCON':>6s} {'ERROR':>6s}")
    for t in tests:
        counts = {}
        for r in rows:
            counts[r.get(f"{t}_status")] = counts.get(r.get(f"{t}_status"), 0) + 1
        print(f"{t:<12s} {counts.get('PASS', 0):>5d} {counts.get('FLAG', 0):>5d} "
              f"{counts.get('FAIL', 0):>5d} {counts.get('INCONCLUSIVE', 0):>6d} "
              f"{counts.get('ERROR', 0):>6d}")


def _first(d, names):
    for n in names:
        if n in d and d[n] is not None and d[n] == d[n] and str(d[n]).strip():
            return d[n]
    return None


def _tic(x):
    """TIC from a table cell: 'TIC 100', 100, or the 100.0 a float column makes of it."""
    return None if x is None else _int(str(x).replace("TIC", "").strip())


def _float(x):
    try:
        import math
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _int(x):
    """Exactly, when ``x`` is already an integer or its string: a Gaia source_id
    does not survive float(), which rounds 78% of them by up to 512."""
    try:
        return int(x)
    except (TypeError, ValueError):
        pass
    v = _float(x)
    return int(v) if v is not None else None


if __name__ == "__main__":
    sys.exit(main())

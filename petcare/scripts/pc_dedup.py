"""pc_dedup -- the same signal appearing on several stars.

A cross-target test.  It does not fit the per-candidate schema and does not
produce one JSON per candidate; it runs over the whole candidate table once
and annotates it.

**The problem.**  A bright eclipsing binary contaminates every star within a
few arcminutes.  The search then finds an apparently independent candidate on
each of them, with the same period and the same epoch, and every one of them
looks like a discovery.  At 48 targets a human notices.  At 30k they do not,
and a single missed cluster puts a dozen fake candidates on a follow-up list.

**The method.**  Group candidates that share a period *and* an epoch *and* a
patch of sky:

1. period within a few times its own uncertainty -- or, absent one, within a
   tolerance scaled to how far the fold drifts over the baseline;
2. transit times that actually coincide, checked by **overlap** rather than by
   epoch subtraction, so a group is not broken by one member's stale t0;
3. sky separation inside ``--radius`` arcminutes.

Within each group the true source is normally the star with the deepest signal
and, where a centroid result exists, the best-centred one.  The rest are
almost certainly NEBs.  This is close to free and it runs before any
pixel-level work.

**It ranks, it does not delete.**  The group's members are annotated with
``dedup_group``, ``dedup_rank`` and ``dedup_likely_source``; deciding what to
drop is a human's job, and the deepest member is not always the right answer
when apertures differ.

    python -m petcare.scripts.pc_dedup --candidates candidates.csv --out out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import ephem
from ..result import utc_now

TEST = "dedup"
VERSION = "0.1.0"

#: Default sky radius for grouping, in arcminutes.  A few TESS pixels; a
#: contaminating EB much further away than this cannot put a detectable signal
#: in an aperture.
DEFAULT_RADIUS_ARCMIN = 5.0

#: Fractional period tolerance used when no period uncertainty is supplied.
DEFAULT_PERIOD_TOL = 0.002

#: Fraction of a duration inside which two transit times count as coincident.
EPOCH_TOL_DURATIONS = 0.5


def build_parser():
    ap = argparse.ArgumentParser(
        prog="pc_dedup",
        description="Find candidates that are the same signal on several stars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="candidate table (CSV/parquet) with tic, ra, dec, "
                         "period, t0 and ideally duration and depth")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_ARCMIN,
                    help="grouping radius in arcminutes")
    ap.add_argument("--period-tol", type=float, default=DEFAULT_PERIOD_TOL,
                    help="fractional period tolerance when the table has no "
                         "period uncertainty")
    ap.add_argument("--min-group", type=int, default=2,
                    help="report groups with at least this many members")
    ap.add_argument("--quiet", action="store_true")
    return ap


def run(args) -> dict:

    df = _read(args.candidates)
    cols = _resolve_columns(df)
    missing = [k for k in ("tic", "ra", "dec", "period", "t0") if cols.get(k) is None]
    if missing:
        raise SystemExit(
            f"{args.candidates} is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}")

    n = len(df)
    tic = df[cols["tic"]].astype(str).str.replace("TIC", "", regex=False).str.strip()
    ra = df[cols["ra"]].astype(float).to_numpy()
    dec = df[cols["dec"]].astype(float).to_numpy()
    period = df[cols["period"]].astype(float).to_numpy()
    t0 = df[cols["t0"]].astype(float).to_numpy()
    t0 = np.where(t0 > 2_400_000, t0 - 2_457_000.0, t0)
    dur = (df[cols["duration"]].astype(float).to_numpy() if cols.get("duration")
           else np.full(n, 3.0))
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, 3.0)
    depth = (df[cols["depth"]].astype(float).to_numpy() if cols.get("depth")
             else np.full(n, np.nan))
    p_err = (df[cols["period_err"]].astype(float).to_numpy()
             if cols.get("period_err") else np.full(n, np.nan))

    radius_deg = float(args.radius) / 60.0
    groups = _group(ra, dec, period, t0, dur, p_err, radius_deg,
                    float(args.period_tol))

    # ---- annotate ---------------------------------------------------------
    out = df.copy()
    out["dedup_group"] = -1
    out["dedup_group_size"] = 1
    out["dedup_rank"] = 1
    out["dedup_likely_source"] = True
    out["dedup_note"] = ""

    reports = []
    for gi, members in enumerate(g for g in groups if len(g) >= args.min_group):
        idx = np.array(sorted(members))
        # deepest first; NaN depths sort last so a table with no depths keeps
        # its own order rather than being ranked arbitrarily
        d = depth[idx]
        order = idx[np.argsort(np.where(np.isfinite(d), -d, np.inf), kind="stable")]
        sep_from_lead = _sep_arcsec(ra[order[0]], dec[order[0]], ra[order], dec[order])

        for rank, j in enumerate(order, start=1):
            out.loc[out.index[j], "dedup_group"] = gi
            out.loc[out.index[j], "dedup_group_size"] = len(order)
            out.loc[out.index[j], "dedup_rank"] = rank
            out.loc[out.index[j], "dedup_likely_source"] = (rank == 1)
            out.loc[out.index[j], "dedup_note"] = (
                f"shares P={period[j]:.5f} d and epoch with "
                f"{len(order) - 1} other star(s) within {args.radius:g} arcmin; "
                + ("deepest member, so the likely true source"
                   if rank == 1 else
                   f"rank {rank} by depth behind TIC {tic.iloc[order[0]]} at "
                   f"{sep_from_lead[rank - 1]:.0f} arcsec -- probably a NEB"))

        reports.append({
            "group": gi,
            "n_members": len(order),
            "period_days": float(np.median(period[order])),
            "period_spread_frac": float(
                (np.max(period[order]) - np.min(period[order])) /
                np.median(period[order])),
            "max_separation_arcsec": float(np.max(sep_from_lead)),
            "likely_source": {
                "tic": tic.iloc[order[0]], "depth_ppm": _f(depth[order[0]]),
                "ra": float(ra[order[0]]), "dec": float(dec[order[0]])},
            "members": [
                {"tic": tic.iloc[j], "rank": rank,
                 "depth_ppm": _f(depth[j]), "period": float(period[j]),
                 "t0_btjd": float(t0[j]),
                 "sep_from_source_arcsec": float(sep_from_lead[rank - 1])}
                for rank, j in enumerate(order, start=1)],
        })

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    table_path = outdir / "dedup_annotated.csv"
    out.to_csv(table_path, index=False)

    summary = {
        "test": TEST, "version": VERSION, "run_utc": utc_now(),
        "candidates_file": str(args.candidates),
        "n_candidates": int(n),
        "n_groups": len(reports),
        "n_candidates_in_groups": int(sum(r["n_members"] for r in reports)),
        "n_likely_nebs": int(sum(r["n_members"] - 1 for r in reports)),
        "radius_arcmin": float(args.radius),
        "period_tolerance_frac": float(args.period_tol),
        "epoch_tolerance_durations": EPOCH_TOL_DURATIONS,
        "annotated_table": str(table_path),
        "groups": reports,
        "note": ("ranking only -- nothing is deleted. The deepest member is "
                 "the likely source, but that is a heuristic: apertures "
                 "differ between stars and a fainter star with a tighter "
                 "aperture can show the deeper signal."),
    }
    summary_path = outdir / "dedup.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    if not args.quiet:
        print(f"{n} candidates, {len(reports)} shared-signal group(s), "
              f"{summary['n_likely_nebs']} likely NEB(s)")
        for rep in reports[:20]:
            lead = rep["likely_source"]
            print(f"  group {rep['group']}: P = {rep['period_days']:.5f} d, "
                  f"{rep['n_members']} stars within "
                  f"{rep['max_separation_arcsec']:.0f} arcsec; likely source "
                  f"TIC {lead['tic']}"
                  + (f" ({lead['depth_ppm']:.0f} ppm)"
                     if lead["depth_ppm"] else ""))
        if len(reports) > 20:
            print(f"  ... and {len(reports) - 20} more")
        print(f"  -> {table_path}")
        print(f"  -> {summary_path}")
    return summary


# --------------------------------------------------------------------------


def _group(ra, dec, period, t0, dur, p_err, radius_deg, period_tol):
    """Union-find over pairs that share sky position, period and epoch."""
    n = len(ra)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # sky first: it is the cheapest filter and removes almost every pair
    order = np.argsort(dec)
    for a_pos in range(n):
        i = order[a_pos]
        for b_pos in range(a_pos + 1, n):
            j = order[b_pos]
            if dec[j] - dec[i] > radius_deg:
                break
            if _sep_arcsec(ra[i], dec[i], ra[j], dec[j]) > radius_deg * 3600:
                continue
            if not _same_period(period[i], period[j], p_err[i], p_err[j],
                                period_tol):
                continue
            if not _same_epoch(period[i], t0[i], dur[i], period[j], t0[j], dur[j]):
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def _sep_arcsec(ra1, dec1, ra2, dec2):
    dra = (np.asarray(ra2, float) - ra1) * np.cos(np.radians(dec1)) * 3600.0
    ddec = (np.asarray(dec2, float) - dec1) * 3600.0
    return np.hypot(dra, ddec)


#: Period ratios accepted as "the same signal".  Harmonics are included on
#: purpose: the search routinely proposes P/2 or 2P on a contaminated star,
#: and requiring equal periods breaks exactly the groups this test exists to
#: find.  The ratio only gets a candidate pair as far as the epoch-overlap
#: test, which is what actually decides.
PERIOD_RATIOS = (1.0, 2.0, 0.5, 3.0, 1 / 3)


def _same_period(p1, p2, e1, e2, tol) -> bool:
    if not (np.isfinite(p1) and np.isfinite(p2) and p1 > 0 and p2 > 0):
        return False
    err = np.nanmax([e1 if np.isfinite(e1) else 0.0,
                     e2 if np.isfinite(e2) else 0.0])
    for ratio in PERIOD_RATIOS:
        target = p2 * ratio
        allowed = max(3.0 * err * max(ratio, 1.0), tol * 0.5 * (p1 + target))
        if abs(p1 - target) <= allowed:
            return True
    return False


def _same_epoch(p1, t01, d1, p2, t02, d2) -> bool:
    """Do the two transit trains actually coincide?

    Compared as *sets of transit times* over a common window, not by
    subtracting epochs.  One member of a real group often has a stale t0 or a
    period off by a cycle count, and epoch subtraction breaks the group
    exactly there -- which is the case the test exists to catch.
    """
    span = 10 * max(p1, p2)
    t_min = min(t01, t02)
    tt1 = ephem.transit_times(p1, t01, t_min, t_min + span)
    tt2 = ephem.transit_times(p2, t02, t_min, t_min + span)
    if tt1.size == 0 or tt2.size == 0:
        return False
    tol = EPOCH_TOL_DURATIONS * min(d1, d2) / 24.0
    # Both directions, and take the better.  At a 2:1 harmonic only half of
    # the shorter-period train can coincide, so a one-directional test scores
    # 0.5 and throws the pair away -- which is the harmonic case the grouping
    # is supposed to catch.
    f12 = float(np.mean(np.min(np.abs(tt1[:, None] - tt2[None, :]), axis=1) <= tol))
    f21 = float(np.mean(np.min(np.abs(tt2[:, None] - tt1[None, :]), axis=1) <= tol))
    return bool(max(f12, f21) >= 0.8)


_COLUMN_ALIASES = {
    "tic": ["tic", "tic_id", "TIC", "TIC ID", "ticid"],
    "ra": ["ra", "RA", "ra_deg", "RA (deg)"],
    "dec": ["dec", "Dec", "DEC", "dec_deg", "Dec (deg)"],
    "period": ["period", "per", "P", "period_days", "Period", "pl_orbper"],
    "period_err": ["period_err", "e_period", "period_error"],
    "t0": ["t0", "t0_btjd", "T0", "epoch", "pl_tranmid"],
    "duration": ["duration_h", "duration", "T14", "dur_hours", "pl_trandur"],
    "depth": ["depth_ppm", "depth", "Depth", "pl_trandep"],
}


def _resolve_columns(df) -> dict:
    lower = {str(c).lower(): c for c in df.columns}
    out = {}
    for key, names in _COLUMN_ALIASES.items():
        out[key] = next((lower[n.lower()] for n in names if n.lower() in lower),
                        None)
    return out


def _read(path):
    import pandas as pd
    p = str(path)
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

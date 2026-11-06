"""pc_known -- is this already a known object?

A cross-target test: it runs over the whole candidate table once and
annotates it, rather than producing one JSON per candidate.

Cross-matches every candidate against:

* the **ExoFOP TOI table** -- TOI number, TFOPWG disposition, and the
  follow-up already taken.  Matched by transit overlap, not period, so a
  candidate at half or twice a TOI's period is still recognised as the same
  object;
* **Gaia DR3 variability** -- the classifier's best class and the eclipsing
  binary catalogue, queried in batches of 500 source_ids rather than one per
  target, because at 30k targets the round trip *is* the cost;
* optionally **SIMBAD**, for an object type.  Off by default, and throttled to
  one query every two seconds when on: SIMBAD is at CDS, CDS is rate-limited
  per IP, and an IP it has filtered breaks ordinary interactive use for days.

Two things come out of it.  The obvious one is not rediscovering a catalogued
eclipsing binary.  The more useful one is the opposite: a candidate that
matches a **confirmed planet** is a check on the pipeline itself, and one that
matches a **TFOPWG false positive** is a chance to read somebody's notes
before spending a night of telescope time.

**Which catalogues were consulted is recorded in the output**, including the
ones that were not.  ASAS-SN and ZTF variable-star catalogues are *not* wired
in here; the record says so rather than leaving a reader to assume they were.

    python -m petcare.scripts.pc_known --candidates candidates.csv --out out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import catalogs, ephem
from ..result import utc_now

TEST = "known"
VERSION = "0.1.0"

#: Gaia variability classes that mean "this is an eclipsing binary".
EB_CLASSES = {"ECL", "ECLIPSING", "EB"}

#: Catalogues this test does not consult.  Named so that nobody has to guess.
NOT_CONSULTED = [
    "ASAS-SN variable star catalogue",
    "ZTF variable star catalogue",
    "Kepler/K2 EB catalogues",
    "the TCE list (only the TOI list is used)",
]


def build_parser():
    ap = argparse.ArgumentParser(
        prog="pc_known",
        description="Cross-match candidates against TOIs and variable-star "
                    "catalogues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="candidate table (CSV/parquet) with tic, period, t0")
    ap.add_argument("--out", default="out")
    ap.add_argument("--simbad", action="store_true",
                    help="also query SIMBAD for an object type. Throttled to "
                         "one query every 2 s (CDS is rate-limited per IP); "
                         "do not use this on a 30k-row table.")
    ap.add_argument("--max-age-days", type=float, default=7.0,
                    help="re-download the TOI table if the cache is older")
    ap.add_argument("--quiet", action="store_true")
    return ap


def run(args) -> dict:
    from .pc_dedup import _read, _resolve_columns

    df = _read(args.candidates)
    cols = _resolve_columns(df)
    if cols.get("tic") is None:
        raise SystemExit(f"{args.candidates} has no TIC column; found "
                         f"{list(df.columns)}")

    n = len(df)
    tic = (df[cols["tic"]].astype(str).str.replace("TIC", "", regex=False)
           .str.strip())
    period = (df[cols["period"]].astype(float).to_numpy()
              if cols.get("period") else np.full(n, np.nan))
    t0 = (df[cols["t0"]].astype(float).to_numpy()
          if cols.get("t0") else np.full(n, np.nan))
    t0 = np.where(t0 > 2_400_000, t0 - 2_457_000.0, t0)
    dur = (df[cols["duration"]].astype(float).to_numpy()
           if cols.get("duration") else np.full(n, 3.0))
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, 3.0)

    consulted, failed = [], []

    # ---- TOI table --------------------------------------------------------
    toi_by_tic = {}
    try:
        toi = catalogs.toi_table(max_age_days=float(args.max_age_days))
        consulted.append("ExoFOP TOI table")
        for _, row in toi.iterrows():
            key = str(int(row["TIC ID"])) if np.isfinite(row["TIC ID"]) else None
            if key:
                toi_by_tic.setdefault(key, []).append(row.to_dict())
    except Exception as exc:  # noqa: BLE001 - an outage is not a verdict
        failed.append(f"ExoFOP TOI table: {exc}")

    # ---- Gaia variability, batched ---------------------------------------
    gaia_ids = {}
    gaia_col = next((c for c in df.columns
                     if str(c).lower() in ("gaia_dr3", "gaia_dr3_source_id",
                                           "source_id", "gaia")), None)
    if gaia_col is not None:
        for i, v in enumerate(df[gaia_col].tolist()):
            try:
                gaia_ids[i] = int(v)
            except (TypeError, ValueError):
                pass
    vari = {}
    if gaia_ids:
        try:
            vari = catalogs.gaia_variability(gaia_ids.values())
            consulted.append("Gaia DR3 vari_classifier_result and "
                             "vari_eclipsing_binary")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"Gaia variability: {exc}")
    else:
        failed.append("Gaia variability: the candidate table has no Gaia "
                      "source_id column, so no variability cross-match was "
                      "possible")

    # ---- annotate ---------------------------------------------------------
    out = df.copy()
    for col, default in (("known_toi", ""), ("known_disposition", ""),
                         ("known_match_relation", ""), ("known_gaia_class", ""),
                         ("known_is_gaia_eb", False),
                         ("known_n_imaging", 0), ("known_n_spectroscopy", 0),
                         ("known_note", "")):
        out[col] = default

    counts = {"matched_toi": 0, "confirmed": 0, "refuted": 0, "gaia_eb": 0,
              "toi_on_star_but_different_signal": 0, "nothing": 0}
    details = []

    for i in range(n):
        key = tic.iloc[i]
        rows = toi_by_tic.get(key, [])
        note_bits = []
        match = None

        if rows and np.isfinite(period[i]) and np.isfinite(t0[i]):
            known = []
            for row in rows:
                p_r = row.get("Period (days)")
                e_r = row.get("Epoch (BJD)")
                if not (p_r and np.isfinite(p_r) and e_r and np.isfinite(e_r)):
                    continue
                known.append({"label": f"TOI {row.get('TOI')}",
                              "period": float(p_r),
                              "t0_btjd": float(e_r) - 2457000.0,
                              "duration_h": float(row.get("Duration (hours)")
                                                  or dur[i])})
            if known:
                span = (min(t0[i], min(k["t0_btjd"] for k in known)),
                        max(t0[i], max(k["t0_btjd"] for k in known)) + 200.0)
                match = ephem.match_known(period[i], t0[i], dur[i], known,
                                          span[0], span[1])

        if match:
            row = next(r for r in rows if f"TOI {r.get('TOI')}" == match["label"])
            disp = str(row.get("TFOPWG Disposition") or "")
            out.loc[out.index[i], "known_toi"] = str(row.get("TOI"))
            out.loc[out.index[i], "known_disposition"] = disp
            out.loc[out.index[i], "known_match_relation"] = match["relation"]
            out.loc[out.index[i], "known_n_imaging"] = _i(
                row.get("Imaging Observations"))
            out.loc[out.index[i], "known_n_spectroscopy"] = _i(
                row.get("Spectroscopy Observations"))
            note_bits.append(
                f"{match['label']} ({match['relation']}), disposition {disp} = "
                f"{catalogs.TFOPWG_DISPOSITIONS.get(disp, 'unknown')}")
            counts["matched_toi"] += 1
            if disp in ("CP", "KP"):
                counts["confirmed"] += 1
            elif disp in ("FP", "FA"):
                counts["refuted"] += 1
        elif rows:
            counts["toi_on_star_but_different_signal"] += 1
            note_bits.append(
                f"{len(rows)} TOI(s) on this star "
                f"({', '.join(str(r.get('TOI')) for r in rows)}) but none "
                f"share transit times with this candidate")

        g = gaia_ids.get(i)
        v = vari.get(g) if g else None
        if v:
            cls = str(v.get("class") or "")
            out.loc[out.index[i], "known_gaia_class"] = cls
            is_eb = bool(v.get("eclipsing_binary")) or cls.upper() in EB_CLASSES
            out.loc[out.index[i], "known_is_gaia_eb"] = is_eb
            if is_eb:
                counts["gaia_eb"] += 1
                note_bits.append(
                    "Gaia DR3 eclipsing binary"
                    + (f" at P = {v['eb_period_days']:.5f} d"
                       if v.get("eb_period_days") else ""))
            elif cls:
                note_bits.append(f"Gaia variability class {cls}"
                                 + (f" (score {v['class_score']:.2f})"
                                    if v.get("class_score") else ""))

        if not note_bits:
            counts["nothing"] += 1
        out.loc[out.index[i], "known_note"] = "; ".join(note_bits)
        if note_bits:
            details.append({"row": int(i), "tic": key,
                            "note": "; ".join(note_bits)})

    # ---- optional SIMBAD --------------------------------------------------
    if args.simbad:
        simbad = _simbad_types(df, cols, out)
        consulted.append("SIMBAD (throttled to one query every 2 s)")
        for i, otype in simbad.items():
            out.loc[out.index[i], "known_note"] = (
                (out.loc[out.index[i], "known_note"] + "; " if
                 out.loc[out.index[i], "known_note"] else "")
                + f"SIMBAD type {otype}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    table_path = outdir / "known_annotated.csv"
    out.to_csv(table_path, index=False)

    summary = {
        "test": TEST, "version": VERSION, "run_utc": utc_now(),
        "candidates_file": str(args.candidates),
        "n_candidates": int(n),
        "counts": counts,
        "catalogues_consulted": consulted,
        "catalogues_that_failed": failed,
        "catalogues_not_consulted": NOT_CONSULTED,
        "annotated_table": str(table_path),
        "details": details[:500],
        "details_truncated": len(details) > 500,
        "note": ("matching is by transit overlap, never by period proximity: "
                 "a candidate at half or twice a TOI's period is the same "
                 "object and is recognised as one"),
    }
    (outdir / "known.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=str))

    if not args.quiet:
        print(f"{n} candidates cross-matched")
        print(f"  {counts['matched_toi']} match a TOI "
              f"({counts['confirmed']} confirmed/known planets, "
              f"{counts['refuted']} TFOPWG false positives)")
        print(f"  {counts['toi_on_star_but_different_signal']} sit on a star "
              f"that has a TOI, but are a different signal")
        print(f"  {counts['gaia_eb']} are catalogued Gaia eclipsing binaries")
        print(f"  {counts['nothing']} match nothing")
        if failed:
            for f in failed:
                print(f"  NOT CHECKED: {f}")
        print(f"  not consulted: {', '.join(NOT_CONSULTED)}")
        print(f"  -> {table_path}")
        print(f"  -> {outdir / 'known.json'}")
    return summary


def _simbad_types(df, cols, out) -> dict:
    """SIMBAD object types, one throttled query per row.

    Deliberately awkward to reach (``--simbad``) and deliberately slow.  CDS
    is rate-limited per IP and being filtered by it breaks ordinary
    interactive use for days, so this is not something to point at a 30k-row
    table.
    """
    from ..cache import cached, cds_throttle

    ra_col, dec_col = cols.get("ra"), cols.get("dec")
    if ra_col is None or dec_col is None:
        return {}

    def query(ra, dec):
        cds_throttle()
        from astroquery.simbad import Simbad
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        s = Simbad()
        t = s.query_region(SkyCoord(ra, dec, unit="deg"), radius=10 * u.arcsec)
        if t is None or len(t) == 0:
            return None
        col = next((c for c in t.colnames if c.lower() in ("otype", "main_type")),
                   None)
        return str(t[col][0]) if col else None

    out_types = {}
    for i in range(len(df)):
        ra, dec = float(df[ra_col].iloc[i]), float(df[dec_col].iloc[i])
        try:
            otype = cached("simbad", round(ra, 5), round(dec, 5))(
                lambda: query(ra, dec))
        except Exception:  # noqa: BLE001 - one failure must not stop the run
            continue
        if otype:
            out_types[i] = otype
    return out_types


def _i(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return 0
    return v


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

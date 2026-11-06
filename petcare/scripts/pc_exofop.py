"""pc_exofop -- does follow-up already exist, and is this already a TOI?

Reads the ExoFOP TOI table and reports what the community already knows about
this star: whether it is a TOI, what the TFOPWG disposition is, and how much
high-resolution imaging, spectroscopy and time-series follow-up has already
been taken.

Two uses, and the second is the one that saves time.

**Do not re-derive what is known.**  A star already dispositioned FP by TFOPWG
does not need a night of Goodman time.  A star with five existing speckle
observations does not need a sixth before someone looks at the ones that exist.

**Is the candidate the TOI, or something else?**  Matching is by transit
overlap, not by period, so a candidate at half or twice a TOI's period is
recognised as the same object.  Period matching fails at exactly the harmonics
the search produces, and that failure previously cost 23% of the known planets
in this sample before any vetting statistic was computed.

The whole TOI table is downloaded once and cached for a week rather than
queried per target: ExoFOP's per-target endpoints time out often enough that
pulling 8000 rows and looking them up locally is both faster and kinder.

    python -m petcare.scripts.pc_exofop --tic 410214986 --period 8.1383 --t0 1332.31
"""

from __future__ import annotations

import sys

import numpy as np

from .. import catalogs, cli, ephem
from ..result import Result

TEST = "exofop"
VERSION = "0.1.0"

#: Dispositions that mean somebody has already decided this is not a planet.
REFUTED = {"FP", "FA"}

#: ...and that it is one.
CONFIRMED = {"CP", "KP"}

#: Depth below which no ground-based facility will confirm a transit.
#: Correlated wiggles at the few-hundred-ppm level do not bin down, so
#: anything under roughly a millimagnitude needs space photometry.  Reported
#: here so nobody discovers it after the observing time is spent.
GROUND_LIMIT_PPM = 1000.0


def add_arguments(ap):
    g = ap.add_argument_group("exofop")
    g.add_argument("--max-age-days", type=float, default=7.0,
                   help="re-download the TOI table if the cache is older")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.target.tic is not None,
                       "pc_exofop needs a TIC: ExoFOP is indexed by it"):
        return r

    try:
        rows = catalogs.toi_rows(int(ctx.target.tic),
                                 max_age_days=float(getattr(a, "max_age_days", 7.0)))
    except Exception as exc:  # noqa: BLE001 - an ExoFOP outage is not a verdict
        return r.inconclusive(f"could not read the ExoFOP TOI table: {exc}")

    r.set_provenance(lightcurve=None, source="ExoFOP TOI table",
                     url=catalogs.TOI_URL)
    r.set_noise(None, "not applicable: this is a catalogue test")

    tois = []
    for row in rows:
        tois.append({
            "toi": row.get("TOI"),
            "tfopwg_disposition": row.get("TFOPWG Disposition"),
            "tfopwg_meaning": catalogs.TFOPWG_DISPOSITIONS.get(
                str(row.get("TFOPWG Disposition")), "unknown"),
            "tess_disposition": row.get("TESS Disposition"),
            "period_days": _f(row.get("Period (days)")),
            "epoch_btjd": (_f(row.get("Epoch (BJD)")) - 2457000.0
                           if _f(row.get("Epoch (BJD)")) else None),
            "duration_h": _f(row.get("Duration (hours)")),
            "depth_ppm": _f(row.get("Depth (ppm)")),
            "planet_name": row.get("Planet Name"),
            "imaging_observations": row.get("Imaging Observations"),
            "spectroscopy_observations": row.get("Spectroscopy Observations"),
            "time_series_observations": row.get("Time Series Observations"),
            "comments": row.get("Comments"),
            "source": row.get("Source"),
        })

    n_imaging = max((t["imaging_observations"] or 0) for t in tois) if tois else 0
    n_spec = max((t["spectroscopy_observations"] or 0) for t in tois) if tois else 0
    n_ts = max((t["time_series_observations"] or 0) for t in tois) if tois else 0

    # ---- does our candidate correspond to one of these? -------------------
    match = None
    if ctx.candidate.period and ctx.candidate.t0_btjd is not None:
        dur = ctx.candidate.duration_h or 3.0
        known = [{"label": f"TOI {t['toi']}", "period": t["period_days"],
                  "t0_btjd": t["epoch_btjd"],
                  "duration_h": t["duration_h"] or dur}
                 for t in tois
                 if t["period_days"] and t["epoch_btjd"] is not None]
        if known:
            span = _time_span(ctx)
            match = ephem.match_known(ctx.candidate.period, ctx.candidate.t0_btjd,
                                      dur, known, span[0], span[1])

    r.metrics.update({
        "n_tois": len(tois), "tois": tois,
        "n_imaging_observations": n_imaging,
        "n_spectroscopy_observations": n_spec,
        "n_time_series_observations": n_ts,
        "dispositions": sorted({str(t["tfopwg_disposition"]) for t in tois
                                if t["tfopwg_disposition"]}),
        "candidate_matches_toi": match,
    })

    depth = ctx.candidate.depth_ppm
    if depth:
        r.metrics["confirmable_from_ground"] = bool(depth >= GROUND_LIMIT_PPM)
        r.metrics["ground_limit_ppm"] = GROUND_LIMIT_PPM

    r.set_self_check(True, f"{len(tois)} TOI row(s) for TIC {ctx.target.tic}")

    # ---- the verdict ------------------------------------------------------
    followup = (f"{n_imaging} imaging, {n_spec} spectroscopy and {n_ts} "
                f"time-series observations already on ExoFOP")
    ground = ""
    if depth:
        ground = (f". At {depth:.0f} ppm this is "
                  + ("above" if depth >= GROUND_LIMIT_PPM else "below")
                  + f" the ~{GROUND_LIMIT_PPM:.0f} ppm floor for ground-based "
                    f"confirmation"
                  + ("" if depth >= GROUND_LIMIT_PPM else
                     " -- correlated wiggles at this level do not bin down, "
                     "so it needs space photometry"))

    if not tois:
        r.passed(f"TIC {ctx.target.tic} is not a TOI and has no ExoFOP "
                 f"follow-up recorded{ground}")
    elif match:
        t = next((x for x in tois if f"TOI {x['toi']}" == match["label"]), {})
        disp = str(t.get("tfopwg_disposition"))
        meaning = catalogs.TFOPWG_DISPOSITIONS.get(disp, "unknown disposition")
        detail = (f"the candidate is {match['label']} ({match['relation']}), "
                  f"dispositioned {disp} = {meaning}")
        if disp in REFUTED:
            r.flag(f"{detail}. Somebody has already refuted this; read their "
                   f"notes before spending anything on it. {followup}{ground}")
        elif disp in CONFIRMED:
            r.passed(f"{detail}. Recovering a confirmed planet is a working "
                     f"pipeline, not a discovery. {followup}{ground}")
        else:
            r.flag(f"{detail}. {followup}{ground}")
    else:
        names = ", ".join(f"TOI {t['toi']} ({t['tfopwg_disposition']})"
                          for t in tois)
        r.flag(f"this star has {len(tois)} TOI(s) -- {names} -- but none of "
               f"them share transit times with this candidate, so it is a "
               f"separate signal on a star somebody is already watching. "
               f"{followup}{ground}")
    return r


def _time_span(ctx):
    """The time range to compare transit trains over.

    The light curve's own span when there is one; otherwise a generous window
    around the candidate epoch, which is enough for the overlap test since it
    only needs several cycles of each ephemeris.
    """
    try:
        lc = ctx.lc
        return float(lc.t.min()), float(lc.t.max())
    except Exception:  # noqa: BLE001 - no light curve is fine here
        t0 = float(ctx.candidate.t0_btjd)
        return t0 - 500.0, t0 + 500.0


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Is this a TOI, what did TFOPWG decide, and what follow-up "
              "already exists?"))
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

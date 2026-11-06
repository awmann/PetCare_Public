"""pc_instrument -- does the signal track the spacecraft rather than the star?

The category the previous pipelines lacked entirely, and at 30k targets it is
expected to be a leading false-positive mode.  Periods near harmonics of the
~13.7 d TESS orbit, or of the ~27 d sector length, manufacture convincing
transits out of scattered light, downlink gaps and momentum dumps.

What this test asks:

* **Is the period an alias of an instrumental one?**  The spacecraft orbit and
  its harmonics, the sector length, and the interval between the data gaps
  this light curve actually has -- the last measured from the data rather than
  assumed, since downlink cadence changed between mission phases.
* **Do the transits sit on the gaps?**  What fraction of epochs have ingress
  or egress in a gap, sit at a sector edge, or fall within a downlink window.
  A signal whose every transit hides behind a gap edge is a detrending
  artefact.
* **Is the depth the same across sectors, cameras and CCDs?**  A real transit
  does not care which detector recorded it.  A blend does: the contaminating
  flux fraction changes with the aperture, which changes with the camera.

**What is not available here is stated, not silently skipped.**  The tessquicklook
extractions do not carry per-cadence quality flags, scattered-light
backgrounds or momentum-dump times, so the tests that need them return
explicit "not available" entries rather than passing.  A test that quietly
does nothing is the failure mode that cost this project an entire analysis.

    python -m petcare.scripts.pc_instrument --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, ephem, prep as pcprep
from ..plotting import (C_BAD, C_BINNED, C_MODEL, C_OK, annotate_status,
                        figure, save)
from ..result import Result

TEST = "instrument"
VERSION = "0.1.0"

#: The TESS spacecraft orbit in days, and the nominal sector length.
TESS_ORBIT_DAYS = 13.7
TESS_SECTOR_DAYS = 27.4

#: Fractional tolerance for calling a period an alias of an instrumental one.
ALIAS_TOL = 0.02

#: Fraction of epochs affected by gaps or sector edges worth flagging.
GAP_EPOCH_FLAG = 0.5

#: Significance at which sector-to-sector depth differences are worth flagging.
SECTOR_CHI2_P_FLAG = 1e-3

#: Minimum sectors before a cross-sector depth comparison means anything.
MIN_SECTORS = 2


def add_arguments(ap):
    g = ap.add_argument_group("instrument")
    g.add_argument("--bin", type=float, default=600.0)
    g.add_argument("--gap-days", type=float, default=0.2,
                   help="a break in the time series longer than this counts "
                        "as a gap")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r
    if not cli.require(r, ctx.candidate.duration_h,
                       "no transit duration: pass --duration or run pc_fit first"):
        return r

    try:
        lc = ctx.lc
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"no usable light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)

    period = float(ctx.candidate.period)
    t0 = float(ctx.candidate.t0_btjd)
    dur = ctx.candidate.duration_days()

    # ---- 1. period aliases ------------------------------------------------
    gaps = _find_gaps(lc.t, float(getattr(a, "gap_days", 0.2)))
    gap_interval = _typical_gap_interval(gaps)
    aliases = _alias_check(period, gap_interval)
    r.metrics["gaps"] = {
        "n_gaps": len(gaps),
        "median_gap_days": _f(np.median([g[1] - g[0] for g in gaps])) if gaps else None,
        "typical_interval_days": _f(gap_interval),
        "note": "gaps measured from this light curve, not assumed: the "
                "downlink cadence changed between mission phases",
    }
    r.metrics["period_aliases"] = aliases

    # ---- 2. epochs against gaps, edges and downlinks ----------------------
    cov = ephem.epoch_coverage(lc, period, t0, dur)
    usable = [c for c in cov if c.usable]
    n = len(usable)
    if not cli.require(r, n >= 1, "no usable epochs to examine"):
        r.set_self_check(False, "no usable epochs")
        return r

    near_gap = [c for c in usable
                if any(abs(c.t_center - g[0]) < dur or abs(c.t_center - g[1]) < dur
                       or g[0] < c.t_center < g[1] for g in gaps)]
    partial = [c for c in usable if not (c.ingress_covered and c.egress_covered)]
    edge = [c for c in usable if c.near_sector_edge]
    holed = [c for c in usable if c.in_gap]

    r.metrics["epochs"] = {
        "n_usable": n,
        "n_near_a_gap": len(near_gap),
        "frac_near_a_gap": len(near_gap) / n,
        "n_missing_ingress_or_egress": len(partial),
        "frac_missing_ingress_or_egress": len(partial) / n,
        "n_near_sector_edge": len(edge),
        "frac_near_sector_edge": len(edge) / n,
        "n_with_internal_gap": len(holed),
        "epochs_near_gap": [c.epoch for c in near_gap],
    }

    # ---- 3. depth across sectors, cameras and CCDs ------------------------
    by_detector = {"status": "not run", "reason": "no duration/ephemeris"}
    try:
        p = pcprep.prepare(ctx, bin_seconds=float(getattr(a, "bin", 600.0)))
        ctx.notes.extend(p.notes)
        eds = pcprep.epoch_depths(p, lc=lc)
        r.set_noise(**p.noise_kwargs())
        by_detector = _depth_by_detector(eds, lc)
    except Exception as exc:  # noqa: BLE001
        by_detector = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    r.metrics["depth_by_detector"] = by_detector

    # ---- 4. what this extraction cannot tell us ---------------------------
    unavailable = _unavailable(lc)
    r.metrics["not_available"] = unavailable

    # ---- self-check -------------------------------------------------------
    checks = [f"{n} usable epochs across {lc.n_sectors} sector(s)",
              f"{len(gaps)} gaps longer than {getattr(a, 'gap_days', 0.2)} d"]
    ok = True
    if lc.n_sectors < MIN_SECTORS:
        checks.append(f"only {lc.n_sectors} sector: no cross-sector comparison "
                      f"is possible")
    if unavailable:
        checks.append("no quality flags, background or momentum-dump times in "
                      "this extraction: " + ", ".join(sorted(unavailable)))
    r.set_self_check(ok, "; ".join(checks))

    # ---- verdict ----------------------------------------------------------
    alias_hit = [x for x in aliases if x["is_alias"]]
    sector_p = by_detector.get("sector", {}).get("chi2_p_value")
    frac_partial = r.metrics["epochs"]["frac_missing_ingress_or_egress"]
    frac_gap = r.metrics["epochs"]["frac_near_a_gap"]

    if alias_hit and frac_gap > GAP_EPOCH_FLAG:
        best = alias_hit[0]
        r.flag(f"the period {period:.5f} d is {best['ratio_text']} of "
               f"{best['name']} ({best['reference_days']:.3f} d) AND "
               f"{frac_gap:.0%} of epochs sit next to a data gap. That "
               f"combination is how the instrument manufactures a transit.")
    elif alias_hit:
        best = alias_hit[0]
        r.flag(f"the period {period:.5f} d is {best['ratio_text']} of "
               f"{best['name']} ({best['reference_days']:.3f} d), within "
               f"{best['difference_frac']:.2%}. Not disqualifying on its own "
               f"-- real planets land on these periods too -- but the "
               f"per-epoch depths deserve a look.")
    elif frac_partial > GAP_EPOCH_FLAG:
        r.flag(f"{frac_partial:.0%} of epochs are missing ingress or egress: "
               f"the depth and duration rest on partial transits, and a "
               f"detrending ramp across a gap looks exactly like one")
    elif sector_p is not None and sector_p < SECTOR_CHI2_P_FLAG:
        s = by_detector["sector"]
        r.flag(f"the depth differs between sectors: chi2 = {s['chi2']:.1f} for "
               f"{s['dof']} dof (p = {sector_p:.1e}), "
               f"{s['min_ppm']:.0f}-{s['max_ppm']:.0f} ppm. A real transit "
               f"does not care which detector recorded it; a blend does, "
               f"because the contaminating fraction moves with the aperture.")
    else:
        bits = [f"no instrumental alias (closest: "
                f"{aliases[0]['name']} at {aliases[0]['difference_frac']:.1%})"]
        bits.append(f"{frac_partial:.0%} of epochs missing ingress/egress")
        if sector_p is not None:
            bits.append(f"sector-to-sector depths consistent (p = {sector_p:.2f})")
        r.passed("; ".join(bits))

    if ctx.make_plots:
        try:
            _plot(ctx, r, lc, cov, gaps, by_detector, period)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


def _find_gaps(t, min_days: float):
    """Breaks in the time series longer than ``min_days``, as (start, end)."""
    t = np.sort(np.asarray(t, float))
    if t.size < 3:
        return []
    d = np.diff(t)
    idx = np.flatnonzero(d > min_days)
    return [(float(t[i]), float(t[i + 1])) for i in idx]


def _typical_gap_interval(gaps):
    """Median spacing between gap starts: the light curve's own downlink period.

    Measured rather than assumed.  The downlink cadence is close to half the
    spacecraft orbit for most of the mission but not all of it, and a period
    alias test built on a hardcoded 13.7/2 will miss the ones that matter on
    the sectors where it differs.
    """
    if len(gaps) < 3:
        return np.nan
    starts = np.array([g[0] for g in gaps])
    d = np.diff(starts)
    d = d[d > 1.0]          # ignore adjacent gaps inside one downlink
    return float(np.median(d)) if d.size else np.nan


def _alias_check(period, gap_interval) -> list:
    """Is the candidate period a low harmonic of an instrumental period?"""
    refs = [("the TESS orbit", TESS_ORBIT_DAYS),
            ("the sector length", TESS_SECTOR_DAYS)]
    if np.isfinite(gap_interval):
        refs.append(("this light curve's own gap interval", gap_interval))

    out = []
    for name, ref in refs:
        best = None
        for num in (1, 2, 3, 4):
            for den in (1, 2, 3, 4):
                target = ref * num / den
                d = abs(period - target) / period
                if best is None or d < best["difference_frac"]:
                    best = {"name": name, "reference_days": float(ref),
                            "ratio_text": f"{num}/{den}" if den > 1 else f"{num}x",
                            "target_days": float(target),
                            "difference_frac": float(d),
                            "is_alias": bool(d < ALIAS_TOL)}
        out.append(best)
    out.sort(key=lambda x: x["difference_frac"])
    return out


def _depth_by_detector(eds, lc) -> dict:
    """Depth grouped by sector, camera and CCD, with a chi-square for each."""
    from ..prep import chi2_pvalue, weighted_mean

    sector_meta = {s.sector: s for s in lc.sectors}
    out = {"status": "ok"}
    for key, getter in (("sector", lambda e: e.sector),
                        ("camera", lambda e: getattr(
                            sector_meta.get(e.sector), "camera", None)),
                        ("ccd", lambda e: getattr(
                            sector_meta.get(e.sector), "ccd", None))):
        groups = {}
        for e in eds:
            g = getter(e)
            if g is None:
                continue
            groups.setdefault(g, []).append(e)
        if len(groups) < 2:
            out[key] = {"n_groups": len(groups),
                        "note": f"only {len(groups)} distinct {key}(s); "
                                f"no comparison possible"}
            continue
        means, errs, detail = [], [], {}
        for g, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
            m, e_, _, k = weighted_mean([x.depth_ppm for x in items],
                                        [x.depth_err_ppm for x in items])
            means.append(m)
            errs.append(e_)
            detail[str(g)] = {"depth_ppm": _f(m), "err_ppm": _f(e_),
                              "n_epochs": k}
        _, _, chi2, nn = weighted_mean(means, errs)
        dof = max(nn - 1, 1)
        out[key] = {"n_groups": len(groups), "groups": detail,
                    "chi2": _f(chi2), "dof": int(dof),
                    "chi2_p_value": _f(chi2_pvalue(chi2, dof)),
                    "min_ppm": _f(np.nanmin(means)), "max_ppm": _f(np.nanmax(means)),
                    "spread_frac": _f((np.nanmax(means) - np.nanmin(means)) /
                                      abs(np.nanmean(means))
                                      if np.nanmean(means) else np.nan)}
    return out


def _unavailable(lc) -> list:
    """Which instrumental diagnostics this extraction cannot supply.

    Named explicitly so that a reader knows the test did not run them, rather
    than assuming a silent pass.
    """
    missing = []
    meta = {}
    for s in lc.sectors:
        meta.update(s.meta or {})
    if not any(k.startswith("quality") for k in meta):
        missing.append("per-cadence quality flags")
    if not any("background" in k or "scatter_light" in k for k in meta):
        missing.append("scattered-light background")
    missing.append("momentum-dump and attitude-tweak times")
    # the extraction does record whether scattered light was allowed in
    settings = str(meta.get("settings_json", ""))
    if "allowscatteredlight=False" in settings:
        missing.append("scattered-light cadences (excluded by the extraction, "
                       "so they cannot be tested here -- see "
                       "allowscatteredlight=False in settings_json)")
    return missing


def _plot(ctx, r, lc, cov, gaps, by_detector, period):
    fig, axes = figure(2, 1, figsize=(10, 6.8))

    ax = axes[0]
    ax.plot(lc.t, (lc.f - 1) * 1e6, ".", ms=0.7, color="#4c4c4c", alpha=0.3,
            rasterized=True)
    for g in gaps:
        ax.axvspan(g[0], g[1], color=C_BAD, alpha=0.18)
    for c in cov:
        if c.usable:
            ax.axvline(c.t_center, color=C_OK, lw=0.8, alpha=0.7)
        else:
            ax.axvline(c.t_center, color=C_MODEL, lw=0.6, ls=":", alpha=0.6)
    ax.plot([], [], color=C_OK, lw=1.2, label="usable transit")
    ax.plot([], [], color=C_MODEL, lw=1.0, ls=":", label="transit without enough data")
    ax.plot([], [], color=C_BAD, lw=5, alpha=0.3, label="gap")
    ax.set_xlabel("BTJD")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title(f"transits against the data gaps ({len(gaps)} gaps)")
    ax.legend(ncol=3, loc="upper right")

    ax = axes[1]
    sec = by_detector.get("sector", {}).get("groups", {})
    if sec:
        keys = sorted(sec, key=lambda k: int(k) if k.lstrip("-").isdigit() else 0)
        x = np.arange(len(keys))
        y = [sec[k]["depth_ppm"] for k in keys]
        e = [sec[k]["err_ppm"] for k in keys]
        ax.errorbar(x, y, yerr=e, fmt="o", ms=5, color=C_BINNED)
        cams = {s.sector: s.camera for s in lc.sectors}
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{k}\ncam{cams.get(int(k), '?')}" for k in keys],
                           fontsize=7)
        finite = [v for v in y if v is not None]
        if finite:
            ax.axhline(float(np.mean(finite)), color=C_OK, lw=1.0)
        p = by_detector["sector"].get("chi2_p_value")
        ax.set_title("depth by sector"
                     + (f"  (chi2 p = {p:.2g})" if p is not None else ""))
    else:
        ax.text(0.5, 0.5, "only one sector: no cross-sector comparison",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("depth by sector")
    ax.set_ylabel("depth [ppm]")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("instrument")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Does the signal track the spacecraft? Period aliases, gaps, "
              "sector edges, and depth consistency across detectors."))
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

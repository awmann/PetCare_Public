"""pc_flares -- is the fold carried by flare-adjacent cadences?

Flares are the single most productive artefact generator in this sample.  One
flare complex on one star produced a "9.3 sigma" detection at P = 15.900 d and
an "11.5 sigma" detection at P = 11.4264 d *simultaneously*; both died as soon
as the epochs were looked at individually.

So this test does three things:

1. **Find the flares**, with a threshold that works on raw young-star data.
   This is where the previous attempt failed twice.  A sigma-clip whose
   threshold comes from the scatter of the series it is handed flags **zero**
   points on raw data from a strongly flaring M dwarf, because its 15000 ppm
   scatter *is* the flares.  A rolling-median-then-clip flagged 70% of points because the grow
   radius was too wide for the number of outliers.  Here the threshold is
   measured on the **variability-removed residual**, using a **one-sided**
   robust scatter estimated from the *negative* excursions only -- flares are
   positive, so the negative half of the distribution is uncontaminated by
   them.  The grow radius is tied to the measured decay time and hard-capped.
2. **Ask whether they touch the transits**: what fraction of in-transit
   cadences are flare-affected, and are they concentrated in a few epochs.
3. **Re-measure the depth with flares masked** and report the change.  That
   is the number that decides the question, and it is reported whether it
   moves or not.

The flare mask itself is written to the result so that other tests, and any
later re-fit, can use the same one.  A mask referenced by code but never
actually generated cost this project an entire multi-phase analysis.

    python -m petcare.scripts.pc_flares --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, ephem, prep as pcprep
from ..flares import (FLARE_SIGMA, MAX_MASKED_FRACTION, detect_flares)
from ..plotting import (C_BAD, C_BINNED, C_MODEL, C_OK, annotate_status,
                        figure, save)
from ..result import Result

TEST = "flares"
VERSION = "0.2.1"

#: Depth change, as a fraction of the unmasked depth, worth flagging.
DEPTH_CHANGE_FLAG = 0.25

#: In-transit flare contamination worth flagging on its own.
IN_TRANSIT_FLARE_FLAG = 0.10

#: Egress rise: a dip that folds onto a flare or a gap shows a rise just after
#: egress, even in the fold.  Flag when the mean post-egress residual exceeds
#: EGRESS_RISE_SIGMA *and* the epochs whose own post-egress rise exceeds
#: EGRESS_EPOCH_SIGMA carry at least EGRESS_DEPTH_FRAC of the depth.  On a set of
#: eye-vetted candidates it caught 13 of 18 "reject -
#: flares" and 1 of 25 planet-like (medians 13.9 vs -0.2 sigma, depth share
#: 0.76 vs 0.04).  A timing offset pulls the PRE-ingress window down instead,
#: which is why this is one-sided and not an egress-minus-ingress difference.
#: A pre-ingress window raised by more than EGRESS_RISE_SIGMA as well is a hump
#: on both sides -- a baseline or ephemeris problem, not a flare -- and does
#: not count: one planet-like candidate at P = 24.17 d rose 12 sigma after egress and
#: 14 before at its vetted ephemeris, and 0.4 at the refitted one.
EGRESS_RISE_SIGMA = 5.0
EGRESS_EPOCH_SIGMA = 3.0
EGRESS_DEPTH_FRAC = 0.4


def add_arguments(ap):
    g = ap.add_argument_group("flares")
    g.add_argument("--bin", type=float, default=120.0,
                   help="bin the light curve to this many seconds; finer than "
                        "the other tests because flares are short")
    g.add_argument("--flare-sigma", type=float, default=FLARE_SIGMA)
    g.add_argument("--write-mask", metavar="PATH", default=None,
                   help="also write the flare mask as a two-column "
                        "time,is_flare file")
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
        p = pcprep.prepare(ctx, bin_seconds=float(getattr(a, "bin", 120.0)))
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)

    thresh = float(getattr(a, "flare_sigma", FLARE_SIGMA))
    fl = detect_flares(p.t, p.f_detrended, sigma=thresh)
    r.set_noise(fl["sigma_ppm"], fl["estimator"],
                one_sided=True,
                symmetric_scatter_ppm=fl["symmetric_sigma_ppm"],
                note="the symmetric scatter is inflated by the flares "
                     "themselves; the one-sided value is the one used")

    mask = fl["mask"]
    frac = float(mask.mean())
    r.metrics.update({
        "n_flare_cadences": int(mask.sum()),
        "masked_fraction": frac,
        "n_flare_events": fl["n_events"],
        "flare_rate_per_day": _f(fl["n_events"] / max(lc.baseline_days, 1e-6)),
        "largest_flare_ppm": fl["largest_ppm"],
        "decay_time_days": fl["decay_days"],
        "grow_days": fl["grow_days"],
        "threshold_sigma": thresh,
        "threshold_ppm": fl["sigma_ppm"] * thresh,
        "bin_seconds": float(getattr(a, "bin", 120.0)),
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{fl['n_events']} flare events over "
                  f"{lc.baseline_days:.0f} d, masking {frac:.2%} of cadences"], True
    if frac > MAX_MASKED_FRACTION:
        checks.append(f"masking {frac:.0%} of the light curve is a detrending "
                      f"failure, not a flare mask")
        ok = False
    var_amp = fl["symmetric_sigma_ppm"]
    if fl["n_events"] == 0 and var_amp > 3000:
        # the exact failure the previous flare masker had on a flaring M dwarf: a star
        # with percent-level scatter and no flares found is a broken threshold
        checks.append(f"no flares found on a star with {var_amp:.0f} ppm "
                      f"residual scatter; the threshold is not working")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    # ---- do the flares touch the transits? --------------------------------
    intr = ephem.in_transit(p.t, p.period, p.t0, p.duration_days)
    in_frac = float(mask[intr].mean()) if intr.any() else np.nan
    ep = p.epochs()
    per_epoch = {}
    for e in np.unique(ep[intr]):
        sel = intr & (ep == e)
        if sel.sum():
            per_epoch[int(e)] = float(mask[sel].mean())
    worst = max(per_epoch.items(), key=lambda kv: kv[1], default=(None, 0.0))

    r.metrics.update({
        "in_transit_flare_fraction": _f(in_frac),
        "per_epoch_flare_fraction": per_epoch,
        "worst_epoch": worst[0],
        "worst_epoch_flare_fraction": _f(worst[1]),
        "n_epochs_over_half_flare": sum(1 for v in per_epoch.values() if v > 0.5),
    })

    # ---- the depth, with and without the flares ---------------------------
    d_all = _fold_depth(p, np.zeros_like(mask, dtype=bool))
    d_masked = _fold_depth(p, mask)
    change = None
    if d_all["depth_ppm"] and d_masked["depth_ppm"]:
        change = (d_masked["depth_ppm"] - d_all["depth_ppm"]) / abs(d_all["depth_ppm"])
    r.metrics.update({
        "depth_all_ppm": d_all["depth_ppm"], "depth_all_err_ppm": d_all["err_ppm"],
        "depth_flare_masked_ppm": d_masked["depth_ppm"],
        "depth_flare_masked_err_ppm": d_masked["err_ppm"],
        "depth_fractional_change": _f(change),
        "n_in_transit_kept": d_masked["n_in"],
    })

    if getattr(a, "write_mask", None):
        _write_mask(a.write_mask, p.t, mask)
        r.provenance["flare_mask_file"] = str(a.write_mask)
    r.metrics["flare_intervals"] = fl["intervals"][:200]
    r.metrics["flare_intervals_truncated"] = len(fl["intervals"]) > 200

    er = _egress_rise(lc.t, lc.f, p.period, p.t0, p.duration_days)
    r.metrics.update({f"egress_{k}": v for k, v in er.items()})
    rise = bool(er.get("post_rise_sig") is not None
                and er["post_rise_sig"] > EGRESS_RISE_SIGMA
                and (er.get("rise_epoch_depth_frac") or 0) >= EGRESS_DEPTH_FRAC
                and (er.get("pre_drop_sig") is None or er["pre_drop_sig"] > -EGRESS_RISE_SIGMA))
    r.metrics["egress_rise"] = rise
    rise_msg = (f"the flux rises {er.get('post_rise_sig', 0):.1f} sigma above the "
                f"local baseline just after egress, and the epochs that do this "
                f"carry {er.get('rise_epoch_depth_frac') or 0:.0%} of the depth: "
                f"the dip folds onto a flare or a data gap. ") if rise else ""

    note = (f"threshold {thresh:.1f} sigma on a one-sided residual scatter of "
            f"{fl['sigma_ppm']:.0f} ppm; symmetric scatter "
            f"{fl['symmetric_sigma_ppm']:.0f} ppm")

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif rise:
        r.flag(rise_msg + f"Depth with flares masked {0 if change is None else change:+.0%}; "
               f"{0 if not np.isfinite(in_frac) else in_frac:.0%} of in-transit "
               f"cadences flare-affected. {note}")
    elif change is not None and abs(change) > DEPTH_CHANGE_FLAG:
        # Direction matters, and they mean opposite things.  Shallower with
        # flares masked means the flares were *making* the signal, which is
        # the dangerous case.  Deeper means flares were filling the transit
        # in, so the masked depth is the better measurement -- on one known
        # planet that took 1796 ppm to 2517 ppm against a literature 2490 ppm.
        if change > 0:
            r.flag(f"the transit is {change:+.0%} deeper with flares masked "
                   f"({d_all['depth_ppm']:.0f} -> {d_masked['depth_ppm']:.0f} ppm): "
                   f"flares were filling the transit in, and the masked depth "
                   f"is the one to use. {note}")
        else:
            r.flag(f"the transit is {change:+.0%} shallower with flares masked "
                   f"({d_all['depth_ppm']:.0f} -> {d_masked['depth_ppm']:.0f} ppm): "
                   f"flare-affected cadences are contributing to the signal "
                   f"rather than diluting it, which is how a flare complex "
                   f"manufactures a candidate. {note}")
    elif np.isfinite(in_frac) and in_frac > IN_TRANSIT_FLARE_FLAG:
        r.flag(f"{in_frac:.0%} of in-transit cadences are flare-affected "
               f"(worst epoch {worst[0]}: {worst[1]:.0%}), though masking them "
               f"changes the depth by only {0 if change is None else change:+.0%}. "
               f"{note}")
    else:
        r.passed(f"{fl['n_events']} flares over {lc.baseline_days:.0f} d; "
                 f"{0 if not np.isfinite(in_frac) else in_frac:.1%} of "
                 f"in-transit cadences affected, and masking them moves the "
                 f"depth by {0 if change is None else change:+.1%} "
                 + ("(the transit falls in a gap, so no depth to compare). "
                    if change is None or d_all["depth_ppm"] is None or d_masked["depth_ppm"] is None else
                    f"({d_all['depth_ppm']:.0f} -> {d_masked['depth_ppm']:.0f} ppm). ")
                 + f"{note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, p, mask, fl)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


def _fold_depth(p, exclude) -> dict:
    """Folded depth with ``exclude`` removed, against a local baseline."""
    keep = ~np.asarray(exclude, bool)
    ph = p.phase()[keep]
    f = p.f_detrended[keep]
    in_m = np.abs(ph) < 0.4 * p.duration_days
    oot_m = (np.abs(ph) >= 0.75 * p.duration_days) & \
            (np.abs(ph) < 3.0 * p.duration_days)
    if in_m.sum() < 5 or oot_m.sum() < 20:
        return {"depth_ppm": None, "err_ppm": None, "n_in": int(in_m.sum())}
    depth = (float(np.median(f[oot_m]) - np.mean(f[in_m]))) * 1e6
    scatter = float(1.4826 * np.median(np.abs(f[oot_m] - np.median(f[oot_m])))) * 1e6
    err = scatter * np.sqrt(1 / in_m.sum() + 1 / oot_m.sum())
    return {"depth_ppm": depth, "err_ppm": err, "n_in": int(in_m.sum())}


def _egress_rise(t, f, P, t0, w) -> dict:
    """Post-egress rise and the share of the depth carried by epochs that rise.

    Young stars vary on the transit timescale, so every epoch gets its own
    quadratic baseline through [2, 4] T14 on both sides of mid-transit and
    everything is measured on the residual.  Uses the unbinned stitched light
    curve, as the eye vetters see it.
    """
    t, f = np.asarray(t, float), np.asarray(f, float)
    ok0 = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok0], f[ok0]
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    ep = np.round((t - t0) / P).astype(int)
    near = np.abs(ph) < 4 * w
    r = np.full_like(f, np.nan)
    for e in np.unique(ep[near]):
        m = (ep == e) & near
        b = m & (np.abs(ph) > 2 * w)
        if (b & (ph < 0)).sum() >= 5 and (b & (ph > 0)).sum() >= 5:
            r[m] = f[m] - np.polyval(np.polyfit(ph[b], f[b], 2), ph[m])
    ok = np.isfinite(r)
    pre = ok & (ph > -1.5 * w) & (ph < -0.5 * w)
    post = ok & (ph > 0.5 * w) & (ph < 1.5 * w)
    base = ok & (np.abs(ph) > 2 * w) & (np.abs(ph) < 4 * w)
    intr = ok & (np.abs(ph) < 0.5 * w)
    if pre.sum() < 3 or post.sum() < 3 or base.sum() < 10:
        return {"post_rise_sig": None, "pre_drop_sig": None,
                "rise_epoch_depth_frac": None, "n_epochs": 0}
    sd = 1.4826 * np.median(np.abs(r[base] - np.median(r[base])))
    if not sd > 0:
        return {"post_rise_sig": None, "pre_drop_sig": None,
                "rise_epoch_depth_frac": None, "n_epochs": 0}
    post_rise = float(np.mean(r[post]) / (sd / np.sqrt(post.sum())))
    pre_drop = float(-np.mean(r[pre]) / (sd / np.sqrt(pre.sum())))
    tot = rise_dep = 0.0
    n = 0
    for e in np.unique(ep[near]):
        mi = (ep == e) & intr
        if mi.sum() < 2:
            continue
        n += 1
        dfc = max(0.0, -float(np.mean(r[mi]))) * mi.sum()
        tot += dfc
        mp = (ep == e) & post
        if mp.sum() >= 3 and np.mean(r[mp]) / (sd / np.sqrt(mp.sum())) > EGRESS_EPOCH_SIGMA:
            rise_dep += dfc
    return {"post_rise_sig": round(post_rise, 2), "pre_drop_sig": round(pre_drop, 2),
            "rise_epoch_depth_frac": round(rise_dep / tot, 3) if tot > 0 else None,
            "n_epochs": n}


def _write_mask(path, t, mask):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_btjd", "is_flare"])
        for ti, mi in zip(t, mask):
            w.writerow([f"{ti:.8f}", int(mi)])


def _plot(ctx, r, p, mask, fl):
    fig, axes = figure(2, 1, figsize=(10, 6.5))

    ax = axes[0]
    y = (p.f_detrended - 1) * 1e6
    ax.plot(p.t, y, ".", ms=1.0, color="#4c4c4c", alpha=0.4, rasterized=True,
            label="variability removed")
    if mask.any():
        ax.plot(p.t[mask], y[mask], ".", ms=1.6, color=C_BAD, alpha=0.8,
                label=f"flare-masked ({mask.mean():.1%})")
    thr = fl["sigma_ppm"] * r.metrics["threshold_sigma"]
    ax.axhline(thr, color=C_MODEL, ls="--", lw=0.9,
               label=f"threshold {thr:.0f} ppm")
    intr = ephem.in_transit(p.t, p.period, p.t0, p.duration_days)
    if intr.any():
        ax.plot(p.t[intr], y[intr], ".", ms=1.6, color=C_OK, alpha=0.7,
                label="in transit")
    ax.set_ylim(np.percentile(y, 0.2), max(np.percentile(y, 99.9), thr * 1.5))
    ax.set_xlabel("BTJD")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title("flare detection")
    ax.legend(ncol=4, loc="upper right")

    ax = axes[1]
    ph = p.phase()
    win = min(3.0 * p.duration_days, 0.45 * p.period)
    sel = np.abs(ph) < win
    from ..plotting import bin_series
    for label, m, colour in (("all cadences", sel, C_BINNED),
                             ("flares masked", sel & ~mask, C_OK)):
        if m.sum() < 10:
            continue
        xb, yb, eb, _ = bin_series(ph[m] * 24, (p.f_detrended[m] - 1) * 1e6,
                                   max(p.duration_h / 12, 0.1), min_points=3)
        ax.errorbar(xb, yb, yerr=eb, fmt="o-", ms=3, lw=1.1, color=colour,
                    label=label)
    for s in (-0.5, 0.5):
        ax.axvline(s * p.duration_h, color=C_MODEL, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux - 1 [ppm]")
    d0, d1 = r.metrics.get("depth_all_ppm"), r.metrics.get("depth_flare_masked_ppm")
    ax.set_title("fold with and without flare-affected cadences"
                 + (f"  ({d0:.0f} -> {d1:.0f} ppm)" if d0 and d1 else ""))
    ax.legend()

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("flares")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Find flares and ask whether the transit fold depends on them."))
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

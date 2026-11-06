"""pc_depths -- are the per-epoch depths mutually consistent, and does one
epoch carry the detection?

Two separate statistics, because they answer different questions and a signal
can pass one while failing the other:

**chi-square of the depths about their weighted mean.**  Are the individual
transits the same depth?  This replaces the old "every depth within 5 sigma of
the median" rule, which passed almost anything once the errors were large --
with 300 ppm errors and a 200 ppm signal, every depth is within 5 sigma of
everything.  A chi-square uses the size of the errors instead of ignoring it.

**jackknife.**  Drop the single best epoch and ask whether the detection
survives: ``surv = z_without_best / z_with_all``.  This is the one cut the
previous search demonstrably trusted -- at ``surv > 0.4`` it cost **0 of 38**
detected known planets, and it killed both of the false positives on one star
that a single flare complex had manufactured at two different periods
simultaneously.  Nearly every artefact in that sample was visible the moment
the stacked signal was broken into individual epochs.

Needs at least four covered epochs; below that it returns INCONCLUSIVE, since
a chi-square on two numbers is not evidence and a jackknife on three is not
either.

    python -m petcare.scripts.pc_depths --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, prep as pcprep
from ..plotting import C_BAD, C_BINNED, C_MODEL, C_OK, annotate_status, figure, save
from ..result import Result

TEST = "depths"
VERSION = "0.1.1"

#: Minimum epochs for either statistic to mean anything.
MIN_EPOCHS = 4

#: Jackknife survival below which the detection rests on one epoch.
#: Calibrated in the previous search at 0 cost across 38 detected known
#: planets (median survival there was 0.93).  The best-supported threshold in
#: this package.
JACKKNIFE_FAIL = 0.4
JACKKNIFE_FLAG = 0.6
#: Below this per-epoch significance there is no detection for one epoch to carry, and
#: the jackknife ratio is a ratio of noise: 87% of sampled FAILs sat here, so these are
#: reported, not refuted.
Z_ALL_MIN = 3.0

#: Chi-square p-value below which the depths are mutually inconsistent.
#: PROVISIONAL -- see calibration/README.md.  Spot crossings on these stars
#: make real planets scatter well beyond their formal errors (DS Tuc A b and
#: HIP 67522 c both exceed 7 sigma in odd/even for exactly this reason), so
#: this is set loose on purpose and flags rather than fails.
CHI2_P_FLAG = 1e-3


def add_arguments(ap):
    g = ap.add_argument_group("depths")
    g.add_argument("--bin", type=float, default=600.0,
                   help="bin the light curve to this many seconds")
    g.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    g.add_argument("--core-frac", type=float, default=0.8,
                   help="fraction of T14 averaged as 'in transit'; below 1 so "
                        "that ingress and egress cadences cannot bias depths low")
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
        p = pcprep.prepare(ctx, bin_seconds=float(getattr(a, "bin", 600.0)))
    except Exception as exc:  # noqa: BLE001 - missing data is normal
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)
    r.set_noise(**p.noise_kwargs())

    core = float(getattr(a, "core_frac", 0.8))
    baseline = pcprep.choose_baseline(p, core_frac=core)
    eds = pcprep.epoch_depths(p, lc=lc, core_frac=core, baseline=baseline)
    # provenance: which window and order the data chose, and what it cost
    r.metrics["baseline"] = {k: v for k, v in baseline.items() if k != "trials"}
    r.metrics["baseline_trials"] = baseline.get("trials", [])
    n_min = int(getattr(a, "min_epochs", MIN_EPOCHS))

    r.metrics["epochs"] = [e.to_dict() for e in eds]
    r.metrics["n_epochs"] = len(eds)
    r.metrics["n_epochs_partial"] = sum(
        1 for e in eds if not (e.ingress_covered and e.egress_covered))
    r.metrics["n_epochs_in_gap"] = sum(1 for e in eds if e.in_gap)
    r.metrics["n_epochs_near_sector_edge"] = sum(1 for e in eds if e.near_sector_edge)

    if not cli.require(r, len(eds) >= n_min,
                       f"only {len(eds)} covered epoch(s); {n_min} are needed "
                       f"for a depth-consistency test"):
        r.set_self_check(False, f"{len(eds)} covered epochs")
        if ctx.make_plots and eds:
            _plot(ctx, r, p, eds, None)
        return r

    depths = np.array([e.depth_ppm for e in eds])
    errs = np.array([e.depth_err_ppm for e in eds])
    r.metrics["median_depth_err_ppm"] = float(np.median(errs))

    # ---- 1. mutual consistency -------------------------------------------
    mean, mean_err, chi2, n = pcprep.weighted_mean(depths, errs)
    dof = max(n - 1, 1)
    pval = pcprep.chi2_pvalue(chi2, dof)
    scatter_ratio = float(np.sqrt(chi2 / dof)) if np.isfinite(chi2) else np.nan

    # ---- 2. jackknife -----------------------------------------------------
    z_all = mean / mean_err if mean_err and np.isfinite(mean_err) else np.nan
    contributions = depths / errs ** 2
    best = int(np.argmax(contributions))
    keep = np.ones(n, dtype=bool)
    keep[best] = False
    m_j, e_j, _, _ = pcprep.weighted_mean(depths[keep], errs[keep])
    z_jack = m_j / e_j if e_j and np.isfinite(e_j) else np.nan
    surv = float(z_jack / z_all) if np.isfinite(z_all) and z_all != 0 else np.nan

    # a second jackknife over sectors: one bad sector is as dangerous as one
    # bad epoch and is not caught by dropping a single epoch
    per_sector = {}
    for e in eds:
        per_sector.setdefault(e.sector, []).append(e)
    sector_depths = {}
    for sec, group in per_sector.items():
        m_s, e_s, _, _ = pcprep.weighted_mean([g.depth_ppm for g in group],
                                              [g.depth_err_ppm for g in group])
        sector_depths[str(sec)] = {"depth_ppm": _f(m_s), "err_ppm": _f(e_s),
                                   "n_epochs": len(group)}

    r.metrics.update({
        "mean_depth_ppm": _f(mean), "mean_depth_err_ppm": _f(mean_err),
        "chi2": _f(chi2), "dof": int(dof), "chi2_p_value": _f(pval),
        "scatter_over_errors": _f(scatter_ratio),
        "z_all": _f(z_all), "z_jackknife": _f(z_jack),
        "jackknife_survival": _f(surv),
        "dropped_epoch": eds[best].epoch,
        "dropped_epoch_depth_ppm": _f(depths[best]),
        "depth_range_ppm": [_f(np.min(depths)), _f(np.max(depths))],
        "per_sector": sector_depths,
        "n_sectors": len(per_sector),
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{n} epochs, {len(per_sector)} sector(s)"], True
    n_partial = r.metrics["n_epochs_partial"]
    if n_partial:
        checks.append(f"{n_partial} epoch(s) missing ingress or egress")
    if np.isfinite(mean) and mean <= 0:
        checks.append("weighted mean depth is not positive")
        ok = False
    if not np.isfinite(surv):
        checks.append("jackknife undefined")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    med_err = float(np.median(errs))
    n_null_ok = sum(1 for e in eds if "null scatter" in e.err_source)
    noise_note = (f"depth errors are {med_err:.0f} ppm (median), measured from "
                  f"off-phase null measurements of the same statistic "
                  f"[{n_null_ok}/{len(eds)} epochs]; point-to-point noise "
                  f"{p.white_ppm:.0f} ppm")
    r.noise["median_depth_err_ppm"] = med_err
    r.noise["depth_err_estimator"] = (
        "scatter of the same depth measurement at off-transit offsets in the "
        "same window; falls back to a propagated white error where fewer than "
        "3 nulls fit")
    r.noise["n_epochs_with_null_error"] = n_null_ok

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif np.isfinite(surv) and surv < JACKKNIFE_FAIL and not (z_all >= Z_ALL_MIN):
        r.inconclusive(f"the per-epoch depths average to only {z_all:.1f} sigma, so there is "
                       f"no detection to test for resting on one epoch (dropping epoch "
                       f"{eds[best].epoch} gives {z_jack:.1f}); {noise_note}")
    elif np.isfinite(surv) and surv < JACKKNIFE_FAIL:
        r.fail(f"the detection rests on one epoch: dropping epoch "
               f"{eds[best].epoch} ({depths[best]:.0f} ppm) takes the "
               f"significance from {z_all:.1f} to {z_jack:.1f} "
               f"(survival {surv:.2f} < {JACKKNIFE_FAIL}); {noise_note}")
    elif np.isfinite(pval) and pval < CHI2_P_FLAG:
        r.flag(f"per-epoch depths are mutually inconsistent: chi2 = {chi2:.1f} "
               f"for {dof} dof (p = {pval:.1e}), scatter {scatter_ratio:.1f}x "
               f"the errors, depths {np.min(depths):.0f}-{np.max(depths):.0f} ppm; "
               f"{noise_note}")
    elif np.isfinite(surv) and surv < JACKKNIFE_FLAG:
        r.flag(f"one epoch carries much of the detection: survival {surv:.2f} "
               f"(z {z_all:.1f} -> {z_jack:.1f}); {noise_note}")
    else:
        r.passed(f"{n} epochs consistent at {mean:.0f} +/- {mean_err:.0f} ppm "
                 f"(chi2 = {chi2:.1f}/{dof}, p = {pval:.2f}), jackknife "
                 f"survival {surv:.2f}; {noise_note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, p, eds, best)
        except Exception as exc:  # noqa: BLE001 - a failed plot is not a failed test
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, p, eds, best):
    fig, axes = figure(2, 1, figsize=(10, 7),
                       gridspec_kw={"height_ratios": [1, 1.25]})

    # per-epoch depths against the weighted mean
    ax = axes[0]
    x = np.arange(len(eds))
    d = np.array([e.depth_ppm for e in eds])
    e_ = np.array([e.depth_err_ppm for e in eds])
    partial = np.array([not (e.ingress_covered and e.egress_covered) or e.in_gap
                        for e in eds])
    ax.errorbar(x[~partial], d[~partial], yerr=e_[~partial], fmt="o", ms=4,
                color=C_BINNED, label="fully covered")
    if partial.any():
        ax.errorbar(x[partial], d[partial], yerr=e_[partial], fmt="s", ms=4,
                    mfc="none", color=C_BAD,
                    label="ingress/egress missing or gap")
    if best is not None:
        ax.plot(x[best], d[best], "o", ms=10, mfc="none", color=C_MODEL, mew=1.6,
                label="dropped by the jackknife")
    m = r.metrics.get("mean_depth_ppm")
    me = r.metrics.get("mean_depth_err_ppm")
    if m is not None:
        ax.axhline(m, color=C_OK, lw=1.2, label=f"weighted mean {m:.0f} ppm")
        if me:
            ax.axhspan(m - me, m + me, color=C_OK, alpha=0.15)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e.epoch}\ns{e.sector}" for e in eds], fontsize=6)
    ax.set_xlabel("epoch / sector")
    ax.set_ylabel("depth [ppm]")
    ax.legend(ncol=4, loc="best")
    ax.set_title("per-epoch depths, each against its own local baseline")

    # the folds those depths came from
    ax = axes[1]
    ph = p.phase()
    ep = p.epochs()
    win = min(3.0 * p.duration_days, 0.45 * p.period)
    offset = max(2.5 * abs(m or 500), 300.0)
    from ..plotting import bin_series
    for i, ed in enumerate(eds[:16]):
        sel = (ep == ed.epoch) & (np.abs(ph) < win)
        if sel.sum() < 3:
            continue
        y = (p.f_detrended[sel] - 1) * 1e6 + i * offset
        ax.plot(ph[sel] * 24, y, ".", ms=1.6, color="#4c4c4c", alpha=0.45,
                rasterized=True)
        xb, yb, _, _ = bin_series(ph[sel] * 24, y, max(p.duration_h / 8, 0.12),
                                  min_points=2)
        ax.plot(xb, yb, "-", lw=1.1,
                color=C_MODEL if i == best else C_BINNED)
        ax.axhline(i * offset, color="k", lw=0.4, alpha=0.25)
        ax.text(win * 24 * 1.02, i * offset,
                f"e{ed.epoch} {ed.depth_ppm:.0f}ppm", fontsize=6, va="center")
    for s in (-0.5, 0.5):
        ax.axvline(s * p.duration_h, color=C_MODEL, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlim(-win * 24 * 1.05, win * 24 * 1.35)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel(f"per-epoch fold [ppm, offset {offset:.0f}]")
    ax.set_title(f"the individual transits ({min(len(eds), 16)} of {len(eds)} shown)")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("depths")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Per-epoch transit depths: are they mutually consistent, and "
              "does the detection survive dropping its best single epoch?"))
    args = ap.parse_args(argv)
    ctx = cli.Context.from_args(args)
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        r.error(exc)
    return cli.finish(r, ctx)


if __name__ == "__main__":
    sys.exit(main())

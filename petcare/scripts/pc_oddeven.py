"""pc_oddeven -- do odd and even transits have the same depth?

The classic eclipsing-binary test: if the true period is twice the candidate's,
alternate "transits" are primary and secondary eclipses of a binary and have
different depths.

**This test flags; it does not fail.**  At the textbook 3-sigma threshold it
would delete 18% of the known planets in this sample, and at 2 sigma, 32%.
The reason is specific to young stars: spot crossings make epoch-to-epoch
depth vary far beyond the formal error, and the two worst offenders in that
calibration -- DS Tuc A b at 8.1 sigma and HIP 67522 c at 7.2 sigma -- are
both real planets.  A difference here is a reason for a human to look at the
per-epoch depths, not a refutation.

The depths come from :mod:`petcare.prep`, so they are measured the same way as
in ``pc_depths``: a GP conditioned on out-of-transit cadences, a local
polynomial baseline per transit, and an uncertainty measured from off-phase
nulls rather than propagated.  That matters here more than anywhere: with a
propagated white error every candidate looks inconsistent.

    python -m petcare.scripts.pc_oddeven --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, prep as pcprep
from ..plotting import (C_ALT, C_BINNED, C_MODEL, annotate_status,
                        bin_series, figure, save)
from ..result import Result

TEST = "odd_even"
VERSION = "0.1.0"

#: Sigma above which a human should look.  Set from the cost measured on this
#: sample: 2 sigma would flag 32% of known planets, 3 sigma 18%, 5 sigma 5%.
#: 3 is the conventional value and is kept because this is a flag, not a cut.
ODD_EVEN_FLAG_SIGMA = 3.0

#: Above this the difference is large enough to name in the reason line as a
#: likely binary, still without failing the candidate.
ODD_EVEN_STRONG_SIGMA = 5.0

#: Both parities need at least this many transits, or the comparison is
#: between a mean and a single measurement.
MIN_PER_PARITY = 2


def add_arguments(ap):
    g = ap.add_argument_group("odd/even")
    g.add_argument("--bin", type=float, default=600.0,
                   help="bin the light curve to this many seconds")
    g.add_argument("--core-frac", type=float, default=0.8,
                   help="fraction of T14 averaged as 'in transit'")
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
    r.metrics["baseline"] = {k: v for k, v in baseline.items() if k != "trials"}

    odd = [e for e in eds if e.epoch % 2 != 0]
    even = [e for e in eds if e.epoch % 2 == 0]
    r.metrics["n_odd"] = len(odd)
    r.metrics["n_even"] = len(even)
    r.metrics["odd_epochs"] = [e.epoch for e in odd]
    r.metrics["even_epochs"] = [e.epoch for e in even]

    if not cli.require(
            r, len(odd) >= MIN_PER_PARITY and len(even) >= MIN_PER_PARITY,
            f"need at least {MIN_PER_PARITY} transits of each parity; "
            f"have {len(odd)} odd and {len(even)} even"):
        r.set_self_check(False, f"{len(odd)} odd, {len(even)} even")
        return r

    d_odd, e_odd, chi_odd, n_odd = pcprep.weighted_mean(
        [e.depth_ppm for e in odd], [e.depth_err_ppm for e in odd])
    d_even, e_even, chi_even, n_even = pcprep.weighted_mean(
        [e.depth_ppm for e in even], [e.depth_err_ppm for e in even])

    diff = d_odd - d_even
    diff_err = float(np.hypot(e_odd, e_even))
    sigma = abs(diff) / diff_err if diff_err > 0 else np.nan
    ratio = (d_odd / d_even if d_even and np.isfinite(d_even) and d_even != 0
             else np.nan)

    # The depth a secondary would need in order to be the shallower of the
    # two: useful when deciding whether a ground-based follow-up could tell.
    shallower = min(d_odd, d_even)
    deeper = max(d_odd, d_even)

    r.metrics.update({
        "odd_depth_ppm": _f(d_odd), "odd_err_ppm": _f(e_odd),
        "even_depth_ppm": _f(d_even), "even_err_ppm": _f(e_even),
        "difference_ppm": _f(diff), "difference_err_ppm": _f(diff_err),
        "difference_sigma": _f(sigma),
        "depth_ratio_odd_over_even": _f(ratio),
        "deeper_ppm": _f(deeper), "shallower_ppm": _f(shallower),
        "odd_chi2": _f(chi_odd), "even_chi2": _f(chi_even),
        "implied_period_if_binary": _f(2 * p.period),
        "per_epoch": [e.to_dict() for e in eds],
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{n_odd} odd and {n_even} even transits covered"], True
    n_partial = sum(1 for e in eds if not (e.ingress_covered and e.egress_covered))
    if n_partial:
        checks.append(f"{n_partial} epoch(s) missing ingress or egress")
    # A parity split that lines up with a sector split is not an odd/even
    # test at all -- it is a sector-to-sector depth difference wearing a
    # disguise, and the instrument test is the one that should answer it.
    odd_sectors = {e.sector for e in odd}
    even_sectors = {e.sector for e in even}
    confounded = bool(odd_sectors and even_sectors and not (odd_sectors & even_sectors))
    if confounded:
        checks.append(f"parities do not share a sector (odd: {sorted(odd_sectors)}, "
                      f"even: {sorted(even_sectors)}); any difference is "
                      f"indistinguishable from a sector-to-sector one")
        ok = False
    if not np.isfinite(sigma):
        checks.append("difference significance undefined")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    noise_note = (f"depth errors {np.median([e.depth_err_ppm for e in eds]):.0f} ppm "
                  f"(median) from off-phase nulls; point-to-point noise "
                  f"{p.white_ppm:.0f} ppm")

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif sigma >= ODD_EVEN_STRONG_SIGMA:
        r.flag(f"odd and even depths differ by {sigma:.1f} sigma "
               f"({d_odd:.0f} +/- {e_odd:.0f} vs {d_even:.0f} +/- {e_even:.0f} ppm): "
               f"consistent with an eclipsing binary at P = {2 * p.period:.5f} d. "
               f"Not a refutation -- two known planets in this sample exceed "
               f"7 sigma here from spot crossings. {noise_note}")
    elif sigma >= ODD_EVEN_FLAG_SIGMA:
        r.flag(f"odd and even depths differ by {sigma:.1f} sigma "
               f"({d_odd:.0f} +/- {e_odd:.0f} vs {d_even:.0f} +/- {e_even:.0f} ppm); "
               f"look at the per-epoch depths. {noise_note}")
    else:
        r.passed(f"odd and even depths agree to {sigma:.1f} sigma "
                 f"({d_odd:.0f} +/- {e_odd:.0f} vs {d_even:.0f} +/- {e_even:.0f} ppm) "
                 f"over {n_odd} odd and {n_even} even transits. {noise_note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, p, odd, even)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, p, odd, even):
    fig, axes = figure(1, 2, figsize=(11, 4.6))

    # the two folds, overlaid
    ax = axes[0]
    ph = p.phase()
    ep = p.epochs()
    win = min(3.0 * p.duration_days, 0.45 * p.period)
    for label, group, colour in (("odd", odd, C_ALT), ("even", even, C_BINNED)):
        epochs = {e.epoch for e in group}
        sel = np.isin(ep, list(epochs)) & (np.abs(ph) < win)
        if sel.sum() < 5:
            continue
        y = (p.f_detrended[sel] - 1) * 1e6
        ax.plot(ph[sel] * 24, y, ".", ms=1.4, color=colour, alpha=0.20,
                rasterized=True)
        xb, yb, eb, _ = bin_series(ph[sel] * 24, y, max(p.duration_h / 10, 0.1),
                                   min_points=3)
        ax.errorbar(xb, yb, yerr=eb, fmt="o-", ms=3, lw=1.2, color=colour,
                    label=f"{label} ({len(epochs)} transits)")
    m = r.metrics
    for key, colour in (("odd_depth_ppm", C_ALT), ("even_depth_ppm", C_BINNED)):
        if m.get(key) is not None:
            ax.axhline(-m[key], color=colour, ls="--", lw=0.9, alpha=0.7)
    for s in (-0.5, 0.5):
        ax.axvline(s * p.duration_h, color=C_MODEL, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title("odd and even folds")
    ax.legend(loc="lower right")

    # depth per epoch, coloured by parity
    ax = axes[1]
    for label, group, colour in (("odd", odd, C_ALT), ("even", even, C_BINNED)):
        if not group:
            continue
        ax.errorbar([e.epoch for e in group], [e.depth_ppm for e in group],
                    yerr=[e.depth_err_ppm for e in group], fmt="o", ms=4,
                    color=colour, label=label)
    for key, colour in (("odd_depth_ppm", C_ALT), ("even_depth_ppm", C_BINNED)):
        if m.get(key) is not None:
            ax.axhline(m[key], color=colour, lw=1.1)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("depth [ppm]")
    sig = m.get("difference_sigma")
    ax.set_title("per-epoch depths by parity"
                 + (f"  ({sig:.1f} sigma apart)" if sig is not None else ""))
    ax.legend()

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("odd_even")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Do odd and even transits have the same depth? Flags a possible "
              "eclipsing binary at twice the period; never fails a candidate."))
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

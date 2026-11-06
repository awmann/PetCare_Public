"""pc_interp -- is the dip bigger than what the detrending invents on its own?

Before any transit is measured, the window around it is masked and a GP
interpolates the star's variability across the gap.  On a quiet star that
interpolation is excellent.  On a fast, spotted rotator it is a guess: if the
mask spans a sizeable fraction of a rotation, nothing inside the gap constrains
the GP, and data minus guess can be anything -- a dip, a bump, or a dip followed
by a bump.  Fold enough of those at the right trial period and the average is a
clean, flat-bottomed "transit": the GP over-fits or distorts the curve until a
transit comes out.

So this test measures the interpolation error *on this star*, empirically.  The
same mask is dropped at many random out-of-transit times, the identical GP is
conditioned on everything else, and the "depth" it leaves behind is recorded
exactly as a real transit's would be.  That distribution is what the detrending
makes from nothing.  The real per-epoch depths are then compared with it:

* **the mean real depth** against the mean of the same number of random fakes
  (a bootstrap, so heavy tails are respected rather than assumed away), and
* **the shoulders** -- the masked-but-out-of-transit points either side.  A
  transit leaves them at zero; an interpolation error swings them the other way.

Nothing here needs a rotation period, so a wrong one cannot break it, and the
test scales itself: on a 5-day rotator the fakes are tiny, on a 5-hour one they are
tens of thousands of ppm.

What it cannot see: interpolation errors that repeat with the candidate's own
period (a rotation harmonic).  The random fakes are independent by
construction, so a commensurate signal looks coherent here.  That case is
pc_phasecurve's.

    python -m petcare.scripts.pc_interp --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, ephem, power
from .. import prep as pcprep
from ..fitting import gp_trend_masked
from ..plotting import C_ALT, C_BAD, C_BINNED, C_MODEL, annotate_status, figure, save
from ..result import Result

TEST = "interp"
VERSION = "0.2.0"

#: Mask half-width in transit durations -- must match what prep.prepare masks
#: (in_transit width=2.5 is a full width of 2.5 T14, so 1.25 T14 either side).
MASK_HALF_T14 = 1.25
#: Points within this many T14 of centre are "in transit" for the depth.
CORE_HALF_T14 = 0.4
#: The shoulders: masked, but outside the transit.
SHOULDER_T14 = (0.6, MASK_HALF_T14)
#: Data needed either side of a window, out to this many T14, for an epoch
#: (real or fake) to count -- so fakes are drawn from the same coverage as reals.
FLANK_T14 = 3.0
#: Fraction of the expected in-window cadences that must be present.
MIN_COVERAGE = 0.6
#: Spacing of fake windows within one GP evaluation, in T14.  Far enough apart
#: that each gap is conditioned on its own surroundings.
FAKE_SPACING_T14 = 10.0
#: Fakes are kept away from real transits by this much, in T14.
FAKE_CLEARANCE_T14 = 3.0
TARGET_FAKES = 300
MAX_BATCHES = 12
N_BOOT = 20000

#: Significance of the shoulder-corrected mean depth against the mean of as
#: many fakes.  Calibrated on a benchmark of known young planets: every planet
#: with a dip at its ephemeris scores 7.0-61 sigma (the lowest with two epochs;
#: HIP 67522 b 15.8 despite interpolation noise 55x white).  Two planets first
#: failed because their archive t0s missed the transit by 1.9 and 1.5 d; refit
#: on TESS they score 20+ sigma, so all now pass.  Note the search picked each candidate as the best of many
#: trial periods, so 3-4 sigma here is what noise reaches -- hence FAIL < 5.
PASS_SIGMA = 7.0
#: Below this interpolation inflation (fake-depth scatter over the white-noise
#: expectation) the detrending is not the problem, so a low score only flags.
ARTEFACT_INFLATION = 3.0
FAIL_SIGMA = 5.0


def add_arguments(ap):
    g = ap.add_argument_group("interp")
    g.add_argument("--bin", type=float, default=None,
                   help="bin seconds (default: a fifth of the duration, 120-600 s)")
    g.add_argument("--seed", type=int, default=1,
                   help="random seed for the fake windows")
    return ap


def _measure(t, y, centre, t14, cadence):
    """Depth (positive = dip) and shoulder level at one window, or None."""
    dt = t - centre
    inwin = np.abs(dt) < MASK_HALF_T14 * t14
    expected = 2 * MASK_HALF_T14 * t14 / cadence
    if inwin.sum() < MIN_COVERAGE * expected:
        return None
    left = (dt < -MASK_HALF_T14 * t14) & (dt > -FLANK_T14 * t14)
    right = (dt > MASK_HALF_T14 * t14) & (dt < FLANK_T14 * t14)
    if left.sum() < 3 or right.sum() < 3:
        return None
    core = np.abs(dt) < CORE_HALF_T14 * t14
    sh = (np.abs(dt) > SHOULDER_T14[0] * t14) & (np.abs(dt) < SHOULDER_T14[1] * t14)
    if core.sum() < 2 or sh.sum() < 2:
        return None
    base = np.median(y[left | right])
    return -(np.mean(y[core]) - base), np.mean(y[sh]) - base


def _profile(t, y, centres, t14, edges):
    """Median residual in bins of time-from-centre, stacked over windows."""
    rows = []
    for c in centres:
        dt = (t - c) / t14
        m = np.abs(dt) < edges[-1]
        if m.sum() < 5:
            continue
        idx = np.digitize(dt[m], edges) - 1
        row = np.full(edges.size - 1, np.nan)
        for k in range(edges.size - 1):
            s = y[m][idx == k]
            if s.size:
                row[k] = np.mean(s)
        rows.append(row)
    return np.array(rows)


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r
    if not cli.require(r, ctx.candidate.duration_h,
                       "no transit duration: pass --duration or run pc_fit first"):
        return r

    period = float(ctx.candidate.period)
    t0 = float(ctx.candidate.t0_btjd)
    t14 = float(ctx.candidate.duration_h) / 24.0
    bin_s = getattr(a, "bin", None) or float(np.clip(t14 * 86400 / 5, 120.0, 600.0))
    try:
        lc = ctx.lc
        p = pcprep.prepare(ctx, bin_seconds=float(bin_s))
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)
    r.set_noise(**p.noise_kwargs())
    if p.gp is None or not getattr(p.gp, "success", False):
        return r.inconclusive("no GP variability model, so no interpolation to test")

    t, f, err = p.t, p.f, p.err
    cadence = float(np.median(np.diff(t))) if t.size > 1 else bin_s / 86400
    y_real = p.f_detrended - 1.0

    # ---- real epochs ------------------------------------------------------
    ep_all = np.unique(ephem.epoch_number(t, period, t0))
    real = []
    for e in ep_all:
        c = float(ephem.transit_time(e, period, t0))
        m = _measure(t, y_real, c, t14, cadence)
        if m is not None:
            real.append((int(e), c, *m))
    if len(real) < 1:
        return r.inconclusive("no transit has enough data inside and either side of "
                              "its window to measure")
    real_depth = np.array([x[2] for x in real])
    real_sh = np.array([x[3] for x in real])

    # ---- fake windows -----------------------------------------------------
    real_mask = ephem.in_transit(t, period, t0, t14, width=2 * MASK_HALF_T14)
    ph = ephem.phase(t, period, t0)
    clear_ok = np.abs(ph) > FAKE_CLEARANCE_T14 * t14
    rng = np.random.default_rng(getattr(a, "seed", 1))
    spacing = FAKE_SPACING_T14 * t14
    fakes, last_batch = [], None
    for _ in range(MAX_BATCHES):
        if len(fakes) >= TARGET_FAKES:
            break
        start = t.min() + rng.uniform(0, spacing)
        cand = np.arange(start, t.max(), spacing)
        # centre must be clear of real transits (in phase), with data near it
        cph = ephem.phase(cand, period, t0)
        cand = cand[np.abs(cph) > FAKE_CLEARANCE_T14 * t14]
        near = np.searchsorted(t, cand)
        near = np.clip(near, 1, t.size - 1)
        gap = np.minimum(np.abs(t[near] - cand), np.abs(t[near - 1] - cand))
        cand = cand[gap < 2 * cadence]
        if cand.size == 0:
            continue
        fmask = np.zeros(t.size, bool)
        for c in cand:
            fmask |= np.abs(t - c) < MASK_HALF_T14 * t14
        trend = gp_trend_masked(t, f, err, real_mask | fmask, p.gp)
        y = f - trend
        kept = []
        for c in cand:
            m = _measure(t, y, c, t14, cadence)
            if m is not None and clear_ok[np.argmin(np.abs(t - c))]:
                fakes.append(m)
                kept.append(c)
        if kept:
            last_batch = (kept, y)   # one batch's residuals, for the profile plot
    if len(fakes) < 30:
        return r.inconclusive(f"only {len(fakes)} random windows fit between the "
                              f"transits -- too few to know what the detrending "
                              f"does on its own (period {period:.3f} d vs "
                              f"{FAKE_CLEARANCE_T14 * 2:g} T14 needed)")
    fake = np.array(fakes)
    fd, fs = fake[:, 0], fake[:, 1]

    # ---- statistics ---------------------------------------------------------
    n = real_depth.size
    mu = float(np.median(fd))
    sig = float(1.4826 * np.median(np.abs(fd - mu)))
    sig = max(sig, float(np.std(fd)) * 0.5, 1e-7)
    mean_real = float(np.mean(real_depth))
    z = (mean_real - mu) / (sig / np.sqrt(n))
    boot = rng.choice(fd, size=(N_BOOT, n), replace=True).mean(axis=1)
    p_boot = float((np.sum(boot >= mean_real) + 1) / (N_BOOT + 1))
    # Shoulder-corrected depth.  On the fakes, depth and shoulder are tightly
    # anticorrelated: an interpolation that guesses high in the middle guesses
    # low at the edges.  So the shoulders predict how much "depth" an epoch's
    # interpolation error alone would leave, and a real transit is the excess
    # over that prediction -- it deepens the centre without moving the edges.
    # A robust line (two passes, 3-sigma clip) fitted to the fakes only.
    k_fit = np.polyfit(fs, fd, 1)
    for _ in range(2):
        res = fd - np.polyval(k_fit, fs)
        s_r = 1.4826 * np.median(np.abs(res - np.median(res)))
        ok = np.abs(res - np.median(res)) < 3 * max(s_r, 1e-9)
        if ok.sum() > 10:
            k_fit = np.polyfit(fs[ok], fd[ok], 1)
    fres = fd - np.polyval(k_fit, fs)
    mu_c = float(np.median(fres))
    sig_c = float(1.4826 * np.median(np.abs(fres - mu_c)))
    sig_c = max(sig_c, float(np.std(fres)) * 0.5, 1e-7)
    real_c = real_depth - np.polyval(k_fit, real_sh)
    mean_c = float(np.mean(real_c))
    z_c = (mean_c - mu_c) / (sig_c / np.sqrt(n))
    boot_c = rng.choice(fres, size=(N_BOOT, n), replace=True).mean(axis=1)
    p_boot_c = float((np.sum(boot_c >= mean_c) + 1) / (N_BOOT + 1))
    # the white-noise expectation for one epoch's core, for scale
    n_core = max(1, int(2 * CORE_HALF_T14 * t14 / cadence))
    white_epoch = (p.white_ppm / 1e6) / np.sqrt(n_core) if np.isfinite(p.white_ppm) else np.nan
    inflation = sig / white_epoch if white_epoch and np.isfinite(white_epoch) else np.nan
    # shoulders: a transit leaves them at zero, an interpolation error swings
    overshoot = float(np.mean(real_sh) / mean_real) if mean_real > 0 else np.nan
    fake_corr = float(np.corrcoef(fd, fs)[0, 1]) if fd.size > 3 else np.nan

    r.metrics.update({
        "n_real_epochs": int(n), "n_fakes": int(fd.size),
        "real_depth_ppm_mean": mean_real * 1e6,
        "real_depth_ppm": [float(x * 1e6) for x in real_depth],
        "real_shoulder_ppm": [float(x * 1e6) for x in real_sh],
        "real_epochs": [int(x[0]) for x in real],
        "fake_depth_median_ppm": mu * 1e6, "fake_depth_sigma_ppm": sig * 1e6,
        "fake_depth_p95_ppm": float(np.percentile(fd, 95) * 1e6),
        "white_per_epoch_ppm": float(white_epoch * 1e6) if np.isfinite(white_epoch) else None,
        "interp_inflation": float(inflation) if np.isfinite(inflation) else None,
        "z_mean_depth": float(z), "p_bootstrap": p_boot,
        "shoulder_over_depth": overshoot if np.isfinite(overshoot) else None,
        "fake_depth_shoulder_corr": fake_corr if np.isfinite(fake_corr) else None,
        "frac_epochs_above_3sig": float(np.mean((real_depth - mu) > 3 * sig)),
        "shoulder_slope": float(k_fit[0]),
        "shoulder_intercept_ppm": float(k_fit[1] * 1e6),
        "corrected_depth_ppm_mean": mean_c * 1e6,
        "corrected_sigma_ppm": sig_c * 1e6,
        "z_corrected": float(z_c), "p_bootstrap_corrected": p_boot_c,
    })
    r.set_self_check(True, f"{fd.size} random windows with the same mask, same GP "
                           f"and the same coverage rule as the {n} real epochs")

    desc = (f"mean depth {mean_real * 1e6:.0f} ppm over {n} epoch(s); the same "
            f"detrending at {fd.size} random times leaves {mu * 1e6:+.0f} +/- "
            f"{sig * 1e6:.0f} ppm per window ({inflation:.1f}x the white noise), "
            f"so the mean stands at {z:.1f} sigma (bootstrap p = {p_boot:.1g}). "
            f"Shoulders average {overshoot:+.2f} of the depth. Correcting each "
            f"epoch for what its shoulders predict, the excess is "
            f"{mean_c * 1e6:.0f} +/- {sig_c * 1e6 / np.sqrt(n):.0f} ppm "
            f"({z_c:.1f} sigma).")
    z = z_c   # the verdict uses the shoulder-corrected statistic
    if z >= PASS_SIGMA:
        r.passed(f"the dip is far larger than anything the detrending makes on its "
                 f"own. {desc}")
    elif z >= FAIL_SIGMA:
        r.flag(f"the dip is only marginally larger than what the detrending makes "
               f"on its own on this star. {desc}")
    elif np.isfinite(inflation) and inflation < ARTEFACT_INFLATION:
        # The interpolation is nearly as good as the white noise here, so a weak
        # result is a weak dip, not a manufactured one.  Not this test's call.
        r.flag(f"the dip is weak, but not because of the detrending: the GP "
               f"interpolates this star almost perfectly. {desc}")
    else:
        power.gate(r, ctx, run, f"the dip is no larger than what the GP interpolation "
                   f"invents at random times on this star -- the 'transit' is consistent "
                   f"with detrending error. {desc}")

    if ctx.make_plots:
        try:
            r.add_plot(_plot(ctx, r, t, y_real, real, fd, fs, mu, sig, mean_real,
                             last_batch, t14))
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"plot failed: {type(exc).__name__}: {exc}")
    return r


def _plot(ctx, r, t, y_real, real, fd, fs, mu, sig, mean_real, last_batch, t14):
    fig, ax = figure(nrows=1, ncols=3, figsize=(16, 5.0))
    rd = np.array([x[2] for x in real]) * 1e6
    rs = np.array([x[3] for x in real]) * 1e6
    n = rd.size
    # 1: what the detrending makes from nothing, and the real epochs on it
    lo, hi = np.percentile(np.concatenate([fd * 1e6, rd]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    ax[0].hist(fd * 1e6, bins=bins, color=C_BINNED, alpha=0.45,
               label=f"{fd.size} random windows")
    for x in rd:
        ax[0].axvline(x, color=C_BAD, lw=0.8, alpha=0.7)
    ax[0].axvline(mean_real * 1e6, color=C_BAD, lw=2.5, label=f"real mean ({n} epochs)")
    ax[0].axvspan((mu - 3 * sig / np.sqrt(n)) * 1e6, (mu + 3 * sig / np.sqrt(n)) * 1e6,
                  color=C_ALT, alpha=0.2, label="3 sigma for a mean of that many fakes")
    ax[0].set_xlabel("depth left by the detrending [ppm]")
    ax[0].set_title("what the detrending invents vs the real epochs", fontsize=10)
    ax[0].legend(fontsize=7)
    # 2: depth vs shoulder
    ax[1].scatter(fd * 1e6, fs * 1e6, s=6, color=C_BINNED, alpha=0.4, label="random windows")
    ax[1].scatter(rd, rs, s=28, color=C_BAD, label="real epochs", zorder=3)
    ax[1].axhline(0, color="0.5", lw=0.6)
    k = r.metrics.get("shoulder_slope")
    if k is not None:
        ss = np.linspace(np.percentile(fs, 1), np.percentile(fs, 99), 10)
        b0 = r.metrics["shoulder_intercept_ppm"] / 1e6
        ax[1].plot((b0 + k * ss) * 1e6, ss * 1e6, "-", color=C_ALT, lw=1.5,
                   label="what interpolation error alone gives")
    ax[1].set_xlabel("depth [ppm]")
    ax[1].set_ylabel("shoulder level [ppm] (a transit leaves 0)")
    ax[1].set_title("interpolation errors swing the shoulders; transits do not",
                    fontsize=10)
    ax[1].legend(fontsize=7)
    # 3: stacked profile, real vs fakes
    edges = np.linspace(-FLANK_T14, FLANK_T14, 37)
    cen = 0.5 * (edges[1:] + edges[:-1])
    yr = _profile(t, y_real, [x[1] for x in real], t14, edges)
    if last_batch is not None:
        fp = _profile(t, last_batch[1], last_batch[0], t14, edges)
        if fp.size:
            q = np.nanpercentile(fp, [16, 50, 84], axis=0) * 1e6
            ax[2].fill_between(cen, q[0], q[2], color=C_BINNED, alpha=0.25,
                               label="random windows, 16-84%")
    for row in yr:
        ax[2].plot(cen, row * 1e6, "-", color=C_BAD, lw=0.5, alpha=0.35)
    if yr.size:
        ax[2].plot(cen, np.nanmean(yr, axis=0) * 1e6, "-", color=C_MODEL, lw=2,
                   label="real epochs, mean")
    for s in (-MASK_HALF_T14, MASK_HALF_T14):
        ax[2].axvline(s, color="0.5", ls=":", lw=0.8)
    ax[2].axhline(0, color="0.5", lw=0.6)
    ax[2].set_xlabel("time from centre [T14]  (dotted: masked window)")
    ax[2].set_ylabel("detrended flux [ppm]")
    ax[2].set_title("each real epoch (thin) against the detrending's own scatter",
                    fontsize=10)
    ax[2].legend(fontsize=7)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Is the dip bigger than what the GP detrending invents on its own "
              "at random times on this star?"))
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

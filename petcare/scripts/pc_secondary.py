"""pc_secondary -- is there a secondary eclipse, at any phase?

**This test can FAIL a candidate**, and it is one of only three that can.  A
secondary eclipse deeper than
:data:`SECONDARY_FAIL_DEPTH_PPM` cannot be produced by a planet in the TESS
band -- reflection off a hot Jupiter reaches about 100 ppm, and thermal
emission from even a self-luminous young giant does not reach a percent -- so
a confident detection of one refutes the candidate by itself.

Three separate things, because an eclipsing binary can show any one of them
without the others:

**A secondary at any phase, not just 0.5.**  An eccentric binary's secondary
sits off-centre, and that is exactly what was caught by eye on several
candidates in an earlier search -- "secondary eclipse slightly off-center" was
a disposition note there.  A test that only looks at phase 0.5 misses those, so this scans
the whole phase range with the same estimator used for the transit itself and
reports a depth limit at *every* phase.

**Ellipsoidal modulation at P/2.**  Tidal distortion of the primary by a
massive companion.  It frequently survives when no secondary is detectable at
all, which makes it the more sensitive of the two for a close binary.

**Doppler beaming at P.**  Smaller, but it has the opposite phase to
reflection and so is worth reporting separately rather than folded into one
"out-of-transit variation" number.

The noise level is measured, not assumed: the scatter of the same depth
statistic across all the phases where nothing is expected *is* the uncertainty
on the depth at any one phase, and it is printed beside every limit.

    python -m petcare.scripts.pc_secondary --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

import copy

from .. import cli, ephem, prep as pcprep
from ..plotting import (C_ALT, C_BAD, C_BINNED, C_MODEL, C_OK, annotate_status,
                        bin_series, figure, save)
from ..result import Result

TEST = "secondary"
VERSION = "0.1.0"

#: Significance at which a dip away from the transit is worth a human's time.
#: PROVISIONAL: 4 sigma on a phase scan of a few hundred trials expects ~0.01
#: false positives per candidate, which is the right order for 30k targets
#: without being so strict that a real secondary is missed.
SECONDARY_FLAG_SIGMA = 4.0

#: And at which it is strong enough to name as a likely binary.
SECONDARY_STRONG_SIGMA = 6.0

#: Ellipsoidal amplitude significance worth flagging.
ELLIPSOIDAL_FLAG_SIGMA = 4.0

#: Secondary depth above which the companion cannot be a planet, so the
#: candidate is refuted outright rather than flagged.  The physics: in the
#: TESS band reflection off a hot Jupiter reaches roughly
#: ``Ag (Rp/a)^2`` ~ 100 ppm, and thermal emission from even a self-luminous
#: young giant -- 3000 K against a 4000 K host, Rp/R* = 0.1 -- is a few
#: thousand ppm at the absolute limit.  Anything deeper is a star.
#: PROVISIONAL: 2000 ppm is deliberately conservative, leaving the ambiguous
#: band 2000 ppm and below as a FLAG rather than a refutation.
SECONDARY_FAIL_DEPTH_PPM = 2000.0

#: Epochs a trial phase needs before its depth is a measurement.  A real
#: eclipse repeats; below this there is no way to tell one from a single bad
#: baseline fit, and the estimator is a median over epochs for the same
#: reason.
MIN_EPOCHS_PER_PHASE = 4

#: Epochs further than this many MAD from the median depth are dropped before
#: the weighted mean.  3 is the usual clip; the failure it guards against is a
#: single epoch's polynomial baseline going wrong, which is far outside this.
EPOCH_CLIP_MAD = 3.0

#: Fraction of an injected eclipse the scan must recover before its limits
#: mean anything.  Below this the variability model is absorbing dips and a
#: non-detection says nothing.  PROVISIONAL, but the failure it guards is not
#: marginal: the old scan recovered 1% of a 15000 ppm injection.
MIN_RECOVERY = 0.5

#: Depth injected for that check, as a multiple of the candidate's own depth
#: (a secondary is at most about as deep as the primary), floored so a shallow
#: candidate still gets a meaningful test.
RECOVERY_DEPTH_FRAC = 0.5
RECOVERY_DEPTH_MIN_PPM = 500.0


def add_arguments(ap):
    g = ap.add_argument_group("secondary")
    g.add_argument("--bin", type=float, default=600.0)
    g.add_argument("--n-phase", type=int, default=0,
                   help="phase grid points; 0 chooses ~4 per duration")
    g.add_argument("--guard-durations", type=float, default=2.0,
                   help="phases within this many durations of the transit are "
                        "excluded from the scan and from the noise estimate")
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
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)

    dur = p.duration_days
    period = p.period
    guard = float(getattr(a, "guard_durations", 2.0)) * dur / period  # in phase

    n_phase = int(getattr(a, "n_phase", 0) or 0)
    if n_phase <= 0:
        n_phase = int(np.clip(round(4.0 * period / dur), 40, 600))

    # ---- the phase scan ---------------------------------------------------
    phases = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    depths = np.full(n_phase, np.nan)
    n_in = np.zeros(n_phase, dtype=int)

    ph_frac = ephem.phase_frac(p.t, period, p.t0)
    half = 0.5 * 0.8 * dur / period          # in phase units
    base_lo, base_hi = 0.75 * dur / period, 3.0 * dur / period
    prim = np.abs(_wrap(ph_frac)) < base_lo

    # Everything that is a transit of something already known: the candidate
    # itself, and every other signal on this star.  Without the second part
    # the scan reports a sibling planet as a secondary eclipse -- a known planet
    # in a four-planet system was refuted by a "22983 ppm secondary at phase
    # 0.250", which is one of its three siblings landing there when folded on
    # its period.
    others = [k for k in (ctx.known or [])
              if not ephem.match_known(period, p.t0, dur * 24, [k],
                                       float(p.t.min()), float(p.t.max()))]
    known_m = (ephem.mask_known_transits(p.t, others, width=2.0)
               if others else np.zeros(p.t.size, bool))
    exclude = prim | known_m
    r.metrics["known_signals_masked"] = [
        {"label": getattr(k, "label", None), "period": getattr(k, "period", None)}
        for k in others]
    r.metrics["n_cadences_masked_for_known"] = int(known_m.sum())
    if others:
        ctx.notes.append(
            f"masked {int(known_m.sum())} cadences belonging to "
            f"{len(others)} other known signal(s) before scanning for a "
            f"secondary; a sibling planet folded on this period looks exactly "
            f"like one")

    for i, ph0 in enumerate(phases):
        d = _wrap(ph_frac - ph0)
        in_m = np.abs(d) < half
        oot_m = (np.abs(d) >= base_lo) & (np.abs(d) < base_hi)
        # never let the primary transit into a trial's baseline
        oot_m &= ~prim
        if in_m.sum() < 10 or oot_m.sum() < 30:
            continue
        # a trial phase where a known signal sits is not measurable here
        if in_m.sum() and known_m[in_m].mean() > 0.3:
            continue
        n_in[i] = int(in_m.sum())
        depths[i] = _local_depth_at(p, ph0, half, base_lo, base_hi,
                                    exclude=exclude)

    # How much of an eclipse does this scan actually return?  Measured before
    # anything is read off it, and applied to the depths -- a scan that
    # recovers 70% of what it is given is under-reporting every depth and
    # every limit by the same factor.
    inj_depth = max(RECOVERY_DEPTH_MIN_PPM,
                    RECOVERY_DEPTH_FRAC * (ctx.candidate.depth_ppm or 0.0))
    trial_phases = [ph for ph in (0.25, 0.5, 0.75) if min(ph, 1 - ph) > guard]
    recovery, n_trials = _recovery_check(p, trial_phases, inj_depth,
                                         base_lo, half, prim, exclude)
    r.metrics["recovery"] = {
        "injected_ppm": inj_depth, "phases": trial_phases,
        "n_trials": n_trials, "median_fraction_recovered": _f(recovery),
        "threshold": MIN_RECOVERY,
        "note": ("depths and limits below are divided by this, so they are "
                 "in true-depth units")}
    if n_trials and np.isfinite(recovery) and recovery >= MIN_RECOVERY:
        depths = depths / recovery

    good = np.isfinite(depths)
    away = good & (np.minimum(phases, 1 - phases) > guard)
    if not cli.require(r, away.sum() >= 20,
                       f"only {int(away.sum())} usable phases outside the "
                       f"transit; the scan needs at least 20"):
        r.set_self_check(False, f"{int(away.sum())} usable phases")
        return r

    # The scatter across phases *is* the uncertainty at any one phase: it
    # contains the residual stellar variability and the baseline error, which
    # a photon-noise estimate does not.  But it is floored at the white-noise
    # prediction, because the GP is a conditional mean and has already removed
    # some genuinely random power -- without the floor the scan reported an
    # 18 ppm three-sigma limit on a star whose white noise alone allows 33.
    measured = float(1.4826 * np.median(np.abs(depths[away] -
                                               np.median(depths[away]))))
    n_typical = float(np.median(n_in[away])) if away.any() else np.nan
    white_pred = (p.white_ppm / np.sqrt(n_typical)
                  if np.isfinite(n_typical) and n_typical > 0 else np.nan)
    sigma = float(np.nanmax([measured, white_pred]))
    sigma_source = ("scatter across phases" if measured >= (white_pred or 0)
                    else "white-noise floor (the phase scatter was below it)")
    offset = float(np.median(depths[away]))
    z = (depths - offset) / sigma if sigma > 0 else np.full(n_phase, np.nan)

    i_best = int(np.nanargmax(np.where(away, z, -np.inf)))
    best_phase = float(phases[i_best])
    best_depth = float(depths[i_best] - offset)
    best_z = float(z[i_best])

    # phase 0.5 specifically, for the circular case
    i_half = int(np.argmin(np.abs(phases - 0.5)))
    z_half = float(z[i_half]) if np.isfinite(z[i_half]) else np.nan

    # ---- ellipsoidal and beaming -----------------------------------------
    # These are smooth, full-phase signals, not localised dips, so the ordinary
    # detrended series is the right input for them.
    harm = _harmonics(ph_frac, p.f_detrended, exclude=prim)

    # ---- eccentricity implied by an off-centre secondary ------------------
    ecc_cos_w = None
    if np.isfinite(best_z) and best_z >= SECONDARY_FLAG_SIGMA:
        # e cos(w) = (pi/2) * (phase_sec - 0.5), to first order in e
        ecc_cos_w = float(np.pi / 2 * (best_phase - 0.5))

    r.set_noise(sigma, f"depth uncertainty per phase, from the {sigma_source}",
                phase_scatter_ppm=_f(measured),
                white_prediction_ppm=_f(white_pred),
                n_cadences_per_trial=_f(n_typical),
                n_phase_trials=int(away.sum()),
                phase_grid_points=n_phase,
                point_to_point_ppm=_f(p.white_ppm))

    r.metrics.update({
        "n_phase": n_phase,
        "phase_scan": {"phase": [float(x) for x in phases],
                       "depth_ppm": [_f(x) for x in depths],
                       "z": [_f(x) for x in z]},
        "depth_limit_3sigma_ppm": 3.0 * sigma,
        "best_phase": best_phase,
        "best_depth_ppm": best_depth,
        "best_sigma": best_z,
        "best_phase_offset_from_0p5": float(best_phase - 0.5),
        "z_at_phase_0.5": z_half,
        "depth_at_phase_0.5_ppm": _f(depths[i_half] - offset),
        "ecc_cos_omega_if_secondary": ecc_cos_w,
        "ellipsoidal": harm["ellipsoidal"],
        "beaming": harm["beaming"],
        "primary_depth_ppm": ctx.candidate.depth_ppm,
        "secondary_over_primary": (
            _f(best_depth / ctx.candidate.depth_ppm)
            if ctx.candidate.depth_ppm else None),
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{int(away.sum())} of {n_phase} phases usable"], True
    med_in = float(np.median(n_in[away])) if away.any() else 0
    checks.append(f"{med_in:.0f} cadences per trial (median)")
    if med_in < 20:
        checks.append("too few cadences per phase trial to measure a depth")
        ok = False
    if sigma <= 0 or not np.isfinite(sigma):
        checks.append("phase-scan scatter is not measurable")
        ok = False

    if n_trials and np.isfinite(recovery):
        checks.append(f"recovers {recovery:.0%} of a {inj_depth:.0f} ppm "
                      f"injected eclipse")
        if recovery < MIN_RECOVERY:
            ok = False
    else:
        checks.append("injection-recovery check could not run")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    limit_note = (f"3-sigma depth limit {3 * sigma:.0f} ppm at any phase "
                  f"(1 sigma = {sigma:.0f} ppm, from the {sigma_source})")
    ell = harm["ellipsoidal"]

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif best_z >= SECONDARY_STRONG_SIGMA:
        where = ("at phase 0.5" if abs(best_phase - 0.5) < 0.02 else
                 f"at phase {best_phase:.3f}, off-centre by "
                 f"{best_phase - 0.5:+.3f} (e cos w = {ecc_cos_w:+.3f})")
        # A refutation cannot rest on an ephemeris the package declined to
        # measure.  ctx.candidate.source is "pc_fit" only when a fit converged
        # and passed its own self-check; anything else means the phase axis
        # here came from the input table, and a secondary's *phase* is the
        # whole claim.  Two FAILs in an early run sat on INCONCLUSIVE
        # fits, one of them at phase 0.047 with an implied e cos w of -0.71.
        ephemeris_measured = getattr(ctx.candidate, "source", None) == "pc_fit"
        r.metrics["ephemeris_source"] = getattr(ctx.candidate, "source", None)
        if best_depth >= SECONDARY_FAIL_DEPTH_PPM and not ephemeris_measured:
            r.flag(f"secondary eclipse {where}: {best_depth:.0f} ppm "
                   f"({best_depth / 1e4:.2f}%) at {best_z:.1f} sigma -- deep "
                   f"enough that no planet produces it, but the ephemeris "
                   f"this phase is measured against came from "
                   f"{getattr(ctx.candidate, 'source', 'the input') or 'the input'}, "
                   f"not from a converged fit. Flagging rather than refuting: "
                   f"a secondary's phase is the whole claim. Re-run once "
                   f"pc_fit produces a usable ephemeris. {limit_note}")
        elif best_depth >= SECONDARY_FAIL_DEPTH_PPM:
            r.fail(f"secondary eclipse {where}: {best_depth:.0f} ppm "
                   f"({best_depth / 1e4:.2f}%) at {best_z:.1f} sigma. No "
                   f"planet produces a secondary this deep in the TESS band: "
                   f"reflection off a hot Jupiter reaches ~100 ppm, and even "
                   f"a self-luminous young giant at 3000 K stays under "
                   f"{SECONDARY_FAIL_DEPTH_PPM:.0f}. This is a stellar "
                   f"companion. {limit_note}")
        else:
            r.flag(f"secondary eclipse {where}: {best_depth:.0f} ppm at "
                   f"{best_z:.1f} sigma -- significant, but shallow enough "
                   f"({best_depth:.0f} < {SECONDARY_FAIL_DEPTH_PPM:.0f} ppm) "
                   f"that thermal emission from a very hot young giant is "
                   f"not excluded. {limit_note}")
    elif best_z >= SECONDARY_FLAG_SIGMA:
        r.flag(f"possible secondary at phase {best_phase:.3f}: "
               f"{best_depth:.0f} ppm at {best_z:.1f} sigma. {limit_note}")
    elif ell.get("sigma") is not None and ell["sigma"] >= ELLIPSOIDAL_FLAG_SIGMA:
        r.flag(f"no secondary ({limit_note}) but ellipsoidal modulation at P/2 "
               f"is detected at {ell['sigma']:.1f} sigma "
               f"({ell['amplitude_ppm']:.0f} ppm) -- tidal distortion by a "
               f"massive companion often survives where a secondary does not")
    else:
        r.passed(f"no secondary at any phase and no ellipsoidal modulation: "
                 f"deepest phase {best_phase:.3f} at {best_z:.1f} sigma, "
                 f"{limit_note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, p, phases, depths - offset, z, sigma, guard, harm)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _recovery_check(p, phases, depth_ppm, base_lo, half, prim, exclude=None):
    """Inject a box eclipse and re-measure it with the scan's own machinery.

    A secondary-eclipse limit is a claim about sensitivity, and this is the
    only thing that substantiates it.  Without it the test reported clean
    non-detections on four candidates a human had already annotated
    "secondary eclipse", because the variability model was deleting the
    signal before the scan ran.

    Returns the median recovered fraction over a few trial phases.
    """
    ph_frac = ephem.phase_frac(p.t, p.period, p.t0)
    got = []
    for ph0 in phases:
        d = _wrap(ph_frac - ph0)
        in_m = np.abs(d) < half
        oot_m = (np.abs(d) >= base_lo) & (np.abs(d) < 3.0 * p.duration_days / p.period)
        oot_m &= ~prim
        if in_m.sum() < 10 or oot_m.sum() < 30:
            continue
        injected = copy.copy(p)
        injected.f = p.f.copy()
        injected.f[in_m] -= depth_ppm / 1e6
        base_hi = 3.0 * p.duration_days / p.period
        meas = _local_depth_at(injected, ph0, half, base_lo, base_hi,
                               exclude=exclude)
        base = _local_depth_at(p, ph0, half, base_lo, base_hi, exclude=exclude)
        if not (np.isfinite(meas) and np.isfinite(base)):
            continue
        got.append((meas - base) / depth_ppm)
    return (float(np.median(got)) if got else float("nan")), len(got)


def _local_depth_at(p, phase, half, base_lo, base_hi, poly_order=2,
                    exclude=None):
    """Depth at a trial phase, from a polynomial baseline fitted per epoch.

    Each epoch's window around the trial phase is levelled with a low-order
    polynomial fitted to its own out-of-dip points, and the depths are
    combined.  This is the notch form, and it is used here for the same
    reason ``pc_fit`` uses it: a quadratic over a few durations cannot follow
    a dip that narrow, so it can neither absorb a real eclipse nor
    interpolate a fake one.

    Both failure modes were live.  The original scan ran on a single
    GP-detrended series whose mask covered only the primary, so a secondary
    anywhere else was in the data the GP was conditioned on and was removed
    along with the star -- injected eclipses came back at 40% of their depth
    at 500 ppm and **1%** at 15000 ppm.  Masking each trial phase before
    conditioning fixed the absorption but introduced the opposite problem:
    the GP then interpolates across a gap in every epoch at once, and the
    difference between that interpolation and the data is a broad, correlated
    structure that is not an eclipse.  On one binary it reported 64393 ppm
    where a local baseline measures 21951 -- a real secondary, overestimated
    threefold.
    """
    t, f = p.t, p.f
    period = p.period
    tc = p.t0 + phase * period
    d = ((t - tc + 0.5 * period) % period) - 0.5 * period
    ep = np.round((t - tc) / period).astype(int)
    win = base_hi * period
    keep = np.ones(t.size, bool) if exclude is None else ~np.asarray(exclude, bool)

    ds, ns = [], []
    for e in np.unique(ep[np.abs(d) < win]):
        m = (ep == e) & (np.abs(d) < win) & keep
        if m.sum() < 12:
            continue
        x, y = d[m], f[m]
        inb = np.abs(x) < half * period
        out = np.abs(x) >= base_lo * period
        if inb.sum() < 3 or out.sum() < 8:
            continue
        try:
            c = np.polyfit(x[out], y[out], poly_order)
        except (np.linalg.LinAlgError, ValueError):
            continue
        ds.append(float(np.mean(np.polyval(c, x[inb]) - y[inb])) * 1e6)
        ns.append(int(inb.sum()))
    if len(ds) < MIN_EPOCHS_PER_PHASE:
        return np.nan
    # Combine the epochs the way pc_depths combines the primary's: a weighted
    # mean, but only after throwing out epochs that disagree with the rest.
    #
    # A plain weighted mean let one bad polynomial fit carry the whole
    # measurement -- in a four-planet system with nothing deeper
    # than 5000 ppm, it produced a "22865 ppm secondary at 16 sigma" and
    # refuted a real planet.  A plain median fixed that and cost the real
    # detection on the binary above, because with a 12.6 d period only some
    # epochs have coverage at a given phase and the rest dilute it to nothing.
    # Clipping then weighting keeps both: an eclipse seen in several epochs
    # survives, an artifact seen in one does not.
    d_arr, n_arr = np.asarray(ds, float), np.asarray(ns, float)
    med = float(np.median(d_arr))
    mad = float(1.4826 * np.median(np.abs(d_arr - med)))
    keep_e = (np.abs(d_arr - med) <= EPOCH_CLIP_MAD * mad) if mad > 0 else \
        np.ones(d_arr.size, bool)
    if keep_e.sum() < MIN_EPOCHS_PER_PHASE:
        keep_e = np.ones(d_arr.size, bool)
    return float(np.sum(d_arr[keep_e] * n_arr[keep_e]) / np.sum(n_arr[keep_e]))


def _detrended_for(p, phase_mask):
    """Flux with the variability removed, computed *without* seeing the dip.

    The scan used to run on the single detrended series from
    :func:`petcare.prep.prepare`, whose variability model masks only the
    primary transit.  A secondary at any other phase is therefore left in the
    data the GP is conditioned on, and the GP removes it along with the
    star -- so the search ran on a series the signal had been deleted from.

    Measured on one star by injecting box eclipses into the real light
    curve and re-measuring them: 500 ppm came back as 40%, 5000 ppm as 19%,
    and 15000 ppm as **1%**.  The deeper the eclipse, the harder the GP
    worked to erase it.  That is the same failure as a GP eating a transit,
    and it made every secondary-eclipse non-detection in this package
    meaningless.

    The fix is the same one the primary already gets: mask the thing being
    searched for before conditioning.  One trend per trial phase.
    """
    return p.f - pcprep._trend_at(p, p.t0, extra_mask=phase_mask)


def _wrap(x):
    """Wrap a phase difference into [-0.5, 0.5)."""
    return (np.asarray(x, float) + 0.5) % 1.0 - 0.5


def _harmonics(ph_frac, f, exclude) -> dict:
    """Least-squares amplitudes of ellipsoidal (P/2) and beaming (P) terms.

    Fitted together with a constant and the quadrature terms, on the
    out-of-transit cadences only, so the transit cannot leak into either.
    Uncertainties come from the residual scatter, so they contain the stellar
    variability the GP left behind rather than photon noise alone.
    """
    keep = ~np.asarray(exclude, bool) & np.isfinite(f)
    if keep.sum() < 100:
        return {"ellipsoidal": {"amplitude_ppm": None, "sigma": None,
                                "note": "too few out-of-transit cadences"},
                "beaming": {"amplitude_ppm": None, "sigma": None}}
    x = 2 * np.pi * np.asarray(ph_frac, float)[keep]
    y = (np.asarray(f, float)[keep] - 1.0) * 1e6

    A = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x),
                         np.cos(2 * x), np.sin(2 * x)])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"ellipsoidal": {"amplitude_ppm": None, "sigma": None},
                "beaming": {"amplitude_ppm": None, "sigma": None}}

    resid = y - A @ coef
    s2 = float(np.sum(resid ** 2) / max(len(y) - A.shape[1], 1))
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        err = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)

    # ellipsoidal is the cos(2 phi) term with a *minimum* at quadrature, so a
    # physical signal has a negative coefficient in this convention
    ell_amp = float(-coef[3])
    ell_err = float(err[3])
    beam_amp = float(coef[2])
    beam_err = float(err[2])
    return {
        "ellipsoidal": {
            "amplitude_ppm": ell_amp, "err_ppm": ell_err,
            "sigma": float(ell_amp / ell_err) if ell_err > 0 else None,
            "note": "cos(2*phase) term; positive means the flux dips at "
                    "quadrature as tidal distortion predicts"},
        "beaming": {
            "amplitude_ppm": beam_amp, "err_ppm": beam_err,
            "sigma": float(abs(beam_amp) / beam_err) if beam_err > 0 else None,
            "note": "sin(phase) term"},
        "residual_rms_ppm": float(np.sqrt(s2)),
        "n_used": int(keep.sum()),
    }


def _plot(ctx, r, p, phases, depths, z, sigma, guard, harm):
    fig, axes = figure(2, 1, figsize=(10, 7))

    ax = axes[0]
    ax.plot(phases, depths, "-", color=C_BINNED, lw=1.0)
    ax.axhspan(-3 * sigma, 3 * sigma, color=C_OK, alpha=0.12,
               label=f"+/-3 sigma ({3 * sigma:.0f} ppm)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvspan(0, guard, color=C_MODEL, alpha=0.12)
    ax.axvspan(1 - guard, 1, color=C_MODEL, alpha=0.12, label="primary (excluded)")
    ax.axvline(0.5, color=C_ALT, ls="--", lw=0.9, label="phase 0.5")
    bp = r.metrics.get("best_phase")
    if bp is not None:
        ax.plot([bp], [r.metrics["best_depth_ppm"]], "v", ms=8, color=C_BAD,
                label=f"deepest: {r.metrics['best_sigma']:.1f} sigma")
    # Scale to the *searched* phases only.  The primary leaks into the edge of
    # its own guard band, and left to autoscale it sets the y-limits: the
    # secondary search -- the entire point of the panel -- is then a flat line
    # at the bottom of a 3700 ppm axis.
    searched = (phases > guard) & (phases < 1 - guard)
    if searched.any():
        lo, hi = np.min(depths[searched]), np.max(depths[searched])
        pad = max(0.25 * (hi - lo), 4 * sigma)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("orbital phase (0 = transit)")
    ax.set_ylabel("depth [ppm]")
    ax.set_title("depth of a transit-width box at every phase "
                 "(scaled to the searched phases)")
    ax.legend(ncol=4, loc="lower right", fontsize=7)

    ax = axes[1]
    ph = ephem.phase_frac(p.t, p.period, p.t0)
    y = (p.f_detrended - 1) * 1e6
    xb, yb, eb, _ = bin_series(ph, y, 1.0 / 120, min_points=5)
    ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=2.5, color=C_BINNED, lw=0.8,
                label="binned fold")
    ell = harm.get("ellipsoidal", {})
    if ell.get("amplitude_ppm") is not None:
        xx = np.linspace(0, 1, 400)
        ax.plot(xx, -ell["amplitude_ppm"] * np.cos(4 * np.pi * xx), "-",
                color=C_MODEL, lw=1.3,
                label=f"ellipsoidal {ell['amplitude_ppm']:.0f} ppm"
                      + (f" ({ell['sigma']:.1f} sigma)"
                         if ell.get("sigma") is not None else ""))
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    lim = max(4 * sigma, 50)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title("full phase fold (zoomed to the out-of-transit level)")
    ax.legend(ncol=2)

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("secondary")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Scan every phase for a secondary eclipse, and fit ellipsoidal "
              "modulation and Doppler beaming."))
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

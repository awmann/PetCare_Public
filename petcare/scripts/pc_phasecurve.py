"""pc_phasecurve -- is the dip a transit, or is it the star's own modulation?

The single largest source of false candidates in a young-star search is the
star itself.  A spotted star rotating in 0.35 d leaves a residual after any
detrending window much longer than that, and folding the light curve at two or
four times the rotation period lines that residual up into something a box
search will happily call a transit.  On one star our search found a
"transit" at 0.70583 d -- twice the 0.3528 d rotation period to 0.04% -- and
PetCare's fold of it showed not a transit but a sinusoid, with the model's
transit sitting in the bottom of a rotation trough.  Every other test passed
it, because each measures its depth against a local baseline that partly
absorbs the modulation.

The question this test asks is whether the dip is a *sharp* feature or part of
a *smooth* periodic signal at the same period.

**The method.**  Fold over the full phase, fit a low-order Fourier series to
everything *outside* the transit window, and interpolate it across the window.
Then measure the dip twice: against the median, which is what the search saw,
and against the Fourier model, which is what is left once the star's smooth
periodic behaviour is accounted for.

* A planet on a quiet stretch of phase leaves the out-of-transit fold flat, the
  model interpolates flat across the window, and the whole dip survives.
* A rotation residual is smooth.  The out-of-transit fold descends into the
  window from both sides, the model follows it down, and nothing is left.

The Fourier order is capped so the model *cannot* represent a feature as short
as the transit: the highest harmonic has a timescale of at least three transit
durations.  And because the window is excluded from the fit, the model is never
pulled toward the dip -- it only interpolates across it.

**Both views are measured.**  On PetCare's detrended flux, and on the raw
normalised flux.  They fail in opposite places.  Detrending can remove most of
a rapid rotator's modulation and the "dip" with it, leaving nothing to
attribute; the raw flux keeps the rotation coherent exactly when the candidate
period is a harmonic of it, which is the case this test exists for.  For a real
planet on an active star at a period unrelated to rotation, the raw fold is
noisy and insensitive, and the detrended one carries the measurement.  A sharp
residual in *either* view is enough to save a candidate.

**Errors come from the light curve, not from a noise model.**  The same
residual-dip statistic is measured at sixty phases away from the transit, and
its scatter there is the uncertainty.  That absorbs red noise and incoherent
rotation without having to model either, and because the Fourier model was fit
including those off-phase points it slightly underestimates the scatter -- the
direction that favours keeping a candidate, not refuting it.

**The self-check is an injection.**  A box of the observed depth and duration is
added at an unrelated phase and recovered through exactly the same machinery.
If the model eats an injected transit, it would eat a real one, and the test
reports nothing.

**Across sectors.**  Spots evolve and planets do not.  The residual dip and the
amplitude of the smooth modulation are measured sector by sector, so a signal
that is really the rotation shows up as a modulation whose amplitude wanders
over the years while the "transit" comes and goes with it.

    python -m petcare.scripts.pc_phasecurve --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys
from itertools import pairwise

import numpy as np
from scipy import stats

from .. import cli, ephem
from .. import prep as pcprep
from ..plotting import (
    C_ALT,
    C_BAD,
    C_BINNED,
    C_DATA,
    C_MODEL,
    C_OK,
    annotate_status,
    figure,
    save,
)
from ..result import Result

TEST = "phasecurve"
VERSION = "0.1.1"

#: The raw dip has to be real before there is anything to attribute.  The
#: search found these at det_sigma > 8, so a dip that does not reach this in
#: the light curve PetCare reads is reported, not explained.
RAW_SIGMA_MIN = 5.0

#: Significance of the smooth out-of-transit modulation, as a delta-chi2 over a
#: flat line, needed before the dip can be blamed on it.
SMOOTH_SIGMA_MIN = 5.0

#: A residual dip at least this significant is a sharp feature beyond the
#: modulation, and saves the candidate whatever else is true.
RESID_SIGMA_MIN = 3.0

#: Fraction of the raw dip that the smooth model has to account for before the
#: dip is called modulation.
EXPLAINED_FRAC = 0.5

#: Half-width of the excluded window, in transit durations: the transit plus a
#: quarter-duration margin either side.
GAP_HALF_T14 = 0.75

#: Half-width of the "in transit" core the dip is averaged over.
CORE_HALF_T14 = 0.4

#: Highest Fourier harmonic has a timescale of at least this many durations, so
#: the model cannot represent the transit itself.
FOURIER_MIN_T14 = 3.0
FOURIER_MAX_ORDER = 25

#: Off-phase positions used to measure the scatter of the statistic.
N_NULL = 60

#: Injection recovery window for the self-check.
RECOVERY_RANGE = (0.5, 1.5)
INJECT_PHASE = 0.37

#: Sibling test: the dip, against the same dip where the star repeats.
#:
#: The Fourier model above has a ceiling -- it may not go fine enough to
#: represent the transit -- and a rotation pattern above that ceiling slips
#: through it.  One candidate is 6.0004 x its 0.7031 d rotation; the raw fold at
#: the candidate period is six identical rotation cycles, the "transit" is one
#: of the six minima, and it only looks like a transit after detrending, because
#: PetCare masks the candidate window while fitting the variability model and so
#: interpolates the trend *across* that one minimum and leaves it behind.  Both
#: views called it sharp.
#:
#: So: fold the raw flux again at P/n, excluding the candidate's own window, and
#: measure the same dip at the sibling phases where the star repeats.  n is
#: found from the data, not from a catalogue Prot, so a wrong Prot can only make
#: this miss a false positive, never refute a planet.  A planet sitting on a
#: rotation harmonic survives: its dip is the trough *plus* the transit, and the
#: excess over the siblings is the transit.
SIBLING_N_MAX = 15
SIBLING_SIGMA = 5.0            # siblings must be this significant (14 trials)
SIBLING_FRAC = 0.5             # ...and this large a fraction of the dip
SIBLING_EXCESS_SIGMA = 3.0     # a transit-sized excess saves the candidate

#: Phase bins per transit duration, for the fold and the fit.
BINS_PER_T14 = 4

#: How many individual transits to draw before and after detrending.
N_TRANSIT_PANELS = 12

#: Sector-to-sector stability.  Spots evolve and planets do not, so a
#: "transit" that is really rotation comes and goes over the years.  But
#: *statistically* significant changes are not the signature -- they are
#: always there.  The light curves' de-dilution factor and aperture differ
#: by sector, so a real planet's depth wanders by ten or fifteen percent
#: against white-noise errors of a few tens of ppm: HIP 67522 b reads
#: 5275-6636 ppm across four sectors at +/-25 ppm, chi2 in the tens of
#: thousands, and the first version of this test flagged every known planet
#: it was shown.  What a spot does that a planet cannot is change by a large
#: factor or change sign -- the 0.35 d rotator above, at four times its
#: period, reads 1145, -459, 99, -416, 750, 534 and 641 ppm.  So a
#: systematic floor is added in quadrature, and the single most discrepant
#: sector is set aside when there are enough to spare one: a stale ephemeris
#: drifts the transit out of the window in the latest sector (HIP 67522 b's
#: sector 102 reads 55 ppm), and that is a property of the period, not of
#: the star.
SECTOR_FLOOR_FRAC = 0.20
SECTOR_MIN_FOR_DROP = 4

#: A sector enters the stability test only if it holds this many transits.  A
#: per-sector depth from one or two events -- possibly partial, possibly moved
#: by TTVs -- is not a stability measurement: two real planets read 219
#: against 1859 ppm and 2812 against 8326 between sectors,
#: both with one or two transits per sector and known timing variations.  The
#: cost is nothing where it matters, because the false positives this test is
#: for are rotation harmonics at P = n x Prot with n <= 4, short periods with
#: many transits in every sector.
SECTOR_MIN_TRANSITS = 3
SECTOR_P_FLAG = 1e-3

#: Only these rotation harmonics are treated as coherent in the fold.  At large
#: multiples any period lies within 3% of *some* integer times Prot -- a
#: (wrong) extracted 0.107 d made every planet of one system "77x" to "456x"
#: the rotation -- so the ratio says nothing there.  pc_rotation tests to 4x.
MAX_HARMONIC = 4


def add_arguments(ap):
    g = ap.add_argument_group("phasecurve")
    g.add_argument("--bin", type=float, default=None,
                   help="bin the light curve to this many seconds (default: "
                        "a fifth of the duration, 120-600 s)")
    return ap


# --------------------------------------------------------------------------


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
    dur = float(ctx.candidate.duration_h) / 24.0
    bin_s = getattr(a, "bin", None)
    if bin_s is None:
        bin_s = float(np.clip(dur * 86400.0 / 5.0, 120.0, 600.0))

    try:
        lc = ctx.lc
        p = pcprep.prepare(ctx, bin_seconds=float(bin_s))
    except Exception as exc:  # noqa: BLE001 - missing data is normal
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)
    r.set_noise(**p.noise_kwargs())

    k_max = int(np.floor(period / (FOURIER_MIN_T14 * dur)))
    order = int(min(max(k_max, 0), FOURIER_MAX_ORDER))
    rot = lc.rotation_period() or {}
    prot = rot.get("prot_days")
    n_rot = None
    if prot and prot > 0:
        ratio = period / prot
        n = round(ratio)
        if 1 <= n <= MAX_HARMONIC and abs(ratio - n) / ratio < 0.03:
            n_rot = n
    r.metrics.update({"fourier_order": order, "prot_days": _f(prot),
                      "rotation_cycles_per_period": n_rot,
                      "bin_seconds": float(bin_s)})

    gap_frac = 2 * GAP_HALF_T14 * dur / period
    if order < 2 or gap_frac > 0.3:
        return r.inconclusive(
            f"the transit is too long for its period to separate from a smooth "
            f"phase curve: the excluded window is {gap_frac:.0%} of the orbit "
            f"and the model could go no higher than harmonic {order}")
    if n_rot is not None and n_rot > order:
        return r.inconclusive(
            f"the period is {n_rot}x the {prot:.4f} d rotation, so modelling the "
            f"rotation needs harmonic {n_rot}, but a model that fine would also "
            f"fit a {dur * 24:.2f} h transit (limit {order})")

    sec = _sector_of(p.t, lc)
    views = {"detrended": p.f_detrended, "raw": p.f}
    meas = {}
    for name, flux in views.items():
        meas[name] = _measure(p.t, flux, period, t0, dur, order)
    r.metrics["views"] = {k: _public(v) for k, v in meas.items()}

    sib = _siblings(p.t, p.f, period, t0, dur)
    sib_fail = False
    if sib is not None:
        excess = sib["dip0"] - sib["sibling_dip"]
        e_ex = float(np.hypot(sib["sigma0"], sib["sigma"]))
        sib.update({"excess": excess, "excess_sigma": excess / e_ex if e_ex > 0 else None,
                    "frac": sib["sibling_dip"] / sib["dip0"] if sib["dip0"] > 0 else None})
        sib_fail = bool(sib["significance"] >= SIBLING_SIGMA
                        and sib["frac"] is not None and sib["frac"] >= SIBLING_FRAC
                        and sib["excess_sigma"] is not None
                        and sib["excess_sigma"] < SIBLING_EXCESS_SIGMA)
        r.metrics["siblings"] = {k: (_f(v * 1e6) if k in ("sibling_dip", "sigma", "dip0", "excess", "sigma0")
                                     else (_f(v) if isinstance(v, float) else v))
                                 for k, v in sib.items()}
    r.metrics["sibling_fail"] = sib_fail

    # ---- self-check: does a sharp transit survive the machinery? -----------
    rec = {}
    for name, flux in views.items():
        m = meas[name]
        inj_depth = max(m["raw_dip"], 3 * m["sigma"]) if np.isfinite(m["sigma"]) else None
        if not inj_depth or not np.isfinite(inj_depth) or inj_depth <= 0:
            continue
        ph = ephem.phase_frac(p.t, period, t0)
        dph = ((ph - INJECT_PHASE + 0.5) % 1.0) - 0.5
        box = np.abs(dph * period) < dur / 2
        mi = _measure(p.t, flux - inj_depth * box, period, t0, dur, order,
                      centre=INJECT_PHASE, extra_gap=0.0)
        rec[name] = {"injected_ppm": inj_depth * 1e6,
                     "recovered_ppm": mi["resid_dip"] * 1e6,
                     "fraction": (mi["resid_dip"] / inj_depth) if inj_depth else None}
    r.metrics["injection"] = rec
    ok_views = [k for k, v in rec.items()
                if v.get("fraction") is not None
                and RECOVERY_RANGE[0] <= v["fraction"] <= RECOVERY_RANGE[1]]
    r.set_self_check(
        bool(ok_views),
        "; ".join(f"{k}: injected {v['injected_ppm']:.0f} ppm, recovered "
                  f"{v['recovered_ppm']:.0f} ({(v['fraction'] or 0):.0%})"
                  for k, v in rec.items()) or "no injection was possible")

    # ---- per sector --------------------------------------------------------
    per_sector = []
    for s in sorted({int(x) for x in sec if x >= 0}):
        msk = sec == s
        if msk.sum() < 50:
            continue
        m = _measure(p.t[msk], p.f_detrended[msk], period, t0, dur, order)
        mr = _measure(p.t[msk], p.f[msk], period, t0, dur, order)
        if m["n_core"] < 3:
            continue
        per_sector.append({
            "sector": s, "t_mid": float(np.median(p.t[msk])),
            "resid_dip_ppm": m["resid_dip"] * 1e6, "sigma_ppm": m["sigma"] * 1e6,
            "raw_dip_ppm": m["raw_dip"] * 1e6,
            "smooth_amp_raw_ppm": mr["smooth_amp"] * 1e6,
            "smooth_amp_detrended_ppm": m["smooth_amp"] * 1e6,
            "n_transits": m["n_epochs"]})
    dil = {si.sector: si.dilution for si in lc.sectors
           if si.dilution is not None and np.isfinite(si.dilution) and si.dilution > 0}
    for x in per_sector:
        c = dil.get(x["sector"])
        x["dilution"] = _f(c)
    r.metrics["per_sector"] = per_sector
    sec_chi2 = sec_p = None
    dropped = None
    good = [x for x in per_sector if x["sigma_ppm"] and np.isfinite(x["sigma_ppm"])
            and x["n_transits"] >= SECTOR_MIN_TRANSITS]
    r.metrics["sectors_in_stability_test"] = [x["sector"] for x in good]
    if len(good) >= 2:
        d = np.array([x["resid_dip_ppm"] for x in good])
        e = np.array([x["sigma_ppm"] for x in good])
        med = float(np.median(d))
        e_eff = np.hypot(e, SECTOR_FLOOR_FRAC * abs(med))
        keep = np.ones(d.size, dtype=bool)
        if d.size >= SECTOR_MIN_FOR_DROP:
            worst = int(np.argmax(np.abs(d - med) / e_eff))
            keep[worst] = False
            dropped = good[worst]["sector"]
        _, _, chi2, n = pcprep.weighted_mean(list(d[keep]), list(e_eff[keep]))
        sec_chi2, sec_p = float(chi2), pcprep.chi2_pvalue(chi2, n - 1)
        amps = np.array([x["smooth_amp_raw_ppm"] for x in good])
        r.metrics["smooth_amp_raw_range_ppm"] = [float(amps.min()), float(amps.max())]
    r.metrics["sector_chi2"] = _f(sec_chi2)
    r.metrics["sector_p"] = _f(sec_p)
    r.metrics["sector_dropped"] = dropped
    r.metrics["sector_floor_frac"] = SECTOR_FLOOR_FRAC

    # ---- verdict -----------------------------------------------------------
    D, R = meas["detrended"], meas["raw"]

    def sig(m, k):
        return m[k] / m["sigma"] if m["sigma"] and np.isfinite(m["sigma"]) else 0.0

    sharp = {k: (sig(m, "raw_dip") >= RAW_SIGMA_MIN
                 and sig(m, "resid_dip") >= RESID_SIGMA_MIN)
             for k, m in meas.items() if k in ok_views}
    modulation = {k: (sig(m, "raw_dip") >= RAW_SIGMA_MIN
                      and m["smooth_sigma"] >= SMOOTH_SIGMA_MIN
                      and m["explained"] is not None
                      and m["explained"] >= EXPLAINED_FRAC)
                  for k, m in meas.items() if k in ok_views}
    r.metrics["sharp_in"] = sorted(k for k, v in sharp.items() if v)
    r.metrics["modulation_in"] = sorted(k for k, v in modulation.items() if v)

    def describe(k):
        m = meas[k]
        return (f"{k}: dip {m['raw_dip'] * 1e6:.0f} ppm against the median "
                f"({sig(m, 'raw_dip'):.1f} sigma), {m['resid_dip'] * 1e6:.0f} ppm "
                f"against the smooth phase curve ({sig(m, 'resid_dip'):.1f} sigma); "
                f"smooth modulation {m['smooth_amp'] * 1e6:.0f} ppm peak-to-peak "
                f"at {m['smooth_sigma']:.1f} sigma")

    harm = (f" The period is {n_rot}x the {prot:.4f} d rotation."
            if n_rot else "")
    if not ok_views:
        r.inconclusive("an injected transit did not survive the Fourier model in "
                       "either view, so the test cannot tell a transit from the "
                       "modulation on this star. " + r.self_check["detail"])
    elif sib_fail:
        r.fail(f"the dip repeats where the star repeats: folding the raw flux at "
               f"P/{sib['n']} = {sib['sibling_period_days']:.5f} d, away from the "
               f"candidate's own window, gives a {sib['sibling_dip'] * 1e6:.0f} ppm "
               f"dip at the sibling phases ({sib['significance']:.1f} sigma) against "
               f"{sib['dip0'] * 1e6:.0f} ppm at the candidate's phase -- "
               f"{sib['frac']:.0%} of it, with an excess of only "
               f"{sib['excess_sigma']:.1f} sigma. A transit happens once per orbit; "
               f"this happens {sib['n']} times. Either it is the star's own "
               f"periodicity{harm.lower() if harm else ''}, or the period is "
               f"P/{sib['n']} and this is an alias -- check for that candidate.")
    elif any(sharp.values()):
        k = next(k for k, v in sharp.items() if v)
        if sec_p is not None and sec_p < SECTOR_P_FLAG:
            r.flag(f"a sharp dip survives the smooth phase curve, but it is not "
                   f"stable between sectors even allowing "
                   f"{SECTOR_FLOOR_FRAC:.0%} for calibration"
                   + (f" and setting aside sector {dropped}" if dropped else "")
                   + f" (chi2 = {sec_chi2:.1f}, p = {sec_p:.1e}): a planet's "
                   f"depth does not change with time, a spot's does. "
                   f"{describe(k)}.{harm}")
        elif any(modulation.values()):
            r.flag(f"a sharp dip survives, but most of what the search measured "
                   f"is smooth modulation at the same period: {describe(k)}.{harm}")
        else:
            r.passed(f"the dip is a sharp feature, not the phase curve. "
                     f"{describe(k)}.{harm}")
    elif any(modulation.values()):
        k = next(k for k, v in modulation.items() if v)
        m = meas[k]
        r.fail(f"the dip is the star's own modulation at this period, not a "
               f"transit: {m['explained']:.0%} of it is accounted for by a smooth "
               f"phase curve that cannot represent anything as short as the "
               f"transit, and what remains is {sig(m, 'resid_dip'):.1f} sigma. "
               f"{describe(k)}.{harm} An injected transit of the same depth "
               f"survives the same model ({r.self_check['detail']}), so a real "
               f"one would have.")
    elif max(sig(D, "raw_dip"), sig(R, "raw_dip")) < RAW_SIGMA_MIN:
        r.inconclusive(f"there is no significant dip at this ephemeris in the "
                       f"light curve PetCare reads, so nothing to attribute: "
                       f"{describe('detrended')}; {describe('raw')}")
    else:
        r.inconclusive(f"the dip is significant but neither clearly sharp nor "
                       f"clearly modulation: {describe('detrended')}.{harm}")

    if ctx.make_plots:
        _plots(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur)
    return r


# --------------------------------------------------------------------------


def _measure(t, f, period, t0, dur, order, centre=0.0, extra_gap=None):
    """Dip against the median and against an out-of-transit Fourier model.

    ``centre`` is the phase being tested; the candidate's own window at phase 0
    is always excluded from the fit as well, so an injection does not borrow
    the real dip's exclusion or fall into it.
    """
    ph = ephem.phase_frac(t, period, t0)
    dt_c = (((ph - centre + 0.5) % 1.0) - 0.5) * period
    dt_0 = ((ph + 0.5) % 1.0 - 0.5) * period
    gap = (np.abs(dt_c) < GAP_HALF_T14 * dur) | (np.abs(dt_0) < GAP_HALF_T14 * dur)
    core = np.abs(dt_c) < CORE_HALF_T14 * dur
    ok = np.isfinite(f)
    oot = ok & ~gap

    out = {"raw_dip": np.nan, "resid_dip": np.nan, "sigma": np.nan,
           "smooth_amp": np.nan, "smooth_sigma": 0.0, "explained": None,
           "n_core": int((core & ok).sum()), "n_epochs": 0, "model": None,
           "phase_grid": None}
    if oot.sum() < 4 * order + 10 or (core & ok).sum() < 3:
        return out

    y = f - np.nanmedian(f[oot])
    X = _design(ph, order)
    # bin out-of-transit points in phase for the fit, so a sector with ten
    # times the data does not dominate the shape
    nb = int(np.clip(np.round(period / (dur / BINS_PER_T14)), 40, 2000))
    idx = np.minimum((ph * nb).astype(int), nb - 1)
    xb, yb, wb = [], [], []
    for b in np.unique(idx[oot]):
        m = oot & (idx == b)
        if m.sum() < 2:
            continue
        v = y[m]
        mad = 1.4826 * np.median(np.abs(v - np.median(v)))
        se = mad / np.sqrt(m.sum()) if mad > 0 else np.std(v) / np.sqrt(m.sum())
        if not np.isfinite(se) or se <= 0:
            continue
        xb.append((b + 0.5) / nb); yb.append(np.median(v)); wb.append(1 / se ** 2)
    if len(xb) < 2 * order + 4:
        return out
    xb, yb, wb = map(np.asarray, (xb, yb, wb))
    Xb = _design(xb, order)
    W = np.sqrt(wb)
    coef, *_ = np.linalg.lstsq(Xb * W[:, None], yb * W, rcond=None)
    model = X @ coef

    chi_flat = np.sum(wb * (yb - np.sum(wb * yb) / np.sum(wb)) ** 2)
    chi_four = np.sum(wb * (yb - Xb @ coef) ** 2)
    dchi = max(chi_flat - chi_four, 0.0)
    dof = 2 * order
    p_smooth = stats.chi2.sf(dchi, dof)
    smooth_sigma = float(stats.norm.isf(max(p_smooth, 1e-300))) if p_smooth < 0.5 else 0.0

    grid = np.linspace(0, 1, 1000, endpoint=False)
    mg = _design(grid, order) @ coef
    raw_dip = -np.mean(y[core & ok])
    resid_dip = np.mean(model[core & ok] - y[core & ok])

    # scatter of the same statistic at phases away from both windows
    nulls = []
    for c in np.linspace(0, 1, N_NULL, endpoint=False):
        dtn = (((ph - c + 0.5) % 1.0) - 0.5) * period
        dist0 = abs(((c + 0.5) % 1.0 - 0.5) * period)
        distc = abs(((c - centre + 0.5) % 1.0 - 0.5) * period)
        if min(dist0, distc) < 2 * dur:
            continue
        cm = ok & (np.abs(dtn) < CORE_HALF_T14 * dur)
        if cm.sum() < 3:
            continue
        nulls.append(np.mean(model[cm] - y[cm]))
    sigma = (1.4826 * np.median(np.abs(np.asarray(nulls) - np.median(nulls)))
             if len(nulls) >= 10 else np.nan)

    epochs = ephem.epoch_number(t[core & ok], period, t0)
    out.update({
        "raw_dip": float(raw_dip), "resid_dip": float(resid_dip),
        "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
        "smooth_amp": float(mg.max() - mg.min()),
        "smooth_sigma": smooth_sigma,
        "explained": (float(1 - resid_dip / raw_dip) if raw_dip > 0 else None),
        "n_epochs": int(np.unique(epochs).size),
        "model": model, "phase_grid": (grid, mg), "y": y, "phase": ph,
        "binned": (xb, yb, 1 / np.sqrt(wb)), "n_null": len(nulls),
    })
    return out


def _siblings(t, f, period, t0, dur):
    """The dip at phase 0 against the same dip where the star repeats.

    Every occurrence is measured separately: each transit at phase 0, and each
    sibling at t0 + e P + k P/n for k = 1..n-1.  Their means are compared using
    their *own* occurrence-to-occurrence scatter.  An earlier version took the
    error from the scatter across a P/n fold, which sweeps straight through the
    rotation pattern and so counted the signal as noise: one candidate's
    siblings repeated at 96% of the dip and still read as 2.6 sigma.

    The n whose siblings dip most significantly is returned.  For a planet at a
    period unrelated to the star's, siblings land on random rotation phases and
    average to nothing, whatever n is tried.
    """
    ok = np.isfinite(f)
    if ok.sum() < 100:
        return None
    ph = ephem.phase_frac(t, period, t0)
    dt0 = (((ph + 0.5) % 1.0) - 0.5) * period
    oot = ok & (np.abs(dt0) > GAP_HALF_T14 * dur)
    y = f - np.nanmedian(f[oot])
    half = CORE_HALF_T14 * dur

    def occurrences(spacing, offsets):
        """Mean of y in the core window around each occurrence."""
        e = np.round((t - t0) / spacing)
        dt = t - (t0 + e * spacing)
        inn = ok & (np.abs(dt) < half)
        vals = []
        for k in np.unique(e[inn]):
            m = inn & (e == k)
            # which sibling this is: k mod n, from the offsets allowed
            if m.sum() >= 3 and offsets(int(k)):
                vals.append(float(-np.mean(y[m])))
        return np.asarray(vals)

    d0 = occurrences(period, lambda k: True)
    if d0.size < 2:
        return None
    best = None
    for n in range(2, SIBLING_N_MAX + 1):
        pn = period / n
        if pn < 4 * dur:
            break
        ds = occurrences(pn, lambda k, n=n: k % n != 0)
        if ds.size < 4:
            continue
        se_s = ds.std(ddof=1) / np.sqrt(ds.size)
        if not se_s > 0:
            continue
        cand = {"n": n, "sibling_period_days": float(pn),
                "sibling_dip": float(ds.mean()), "sigma": float(se_s),
                "significance": float(ds.mean() / se_s), "n_siblings": int(ds.size),
                "dip0": float(d0.mean()),
                "sigma0": float(d0.std(ddof=1) / np.sqrt(d0.size)), "n_transits": int(d0.size)}
        if best is None or cand["significance"] > best["significance"]:
            best = cand
    return best


def _design(ph, order):
    cols = [np.ones_like(ph)]
    for k in range(1, order + 1):
        cols += [np.cos(2 * np.pi * k * ph), np.sin(2 * np.pi * k * ph)]
    return np.vstack(cols).T


def _public(m):
    keep = ("raw_dip", "resid_dip", "sigma", "smooth_amp", "smooth_sigma",
            "explained", "n_core", "n_epochs", "n_null")
    out = {}
    for k in keep:
        v = m.get(k)
        if k in ("raw_dip", "resid_dip", "sigma", "smooth_amp"):
            out[k + "_ppm"] = _f(v * 1e6) if v is not None else None
        else:
            out[k] = _f(v) if isinstance(v, (float, np.floating)) else v
    return out


def _sector_of(t, lc):
    """Per-point sector for a binned series, from the stitched curve's times."""
    order = np.argsort(lc.t)
    ts, ss = lc.t[order], lc.sector_id[order]
    i = np.clip(np.searchsorted(ts, t), 0, ts.size - 1)
    return ss[i]


# --------------------------------------------------------------------------
# plots: the test's own figure, plus the light-curve views a vetter needs


def _plots(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur):
    for fn in (_plot_phasecurve, _plot_sector_folds, _plot_lightcurve_sectors,
               _plot_transits):
        try:
            path = fn(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur)
            if path:
                r.add_plot(path)
        except Exception as exc:  # noqa: BLE001 - a failed figure is not a verdict
            ctx.notes.append(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")


def _plot_phasecurve(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur):
    fig, ax = figure(nrows=3, ncols=2, figsize=(13, 11))
    for col, name in enumerate(("raw", "detrended")):
        m = meas[name]
        a0, a1 = ax[0, col], ax[1, col]
        if m.get("binned") is None:
            a0.text(0.5, 0.5, "not measurable", ha="center", transform=a0.transAxes)
            continue
        xb, yb, eb = m["binned"]
        xs = ((xb + 0.5) % 1.0) - 0.5
        o = np.argsort(xs)
        a0.errorbar(xs[o], yb[o] * 1e6, eb[o] * 1e6, fmt=".", color=C_BINNED,
                    ms=3, lw=0.5, alpha=0.8, label="out-of-transit, binned")
        g, mg = m["phase_grid"]
        gs = ((g + 0.5) % 1.0) - 0.5
        og = np.argsort(gs)
        a0.plot(gs[og], mg[og] * 1e6, color=C_MODEL, lw=1.6,
                label=f"Fourier model (to harmonic {r.metrics['fourier_order']})")
        w = GAP_HALF_T14 * dur / period
        a0.axvspan(-w, w, color=C_BAD, alpha=0.12, label="excluded from the fit")
        a0.set_xlabel("phase")
        a0.set_ylabel("flux - median [ppm]")
        a0.set_title(f"{name}: full-phase fold", fontsize=10)
        a0.legend(fontsize=7, loc="lower left")

        # zoom on the transit, unbinned + binned, with the model across the gap
        ph, y, model = m["phase"], m["y"], m["model"]
        dth = (((ph + 0.5) % 1.0) - 0.5) * period * 24
        z = np.abs(dth) < 4 * dur * 24
        a1.plot(dth[z], y[z] * 1e6, ".", color=C_DATA, ms=1.2, alpha=0.25)
        bw = dur * 24 / BINS_PER_T14
        edges = np.arange(-4 * dur * 24, 4 * dur * 24 + bw, bw)
        cen = 0.5 * (edges[1:] + edges[:-1])
        yb2 = [np.median(y[z][(dth[z] >= lo) & (dth[z] < hi)]) * 1e6
               if ((dth[z] >= lo) & (dth[z] < hi)).sum() >= 3 else np.nan
               for lo, hi in pairwise(edges)]
        mb2 = [np.median(model[z][(dth[z] >= lo) & (dth[z] < hi)]) * 1e6
               if ((dth[z] >= lo) & (dth[z] < hi)).sum() >= 3 else np.nan
               for lo, hi in pairwise(edges)]
        a1.plot(cen, yb2, "o", color=C_BINNED, ms=4, label="data, binned")
        a1.plot(cen, mb2, "-", color=C_MODEL, lw=2,
                label="smooth phase curve, interpolated across the window")
        a1.axvspan(-dur * 12, dur * 12, color=C_BAD, alpha=0.08)
        s_r = m["raw_dip"] / m["sigma"] if m["sigma"] else np.nan
        s_s = m["resid_dip"] / m["sigma"] if m["sigma"] else np.nan
        a1.set_title(f"{name}: dip {m['raw_dip'] * 1e6:.0f} ppm ({s_r:.1f}σ) vs "
                     f"median, {m['resid_dip'] * 1e6:.0f} ppm ({s_s:.1f}σ) vs model",
                     fontsize=9)
        a1.set_xlabel("hours from mid-transit")
        a1.set_ylabel("ppm")
        a1.legend(fontsize=7, loc="lower left")

    ps = [x for x in per_sector if x["sigma_ppm"] and np.isfinite(x["sigma_ppm"])]
    a2, a3 = ax[2, 0], ax[2, 1]
    if ps:
        tt = [x["t_mid"] for x in ps]
        a2.errorbar(tt, [x["resid_dip_ppm"] for x in ps],
                    [x["sigma_ppm"] for x in ps], fmt="o", color=C_OK,
                    label="dip against the smooth model")
        a2.plot(tt, [x["raw_dip_ppm"] for x in ps], "s", color=C_ALT, mfc="none",
                label="dip against the median")
        for x in ps:
            a2.annotate(f"S{x['sector']}", (x["t_mid"], x["resid_dip_ppm"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        a2.axhline(0, color="0.5", lw=0.6)
        a2.set_xlabel("BTJD")
        a2.set_ylabel("depth [ppm]")
        chi = r.metrics.get("sector_chi2")
        a2.set_title("does the dip hold still across the years?"
                     + (f"  chi2 = {chi:.1f}, p = {r.metrics['sector_p']:.1e}"
                        if chi is not None else ""), fontsize=9)
        a2.legend(fontsize=7)
        a3.plot(tt, [x["smooth_amp_raw_ppm"] for x in ps], "o-", color=C_BAD,
                label="raw flux")
        a3.plot(tt, [x["smooth_amp_detrended_ppm"] for x in ps], "s--",
                color=C_BINNED, label="detrended")
        a3.set_xlabel("BTJD")
        a3.set_ylabel("peak-to-peak [ppm]")
        a3.set_title("smooth modulation at this period, by sector -- spots "
                     "evolve, a planet does not", fontsize=9)
        a3.legend(fontsize=7)
    else:
        for a_ in (a2, a3):
            a_.text(0.5, 0.5, "too few sectors for a time series",
                    ha="center", transform=a_.transAxes)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path("phasecurve"))


def _plot_sector_folds(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur):
    sectors = sorted({int(x) for x in sec if x >= 0})
    if not sectors:
        return None
    n = len(sectors)
    fig, ax = figure(nrows=1, ncols=2, figsize=(13, 1.6 * n + 2.2))
    ph = ephem.phase_frac(p.t, period, t0)
    ph = ((ph + 0.5) % 1.0) - 0.5
    nb = int(np.clip(np.round(period / (dur / 2)), 40, 400))
    edges = np.linspace(-0.5, 0.5, nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    for col, (name, flux) in enumerate((("raw", p.f), ("detrended", p.f_detrended))):
        a = ax[col]
        spans = []
        for s in sectors:
            m = sec == s
            y = (flux[m] - np.nanmedian(flux[m])) * 1e6
            yb = [np.nanmedian(y[(ph[m] >= lo) & (ph[m] < hi)])
                  if ((ph[m] >= lo) & (ph[m] < hi)).sum() >= 2 else np.nan
                  for lo, hi in pairwise(edges)]
            spans.append(np.nanpercentile(np.abs(yb), 95) if np.isfinite(yb).any() else 0)
        step = 2.5 * (np.nanmax(spans) if spans and np.nanmax(spans) > 0 else 1000)
        for i, s in enumerate(sectors):
            m = sec == s
            y = (flux[m] - np.nanmedian(flux[m])) * 1e6
            yb = np.array([np.nanmedian(y[(ph[m] >= lo) & (ph[m] < hi)])
                           if ((ph[m] >= lo) & (ph[m] < hi)).sum() >= 2 else np.nan
                           for lo, hi in pairwise(edges)])
            a.plot(cen, yb - i * step, "-", color=C_BINNED, lw=1)
            t_mid = float(np.median(p.t[m]))
            a.text(0.5, -i * step, f" S{s}  ({t_mid:.0f})", fontsize=7, va="center")
        w = dur / 2 / period
        a.axvspan(-w, w, color=C_BAD, alpha=0.15)
        a.set_xlim(-0.5, 0.62)
        a.set_yticks([])
        a.set_xlabel("phase")
        a.set_title(f"{name} flux folded at P = {period:.5f} d, one row per "
                    f"sector, oldest at the top", fontsize=9)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path("phasecurve_sectors"))


def _plot_lightcurve_sectors(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur):
    sectors = sorted({int(x) for x in lc.sector_id if x >= 0})
    if not sectors:
        return None
    n = len(sectors)
    fig, ax = figure(nrows=n, ncols=1, figsize=(13, 2.1 * n + 0.8))
    ax = np.atleast_1d(ax)
    for a, s in zip(ax, sectors):
        m = lc.sector_id == s
        t, f = lc.t[m], lc.f[m]
        a.plot(t, (f - 1) * 1e6, ".", color=C_DATA, ms=0.8, alpha=0.35)
        mp = sec == s
        if mp.any():
            a.plot(p.t[mp], (p.f[mp] - 1) * 1e6, ".", color=C_BINNED, ms=1.5,
                   alpha=0.6)
            if p.trend is not None:
                a.plot(p.t[mp], p.trend[mp] * 1e6, "-", color=C_OK, lw=0.8,
                       alpha=0.9)
        tt = ephem.transit_times(period, t0, float(t.min()), float(t.max()))
        lo, hi = np.nanpercentile((f - 1) * 1e6, [0.5, 99.5])
        for x in tt:
            a.axvspan(x - dur / 2, x + dur / 2, color=C_BAD, alpha=0.25, lw=0)
        a.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        a.set_ylabel(f"S{s}\nppm", fontsize=8)
        a.text(0.995, 0.92, f"{len(tt)} predicted transits", ha="right",
               va="top", fontsize=7, transform=a.transAxes)
    ax[-1].set_xlabel("BTJD")
    ax[0].set_title("raw light curve by sector; predicted transits shaded, "
                    "PetCare's variability model in green", fontsize=10)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path("lightcurve_sectors"))


def _plot_transits(ctx, r, lc, p, sec, meas, per_sector, period, t0, dur):
    ep = ephem.epoch_number(p.t, period, t0)
    dt = (p.t - ephem.transit_time(ep, period, t0)) * 24.0
    win = 3.0 * dur * 24.0
    near = np.abs(dt) < win
    if not near.any():
        return None
    # best-covered epochs first, then in time order
    cover = {}
    for e in np.unique(ep[near]):
        m = near & (ep == e)
        cover[int(e)] = int((np.abs(dt[m]) < dur * 12).sum())
    chosen = sorted(sorted(cover, key=lambda e: -cover[e])[:N_TRANSIT_PANELS])
    n = len(chosen)
    if n == 0:
        return None
    fig, ax = figure(nrows=n, ncols=2, figsize=(12, 1.5 * n + 1.4),
                     sharex=True, squeeze=False)
    for i, e in enumerate(chosen):
        m = near & (ep == e)
        a0, a1 = ax[i, 0], ax[i, 1]
        a0.plot(dt[m], (p.f[m] - 1) * 1e6, ".", color=C_DATA, ms=2.5)
        if p.trend is not None:
            a0.plot(dt[m], p.trend[m] * 1e6, "-", color=C_OK, lw=1.2)
        a1.plot(dt[m], (p.f_detrended[m] - 1) * 1e6, ".", color=C_BINNED, ms=2.5)
        a1.axhline(0, color="0.6", lw=0.6)
        # Robust limits per panel: one flare must not flatten the transit.
        for a_, y in ((a0, (p.f[m] - 1) * 1e6), (a1, (p.f_detrended[m] - 1) * 1e6)):
            y = y[np.isfinite(y)]
            if y.size > 5:
                lo, hi = np.percentile(y, [1, 99])
                pad = 0.15 * (hi - lo) if hi > lo else 1.0
                a_.set_ylim(lo - pad, hi + pad)
        for a_ in (a0, a1):
            a_.axvspan(-dur * 12, dur * 12, color=C_BAD, alpha=0.12)
            a_.tick_params(labelsize=6)
        s = int(sec[m][0]) if m.any() else -1
        a0.set_ylabel(f"E{e}\nS{s}", fontsize=7)
    ax[0, 0].set_title("before detrending: flux and variability model", fontsize=9)
    ax[0, 1].set_title("after detrending", fontsize=9)
    ax[-1, 0].set_xlabel("hours from predicted mid-transit")
    ax[-1, 1].set_xlabel("hours from predicted mid-transit")
    total = len(cover)
    fig.suptitle(f"{n} best-covered of {total} transits", y=0.995, fontsize=9)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path("transits_individual"))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Is the dip a sharp transit, or the star's own smooth modulation "
              "at the same period -- and does it hold still across sectors?"))
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

"""pc_uniqueness -- is the dip unique in the folded light curve, and is it a dip?

Two vetters from the Kepler Robovetter lineage, computed here with local
baselines on PetCare's own light curve (exoVetter, Dallas et al. 2026,
arXiv:2610.32949, is used only for its version string; see the note in run()):

**ModShift** (Thompson et al. 2018, A.3.4).  The folded light curve is
convolved with the transit model at every phase.  The primary event is the
candidate; the secondary is the strongest other dip, the tertiary the strongest
dip after that, and "positive" the strongest brightening.  Each is expressed in
units of the folded light curve's own scatter, so a young spotted star's
quasi-periodic residuals count against the candidate exactly as much as they
deserve.  From these:

* ``MS1 = sig_pri / Fred - FA``  -- is the primary significant against red noise?
* ``MS2 = sig_pri - sig_ter - FA2`` -- is it stronger than the next dip that is
  neither it nor a secondary eclipse?  The uniqueness test.
* ``MS3 = sig_pri - sig_pos - FA2`` -- is it stronger than the biggest brightening?

``Fred`` is the ratio of the convolution's scatter to the binned flux's: well
above 1 means correlated noise on the transit's own timescale.

**SHP** (Kunimoto et al. 2025, LEO-vetter 3.10): the strongest brightening
box-depth over (that minus the deepest dip).  Near 0 a dip dominates the fold;
toward 0.5 brightenings (flares, the other side of the rotation) are as strong.

Thresholds were set on the PetCare benchmark of known young planets (see
CALIBRATION below): a test that removes real young planets is not used to
refute.  Until then everything here only flags.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from .. import cli, power
from .. import prep as pcprep
from ..plotting import C_BAD, C_BINNED, C_DATA, C_MODEL, annotate_status, figure, save

TEST = "uniqueness"
VERSION = "0.2.0"

#: CALIBRATION, local-baseline ModShift, on 23 known young planets (one with
#: a single transit is INCONCLUSIVE): weakest MS2 3.6, MS3 5.7, highest SHP
#: 0.32.  On ~2000 candidates from our search, MS2 or MS3 < -3 hits 26% of
#: the strongest candidates, 29% of close calls and 53% of refuted candidates.
#: The fail cut -3 (6.6 sigma below the weakest real planet) applies to
#: candidates with >= 4 transits.
MS2_FAIL = -3.0             # for >= FAIL_MIN_TRANSITS only
MS3_FAIL = -3.0
#: With 2-3 transits every phase is built from the same few windows of spot
#: noise and the benchmark barely covers it (one planet, 3 transits):
#: those keep the result as a flag.
FAIL_MIN_TRANSITS = 4
SHP_FAIL = None              # SHP never refutes
MS2_FLAG = 0.0
MS3_FLAG = 0.0
SHP_FLAG = 0.50
SHP_MIN_MS1 = -1e9           # (no longer gated on MS1: the local-baseline MS1 is Fred-limited)
PROPOSED_FAIL = -3.0
#: Running-median width for the uniform detrending, in transit durations.
WINDOW_T14 = 3.0


def add_arguments(ap):
    return ap


def run(ctx: cli.Context):
    r = ctx.result(TEST, VERSION)
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris"):
        return r
    if not cli.require(r, ctx.candidate.duration_h, "no transit duration"):
        return r
    period, t0 = float(ctx.candidate.period), float(ctx.candidate.t0_btjd)
    dur_h = float(ctx.candidate.duration_h)
    try:
        p = pcprep.prepare(ctx)
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, ctx.lc)
    r.set_noise(**p.noise_kwargs())

    # NOT p.f_detrended: prep masks the candidate's own windows and lets the GP
    # interpolate across them, so the candidate's phase is treated unlike every
    # other phase -- on a fast rotator the GP sagged 40 ppt in the gaps and a
    # 16 ppt transit folded into a +25 ppt bump.  Nor a global
    # running median: wide enough to keep the transit, it cannot follow a 1.4 d
    # rotation (HIP 67522: Fred 21).  So ModShift is done with LOCAL baselines:
    # at every trial phase each window gets its own quadratic baseline, exactly
    # as at the candidate's phase.  f here is the raw binned flux.
    t = np.asarray(p.t, float)
    e = np.asarray(p.err, float)
    f = np.asarray(p.f, float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[ok], f[ok], e[ok]
    if t.size < 50:
        return r.inconclusive(f"only {t.size} usable binned points")

    # ModShift is computed here, not with exoVetter's: v0.0.16 convolves the
    # SUM of the flux in each phase bin but divides by a per-point scatter, so
    # its significances scale with points per bin (HIP 67522 b came out at
    # -14237 sigma).  Same definitions as Thompson et al. 2018, A.3.4.
    s, S_series, phases = modshift(t, f, period, t0, dur_h, return_series=True)
    fa, fa2, fred = s["FA1"], s["FA2"], s["Fred"]
    if (s.get("n_transits") or 0) < 2 and s["sigma_pri"] is not None:
        r.metrics.update({"n_transits": s.get("n_transits"), "modshift": s})
        r.inconclusive(f"only {s.get('n_transits')} transit in the data: a fold of one event "
                       f"cannot say whether the dip is unique (a real single-transit planet can do this)")
        return r
    if s["sigma_pri"] is None:
        r.inconclusive("no trial phase near the candidate has transit windows with baseline "
                       "on both sides; the primary cannot be measured")
        return r
    sp = s["sigma_pri"]
    ms1 = sp / fred - fa if (sp is not None and fred) else None
    ms2 = sp - s["sigma_ter"] - fa2 if None not in (sp, s["sigma_ter"]) else None
    ms3 = sp - s["sigma_pos"] - fa2 if None not in (sp, s["sigma_pos"]) else None
    sp_, spos = s["sigma_pri"], s["sigma_pos"]
    # SHP (Kunimoto et al. 2025) from the same local-baseline series: the
    # strongest brightening against brightening + primary dip
    shp = (spos / (spos + sp_) if (sp_ and sp_ > 0 and spos is not None and spos > 0)
           else (0.0 if (sp_ and sp_ > 0) else None))
    n_tr = s.get("n_transits") or 0
    r.metrics.update({"n_transits": n_tr, "modshift": s, "MS1": ms1, "MS2": ms2, "MS3": ms3, "Fred": fred,
                      "SHP": shp, "n_points": int(t.size), "method": "local quadratic baseline per window"})
    r.set_self_check(ms1 is not None, f"{t.size} binned points; Fred {fred}")

    fails, flags = [], []
    for name, val, fl, fa_ in (("MS2", ms2, MS2_FLAG, MS2_FAIL), ("MS3", ms3, MS3_FLAG, MS3_FAIL)):
        if val is None:
            continue
        if fa_ is not None and val < fa_ and n_tr >= FAIL_MIN_TRANSITS:
            fails.append(f"{name} = {val:.1f} < {fa_:g}")
        elif val < fl:
            flags.append(f"{name} = {val:.1f}")
    # SHP compares the deepest dip with the strongest brightening; with one
    # transit a single flare or rotation crest rivals it (0.30 on a real planet)
    if shp is not None and n_tr >= 2 and ms1 is not None and ms1 >= SHP_MIN_MS1:
        if SHP_FAIL is not None and shp > SHP_FAIL:
            fails.append(f"SHP = {shp:.2f} > {SHP_FAIL:g}")
        elif shp > SHP_FLAG:
            flags.append(f"SHP = {shp:.2f}")
    desc = (f"primary {s['sigma_pri']:.1f} sigma, secondary {s['sigma_sec']:.1f}, next dip "
            f"{s['sigma_ter']:.1f}, biggest brightening {s['sigma_pos']:.1f} (false-alarm "
            f"{fa:.1f}/{fa2:.1f}, Fred {fred:.2f}); "
            f"MS1 {ms1:.1f}, MS2 {ms2:.1f}, MS3 {ms3:.1f}; SHP {shp if shp is None else round(shp, 2)}"
            if None not in (s["sigma_pri"], s["sigma_sec"], s["sigma_ter"], s["sigma_pos"], fa, fred, ms1)
            else "metrics incomplete")
    if ms1 is None:
        r.inconclusive("the primary dip could not be measured in the fold (too few points in "
                       "the transit window; check the fitted duration). " + desc)
    elif fails:
        power.gate(r, ctx, run, "the dip is not unique in the folded light curve: "
                   + "; ".join(fails) + ". " + desc)
    elif flags:
        r.flag("other features in the fold rival the dip (" + "; ".join(flags) + "). " + desc)
    else:
        r.passed("the dip is the strongest feature in the fold and a dip, not a brightening. " + desc)
    try:
        if ctx.make_plots:
            _plot(ctx, r, S_series, phases, s, s.get("FA2") or 0)
    except Exception as exc:  # noqa: BLE001
        r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, S, phases, s, fa2):
    x = (phases + 0.25) % 1 - 0.25
    o = np.argsort(x)
    fig, ax = figure(1, 1, figsize=(12, 4.6))
    ax.plot(x[o], S[o], "-", color=C_BINNED, lw=1)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axhline(fa2, color="0.5", lw=0.6, ls=":", label=f"false-alarm level {fa2:.1f}σ")
    for key, lab, col in (("phase_pri", "primary", C_BAD), ("phase_sec", "secondary", C_MODEL),
                          ("phase_ter", "next dip", "#7b5aa6"), ("phase_pos", "largest brightening", "#2e7d5b")):
        v = s.get(key)
        if v is None:
            continue
        sig = s.get("sigma_" + key.split("_")[1])
        ax.axvline((v + 0.25) % 1 - 0.25, color=col, lw=1.2, ls="--", alpha=0.8,
                   label=f"{lab}: {sig:.1f}σ" if sig is not None else lab)
    ax.set_xlim(-0.25, 0.75)
    ax.set_xlabel("phase (candidate at 0)")
    ax.set_ylabel("dip significance [σ]  (positive = dip)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("a box dip tested at every phase, each transit window against its own baseline: "
                 "is the candidate's the only real one?", fontsize=10)
    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("uniqueness")))


def local_box_significance(t, f, period, t0, dur, phases, half_window=1.5, inner=0.5,
                           base_gap=0.75):
    """Significance of a box dip at each trial phase, each transit window
    measured against its own quadratic baseline (|dt| in [base_gap, half_window]
    durations), then combined over windows by inverse variance.  Vectorised:
    per-window least squares from bincount moments."""
    out = np.full(len(phases), np.nan)
    u = (t - t0) / period
    for k, phi in enumerate(phases):
        x = u - phi
        n = np.round(x)
        dt = (x - n) * period / dur                       # in durations
        sel = np.abs(dt) < half_window
        if sel.sum() < 10:
            continue
        g, inv = np.unique(n[sel], return_inverse=True)
        xs, ys = dt[sel], f[sel]
        base = np.abs(xs) >= base_gap
        inn = np.abs(xs) < inner
        G = len(g)
        def bc(w, m):
            return np.bincount(inv[m], weights=w, minlength=G)
        xb, yb = xs[base], ys[base]
        M = np.stack([np.stack([bc(xb ** (i + j), base) for j in range(3)], -1) for i in range(3)], -2)
        V = np.stack([bc(yb * xb ** i, base) for i in range(3)], -1)
        nl = np.bincount(inv[base & (xs < 0)], minlength=G)
        nr = np.bincount(inv[base & (xs > 0)], minlength=G)
        ni = np.bincount(inv[inn], minlength=G)
        good = (nl >= 3) & (nr >= 3) & (ni >= 3)
        if not good.any():
            continue
        M = M[good] + 1e-12 * np.eye(3)
        coef = np.full((G, 3), np.nan)
        coef[good] = np.linalg.solve(M, V[good][..., None])[..., 0]
        model = coef[inv, 0] + coef[inv, 1] * xs + coef[inv, 2] * xs ** 2
        resid = ys / model - 1
        nb_ = np.bincount(inv[base], minlength=G)
        dep = -np.bincount(inv[inn], weights=resid[inn], minlength=G)
        # one robust point scatter for every window at this phase: a window's own
        # rms from a handful of baseline points can be near zero, and that single
        # window then dominates (one-bin "dips" at 62 sigma on a known planet)
        rgood = good[inv] & base & np.isfinite(resid)
        if rgood.sum() < 10:
            continue
        sig = 1.4826 * np.nanmedian(np.abs(resid[rgood] - np.nanmedian(resid[rgood])))
        with np.errstate(invalid="ignore", divide="ignore"):
            d = dep / ni
            err = sig / np.sqrt(ni) * np.sqrt(1 + ni / np.maximum(nb_, 1))
            w = np.where(good & np.isfinite(err) & (err > 0), 1 / err ** 2, 0)
        if w.sum() <= 0:
            continue
        out[k] = np.nansum(np.where(w > 0, d, 0) * w) / np.sqrt(w.sum())
    return out


def uniform_detrend(t, f, width_days):
    """Divide by a running median of the given full width, the same at every time."""
    order = np.argsort(t)
    ts, fs = t[order], f[order]
    lo = np.searchsorted(ts, ts - width_days / 2)
    hi = np.searchsorted(ts, ts + width_days / 2)
    med = np.array([np.nanmedian(fs[a:b]) if b - a >= 3 else np.nan for a, b in zip(lo, hi)])
    out = np.empty_like(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[order] = fs / med
    return out


def modshift(t, f, period, t0, dur_h, per_dur=10, return_series=False):
    """Box matched filter at every phase of the fold, in units of significance.

    Returns the significance of the primary (the dip nearest phase 0), the
    secondary (strongest dip outside +/-2 T14 of it), the tertiary (strongest
    dip outside +/-2 T14 of both) and the largest brightening (outside
    +/-3 T14 of both), with phases as fractions of the orbit, the red-noise
    ratio Fred and the Robovetter false-alarm thresholds."""
    from scipy.special import erfcinv
    q = dur_h / 24 / period                       # duration as a phase fraction
    nb = int(np.clip(round(per_dur / q), 40, 4000))
    centre = (np.arange(nb) + 0.5) / nb
    S = local_box_significance(t, f / np.nanmedian(f), period, t0, dur_h / 24, centre)
    w = max(1, int(round(q * nb)))
    n_win = np.where(np.isfinite(S), 10, 0)      # kept for the n_win < 3 guard below
    def dist(a, b):
        d = np.abs(a - b) % 1
        return np.minimum(d, 1 - d)
    near_pri = (dist(centre, 0.0) < q) & np.isfinite(S)
    empty = {"sigma_pri": None, "sigma_sec": None, "sigma_ter": None, "sigma_pos": None,
             "n_transits": 0, "Fred": None, "FA1": float(np.sqrt(2) * erfcinv(q / 20000)),
             "FA2": float(np.sqrt(2) * erfcinv(q)), "n_bins": nb}
    if not near_pri.any():
        return (empty, S, centre) if return_series else empty
    i_pri = np.where(near_pri)[0][np.nanargmax(S[near_pri])]
    ph_pri = centre[i_pri]
    ph = ((t - t0) / period) % 1
    # no measurable phase away from the primary: nothing to compare it with
    m2 = (dist(centre, ph_pri) > 2 * q) & np.isfinite(S)
    if not m2.any():
        return (empty, S, centre) if return_series else empty
    i_sec = np.where(m2)[0][np.nanargmax(np.where(m2, S, -np.inf)[m2])]
    ph_sec = centre[i_sec]
    m3 = m2 & (dist(centre, ph_sec) > 2 * q)
    i_ter = np.where(m3)[0][np.nanargmax(np.where(m3, S, -np.inf)[m3])] if m3.any() else None
    # +/-3 T14 around both, narrowing when a long transit on a short period
    # leaves no phase outside that (0.71 d with a 1.7 h transit: q = 0.1)
    for gap in (3, 2, 1.5, 1):
        m4 = (dist(centre, ph_pri) > gap * q) & (dist(centre, ph_sec) > gap * q)
        if m4.sum() >= 3:
            break
    i_pos = np.where(m4)[0][np.nanargmin(np.where(m4, S, np.inf)[m4])] if m4.any() else None
    # Fred: scatter of the significance away from primary and secondary
    # (exactly 1 for white noise; correlated noise on the transit timescale raises it)
    away = m3 & np.isfinite(S)
    fred = float(np.nanstd(S[away])) if away.sum() > 10 else None
    fa1 = float(np.sqrt(2) * erfcinv(q / 20000))   # Robovetter: 20000 trial TCEs
    fa2 = float(np.sqrt(2) * erfcinv(q))
    g = lambda i: None if i is None else _f(S[i])
    n_tr = int(len(np.unique(np.round((t[dist(ph, ph_pri) < 0.5 * q] - t0) / period))))
    out = {"n_transits": n_tr, "sigma_pri": g(i_pri), "sigma_sec": g(i_sec), "sigma_ter": g(i_ter),
            "sigma_pos": None if i_pos is None else _f(-S[i_pos]),
            "phase_pri": float(ph_pri), "phase_sec": float(ph_sec),
            "phase_ter": None if i_ter is None else float(centre[i_ter]),
            "phase_pos": None if i_pos is None else float(centre[i_pos]),
            "Fred": fred, "FA1": fa1, "FA2": fa2, "n_bins": nb, "bins_per_duration": w}
    return (out, S, centre) if return_series else out


def _f(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _version():
    try:
        import exovetter
        return exovetter.__version__
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(TEST, "ModShift uniqueness and SHP (exoVetter)"))
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

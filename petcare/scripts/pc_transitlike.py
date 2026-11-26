"""pc_transitlike -- is the dip shaped like a transit, or like something a star does?

Rotation minima and flare decays both trigger a transit search on young stars.
They leave dips, but not flat-bottomed ones with two matched edges.  This test
fits the RAW flux around every predicted transit with a polynomial baseline
per event plus one of

    T    a limb-darkened transit: depth, duration, impact parameter and one
         timing offset shared by all events.  The duration is searched between
         0.3 and 1.5 times the central-crossing value for the star's density.
         Impact parameters run to 1.03, so V-shaped grazing transits count as
         transits.
    Gs   a Gaussian dip (what a rotation minimum looks like), shared amplitude,
         width and offset
    Fs   an inverted flare (one sharp edge, exponential tail, either way round),
         shared amplitude, decay time and offset
    0    the baseline alone

and compares them by BIC.  ``dBIC_X = BIC_X - BIC_T``: positive favors the
transit.  The alternatives have the same freedom as the transit, so the
comparison is of shape and nothing else.

**Why raw flux.**  A detrended curve has already had a model's opinion of the
variability removed, and that model can carve or fill a dip.  Here each event
gets its own polynomial, whose order (1-7) is chosen by BIC on the points
OUTSIDE the transit core, so it can follow rotation but cannot absorb the dip.

**The rule.**  FAIL unless all of: fitted depth > 0; the transit beats no dip
by ``MIN_DBIC_FLAT``; it beats the flare shape by ``MIN_DBIC_FLARE`` AND by
``MIN_FLARE_FRAC`` of its lead over no dip (a fixed margin alone is an S/N cut
in disguise); and a Gaussian fits better by no more than ``MIN_GAUSS_RATIO`` of
the transit's lead over no dip.

**What it costs.**  Calibrated on our young-star search
(margin >= 5).  Of the 1,871 candidates that no other test refutes it keeps
569, all 10 known planets and all 20 signals judged planet-like by eye; 28 of 62
signals rejected by eye in that set still pass.  The 0.2.0 rule (no grazing
transits, flare margin fixed at 20) kept 390 and lost one planet-like signal.

Every model is averaged over each point's exposure.  chi2 uses a white-noise
sigma per event (MAD of point-to-point differences); BIC = chi2 + k ln N.

Ported from the search-side script that produced the calibration
numbers above.  ``--transitlike-free`` adds the per-event-free Gaussian and
flare alternatives (``dBIC_G``, ``dBIC_F``); they are reported, not used, and
they are slow.

    python -m petcare.scripts.pc_transitlike --tic 123 --period 4.2 --t0 1500.1
"""

from __future__ import annotations

import numpy as np

from .. import cli
from ..plotting import C_ALT, C_BAD, C_BINNED, C_DATA, C_MODEL, annotate_status, bin_series, figure, save
from ..result import Result
from ..stellar import RHO_SUN_CGS

TEST = "transitlike"
VERSION = "0.3.0"       # 0.3.0: grazing transits, flare margin scaled with the dip

#: Transit against no dip at all.
MIN_DBIC_FLAT = 10.0
#: Transit against a flare-shaped dip of equal freedom: both must hold.
MIN_DBIC_FLARE = 5.0
MIN_FLARE_FRAC = 0.05
#: dBIC_Gs / dBIC_0: how much better a Gaussian dip may fit, as a fraction of
#: the transit's lead over no dip.
MIN_GAUSS_RATIO = -0.10
#: Impact parameters tried; 1.0 and 1.03 are grazing (V-shaped).
IMPACT_GRID = (0.0, 0.5, 0.75, 0.9, 1.0, 1.03)

#: Duration search range, as multiples of the central-crossing duration.
T14_RANGE = (0.3, 1.5)
#: Fixed quadratic limb darkening; the shape comparison is insensitive to it.
LIMB_DARKENING = (0.40, 0.25)
#: Half-width of the fitted window around each event: this many central
#: durations, clipped to WINDOW_CLIP days.
WINDOW_T14 = 2.5
WINDOW_CLIP = (0.2, 0.5)
MIN_POINTS_EVENT = 12
MIN_POINTS_CORE = 3

_FINE_STEP = 20.0 / 86400.0
_XF = np.arange(-1.2, 1.2, _FINE_STEP)
_XE = np.concatenate([_XF, [_XF[-1] + _FINE_STEP]])


def central_t14_days(period: float, rho_gcc: float) -> float:
    """Duration of a central transit on a circular orbit, small planet."""
    return (13.0 / 24.0) * (period / 365.25) ** (1.0 / 3.0) * (rho_gcc / RHO_SUN_CGS) ** (-1.0 / 3.0)


def _cum(profile):
    return np.concatenate([[0.0], np.cumsum(profile) * _FINE_STEP])


def _smear(C, x, e):
    lo = np.interp(x - e / 2, _XE, C)
    hi = np.interp(x + e / 2, _XE, C)
    return (hi - lo) / e


def _transit_profile(t14: float, b: float, k: float = 0.05):
    import batman
    P = 100.0
    ar = P / (np.pi * t14) * np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-4))
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a = 0.0, P, k, ar
    pm.inc = np.degrees(np.arccos(min(b / ar, 1.0)))
    pm.ecc, pm.w, pm.u, pm.limb_dark = 0.0, 90.0, list(LIMB_DARKENING), "quadratic"
    s = 1.0 - batman.TransitModel(pm, _XF).light_curve(pm)
    return s / s.max() if s.max() > 0 else s


def _segments(t, f, expo, period, t0, half, core):
    """One entry per event: offsets, exposures, the baseline's orthonormal basis and
    the baseline-projected flux."""
    ep = np.round((t - t0) / period)
    near = np.abs(t - (t0 + ep * period)) < half
    segs = []
    for e in np.unique(ep[near]):
        tc = t0 + e * period
        m = np.where(np.abs(t - tc) < half)[0]
        if len(m) < MIN_POINTS_EVENT:
            continue
        x = t[m] - tc
        if (np.abs(x) < 0.25 * half).sum() < MIN_POINTS_CORE:
            continue
        out = np.abs(x) > 0.8 * core
        order, best = 3, np.inf
        if out.sum() >= 15:
            for od in range(1, 8):
                V = np.vander(x[out] / half, od + 1)
                c, *_ = np.linalg.lstsq(V, f[m][out], rcond=None)
                rss = float(np.sum((f[m][out] - V @ c) ** 2))
                n = int(out.sum())
                bic = n * np.log(max(rss, 1e-300) / n) + (od + 1) * np.log(n)
                if bic < best:
                    order, best = od, bic
        Q, _ = np.linalg.qr(np.vander(x / half, order + 1))
        y = f[m] - Q @ (Q.T @ f[m])
        d = np.diff(f[m])
        sig = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        segs.append(dict(x=x, e=expo[m], Q=Q, y=y, sig=max(float(sig), 1e-6), n=len(m),
                         order=order, epoch=int(e)))
    return segs


def _proj(s, Q):
    return s - Q @ (Q.T @ s)


def _fit_shared(segs, shapes):
    num = sum(np.sum(sg["y"] * (-p) / sg["sig"] ** 2) for sg, p in zip(segs, shapes))
    den = sum(np.sum(p * p / sg["sig"] ** 2) for sg, p in zip(segs, shapes))
    A = num / den if den > 0 else 0.0
    chi = sum(np.sum((sg["y"] + A * p) ** 2 / sg["sig"] ** 2) for sg, p in zip(segs, shapes))
    return float(A), float(chi)


def _fit_free(sg, p):
    den = np.sum(p * p)
    A = np.sum(sg["y"] * (-p)) / den if den > 0 else 0.0
    return float(np.sum((sg["y"] + A * p) ** 2) / sg["sig"] ** 2)


def shape_fit(t, f, expo_days, period, t0, rho_gcc, free: bool = False,
              t14_range=None, n_t14: int = 9) -> dict | None:
    """The four fits.  Returns None when no event has enough points."""
    t14_range = t14_range or T14_RANGE
    T0 = central_t14_days(period, rho_gcc)
    half = float(np.clip(WINDOW_T14 * T0, *WINDOW_CLIP))
    segs = _segments(t, f, expo_days, period, t0, half, T0)
    if not segs:
        return None
    N = sum(sg["n"] for sg in segs)
    chi0 = float(sum(np.sum(sg["y"] ** 2) / sg["sig"] ** 2 for sg in segs))

    best = (np.inf, None, None)
    for fr in np.geomspace(t14_range[0], t14_range[1], n_t14):
        for b in IMPACT_GRID:
            C = _cum(_transit_profile(fr * T0, b))
            for off in np.linspace(-0.4, 0.4, 9) * fr * T0:
                A, chi = _fit_shared(segs, [_proj(_smear(C, sg["x"] - off, sg["e"]), sg["Q"]) for sg in segs])
                if A > 0 and chi < best[0]:
                    best = (chi, (A, fr * T0, b, off), C)
    chiT, parT, CT = best
    if parT is None:
        chiT, parT = chi0, (0.0, np.nan, np.nan, np.nan)

    offs = np.linspace(-0.5, 0.5, 11)
    chiGs, CG = np.inf, None
    for wf in np.geomspace(0.1, 1.5, 8):
        sw = wf * T0 / 2.355
        for c in offs * T0:
            C = _cum(np.exp(-0.5 * ((_XF - c) / sw) ** 2))
            A, chi = _fit_shared(segs, [_proj(_smear(C, sg["x"], sg["e"]), sg["Q"]) for sg in segs])
            if A > 0 and chi < chiGs:
                chiGs, CG = chi, (C, A)
    chiFs, CF = np.inf, None
    taus = np.array([5, 10, 20, 40, 80, 160]) / 1440.0
    for tau in taus:
        for c in offs * T0:
            for sgn in (1, -1):
                z = sgn * (_XF - c)
                C = _cum(np.where(z >= 0, np.exp(-np.clip(z, 0, None) / tau), 0.0))
                A, chi = _fit_shared(segs, [_proj(_smear(C, sg["x"], sg["e"]), sg["Q"]) for sg in segs])
                if A > 0 and chi < chiFs:
                    chiFs, CF = chi, (C, A)
    chiGs, chiFs = min(chiGs, chi0), min(chiFs, chi0)

    E, lnN = len(segs), np.log(N)
    bicT = chiT + 4 * lnN
    out = dict(n_events=E, n_pts=int(N), central_t14_h=24 * T0,
               depth_ppm=1e6 * parT[0],
               t14_h=24 * parT[1] if np.isfinite(parT[1]) else None,
               t14_ratio=parT[1] / T0 if np.isfinite(parT[1]) else None,
               b=parT[2] if np.isfinite(parT[2]) else None,
               offset_h=24 * parT[3] if np.isfinite(parT[3]) else None,
               dBIC_0=chi0 - bicT, dBIC_Gs=chiGs + 3 * lnN - bicT, dBIC_Fs=chiFs + 3 * lnN - bicT,
               chi2_red_T=chiT / N, base_order_med=float(np.median([sg["order"] for sg in segs])))
    if free:
        chiG = np.inf
        for wf in np.geomspace(0.1, 1.5, 8):
            sw = wf * T0 / 2.355
            Cs = [_cum(np.exp(-0.5 * ((_XF - c) / sw) ** 2)) for c in offs * T0]
            chiG = min(chiG, sum(min(_fit_free(sg, _proj(_smear(C, sg["x"], sg["e"]), sg["Q"])) for C in Cs)
                                 for sg in segs))
        chiF = np.inf
        for tau in taus:
            Cs = []
            for c in offs * T0:
                for sgn in (1, -1):
                    z = sgn * (_XF - c)
                    Cs.append(_cum(np.where(z >= 0, np.exp(-np.clip(z, 0, None) / tau), 0.0)))
            chiF = min(chiF, sum(min(_fit_free(sg, _proj(_smear(C, sg["x"], sg["e"]), sg["Q"])) for C in Cs)
                                 for sg in segs))
        out["dBIC_G"] = chiG + (2 * E + 1) * lnN - bicT
        out["dBIC_F"] = chiF + (2 * E + 1) * lnN - bicT
    out["_plot"] = dict(segs=segs, T=(CT, parT), G=CG, F=CF, half=half)
    return out


def judge(m: dict) -> tuple[bool, list]:
    """(passes, the parts of the rule that failed)."""
    why = []
    d0 = m.get("dBIC_0")
    if not (m.get("depth_ppm") or 0) > 0:
        why.append("no positive transit depth fits")
    if not (d0 is not None and d0 > MIN_DBIC_FLAT):
        why.append(f"a transit is no better than no dip (dBIC {d0:.1f}, need > {MIN_DBIC_FLAT:.0f})")
    flare_need = max(MIN_DBIC_FLARE, MIN_FLARE_FRAC * d0) if d0 and d0 > 0 else MIN_DBIC_FLARE
    if not m["dBIC_Fs"] > flare_need:
        why.append(f"a flare-shaped dip fits as well (dBIC {m['dBIC_Fs']:.1f}, need > {flare_need:.1f})")
    ratio = m["dBIC_Gs"] / d0 if d0 and d0 > 0 else None
    m["gauss_ratio"] = ratio
    if d0 and d0 > 0 and not ratio > MIN_GAUSS_RATIO:
        why.append(f"a Gaussian dip fits better (dBIC {m['dBIC_Gs']:.1f}, {ratio:.2f} of the transit's "
                   f"lead over no dip; limit {MIN_GAUSS_RATIO:.2f})")
    return (not why), why


def add_arguments(ap):
    g = ap.add_argument_group("transitlike")
    g.add_argument("--transitlike-free", action="store_true",
                   help="also fit per-event-free Gaussian and flare alternatives (slow; reported only)")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0"):
        return r
    rho, _ = ctx.star.density()
    if not cli.require(r, rho and rho > 0,
                       f"no stellar density for {ctx.star.name}, so the allowed transit durations "
                       f"are unknown. A missing row is normal at this scale and is not a verdict."):
        r.set_self_check(False, "no stellar density")
        return r
    try:
        lc = ctx.lc
    except Exception as exc:  # noqa: BLE001 - missing data is normal
        return r.inconclusive(f"could not load the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)

    ok = np.isfinite(lc.t) & np.isfinite(lc.f)
    o = np.argsort(lc.t[ok])
    t, f = lc.t[ok][o], lc.f[ok][o]
    f = f / np.nanmedian(f)
    expo = np.clip(np.asarray(lc.cadence_s, float)[ok][o], 20.0, 1800.0) / 86400.0
    period, t0 = float(ctx.candidate.period), float(ctx.candidate.t0_btjd)

    m = shape_fit(t, f, expo, period, t0, float(rho), free=bool(getattr(ctx.args, "transitlike_free", False)))
    if m is None:
        r.set_self_check(False, "no event has enough points")
        return r.inconclusive(f"no predicted transit has {MIN_POINTS_EVENT} points around it with "
                              f"{MIN_POINTS_CORE} near mid-transit, so there is no shape to fit")
    plot = m.pop("_plot")
    d = np.diff(f)
    r.set_noise(1e6 * 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2),
                "MAD of point-to-point differences of the raw flux")
    passes, why = judge(m)
    r.metrics.update({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()})
    r.metrics.update(rho_star_gcc=float(rho), thresholds=dict(
        min_dbic_flat=MIN_DBIC_FLAT, min_dbic_flare=MIN_DBIC_FLARE, min_flare_frac=MIN_FLARE_FRAC,
        min_gauss_ratio=MIN_GAUSS_RATIO, impact_grid=list(IMPACT_GRID)))
    r.set_self_check(True, f"{m['n_events']} event(s), {m['n_pts']} points fitted")
    desc = (f"{m['n_events']} event(s): transit depth {m['depth_ppm']:.0f} ppm, T14 "
            f"{(m['t14_h'] or float('nan')):.2f} h ({(m['t14_ratio'] or float('nan')):.2f} of a central crossing); "
            f"dBIC against no dip {m['dBIC_0']:.1f}, Gaussian {m['dBIC_Gs']:.1f}, flare {m['dBIC_Fs']:.1f}")
    if passes:
        r.passed(f"the dip is transit-shaped. {desc}")
    else:
        r.fail(f"the dip is not transit-shaped: {'; '.join(why)}. {desc}")

    if ctx.make_plots:
        try:
            r.add_plot(_plot(ctx, r, plot, m))
        except Exception as exc:  # noqa: BLE001 - a failed figure is not a verdict
            ctx.notes.append(f"transitlike plot failed: {type(exc).__name__}: {exc}")
    return r


def _plot(ctx, r, plot, m):
    segs, half = plot["segs"], plot["half"]
    x = np.concatenate([sg["x"] for sg in segs]) * 24.0
    y = np.concatenate([sg["y"] for sg in segs]) * 1e6
    fig, ax = figure(figsize=(9, 5))
    ax.plot(x, y, ".", color=C_DATA, ms=1.5, alpha=0.25)
    width = max(2 * half * 24.0 / 60.0, 0.05)
    bx, by, _ = bin_series(x, y, width)[:3]
    ax.plot(bx, by, "o", color=C_BINNED, ms=3.5, label="data, baseline removed (binned)")

    def model(C, A, off=0.0):
        mm = np.concatenate([-A * _proj(_smear(C, sg["x"] - off, sg["e"]), sg["Q"]) for sg in segs]) * 1e6
        return bin_series(x, mm, width)[:2]
    CT, parT = plot["T"]
    if CT is not None:
        ax.plot(*model(CT, parT[0], parT[3]), "-", color=C_MODEL, lw=1.8, label="transit")
    if plot["G"] is not None:
        ax.plot(*model(*plot["G"]), "--", color=C_ALT, lw=1.3, label=f"Gaussian (dBIC {m['dBIC_Gs']:.0f})")
    if plot["F"] is not None:
        ax.plot(*model(*plot["F"]), ":", color=C_BAD, lw=1.5, label=f"flare shape (dBIC {m['dBIC_Fs']:.0f})")
    lo, hi = np.nanpercentile(by, [1, 99]) if len(by) > 5 else (np.nanmin(y), np.nanmax(y))
    pad = 0.3 * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("hours from predicted mid-transit")
    ax.set_ylabel("ppm, per-event baseline removed")
    ax.set_title(f"shape of the dip in raw flux: {m['n_events']} event(s) stacked; "
                 f"positive dBIC favors the transit", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(TEST, "is the dip transit-shaped, in raw flux?"))
    ctx = cli.Context.from_args(ap.parse_args(argv))
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # recorded, never swallowed
        r.error(exc)
    return cli.finish(r, ctx)


if __name__ == "__main__":
    raise SystemExit(main())

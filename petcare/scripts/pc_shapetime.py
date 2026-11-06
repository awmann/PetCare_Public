"""pc_shapetime -- does the transit keep its shape and depth over the years?

A planet's transit is the same event every time it happens: the same depth,
the same duration, the same ingress.  Stellar variability folded at the wrong
period is not -- spots evolve, and a "transit" that is really a rotation trough
or a residual of the detrending changes its depth *and its width* from one
year to the next, by far more than a TTV would explain.

This test makes that measurable.  One trapezoid is fitted to every transit at
once -- depth, total duration, ingress fraction, centre -- and then each group
of sectors is refitted with its own depth, duration and centre, the ingress
fraction held at the global value.

Three things are built in so that a real planet does not fail it:

* **The centre floats per group.**  A slightly wrong period, or a TTV, moves the
  transit; folding a moved transit at a fixed ephemeris smears it, and a
  smeared transit looks wider and shallower.  Letting each group find its own
  centre removes that, so a timing problem cannot masquerade as a shape change.
  The shifts are reported, and large ones are a separate signal.
* **The model is integrated over each point's exposure.**  Early sectors are 30
  minute FFIs and later ones 2 minute, and a 30 minute exposure rounds a
  transit's corners.  Comparing the two without the integration would call the
  cadence a shape change.
* **Systematic floors.**  The light curves' de-dilution factor and aperture change
  between sectors, so a real planet's depth wanders by ten or fifteen
  percent; the depth floor absorbs that.  The duration floor absorbs the
  remaining limb-darkening and detrending differences.

The headline numbers are the *extra* scatter -- the fractional floor that
would have to be added to the errors to make the groups agree -- for depth and
for duration.  They are in the units a person reasons in: "the duration varies
by 45% between groups, beyond the noise".

Sectors are grouped, not used one at a time: consecutive sectors within a
month of each other form one group, and thin groups are merged with their
neighbours until each holds at least three transits.  A depth from one partial
event is not a measurement of how the transit changes.

    python -m petcare.scripts.pc_shapetime --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares

from .. import cli, ephem
from .. import prep as pcprep
from ..plotting import C_ALT, C_BINNED, C_MODEL, C_OK, annotate_status, figure, save
from ..result import Result

TEST = "shapetime"
VERSION = "0.1.0"

#: Fit window, in transit durations either side of mid-transit.
WINDOW_T14 = 2.5

#: Gap that starts a new pseudo-sector when the light curve has no sector ids.
PSEUDO_SECTOR_GAP_DAYS = 5.0

#: Group construction.
GROUP_GAP_DAYS = 30.0
MIN_TRANSITS_PER_GROUP = 3

#: Systematic floors added in quadrature before the groups are compared.
DEPTH_FLOOR = 0.20
DURATION_FLOOR = 0.10

#: Extra fractional scatter in DURATION, beyond noise and floors, above which
#: the test flags.  Calibrated on a benchmark of known young planets: the
#: worst real planet is 20% (a 934 ppm transit), most are 0-12%; one reads 47%
#: only because its literature ephemeris misses the transit entirely.  Among
#: our search's candidates, the ones that are stellar variability reach
#: 43-81%.  40% is twice the worst planet.
DURATION_EXTRA_FLAG = 0.40

#: Depth is measured and reported but NOT flagged on.  Spots change a real
#: planet's measured depth: a published candidate (FPP 0.042) varies by 92%
#: beyond the floor, another by 39%, and variability among the candidates
#: spans the same range, 50-214%.  A depth cut that separated them
#: would cost planets.  Shape is what holds still; depth is not.
DEPTH_EXTRA_FLAG = None

#: Sub-samples across each exposure for the integrated model.
N_SUB = 7


def add_arguments(ap):
    g = ap.add_argument_group("shapetime")
    g.add_argument("--bin", type=float, default=None,
                   help="bin seconds (default: a fifth of the duration, 120-600 s)")
    return ap


# --------------------------------------------------------------------------
# the model


def _trapezoid(dt, depth, total, q):
    """Box of ``depth`` and full width ``total`` with linear ingress and egress
    each ``q * total / 2`` long."""
    half = total / 2.0
    ing = max(q * half, 1e-6)
    x = np.abs(dt)
    out = np.zeros_like(dt)
    flat = x <= half - ing
    ramp = (x > half - ing) & (x < half)
    out[flat] = -depth
    out[ramp] = -depth * (half - x[ramp]) / ing
    return out


def _model(dt, exp, depth, total, q, centre, base):
    """The trapezoid averaged over each point's exposure."""
    offs = (np.arange(N_SUB) + 0.5) / N_SUB - 0.5
    sub = (dt - centre)[:, None] + exp[:, None] * offs[None, :]
    m = _trapezoid(sub.ravel(), depth, total, q).reshape(sub.shape).mean(axis=1)
    return base + m


def _fit(dt, y, e, exp, guess, free_q=True, q_fixed=None, t14=None):
    depth0, total0, q0, c0 = guess
    if free_q:
        x0 = [depth0, total0, q0, c0, 0.0]
        lo = [0.0, 0.3 * t14, 0.05, -0.5 * t14, -np.inf]
        hi = [np.inf, 2.5 * t14, 1.0, 0.5 * t14, np.inf]

        def res(p):
            return (y - _model(dt, exp, p[0], p[1], p[2], p[3], p[4])) / e
    else:
        x0 = [depth0, total0, c0, 0.0]
        lo = [0.0, 0.3 * t14, -0.5 * t14, -np.inf]
        hi = [np.inf, 2.5 * t14, 0.5 * t14, np.inf]

        def res(p):
            return (y - _model(dt, exp, p[0], p[1], q_fixed, p[2], p[3])) / e
    x0 = np.clip(x0, [a + 1e-9 if np.isfinite(a) else a for a in lo],
                 [b - 1e-9 if np.isfinite(b) else b for b in hi])
    sol = least_squares(res, x0, bounds=(lo, hi), x_scale="jac")
    dof = max(y.size - len(x0), 1)
    rchi = float(np.sum(sol.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(rchi, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(x0), np.nan)
    return sol.x, err, rchi


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

    # per-point sector and exposure
    order = np.argsort(lc.t)
    i = np.clip(np.searchsorted(lc.t[order], p.t), 0, lc.t.size - 1)
    sec = lc.sector_id[order][i]
    cad = {si.sector: (si.cadence_s or 120.0) for si in lc.sectors}
    pcad = lc.cadence_s[order][i] if lc.cadence_s is not None else None
    if not np.any(sec >= 0):
        # No sector ids (a plain stitched CSV): cut on gaps longer than a TESS
        # orbit's downlink instead.  Groups are what the test needs, not sector
        # numbers, and without this every such light curve is untestable.
        gaps = np.concatenate([[0], np.diff(p.t) > PSEUDO_SECTOR_GAP_DAYS]).cumsum()
        sec = gaps.astype(int)
        r.metrics["pseudo_sectors"] = int(sec.max() + 1)
    exp = np.array([max(cad.get(int(s), float(pc) if pcad is not None else 120.0), bin_s)
                    for s, pc in zip(sec, pcad if pcad is not None else [120.0] * sec.size)]) / 86400.0

    ep = ephem.epoch_number(p.t, period, t0)
    dt = p.t - ephem.transit_time(ep, period, t0)
    y = p.f_detrended - 1.0
    win = (np.abs(dt) < WINDOW_T14 * t14) & np.isfinite(y)
    if win.sum() < 30:
        return r.inconclusive("too few points near the transits to fit a shape")
    white = p.white_ppm / 1e6 if np.isfinite(p.white_ppm) else np.nanstd(y[~win])
    red = p.red_factor if np.isfinite(p.red_factor) and p.red_factor > 0 else 1.0
    e = np.full(y.size, white)

    # ---- global shape ------------------------------------------------------
    inn = win & (np.abs(dt) < 0.4 * t14)
    d_guess = max(-np.nanmedian(y[inn]), 1e-5) if inn.any() else 1e-3
    gx, _, grchi = _fit(dt[win], y[win], e[win], exp[win],
                           (d_guess, t14, 0.3, 0.0), t14=t14)
    D, T, Q, C = gx[:4]
    r.metrics["global"] = {"depth_ppm": _f(D * 1e6), "duration_h": _f(T * 24),
                           "ingress_frac": _f(Q), "centre_h": _f(C * 24),
                           "reduced_chi2": _f(grchi)}
    if D <= 0 or not np.isfinite(D):
        return r.inconclusive("the global fit found no transit to compare across time")

    # ---- groups ------------------------------------------------------------
    groups = _groups(p.t, sec, ep, dt, win, t14)
    r.metrics["n_groups"] = len(groups)
    if len(groups) < 2:
        r.set_self_check(True, f"{len(groups)} group(s) with >= "
                               f"{MIN_TRANSITS_PER_GROUP} transits")
        return r.inconclusive(
            f"only {len(groups)} group of sectors holds {MIN_TRANSITS_PER_GROUP} "
            f"or more transits, so there is no 'over time' to test")

    rows = []
    for gsec, gmask in groups:
        m = gmask & win
        x, err, rc = _fit(dt[m], y[m], e[m], exp[m], (D, T, Q, C),
                          free_q=False, q_fixed=Q, t14=t14)
        err = err * red
        rows.append({
            "sectors": [int(s) for s in gsec], "t_mid": float(np.median(p.t[m])),
            "n_transits": int(np.unique(ep[m & (np.abs(dt) < 0.5 * t14)]).size),
            "depth_ppm": float(x[0] * 1e6), "depth_err_ppm": float(err[0] * 1e6),
            "duration_h": float(x[1] * 24), "duration_err_h": float(err[1] * 24),
            "shift_h": float(x[2] * 24), "shift_err_h": float(err[2] * 24),
            "reduced_chi2": float(rc)})
    r.metrics["groups"] = rows

    d = np.array([g["depth_ppm"] for g in rows]) / (D * 1e6)
    de = np.array([g["depth_err_ppm"] for g in rows]) / (D * 1e6)
    w = np.array([g["duration_h"] for g in rows]) / (T * 24)
    we = np.array([g["duration_err_h"] for g in rows]) / (T * 24)
    depth_extra = _extra(d, de, DEPTH_FLOOR)
    dur_extra = _extra(w, we, DURATION_FLOOR)
    shifts = np.array([g["shift_h"] for g in rows])
    r.metrics.update({
        "depth_ratios": [_f(v) for v in d], "duration_ratios": [_f(v) for v in w],
        "depth_extra_frac": _f(depth_extra), "duration_extra_frac": _f(dur_extra),
        "depth_range": [_f(d.min()), _f(d.max())],
        "duration_range": [_f(w.min()), _f(w.max())],
        "shift_range_h": [_f(shifts.min()), _f(shifts.max())],
        "floors": {"depth": DEPTH_FLOOR, "duration": DURATION_FLOOR},
    })

    ok = bool(np.all(np.isfinite(de)) and np.all(np.isfinite(we)))
    r.set_self_check(ok, f"{len(rows)} groups of sectors, each with >= "
                         f"{MIN_TRANSITS_PER_GROUP} transits; global T14 "
                         f"{T * 24:.2f} h against the input {t14 * 24:.2f} h")

    desc = (f"{len(rows)} groups of sectors: depth {d.min():.2f}-{d.max():.2f} x "
            f"the global value, duration {w.min():.2f}-{w.max():.2f} x; beyond the "
            f"noise and a {DEPTH_FLOOR:.0%}/{DURATION_FLOOR:.0%} floor the depth "
            f"varies by a further {depth_extra:.0%} and the duration by "
            f"{dur_extra:.0%}; each group re-centred, shifts "
            f"{shifts.min():+.2f} to {shifts.max():+.2f} h")
    if not ok:
        r.inconclusive("the per-group fits did not return usable errors. " + desc)
    elif dur_extra > DURATION_EXTRA_FLAG:
        r.flag(f"the transit's duration changes over time by more than "
               f"calibration, cadence or timing can explain -- what stellar "
               f"variability does and a planet does not. {desc}")
    else:
        r.passed(f"the transit keeps its shape and depth over time. {desc}")

    if ctx.make_plots:
        try:
            r.add_plot(_plot(ctx, r, p, dt, y, win, ep, groups, rows, exp, (D, T, Q, C), t14))
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"plot failed: {type(exc).__name__}: {exc}")
    return r


def _groups(t, sec, ep, dt, win, t14):
    """Consecutive sectors within GROUP_GAP_DAYS form a group; thin groups are
    merged into a neighbour until each holds MIN_TRANSITS_PER_GROUP transits."""
    secs = sorted({int(s) for s in sec[win] if s >= 0},
                  key=lambda s: np.median(t[sec == s]))
    if not secs:
        return []
    spans = {s: (t[sec == s].min(), t[sec == s].max()) for s in secs}
    groups = [[secs[0]]]
    for s in secs[1:]:
        if spans[s][0] - spans[groups[-1][-1]][1] < GROUP_GAP_DAYS:
            groups[-1].append(s)
        else:
            groups.append([s])

    def n_tr(g):
        m = np.isin(sec, g) & win & (np.abs(dt) < 0.5 * t14)
        return int(np.unique(ep[m]).size)

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k, g in enumerate(groups):
            if n_tr(g) < MIN_TRANSITS_PER_GROUP:
                j = k - 1 if k == len(groups) - 1 else (
                    k + 1 if k == 0 else
                    (k - 1 if n_tr(groups[k - 1]) <= n_tr(groups[k + 1]) else k + 1))
                lo, hi = min(k, j), max(k, j)
                groups[lo] = groups[lo] + groups[hi]
                del groups[hi]
                changed = True
                break
    groups = [g for g in groups if n_tr(g) >= MIN_TRANSITS_PER_GROUP]
    return [(g, np.isin(sec, g)) for g in groups]


def _extra(v, e, floor):
    """Fractional scatter to add, beyond the errors and the floor, to make the
    values agree (reduced chi2 of 1 about their weighted mean)."""
    v, e = np.asarray(v, float), np.asarray(e, float)
    if v.size < 2:
        return 0.0

    def rchi(s):
        ee = np.sqrt(e ** 2 + floor ** 2 + s ** 2)
        w = 1 / ee ** 2
        mu = np.sum(w * v) / np.sum(w)
        return np.sum(((v - mu) / ee) ** 2) / (v.size - 1)

    if rchi(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rchi(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _plot(ctx, r, p, dt, y, win, ep, groups, rows, exp, gl, t14):
    D, T, Q, _ = gl
    fig, ax = figure(nrows=1, ncols=3, figsize=(16, 5.2))
    cmap = __import__("matplotlib").colormaps["viridis"]
    n = len(rows)
    xs = np.linspace(-WINDOW_T14 * t14, WINDOW_T14 * t14, 400)
    for k, ((gsec, gmask), g) in enumerate(zip(groups, rows)):
        col = cmap(k / max(n - 1, 1))
        m = gmask & win
        bw = t14 / 8
        edges = np.arange(-WINDOW_T14 * t14, WINDOW_T14 * t14 + bw, bw)
        cen = 0.5 * (edges[1:] + edges[:-1])
        yb = [np.median(y[m][(dt[m] >= lo) & (dt[m] < hi)])
              if ((dt[m] >= lo) & (dt[m] < hi)).sum() >= 3 else np.nan
              for lo, hi in pairwise(edges)]
        lab = f"S{gsec[0]}" + (f"-{gsec[-1]}" if len(gsec) > 1 else "")
        ax[0].plot(cen * 24, np.array(yb) * 1e6, "o", ms=3, color=col, label=lab)
        mod = _model(xs, np.full(xs.size, np.median(exp[m])), g["depth_ppm"] / 1e6,
                     g["duration_h"] / 24, Q, g["shift_h"] / 24, 0.0)
        ax[0].plot(xs * 24, mod * 1e6, "-", color=col, lw=1.2)
    ax[0].set_xlabel("hours from mid-transit")
    ax[0].set_ylabel("ppm")
    ax[0].set_title("each group of sectors, with its own fit", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2)
    tm = [g["t_mid"] for g in rows]
    ax[1].errorbar(tm, [g["depth_ppm"] / (D * 1e6) for g in rows],
                   [g["depth_err_ppm"] / (D * 1e6) for g in rows], fmt="o", color=C_BINNED,
                   label="depth")
    ax[1].errorbar(tm, [g["duration_h"] / (T * 24) for g in rows],
                   [g["duration_err_h"] / (T * 24) for g in rows], fmt="s", color=C_ALT,
                   label="duration")
    ax[1].axhline(1, color="0.5", lw=0.7)
    ax[1].axhspan(1 - DEPTH_FLOOR, 1 + DEPTH_FLOOR, color=C_OK, alpha=0.08)
    ax[1].set_xlabel("BTJD")
    ax[1].set_ylabel("relative to the global fit")
    ax[1].set_title(f"depth +{r.metrics['depth_extra_frac']:.0%}, duration "
                    f"+{r.metrics['duration_extra_frac']:.0%} extra scatter", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[2].errorbar(tm, [g["shift_h"] for g in rows], [g["shift_err_h"] for g in rows],
                   fmt="o", color=C_MODEL)
    ax[2].axhline(0, color="0.5", lw=0.7)
    ax[2].set_xlabel("BTJD")
    ax[2].set_ylabel("hours")
    ax[2].set_title("each group's centre against the ephemeris", fontsize=10)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Does the transit keep its depth and shape over the years, as a "
              "planet's does and stellar variability's does not?"))
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

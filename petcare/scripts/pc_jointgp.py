"""pc_jointgp -- each transit fitted jointly with the GP, nothing masked.

The other light-curve tests remove variability first: the transit window is
masked, a GP interpolates across it, and the depth is measured from what is
left.  On a fast rotator that interpolation is unconstrained, and pc_interp
measures how much it can invent.  This test takes the other route:
no mask, and the transit and the variability fitted together.

Per transit, within a window several durations wide:

    flux = c + depth * s(t) + GP

where s(t) is the transit shape from pc_fit (unit depth, fixed), and the GP has
the kernel and hyperparameters fitted to the *out-of-transit* data in prep --
so it cannot reshape itself around this dip.  With the hyperparameters and the
shape fixed, the model is linear in (c, depth), and generalised least squares
against the full GP covariance gives the depth and its error exactly:

    beta = (X^T K^-1 X)^-1 X^T K^-1 y,   cov = (X^T K^-1 X)^-1

That error is the honest one.  Where the star varies on the transit's own
timescale, the GP and the transit are degenerate, and the depth error grows to
say so; where the star is quiet it shrinks to the photometric noise.  A dip the
detrending manufactured shows up as a depth consistent with zero, or as epochs
that scatter far beyond their (already GP-inflated) errors.

Reported: the inverse-variance mean depth and its significance, and the
chi-square of the per-epoch depths about that mean.  Real planets on spotted
stars do vary in depth, so depth consistency only flags.

    python -m petcare.scripts.pc_jointgp --tic 410214986 --period 8.1383 \\
        --t0 1332.31 --duration 3.2 --use-fit
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .. import cli, ephem
from .. import prep as pcprep
from ..fitting import is_segmented, make_gp, transit_model
from ..plotting import C_BAD, C_BINNED, C_MODEL, annotate_status, figure, save
from ..result import Result, Status

TEST = "jointgp"
VERSION = "0.1.2"

#: Half-width of each transit's window, in T14, with a floor in days, so the GP
#: sees enough out-of-transit flux on both sides to pin itself down.
WINDOW_T14 = 4.0
WINDOW_MIN_DAYS = 0.4
#: Points needed in transit and either side for an epoch to count.
MIN_IN = 3
MIN_SIDE = 5
#: Significance of the combined depth.  Calibrated on a benchmark
#: of 22 known young planets: all 21 with more than one transit score
#: 7.1-58 sigma (HIP 67522 b 53 despite the GP inflating its errors 7x).  The
#: consistency flag fires on two heavily spotted hosts, so it is a
#: flag, never a cut.
PASS_SIGMA = 7.0
FAIL_SIGMA = 5.0
#: A FAIL means the dip is not there, which needs a fit able to see it.
#: Below FAIL_SIGMA it is reported, not refuted, when all three hold: the
#: search depth would itself be under FAIL_SIGMA at this error, the measured depth agrees
#: with the search depth, and there is still this much evidence for a dip.  Injections
#: back it: the error on a star predicts recovery, and failed injections keep their depth.
LIMITED_DEPTH_GAP_SIGMA = 2.0
LIMITED_MIN_Z = 3.0
#: Per-epoch errors this many times the white-noise error mean the GP is competing with the
#: transit.  Below it a low z is noise, so the FAIL says that instead.
GP_DEGENERATE = 2.0
#: Above this reduced chi-square of per-epoch depths about the mean, flag.
CONSISTENCY_FLAG_P = 1e-4


def add_arguments(ap):
    g = ap.add_argument_group("jointgp")
    g.add_argument("--bin", type=float, default=None,
                   help="bin seconds (default: a fifth of the duration, 120-600 s)")
    return ap


def _shape(ctx, t, period, t0, t14, exp_days):
    """Unit-depth transit shape (positive in transit) from pc_fit's geometry,
    or a trapezoid with 20% ingress if there is no usable fit."""
    fitf = ctx.out / ctx.target.slug / (
        f"fit_c{ctx.candidate.index}.json" if ctx.candidate.index else "fit.json")
    m = {}
    if fitf.exists():
        try:
            fit = json.loads(fitf.read_text())
            # Same rule as Context._apply_fit_refinement: an inconclusive fit is not
            # a measurement.  A 5800 ppm transit at the T14 of such a fit (8.2 h) gave
            # z = -0.8, and z = 87 at the right one.
            if fit.get("status") not in (Status.ERROR, Status.INCONCLUSIVE):
                m = fit.get("metrics") or {}
        except (OSError, json.JSONDecodeError):
            m = {}
    k, a, b = m.get("rp_rs"), m.get("a_rs"), m.get("b")
    if k and a and b is not None and k > 0 and a > 1:
        lc = transit_model(t, t0, period, k, a, b, exp_time_days=exp_days)
        d = 1.0 - lc
        peak = d.max() if d.size else 0
        # the batman model can be all-zero in a window that misses the
        # transit; fall through to the trapezoid then
        full = 1.0 - transit_model(np.array([t0]), t0, period, k, a, b)[0]
        if full > 0:
            return d / full, "pc_fit geometry"
        if peak > 0:
            return d / peak, "pc_fit geometry"
    dt = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
    ing = 0.2 * t14
    s = np.clip((0.5 * t14 - dt) / ing, 0, 1)
    return s, "trapezoid (no usable pc_fit geometry)"


def _gls(gp, y, X):
    """Generalised least squares against a computed celerite2 GP."""
    KiX = np.column_stack([gp.apply_inverse(X[:, j]) for j in range(X.shape[1])])
    Kiy = gp.apply_inverse(y)
    A = X.T @ KiX
    cov = np.linalg.inv(A)
    beta = cov @ (X.T @ Kiy)
    r = y - X @ beta
    chi2 = float(r @ gp.apply_inverse(r))
    return beta, cov, chi2


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
    spec = p.gp
    if spec is None or not getattr(spec, "success", False):
        return r.inconclusive("no GP variability model trained on the out-of-transit "
                              "data, so there is nothing to fit jointly")

    t, f, err = p.t, p.f, p.err
    shape, shape_src = _shape(ctx, t, period, t0, t14, bin_s / 86400.0)
    half = max(WINDOW_T14 * t14, WINDOW_MIN_DAYS)
    seg_index = spec.index_for(t) if is_segmented(spec) else None

    rows, n_failed = [], 0
    for e in np.unique(ephem.epoch_number(t, period, t0)):
        c = float(ephem.transit_time(e, period, t0))
        w = np.abs(t - c) < half
        intr = w & (np.abs(t - c) < 0.5 * t14)
        if intr.sum() < MIN_IN or (w & (t < c - 0.6 * t14)).sum() < MIN_SIDE \
                or (w & (t > c + 0.6 * t14)).sum() < MIN_SIDE:
            continue
        sub = spec
        if seg_index is not None:
            ids = np.unique(seg_index[w])
            if ids.size != 1 or ids[0] < 0:
                continue
            sub = spec.specs[int(ids[0])]
        tw, yw, ew, sw = t[w], f[w] - np.median(f[w]), err[w], shape[w]
        if sw.max() <= 0:
            continue
        try:
            gp = make_gp(sub, None, tw, ew)
            X = np.column_stack([np.ones(tw.size), -sw])   # depth > 0 is a dip
            beta, cov, chi2 = _gls(gp, yw, X)
        except Exception:  # noqa: BLE001 - one bad window must not sink the test
            n_failed += 1
            continue
        dof = max(tw.size - 2, 1)
        infl = max(1.0, np.sqrt(chi2 / dof))   # if the GP underfits, say so
        d, sd = float(beta[1]), float(np.sqrt(cov[1, 1]) * infl)
        # the transit-free GP prediction under the joint model, for the plot
        try:
            mu = gp.predict(yw - X @ beta, t=tw)
        except Exception:  # noqa: BLE001
            mu = np.zeros(tw.size)
        rows.append({"epoch": int(e), "t_mid": c, "depth": d, "err": sd,
                     "chi2_red": chi2 / dof,
                     "white_err": float(np.sqrt(1.0 / np.sum(sw ** 2 / ew ** 2))),
                     "_t": tw - c, "_y": yw - beta[0] - mu, "_s": sw})
    if not rows:
        return r.inconclusive("no transit has enough data in and around it to fit "
                              "jointly with the GP")

    d = np.array([x["depth"] for x in rows])
    s = np.array([x["err"] for x in rows])
    wts = 1.0 / s ** 2
    mean = float(np.sum(wts * d) / np.sum(wts))
    sig = float(1.0 / np.sqrt(np.sum(wts)))
    z = mean / sig
    n = d.size
    chi2c = float(np.sum(((d - mean) / s) ** 2))
    from scipy.stats import chi2 as _chi2
    p_cons = float(_chi2.sf(chi2c, max(n - 1, 1))) if n > 1 else 1.0
    white = np.array([x["white_err"] for x in rows])
    degeneracy = float(np.median(s / white)) if np.all(np.isfinite(white)) else np.nan

    r.metrics.update({
        "n_epochs": int(n), "depth_ppm": mean * 1e6, "depth_err_ppm": sig * 1e6,
        "z": float(z), "shape_source": shape_src,
        "epoch_depth_ppm": [float(x * 1e6) for x in d],
        "epoch_err_ppm": [float(x * 1e6) for x in s],
        "epochs": [x["epoch"] for x in rows],
        "consistency_chi2": chi2c, "consistency_dof": int(max(n - 1, 0)),
        "consistency_p": p_cons,
        "gp_degeneracy": degeneracy if np.isfinite(degeneracy) else None,
        "window_half_days": half, "gp_kind": getattr(spec, "kind", None),
        "n_windows_failed": n_failed,
    })
    r.set_self_check(True, f"{n} transit(s) fitted jointly with a GP whose "
                           f"hyperparameters were trained out of transit; shape "
                           f"from {shape_src}")
    desc = (f"joint GP + transit fit of {n} transit(s), nothing masked: depth "
            f"{mean * 1e6:.0f} +/- {sig * 1e6:.0f} ppm ({z:.1f} sigma). The GP "
            f"inflates each epoch's depth error {degeneracy:.1f}x over the photometric "
            f"noise (its degeneracy with the transit). Per-epoch depths "
            f"chi2 = {chi2c:.1f} for {max(n - 1, 0)} dof (p = {p_cons:.1g}).")
    search_depth = getattr(ctx, "input_depth_ppm", None) or ctx.candidate.depth_ppm
    limited = bool(search_depth and search_depth > 0 and sig > 0
                   and search_depth / (sig * 1e6) < FAIL_SIGMA
                   and (search_depth - mean * 1e6) / (sig * 1e6) < LIMITED_DEPTH_GAP_SIGMA
                   and z >= LIMITED_MIN_Z)
    if n < 2:
        # One event: pc_fit's geometry is unanchored and nothing repeats, so the
        # joint fit cannot be calibrated here (on our shortlisted candidates it gave
        # negative depths for lone dips pc_interp called 8-10 sigma).
        r.inconclusive(f"single transit: the shape is unconstrained and a lone dip "
                       f"cannot be checked for repetition, so this is reported, "
                       f"not judged. {desc}")
    elif z < FAIL_SIGMA and limited:
        r.metrics["sensitivity_limited"] = True
        r.inconclusive(f"the joint fit cannot settle this at the depth the search found: "
                       f"{search_depth:.0f} ppm would only be {search_depth / (sig * 1e6):.1f} "
                       f"sigma at this error, and the measured depth agrees with it. {desc}")
    elif z < FAIL_SIGMA:
        why = ("the GP explains the dip about as well" if not np.isfinite(degeneracy)
               or degeneracy >= GP_DEGENERATE else
               f"the errors are only {degeneracy:.1f}x white, so the GP is not absorbing it; "
               f"there is just too little signal")
        r.fail(f"fitted jointly with the variability, the transit is not significant: {why}. {desc}")
    elif z < PASS_SIGMA:
        r.flag(f"the transit survives a joint fit with the variability, but only "
               f"marginally. {desc}")
    elif p_cons < CONSISTENCY_FLAG_P:
        r.flag(f"the transit is significant in a joint fit, but its depth varies "
               f"between epochs far beyond the errors -- spots can do some of "
               f"this to a real planet; look at which epochs carry it. {desc}")
    else:
        r.passed(f"the transit holds up fitted jointly with the variability. {desc}")

    if ctx.make_plots:
        try:
            r.add_plot(_plot(ctx, r, rows, mean, sig, t14))
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"plot failed: {type(exc).__name__}: {exc}")
    return r


def _plot(ctx, r, rows, mean, sig, t14):
    fig, ax = figure(nrows=1, ncols=2, figsize=(14, 5.0))
    tm = np.array([x["t_mid"] for x in rows])
    d = np.array([x["depth"] for x in rows]) * 1e6
    s = np.array([x["err"] for x in rows]) * 1e6
    ax[0].errorbar(tm, d, s, fmt="o", color=C_BINNED, ms=4)
    ax[0].axhline(mean * 1e6, color=C_MODEL, lw=1.5, label="weighted mean")
    ax[0].axhspan((mean - sig) * 1e6, (mean + sig) * 1e6, color=C_MODEL, alpha=0.15)
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_xlabel("BTJD")
    ax[0].set_ylabel("depth from joint GP fit [ppm]")
    ax[0].set_title("each transit, fitted with the variability (no mask)", fontsize=10)
    ax[0].legend(fontsize=7)
    tt = np.concatenate([x["_t"] for x in rows]) * 24
    yy = np.concatenate([x["_y"] for x in rows]) * 1e6
    ss = np.concatenate([x["_s"] for x in rows])
    ax[1].plot(tt, yy, ".", color="0.6", ms=2, alpha=0.5)
    o = np.argsort(tt)
    edges = np.linspace(tt.min(), tt.max(), 60)
    idx = np.digitize(tt, edges)
    bc = [0.5 * (edges[i - 1] + edges[i]) for i in range(1, edges.size)]
    bm = [np.median(yy[idx == i]) if (idx == i).sum() >= 3 else np.nan
          for i in range(1, edges.size)]
    ax[1].plot(bc, bm, "o", color=C_BINNED, ms=4, label="binned")
    ax[1].plot(tt[o], -mean * ss[o] * 1e6, "-", color=C_BAD, lw=1.5,
               label="transit at the mean depth")
    lim = np.nanpercentile(np.abs(yy), 99) * 1.2
    ax[1].set_ylim(-lim, lim * 0.6)
    ax[1].set_xlabel("hours from mid-transit")
    ax[1].set_ylabel("flux minus the joint model's GP [ppm]")
    ax[1].set_title(f"stacked, with the GP the joint fit chose (T14 = {t14 * 24:.1f} h)",
                    fontsize=10)
    ax[1].legend(fontsize=7)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Each transit fitted jointly with the GP, nothing masked: does the "
              "depth survive the degeneracy with the variability?"))
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

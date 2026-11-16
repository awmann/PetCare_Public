"""pc_epochaudit -- do the transits agree with each other when each is fitted with the star?

``pc_depths``, ``pc_shapetime`` and ``pc_oddeven`` measure single transits on a
light curve whose variability was removed by a GP fitted with the transits
**masked**.  The GP then has to interpolate across every transit, and on an
active star that interpolation alone makes single epochs deep, long or odd:
one epoch of a candidate on an active young star is 45,000 ppm and 10.6 h on
the masked detrend and 18,000 ppm and 4.7 h when fitted jointly; masked-GP odd/even is 7.6 sigma on
AU Mic b and 6.5 sigma on HIP 67522 b.

This test fits every well-covered transit on its own, **nothing masked**: a GP
with its hyperparameters held at the values ``petcare.prep`` fitted, plus a
trapezoid with free depth, duration and mid-time and an offset.  The ingress
fraction is shared, from a global fit to the masked detrend.  It then asks
three things of the per-epoch depths:

* is the weighted mean depth detected (``MIN_SNR``);
* is the epoch-to-epoch scatter, beyond the errors, small beside the depth
  (``MAX_EXCESS``);
* do odd and even epochs agree (``MAX_ODDEVEN``).

Because it is the better measurement of what those three tests measure, its
verdict **overrules** them (``petcare.verdict``): a PASS here turns their FAIL
into a FLAG, and a FAIL here stands on its own.  The per-test records are never
edited; the merged summary lists what was overruled.

The per-epoch duration and timing chi-squared are reported but **not judged**:
their formal errors are too small and reject real planets at p ~ 1e-20.

It is expensive (a GP likelihood per trial per epoch), so by default it runs
only when one of the three tests it overrules has FAILED; ``--always`` runs it
regardless.

Calibrated 2026-10-31 on 21 benchmark planets, 15 planet-like candidates and
an eclipsing binary: every planet with three or more covered epochs passes
(excess <= 0.27 except TOI-451 b at 0.40; odd/even <= 2.5 sigma), and the
binary fails (odd/even 4.1 sigma jointly fitted, 23 masked).

    python -m petcare.scripts.pc_epochaudit --tic 441420236 --use-fit --always
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .. import cli, ephem, fitting
from .. import prep as pcprep
from ..plotting import C_BINNED, C_MODEL, annotate_status, figure, save
from ..result import Result, Status
from ..verdict import EPOCH_AUDIT_OVERRULES
from . import pc_shapetime as pst

TEST = "epoch_audit"
VERSION = "0.1.0"

#: An epoch counts when this fraction of its in-transit bins has data, in both halves.
MIN_COVERAGE = 0.8
#: Epochs fitted per candidate at most (evenly spread); short periods have hundreds.
MAX_EPOCHS = 40
#: Fewer jointly fitted epochs than this and the audit gives no verdict.
MIN_EPOCHS = 3
#: Excess epoch-to-epoch depth scatter over the errors, as a fraction of the
#: depth.  Benchmark planets <= 0.27 (TOI-451 b 0.40).  A chi-squared p-value is
#: not used: it rejects the bright planets, whose errors are tiny.
MAX_EXCESS = 0.5
#: Odd/even depth difference [sigma].  Benchmark planets <= 2.5; the binary 4.05.
MAX_ODDEVEN = 3.0
#: Weighted mean depth over its error.
MIN_SNR = 5.0
#: Half-width of the window fitted around each transit [days]: three durations,
#: at least six hours, at most 0.77 d.
WINDOW_MAX_DAYS = 0.7723


def add_arguments(ap):
    g = ap.add_argument_group("epoch_audit")
    g.add_argument("--always", action="store_true",
                   help="run even when none of " + ", ".join(sorted(EPOCH_AUDIT_OVERRULES))
                        + " has failed")
    return ap


def _wmean(v, e):
    """Weighted mean, its error and chi-squared of ``v`` about it; None below two values."""
    from scipy.stats import chi2
    v, e = np.asarray(v, float), np.asarray(e, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return None
    w = 1 / e[ok] ** 2
    mu = float(np.sum(w * v[ok]) / w.sum())
    c = float(np.sum(((v[ok] - mu) / e[ok]) ** 2))
    d = int(ok.sum() - 1)
    return {"mean": mu, "err": float(1 / np.sqrt(w.sum())), "chi2": c, "dof": d,
            "p": float(chi2.sf(c, d))}


def _failed_tests(ctx) -> list:
    """Which of the overruled tests have a FAIL on record for this candidate."""
    out = []
    suffix = f"_c{ctx.candidate.index}" if ctx.candidate.index else ""
    for name in sorted(EPOCH_AUDIT_OVERRULES):
        path = ctx.out / ctx.target.slug / f"{name}{suffix}.json"
        try:
            if json.loads(path.read_text()).get("status") == Status.FAIL:
                out.append(name)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def verdict(epochs: list, oddeven_sigma) -> dict | None:
    """Judge the jointly fitted per-epoch depths.

    ``epochs`` is a list of ``{"depth", "depth_err"}`` (fractions).  Returns None
    when there is nothing to judge: fewer than ``MIN_EPOCHS`` epochs or a
    non-finite value.
    """
    E = [e for e in epochs if e and "depth" in e]
    if len(E) < MIN_EPOCHS:
        return None
    d = np.array([e["depth"] for e in E], float)
    s = np.array([e["depth_err"] for e in E], float)
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(s)) and np.all(s > 0)):
        return None
    mu = float(np.average(d, weights=1 / s ** 2))
    snr = mu / float(1 / np.sqrt(np.sum(1 / s ** 2)))
    ex = (float(np.sqrt(max(0.0, np.var(d, ddof=1) - np.median(s) ** 2)) / mu)
          if mu > 0 else float("inf"))
    oe = float(oddeven_sigma) if oddeven_sigma is not None and np.isfinite(oddeven_sigma) else None
    ok = mu > 0 and ex < MAX_EXCESS and snr >= MIN_SNR and (oe is None or oe < MAX_ODDEVEN)
    return {"pass": bool(ok), "n": len(E), "mean_depth": mu, "snr": float(snr),
            "excess": ex, "oddeven": oe}


def run(ctx: cli.Context) -> Result:
    from scipy.optimize import minimize

    r = ctx.result(TEST, VERSION)
    a = ctx.args
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r
    if not cli.require(r, ctx.candidate.duration_h,
                       "no transit duration: pass --duration or run pc_fit first"):
        return r

    failed = _failed_tests(ctx)
    r.metrics["overrules"] = sorted(EPOCH_AUDIT_OVERRULES)
    r.metrics["failed_on_record"] = failed
    if not failed and not getattr(a, "always", False):
        return r.inconclusive(
            "not run: none of " + ", ".join(sorted(EPOCH_AUDIT_OVERRULES))
            + " has failed, so there is nothing to overrule (--always runs it anyway)")

    P, T0 = float(ctx.candidate.period), float(ctx.candidate.t0_btjd)
    T14 = float(ctx.candidate.duration_h) / 24.0
    bin_s = float(np.clip(T14 * 86400 / 5, 120.0, 600.0))
    try:
        lc = ctx.lc
        p = pcprep.prepare(ctx, bin_seconds=bin_s)
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)
    r.set_noise(**p.noise_kwargs())
    if p.gp is None or not getattr(p.gp, "success", False):
        r.set_self_check(False, "no GP was fitted to this light curve")
        return r.inconclusive("no GP fitted: the joint per-epoch fit needs its hyperparameters")

    white = p.white_ppm / 1e6 if np.isfinite(p.white_ppm) else float(np.nanstd(p.f_detrended))
    ep = ephem.epoch_number(p.t, P, T0)
    dt = p.t - ephem.transit_time(ep, P, T0)
    exp = np.full(p.t.size, bin_s / 86400)
    y_masked = p.f_detrended - 1.0            # masked-GP detrend, for comparison
    f_raw = p.f / np.nanmedian(p.f) - 1.0     # what the joint fit sees

    # global shape on the masked detrend, for the shared ingress fraction
    win = (np.abs(dt) < pst.WINDOW_T14 * T14) & np.isfinite(y_masked)
    if win.sum() < 30:
        return r.inconclusive("too few points near the transits to fit a shape")
    inn = win & (np.abs(dt) < 0.4 * T14)
    d0 = max(-np.nanmedian(y_masked[inn]), 1e-5) if inn.any() else 1e-3
    gx, _, _ = pst._fit(dt[win], y_masked[win], np.full(win.sum(), white), exp[win],
                        (d0, T14, 0.3, 0.0), t14=T14)
    Q = float(np.clip(gx[2], 0.05, 1.0))

    # usable epochs: both halves of the transit covered
    n_exp = max(T14 / 2 / (bin_s / 86400), 1)
    usable = []
    for E in np.unique(ep[np.abs(dt) < T14 / 2]):
        m = (ep == E) & np.isfinite(f_raw)
        first = (m & (dt > -T14 / 2) & (dt <= 0)).sum() / n_exp
        second = (m & (dt > 0) & (dt < T14 / 2)).sum() / n_exp
        if min(first, second) >= MIN_COVERAGE:
            usable.append(int(E))
    n_covered = len(usable)
    if len(usable) > MAX_EPOCHS:
        usable = [usable[i] for i in
                  np.linspace(0, len(usable) - 1, MAX_EPOCHS).round().astype(int)]
    r.metrics.update({"n_epochs_covered": n_covered, "n_epochs_fitted": len(usable),
                      "ingress_fraction": Q, "duration_h": T14 * 24, "period": P})
    if len(usable) < MIN_EPOCHS:
        r.set_self_check(False, f"{len(usable)} fully covered epoch(s); {MIN_EPOCHS} needed")
        return r.inconclusive(f"only {len(usable)} fully covered transit(s): "
                              f"{MIN_EPOCHS} are needed to compare them")

    def trap(d, depth, total, centre, e):
        return pst._model(d, e, depth, total, Q, centre, 0.0)

    half = float(min(WINDOW_MAX_DAYS, max(3 * T14, 0.25)))
    rows = []
    for E in usable:
        row = {"epoch": E, "t_center": float(ephem.transit_time(E, P, T0))}
        mA = (ep == E) & (np.abs(dt) < pst.WINDOW_T14 * T14) & np.isfinite(y_masked)
        try:
            x, err, _ = pst._fit(dt[mA], y_masked[mA], np.full(mA.sum(), white), exp[mA],
                                 (gx[0], T14, Q, 0.0), free_q=False, q_fixed=Q, t14=T14)
            row["masked"] = dict(zip(("depth", "T14", "shift"), map(float, x[:3]))) | \
                dict(zip(("depth_err", "T14_err", "shift_err"), map(float, err[:3])))
        except Exception as exc:  # noqa: BLE001
            row["masked"] = {"error": repr(exc)}
        m = (ep == E) & (np.abs(dt) < half) & np.isfinite(f_raw)
        d, y, e = dt[m], f_raw[m], exp[m]
        try:
            gp = fitting.make_gp(p.gp, None, p.t[m], np.full(m.sum(), white))

            def nll(q, d=d, y=y, e=e, gp=gp):
                if not (0 < q[0] < 0.5 and 0.3 * T14 < q[1] < 2.5 * T14
                        and abs(q[2]) < 0.5 * T14):
                    return 1e25
                return -gp.log_likelihood(y - trap(d, q[0], q[1], q[2], e) - q[3])

            best = None
            for c0 in (-0.1 * T14, 0.1 * T14):
                for w0 in (0.85 * T14, 1.15 * T14):
                    s = minimize(nll, [max(gx[0], 1e-4), w0, c0, float(np.median(y))],
                                 method="Nelder-Mead",
                                 options={"xatol": 1e-6, "fatol": 1e-3, "maxiter": 4000})
                    if best is None or s.fun < best.fun:
                        best = s
            x = best.x
            h = np.array([max(abs(x[0]) * 0.02, 2e-5), T14 * 0.02, T14 * 0.01,
                          max(white * 0.1, 1e-5)])
            H = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    ei, ej = np.eye(4)[i] * h[i], np.eye(4)[j] * h[j]
                    H[i, j] = (nll(x + ei + ej) - nll(x + ei - ej) - nll(x - ei + ej)
                               + nll(x - ei - ej)) / (4 * h[i] * h[j])
            err = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
            row["joint"] = dict(zip(("depth", "T14", "shift"), map(float, x[:3]))) | \
                dict(zip(("depth_err", "T14_err", "shift_err"), map(float, err[:3])))
        except Exception as exc:  # noqa: BLE001
            row["joint"] = {"error": repr(exc)}
        rows.append(row)

    r.metrics["epochs"] = rows
    r.metrics["gp_amplitude_over_depth"] = _f(np.exp(p.gp.values[0]) / max(gx[0], 1e-6))
    oddeven = {}
    for tr in ("masked", "joint"):
        good = [q[tr] for q in rows if "depth" in q[tr]]
        for k in ("depth", "T14", "shift"):
            r.metrics[f"{tr}_{k}"] = _wmean([g[k] for g in good], [g[k + "_err"] for g in good])
        odd = [q[tr] for q in rows if "depth" in q[tr] and q["epoch"] % 2]
        even = [q[tr] for q in rows if "depth" in q[tr] and not q["epoch"] % 2]
        o = _wmean([g["depth"] for g in odd], [g["depth_err"] for g in odd])
        ev = _wmean([g["depth"] for g in even], [g["depth_err"] for g in even])
        if len(odd) == 1:
            o = {"mean": odd[0]["depth"], "err": odd[0]["depth_err"]}
        if len(even) == 1:
            ev = {"mean": even[0]["depth"], "err": even[0]["depth_err"]}
        oddeven[tr] = (_f(abs(o["mean"] - ev["mean"]) / np.hypot(o["err"], ev["err"]))
                       if o and ev else None)
        r.metrics[f"{tr}_oddeven_sigma"] = oddeven[tr]

    v = verdict([q["joint"] for q in rows], oddeven["joint"])
    if v is None:
        r.set_self_check(False, "fewer than three epochs gave a finite joint fit")
        return r.inconclusive("the joint fit converged on fewer than "
                              f"{MIN_EPOCHS} epochs: nothing to compare")
    r.metrics.update({
        "n_epochs_judged": v["n"],
        "mean_depth_ppm": _f(v["mean_depth"] * 1e6),
        "depth_snr": _f(v["snr"]),
        "depth_excess_scatter": _f(v["excess"]),
        "oddeven_sigma": v["oddeven"],
        "oddeven_consistent": (v["oddeven"] is not None and v["oddeven"] < MAX_ODDEVEN),
        "thresholds": {"min_epochs": MIN_EPOCHS, "max_excess": MAX_EXCESS,
                       "max_oddeven_sigma": MAX_ODDEVEN, "min_snr": MIN_SNR},
    })
    r.set_self_check(True, f"{v['n']} transits fitted jointly with the GP "
                           f"(of {n_covered} fully covered)")

    oe_txt = "no odd/even comparison" if v["oddeven"] is None else f"odd/even {v['oddeven']:.1f} sigma"
    summary = (f"{v['n']} transits fitted with the star: mean depth "
               f"{v['mean_depth'] * 1e6:.0f} ppm at {v['snr']:.1f} sigma, excess scatter "
               f"{v['excess']:.0%} of the depth, {oe_txt}")
    if v["pass"]:
        over = (f"; overrules the FAIL of {', '.join(failed)}" if failed else "")
        r.passed(summary + over)
    else:
        why = []
        if not v["mean_depth"] > 0:
            why.append("the mean depth is not positive")
        elif v["snr"] < MIN_SNR:
            why.append(f"the mean depth is below {MIN_SNR:g} sigma")
        if v["excess"] >= MAX_EXCESS:
            why.append(f"the depths scatter by {MAX_EXCESS:.0%} of the depth or more beyond their errors")
        if v["oddeven"] is not None and v["oddeven"] >= MAX_ODDEVEN:
            why.append(f"odd and even transits differ by {MAX_ODDEVEN:g} sigma or more")
        r.fail("; ".join(why) + " -- " + summary)

    if ctx.make_plots:
        try:
            _plot(ctx, r, rows, v)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, rows, v):
    fig, ax = figure(figsize=(7.5, 4.6))
    for tr, colour, off, label in (("masked", C_BINNED, -0.12, "transits masked from the GP"),
                                   ("joint", C_MODEL, 0.12, "GP and transit fitted together")):
        g = [(q["epoch"], q[tr]["depth"] * 1e6, q[tr]["depth_err"] * 1e6)
             for q in rows if "depth" in q[tr]]
        if g:
            e, d, s = map(np.array, zip(*g))
            x = np.searchsorted(sorted({q["epoch"] for q in rows}), e) + off
            ax.errorbar(x, d, s, fmt="o", color=colour, ms=4, lw=1, label=label)
    ax.axhline(v["mean_depth"] * 1e6, color=C_MODEL, lw=1, ls="--")
    ax.axhline(0, color="0.6", lw=0.8)
    eps = sorted({q["epoch"] for q in rows})
    step = max(1, len(eps) // 12)
    ax.set_xticks(range(0, len(eps), step))
    ax.set_xticklabels([str(e) for e in eps[::step]])
    ax.set_xlabel("transit epoch")
    ax.set_ylabel("depth [ppm]")
    ax.legend(loc="upper left", fontsize=8)
    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path(TEST)))


def _f(x):
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Do the transits agree with each other when each is fitted "
              "together with the star's variability?"))
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

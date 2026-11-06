"""pc_shape -- is the transit profile planet-like, or V-shaped?

Fits a **trapezoid** to the folded transit: depth, total duration T14, and
ingress duration T12.  The quantity of interest is the ingress fraction
``T12 / T14``, which is small for a planet crossing a stellar disc and tends
to 0.5 for a grazing or blended eclipse that never reaches a flat bottom.

A trapezoid rather than a limb-darkened transit model on purpose.  The
question here is "what shape is the data", not "which planet model fits best",
and a physical model with a free impact parameter will happily answer the
second question with ``b = 1.2`` and an 8% depth -- several 2026 candidates
optimised to exactly that, including one at 83876 ppm.  A trapezoid cannot
hide a bad shape inside a degenerate parameter.

The measured ingress fraction is reported next to the value the *fitted*
geometry predicts, so a reader can see whether the shape is consistent with
the impact parameter the rest of the pipeline settled on.

Limb darkening rounds a real transit's shoulders, so a modest ingress fraction
is expected even at ``b = 0``: for a Sun-like star in the TESS band the
trapezoid equivalent is around 0.15-0.2 for a central crossing.  The threshold
here is set well above that and **flags rather than fails** -- a grazing planet
is still a planet, and the depth-and-radius consequences are for a human.

    python -m petcare.scripts.pc_shape --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import least_squares

from .. import cli, prep as pcprep
from ..plotting import (C_BINNED, C_MODEL, C_OK, annotate_status, bin_series,
                        figure, save)
from ..result import Result

TEST = "shape"
VERSION = "0.1.0"

#: Ingress fraction above which the profile is V-shaped enough to look at.
#: PROVISIONAL.  A central limb-darkened transit gives ~0.15-0.20 as a
#: trapezoid; 0.35 is comfortably above that and below the 0.5 of a true V.
INGRESS_FRACTION_FLAG = 0.35

#: And the value at which there is effectively no flat bottom at all.
INGRESS_FRACTION_STRONG = 0.45

#: Below this many in-transit cadences the trapezoid corners are not resolved
#: and the fit is reporting the binning, not the star.
MIN_IN_TRANSIT = 12


def add_arguments(ap):
    g = ap.add_argument_group("shape")
    g.add_argument("--bin", type=float, default=300.0,
                   help="bin the light curve to this many seconds; finer than "
                        "the other tests because this one measures the corners")
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
        p = pcprep.prepare(ctx, bin_seconds=float(getattr(a, "bin", 300.0)))
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"could not prepare the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    ctx.notes.extend(p.notes)
    r.set_noise(**p.noise_kwargs())

    ph = p.phase()
    win = min(2.5 * p.duration_days, 0.45 * p.period)
    sel = np.abs(ph) < win
    x = ph[sel]
    y = p.f_detrended[sel]
    n_in = int((np.abs(x) < 0.5 * p.duration_days).sum())

    if not cli.require(r, n_in >= MIN_IN_TRANSIT,
                       f"only {n_in} in-transit cadences at this binning; "
                       f"{MIN_IN_TRANSIT} are needed to resolve the ingress"):
        r.set_self_check(False, f"{n_in} in-transit cadences")
        return r

    fit = _fit_trapezoid(x, y, p.duration_days)
    if not cli.require(r, fit["success"],
                       f"trapezoid fit did not converge: {fit['reason']}"):
        return r

    t14 = fit["t14"]
    t12 = fit["t12"]
    frac = t12 / t14 if t14 > 0 else np.nan
    depth_ppm = fit["depth"] * 1e6

    # what the fitted geometry would predict, for comparison
    geom = _expected_ingress_fraction(ctx)

    # a straight comparison of models, so "V-shaped" is not just a number
    flat = _fit_trapezoid(x, y, p.duration_days, force_v=True)
    dchi2 = (flat["chi2"] - fit["chi2"]) if fit["chi2"] and flat["chi2"] else None

    r.metrics.update({
        "t14_h": t14 * 24, "t14_err_h": _f(fit["t14_err"] * 24),
        "t12_h": t12 * 24, "t12_err_h": _f(fit["t12_err"] * 24),
        "ingress_fraction": _f(frac),
        "ingress_fraction_err": _f(
            abs(frac) * np.hypot(fit["t12_err"] / t12 if t12 else np.nan,
                                 fit["t14_err"] / t14 if t14 else np.nan)),
        "trapezoid_depth_ppm": _f(depth_ppm),
        "trapezoid_depth_err_ppm": _f(fit["depth_err"] * 1e6),
        "t0_offset_h": _f(fit["t0_off"] * 24),
        "chi2_trapezoid": _f(fit["chi2"]),
        "chi2_pure_v": _f(flat["chi2"]),
        "delta_chi2_v_minus_trapezoid": _f(dchi2),
        "expected_ingress_fraction": geom,
        "n_in_transit": n_in,
        "bin_seconds": float(getattr(a, "bin", 300.0)),
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{n_in} in-transit cadences at "
                  f"{getattr(a, 'bin', 300.0):.0f} s"], True
    snr = depth_ppm / (fit["depth_err"] * 1e6) if fit["depth_err"] > 0 else np.nan
    checks.append(f"trapezoid depth S/N {snr:.1f}")
    if not np.isfinite(snr) or snr < 5:
        checks.append("depth is not significant enough for the shape to mean anything")
        ok = False
    # ingress shorter than a cadence is a binning artefact, not a measurement
    cad_days = float(getattr(a, "bin", 300.0)) / 86400.0
    if t12 < cad_days:
        checks.append(f"fitted ingress ({t12 * 24 * 60:.1f} min) is shorter "
                      f"than one bin ({cad_days * 24 * 60:.1f} min)")
        ok = False
    if abs(t14 - p.duration_days) > 0.6 * p.duration_days:
        checks.append(f"fitted T14 {t14 * 24:.2f} h disagrees with the input "
                      f"{p.duration_h:.2f} h by more than 60%")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    note = (f"depth {depth_ppm:.0f} +/- {fit['depth_err'] * 1e6:.0f} ppm, "
            f"T14 = {t14 * 24:.2f} h, ingress {t12 * 24 * 60:.0f} min; "
            f"point-to-point noise {p.white_ppm:.0f} ppm")
    if geom.get("value") is not None:
        note += f"; fitted geometry predicts {geom['value']:.2f}"

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif frac >= INGRESS_FRACTION_STRONG:
        r.flag(f"V-shaped: ingress is {frac:.2f} of the total duration, with "
               f"essentially no flat bottom. Grazing or blended eclipses look "
               f"like this -- but so does a grazing planet. {note}")
    elif frac >= INGRESS_FRACTION_FLAG:
        r.flag(f"ingress fraction {frac:.2f} is high for a planet on a "
               f"non-grazing orbit. {note}")
    else:
        r.passed(f"box-like profile: ingress is {frac:.2f} of the duration. {note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, p, x, y, fit)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


def trapezoid(x, t0, depth, t14, t12):
    """Symmetric trapezoid: flat bottom of ``t14 - 2*t12``, linear wings."""
    d = np.abs(np.asarray(x, float) - t0)
    half = 0.5 * t14
    inner = max(half - t12, 0.0)
    y = np.ones_like(d)
    y[d <= inner] = 1.0 - depth
    ramp = (d > inner) & (d < half)
    if t12 > 0:
        y[ramp] = 1.0 - depth * (half - d[ramp]) / t12
    return y


def _fit_trapezoid(x, y, dur_guess, force_v: bool = False):
    """Least-squares trapezoid, or a pure V (``t12 = t14/2``) when forced."""
    depth0 = max(float(np.median(y[np.abs(x) > 0.75 * dur_guess]) -
                       np.mean(y[np.abs(x) < 0.3 * dur_guess])), 1e-5)
    if force_v:
        def resid(theta):
            t0, depth, t14 = theta
            return trapezoid(x, t0, depth, t14, 0.5 * t14) - y
        theta0 = [0.0, depth0, dur_guess]
        bounds = ([-0.4 * dur_guess, 1e-6, 0.2 * dur_guess],
                  [0.4 * dur_guess, 0.5, 3.0 * dur_guess])
    else:
        def resid(theta):
            t0, depth, t14, t12 = theta
            return trapezoid(x, t0, depth, t14, t12) - y
        theta0 = [0.0, depth0, dur_guess, 0.15 * dur_guess]
        bounds = ([-0.4 * dur_guess, 1e-6, 0.2 * dur_guess, 1e-6],
                  [0.4 * dur_guess, 0.5, 3.0 * dur_guess, 1.5 * dur_guess])

    try:
        res = least_squares(resid, theta0, bounds=bounds, max_nfev=4000)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "reason": str(exc), "chi2": None}

    chi2 = float(np.sum(res.fun ** 2))
    # parameter errors from the Jacobian at the solution
    try:
        n, k = len(x), len(theta0)
        s2 = chi2 / max(n - k, 1)
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        err = np.sqrt(np.abs(np.diag(cov)))
    except (np.linalg.LinAlgError, ValueError):
        err = np.full(len(theta0), np.nan)

    if force_v:
        t0, depth, t14 = res.x
        t12, t12_err = 0.5 * t14, 0.5 * err[2]
        errs = (err[0], err[1], err[2], t12_err)
    else:
        t0, depth, t14, t12 = res.x
        errs = tuple(err)

    # A trapezoid with t12 > t14/2 is not a trapezoid; clip and say so.
    capped = t12 > 0.5 * t14
    t12 = min(t12, 0.5 * t14)

    return {"success": True, "reason": "", "t0_off": float(t0),
            "depth": float(depth), "t14": float(t14), "t12": float(t12),
            "t0_err": _f(errs[0]), "depth_err": float(errs[1]) if np.isfinite(errs[1]) else np.nan,
            "t14_err": float(errs[2]) if np.isfinite(errs[2]) else np.nan,
            "t12_err": float(errs[3]) if np.isfinite(errs[3]) else np.nan,
            "chi2": chi2, "ingress_capped_at_v": bool(capped)}


def _expected_ingress_fraction(ctx) -> dict:
    """What ``T12/T14`` the fitted ``(b, Rp/R*)`` geometry implies.

    For a non-grazing transit ``T12/T14 ~ k / sqrt((1+k)^2 - b^2)`` to a good
    approximation, ignoring limb darkening -- which is why the measured value
    is normally a little larger than this.
    """
    import json
    path = ctx.out / ctx.target.slug / (
        f"fit_c{ctx.candidate.index}.json" if ctx.candidate.index else "fit.json")
    if not path.exists():
        return {"value": None, "reason": "no fit.json; run pc_fit first"}
    try:
        m = (json.loads(path.read_text()).get("metrics") or {})
    except (OSError, json.JSONDecodeError) as exc:
        return {"value": None, "reason": f"could not read fit.json: {exc}"}
    k, b = m.get("rp_rs"), m.get("b")
    if k is None or b is None:
        return {"value": None, "reason": "fit.json has no rp_rs/b"}
    arg = (1 + k) ** 2 - b ** 2
    if arg <= 0:
        return {"value": 0.5, "b": b, "rp_rs": k,
                "reason": "b > 1 + k: fully grazing, no flat bottom"}
    return {"value": float(k / np.sqrt(arg)), "b": float(b), "rp_rs": float(k),
            "reason": "k / sqrt((1+k)^2 - b^2), limb darkening neglected"}


def _plot(ctx, r, p, x, y, fit):
    fig, ax = figure(figsize=(8.5, 5))
    yy = (y - 1) * 1e6
    ax.plot(x * 24, yy, ".", ms=1.6, color="#4c4c4c", alpha=0.25, rasterized=True)
    xb, yb, eb, _ = bin_series(x * 24, yy, max(p.duration_h / 25, 0.05), min_points=4)
    ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=3.4, color=C_BINNED, lw=0.9,
                label="binned")

    xx = np.linspace(x.min(), x.max(), 800)
    model = (trapezoid(xx, fit["t0_off"], fit["depth"], fit["t14"], fit["t12"]) - 1) * 1e6
    ax.plot(xx * 24, model, "-", color=C_MODEL, lw=1.7,
            label=(f"trapezoid: T14 {fit['t14'] * 24:.2f} h, "
                   f"ingress {fit['t12'] * 24 * 60:.0f} min "
                   f"({fit['t12'] / fit['t14']:.2f} of T14)"))
    v = (trapezoid(xx, fit["t0_off"], fit["depth"], fit["t14"],
                   0.5 * fit["t14"]) - 1) * 1e6
    ax.plot(xx * 24, v, "--", color="#999999", lw=1.1, label="pure V, same T14")

    half = 0.5 * fit["t14"] * 24
    for s in (-1, 1):
        ax.axvline(s * half, color=C_MODEL, ls=":", lw=0.8, alpha=0.6)
        ax.axvline(s * (half - fit["t12"] * 24), color=C_OK, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title("transit profile and the trapezoid fitted to it")
    ax.legend(loc="lower right")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("shape")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Is the transit profile box-like or V-shaped? Fits a trapezoid "
              "and reports the ingress fraction."))
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

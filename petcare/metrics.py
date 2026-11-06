"""Vetting metrics: one flat set of numbers per candidate, reported and never graded.

The Kepler Robovetter (Thompson et al. 2018) and LEO-vetter (Kunimoto et al.
2025) refute on fixed cuts tuned to old, quiet stars, and several of those cuts
remove real young planets (odd/even at 3 sigma alone costs 18% of the PetCare
benchmark).  So this module only collects: the generic numbers those vetters
use, the ones that matter for young stars, and the scalars the PetCare tests already measure, under one stable
set of names.  Cuts come later, set on known young planets, eye-vetted
candidates, false positives and injections.

Everything here is computed from the tests' own results (a merged petcare.json
record), so it can be recomputed for any candidate without re-running a test.
A metric whose inputs are missing is None.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "0.1.0"

#: Free parameters of the transit model, for the BIC (t0, P, Rp/R*, a/R*, b).
N_TRANSIT_PARAMS = 5


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _m(rec, test, *path):
    """A metric from one test's result, walking nested keys; None if absent."""
    x = ((rec.get("tests") or {}).get(test) or {}).get("metrics") or {}
    for p in path:
        x = x.get(p) if isinstance(x, dict) else None
    return x


def _ratio(a, b):
    a, b = _f(a), _f(b)
    return a / b if (a is not None and b) else None


def _weighted(d, e):
    w = 1.0 / e ** 2
    mean = float(np.sum(w * d) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    return mean, err


def epoch_metrics(epochs: list) -> dict:
    """From pc_depths' per-transit depths: how much the signal rests on a few events.

    LEO-vetter's single-event share and SNR consistency, the k-fold S/N
    (drop each transit in turn), the depth with the worst 1 and 2 transits
    removed (worst = furthest from the mean in its own errors), and the
    scatter of the depths about their mean.
    """
    d = np.array([_f(e.get("depth_ppm")) for e in epochs], dtype=float)
    e = np.array([_f(e.get("depth_err_ppm")) for e in epochs], dtype=float)
    ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
    d, e = d[ok], e[ok]
    n = int(d.size)
    out = {"n_epochs": n}
    if n == 0:
        return out
    mean, err = _weighted(d, e)
    ses = d / e
    out.update({
        "mean_depth_ppm": mean, "mean_depth_err_ppm": err, "snr": mean / err,
        "median_depth_ppm": float(np.median(d)),
        "mean_over_median_depth": _ratio(mean, np.median(d)),
        "frac_negative": float(np.mean(d < 0)),
        "max_single_event_snr": float(ses.max()),
        # the largest single event's share of the combined S/N^2: 1 means one transit is the signal
        "max_event_share": float(ses.max() ** 2 / np.sum(ses ** 2)) if np.any(ses > 0) else None,
        "chi2_red": float(np.sum(((d - mean) / e) ** 2) / (n - 1)) if n > 1 else None,
    })
    if n >= 2:
        loo = np.array([np.divide(*_weighted(np.delete(d, i), np.delete(e, i))) for i in range(n)])
        out["kfold_snr_min"], out["kfold_snr_median"] = float(loo.min()), float(np.median(loo))
    worst = np.argsort(-np.abs((d - mean) / e))
    for k in (1, 2):
        if n - k >= 1:
            keep = np.setdiff1d(np.arange(n), worst[:k])
            m, s = _weighted(d[keep], e[keep])
            out[f"drop{k}_depth_ppm"], out[f"drop{k}_snr"] = m, m / s
    return out


def flare_proximity(rec, epochs: list, t14_days: float | None) -> dict:
    """Time from each transit centre to the nearest flare, in transit durations."""
    iv = _m(rec, "flares", "flare_intervals")
    tc = np.array([_f(e.get("t_center")) for e in epochs if _f(e.get("t_center")) is not None])
    if not iv or tc.size == 0 or not t14_days:
        return {}
    a = np.array([[x[0], x[1]] for x in iv if len(x) >= 2], dtype=float)
    if a.size == 0:
        return {}
    dist = np.array([0.0 if np.any((a[:, 0] <= t) & (t <= a[:, 1]))
                     else np.min(np.minimum(np.abs(a[:, 0] - t), np.abs(a[:, 1] - t))) for t in tc])
    dist = dist / t14_days
    return {"nearest_flare_min_t14": float(dist.min()), "nearest_flare_median_t14": float(np.median(dist)),
            "frac_transits_flare_within_1t14": float(np.mean(dist < 1.0)),
            "flare_list_truncated": bool(_m(rec, "flares", "flare_intervals_truncated"))}


def collect(rec: dict) -> dict:
    """Every metric for one candidate, from its merged record."""
    tests = rec.get("tests") or {}
    fit = (tests.get("fit") or {})
    fit_ok = fit.get("status") in ("PASS", "FLAG")
    noise = fit.get("noise") or {}
    depth, derr = _f(_m(rec, "fit", "depth_ppm")), _f(_m(rec, "fit", "depth_err_ppm"))
    t14_h = _f(_m(rec, "fit", "duration_h")) or _f((rec.get("candidate") or {}).get("duration"))
    n_in = _f(_m(rec, "fit", "fit_detail", "n_in_transit"))
    white = _f(noise.get("white_ppm"))
    dlogl = _f(_m(rec, "fit", "delta_logl_vs_no_transit"))
    n_pts = _f(((tests.get("fit") or {}).get("provenance") or {}).get("n_points"))
    epochs = _m(rec, "depths", "epochs") or []
    ms = _m(rec, "uniqueness", "modshift") or {}
    sp, ss, st_, spos = (_f(ms.get(k)) for k in ("sigma_pri", "sigma_sec", "sigma_ter", "sigma_pos"))
    fred, fa1, fa2 = _f(ms.get("Fred")), _f(ms.get("FA1")), _f(ms.get("FA2"))
    prot = _f(_m(rec, "rotation", "prot_days"))
    period = _f(_m(rec, "fit", "period")) or _f((rec.get("candidate") or {}).get("period"))
    rp_rs, b = _f(_m(rec, "fit", "rp_rs")), _f(_m(rec, "fit", "b"))

    m = {
        # -- the fit, and detection ---------------------------------------------------
        "fit_converged": bool(fit_ok),
        "period_d": period, "t0_btjd": _f(_m(rec, "fit", "t0_btjd")),
        "depth_ppm": depth, "depth_err_ppm": derr,
        "t14_h": t14_h, "rp_rs": rp_rs, "a_rs": _f(_m(rec, "fit", "a_rs")), "b": b,
        "rp_rs_plus_b": rp_rs + b if None not in (rp_rs, b) else None,   # > 1: grazing
        "railed": ",".join(_m(rec, "fit", "fit_detail", "railed") or []) or None,
        # red-noise S/N (the fit's error is inflated for red noise) and the white-noise one
        "snr_red": _ratio(depth, derr),
        "snr_white": (depth / (white / math.sqrt(n_in)) if None not in (depth, white, n_in) and white and n_in
                      else None),
        "beta_red": _f(noise.get("red_noise_factor")),
        "delta_logl": dlogl,
        "delta_bic": (2 * dlogl - N_TRANSIT_PARAMS * math.log(n_pts)) if (dlogl is not None and n_pts) else None,
        "ephemeris_shift_t14": _f(_m(rec, "fit", "ephemeris_drift", "shift_at_tref_durations")),
        # -- shape --------------------------------------------------------------------
        "vshape": _f(_m(rec, "shape", "ingress_fraction")),          # tau / T14
        "vshape_err": _f(_m(rec, "shape", "ingress_fraction_err")),
        "dchi2_v_minus_trapezoid": _f(_m(rec, "shape", "delta_chi2_v_minus_trapezoid")),
        "shape_t0_offset_h": _f(_m(rec, "shape", "t0_offset_h")),
        # -- the star ------------------------------------------------------------------
        "rho_ratio": _ratio(_m(rec, "duration", "rho_transit_gcc"), _m(rec, "duration", "rho_star_gcc")),
        "duration_over_central": _f(_m(rec, "duration", "duration_ratio_observed_over_central")),
        "teff_k": _f(_m(rec, "stellar", "teff_k")), "radius_rsun": _f(_m(rec, "stellar", "radius_rsun")),
        "mass_msun": _f(_m(rec, "stellar", "mass_msun")), "age_myr": _f(_m(rec, "stellar", "age_myr")),
        "tmag": _f(_m(rec, "stellar", "tmag")),
        "stellar_source": _m(rec, "stellar", "parameter_source"),
        "rp_earth": _f(_m(rec, "companion", "rp_earth")),
        "dilution": _f(((tests.get("fit") or {}).get("dilution") or {}).get("factor")),
        # -- rotation and flares --------------------------------------------------------
        "prot_days": prot,
        "period_over_prot": _ratio(period, prot),
        "t14_over_prot": _ratio(t14_h / 24 if t14_h else None, prot),
        "harmonic_offset_frac": _f(_m(rec, "rotation", "closest_harmonic", "difference_frac")),
        "rot_amplitude_ppm": _f(_m(rec, "fit", "gp_amplitude_ppm")),
        "depth_vs_rot_phase_sigma": _f(_m(rec, "rotation", "depth_vs_rotation_phase", "sigma")),
        "flare_rate_per_day": _f(_m(rec, "flares", "flare_rate_per_day")),
        "flare_frac_in_transit": _f(_m(rec, "flares", "in_transit_flare_fraction")),
        "flare_depth_change_frac": _f(_m(rec, "flares", "depth_fractional_change")),
        "egress_rise_sigma": _f(_m(rec, "flares", "egress_post_rise_sig")),
        # -- consistency between transits and methods ----------------------------------
        "oddeven_sigma": _f(_m(rec, "odd_even", "difference_sigma")),
        "secondary_best_sigma": _f(_m(rec, "secondary", "best_sigma")),
        "secondary_best_phase": _f(_m(rec, "secondary", "best_phase")),
        "secondary_sigma_half": _f(_m(rec, "secondary", "z_at_phase_0.5")),
        "secondary_over_primary": _f(_m(rec, "secondary", "secondary_over_primary")),
        "shapetime_depth_extra": _f(_m(rec, "shapetime", "depth_extra_frac")),
        "shapetime_duration_extra": _f(_m(rec, "shapetime", "duration_extra_frac")),
        "jointgp_z": _f(_m(rec, "jointgp", "z")),
        "jointgp_over_fit_depth": _ratio(_m(rec, "jointgp", "depth_ppm"), depth),
        "jointgp_consistency_p": _f(_m(rec, "jointgp", "consistency_p")),
        "jointgp_degeneracy": _f(_m(rec, "jointgp", "gp_degeneracy")),
        "crossmatch_spread_frac": _f(_m(rec, "crossmatch", "depth_spread_fraction")),
        "crossmatch_p": _f(_m(rec, "crossmatch", "chi2_p_value")),
        # -- against the star's own noise -----------------------------------------------
        "interp_z": _f(_m(rec, "interp", "z_corrected")),
        "interp_inflation": _f(_m(rec, "interp", "interp_inflation")),
        "interp_shoulder_over_depth": _f(_m(rec, "interp", "shoulder_over_depth")),
        "interp_power": _f(_m(rec, "interp", "injection_power", "power")),
        "MS1": _f(_m(rec, "uniqueness", "MS1")), "MS2": _f(_m(rec, "uniqueness", "MS2")),
        "MS3": _f(_m(rec, "uniqueness", "MS3")),
        # Thompson et al. 2018 A.3.4: is the secondary itself significant and unique?
        "MS4": ss / fred - fa1 if None not in (ss, fred, fa1) and fred else None,
        "MS5": ss - st_ - fa2 if None not in (ss, st_, fa2) else None,
        "MS6": ss - spos - fa2 if None not in (ss, spos, fa2) else None,
        "Fred": fred, "SHP": _f(_m(rec, "uniqueness", "SHP")),
        "uniqueness_power": _f(_m(rec, "uniqueness", "injection_power", "power")),
        # -- position ---------------------------------------------------------------------
        "centroid_snr": _f(_m(rec, "centroid", "combined_snr")),
        "centroid_offset_arcsec": _f(_m(rec, "centroid", "combined_offset_arcsec")),
        # -- coverage -----------------------------------------------------------------------
        "frac_epochs_near_gap": _f(_m(rec, "instrument", "epochs", "frac_near_a_gap")),
        "frac_epochs_partial": _f(_m(rec, "instrument", "epochs", "frac_missing_ingress_or_egress")),
        "frac_epochs_near_sector_edge": _f(_m(rec, "instrument", "epochs", "frac_near_sector_edge")),
    }
    m.update({f"epochs_{k}": v for k, v in epoch_metrics(epochs).items()})
    m.update(flare_proximity(rec, epochs, t14_h / 24 if t14_h else None))
    m["statuses"] = {t: r.get("status") for t, r in tests.items()}
    return {"version": VERSION, "metrics": m}

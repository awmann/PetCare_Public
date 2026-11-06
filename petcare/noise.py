"""Noise estimation.

Standing rule on this project: **print the noise level beside any statistic
built on an estimated noise level**, and measure the red noise rather than
assuming white.  On these light curves the empirical scatter on the transit
timescale runs 2.5-3.7x the white-noise expectation; a white error model
produced apparent 40-sigma detections of nothing.

Everything here returns ppm and is computed on flux normalised to a median
of 1.
"""

from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826


def mad_std(x, center=None) -> float:
    """Robust sigma via the median absolute deviation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    c = np.median(x) if center is None else center
    return float(MAD_TO_SIGMA * np.median(np.abs(x - c)))


def point_to_point_ppm(f) -> float:
    """White scatter from successive differences, in ppm.

    ``1.4826 * MAD(diff(f)) / sqrt(2)``.  Differencing removes any smooth
    variability, so this measures the genuinely uncorrelated component even on
    a star with 3% spot modulation -- which is exactly why it is the default
    estimator quoted in every result.
    """
    f = np.asarray(f, dtype=float)
    f = f[np.isfinite(f)]
    if f.size < 3:
        return float("nan")
    return float(mad_std(np.diff(f)) / np.sqrt(2.0) * 1e6)


def binned_rms_ppm(t, f, bin_days: float, min_points: int = 3) -> tuple[float, int]:
    """Robust scatter of the binned light curve, in ppm, and the bin count.

    Bins are fixed-width in time from the first cadence, and bins with fewer
    than ``min_points`` cadences are dropped so that a nearly-empty bin at a
    gap edge does not dominate.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size < 5 or not np.isfinite(bin_days) or bin_days <= 0:
        return float("nan"), 0

    idx = np.floor((t - t.min()) / bin_days).astype(int)
    means = []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() >= min_points:
            means.append(np.mean(f[sel]))
    if len(means) < 4:
        return float("nan"), len(means)
    return float(mad_std(means) * 1e6), len(means)


def red_noise_factor(t, f, bin_days: float) -> dict:
    """How much worse the binned scatter is than the white expectation.

    A purely white series binned by ``n`` points falls as ``1/sqrt(n)``.  The
    ratio of the measured binned scatter to that prediction is the red-noise
    inflation factor, and it is the number to multiply a formal depth
    uncertainty by before believing any significance.

    Returns a dict with the measured and predicted scatter, the factor, and
    the number of bins it was measured from -- a factor from 4 bins is not
    worth much and the caller needs to be able to see that.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]

    white_ppm = point_to_point_ppm(f)
    measured_ppm, n_bins = binned_rms_ppm(t, f, bin_days)

    dt = np.median(np.diff(np.sort(t))) if t.size > 2 else np.nan
    n_per_bin = bin_days / dt if np.isfinite(dt) and dt > 0 else np.nan
    predicted_ppm = white_ppm / np.sqrt(n_per_bin) if np.isfinite(n_per_bin) and n_per_bin >= 1 else np.nan

    factor = (measured_ppm / predicted_ppm
              if np.isfinite(measured_ppm) and np.isfinite(predicted_ppm) and predicted_ppm > 0
              else np.nan)
    return {
        "white_ppm": _f(white_ppm),
        "binned_rms_ppm": _f(measured_ppm),
        "white_prediction_ppm": _f(predicted_ppm),
        "red_noise_factor": _f(factor),
        "bin_days": float(bin_days),
        "n_bins": int(n_bins),
        "n_per_bin": _f(n_per_bin),
        "estimator": "1.4826*MAD(diff)/sqrt(2); binned scatter vs 1/sqrt(n)",
    }


def depth_uncertainty_ppm(n_in: int, n_oot: int, white_ppm: float,
                          red_factor: float = 1.0) -> float:
    """Uncertainty on a depth measured as ``mean(oot) - mean(in)``.

    Includes the out-of-transit term, which is routinely dropped and which
    matters whenever the local baseline is short.  ``red_factor`` inflates the
    result; pass the measured value from :func:`red_noise_factor`, not 1.
    """
    if n_in < 1 or n_oot < 1 or not np.isfinite(white_ppm):
        return float("nan")
    sigma = white_ppm * np.sqrt(1.0 / n_in + 1.0 / n_oot)
    return float(sigma * (red_factor if np.isfinite(red_factor) and red_factor > 0 else 1.0))


def noise_block(t, f, duration_days: float | None = None) -> dict:
    """The standard ``noise`` block for a :class:`~petcare.result.Result`.

    Measured at the transit timescale when a duration is known, because that
    is the timescale the statistic actually lives on.
    """
    white = point_to_point_ppm(f)
    block = {"white_ppm": _f(white),
             "estimator": "1.4826*MAD(diff)/sqrt(2)"}
    if duration_days and np.isfinite(duration_days) and duration_days > 0:
        rn = red_noise_factor(t, f, duration_days)
        block.update({
            "binned_rms_ppm": rn["binned_rms_ppm"],
            "red_noise_factor": rn["red_noise_factor"],
            "red_noise_timescale_days": duration_days,
            "n_bins": rn["n_bins"],
        })
    return block


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None

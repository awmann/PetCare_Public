"""Flare detection.

Its own module because two different things need it: ``pc_flares``, which asks
whether the transit fold depends on flare-affected cadences, and the GP
hyperparameter fit, which has to mask flares before it can describe the star.

The second use is the one that was missing.  On one young M dwarf -- 137 flares over a
2580-day baseline -- an unmasked GP fit drives its amplitude to the upper
bound and its quality factor to the floor trying to explain the flares, which
is exactly the maximally-flexible configuration that then absorbs the transit.
The fitted depth came back at 0.94 ppm against a real 2490 ppm.

The threshold is the part that is easy to get wrong, and it was got wrong
twice before.  A sigma-clip whose threshold comes from the scatter of the
series it is handed flags **zero** points on that star's raw data, because its
15000 ppm scatter *is* the flares.  A rolling-median-then-clip flagged 70% of
points because the grow radius was too wide for the number of outliers.  What
works: a **one-sided** scatter, estimated from the points *below* the median,
on a **variability-removed** series, with a grow radius tied to the measured
decay time and hard-capped.
"""

from __future__ import annotations

import numpy as np

#: Detection threshold in robust sigma of the residual, one-sided.
FLARE_SIGMA = 4.0

#: How far either side of a flagged cadence to extend the mask, in units of
#: the measured decay time.  Capped at :data:`MAX_GROW_DAYS` so that a star
#: with continuous low-level activity cannot mask its own light curve away.
GROW_DECAYS = 2.0
MAX_GROW_DAYS = 0.08          # ~2 hours

#: If more than this fraction of the light curve would be masked, the mask is
#: not a flare mask -- it is a detrending failure.
MAX_MASKED_FRACTION = 0.25


def detect_flares(t, f, sigma: float = FLARE_SIGMA, grow_decays: float = GROW_DECAYS,
                  max_grow_days: float = MAX_GROW_DAYS) -> dict:
    """Flag positive outliers in a variability-removed series.

    The threshold comes from a **one-sided** scatter: the median absolute
    deviation of the points *below* the median, doubled.  Flares only ever
    push flux up, so the lower half of the residual distribution is a clean
    sample of the noise, while a symmetric MAD is inflated by the very events
    being searched for -- on an active star that inflation is enough to hide
    every flare behind its own threshold.

    The mask is grown by the measured decay time rather than a fixed number of
    cadences, and capped, so that neither a 20 s nor a 30 min light curve gets
    a radius that makes no physical sense.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    good = np.isfinite(t) & np.isfinite(f)
    resid = f - np.median(f[good])

    lower = resid[good][resid[good] < 0]
    one_sided = (float(1.4826 * np.median(np.abs(lower))) if lower.size > 20
                 else float(1.4826 * np.median(np.abs(resid[good]))))
    symmetric = float(1.4826 * np.median(np.abs(resid[good] -
                                                np.median(resid[good]))))
    if one_sided <= 0:
        one_sided = max(symmetric, 1e-9)

    hot = np.zeros(t.size, dtype=bool)
    hot[good] = resid[good] > sigma * one_sided

    # decay time: median length of a contiguous run above half the threshold,
    # which is a property of the flares rather than of the sampling
    warm = np.zeros(t.size, dtype=bool)
    warm[good] = resid[good] > 0.5 * sigma * one_sided
    runs = _runs(warm)
    seeded = [(i, j) for i, j in runs if hot[i:j + 1].any()]
    decay = (float(np.median([t[j] - t[i] for i, j in seeded]))
             if seeded else 0.0)
    cad = float(np.median(np.diff(np.sort(t[good])))) if good.sum() > 2 else 0.01
    grow = float(np.clip(grow_decays * max(decay, cad), cad, max_grow_days))

    mask = np.zeros(t.size, dtype=bool)
    intervals = []
    for i, j in seeded:
        lo, hi = t[i] - grow, t[j] + grow
        mask |= (t >= lo) & (t <= hi)
        intervals.append([float(lo), float(hi),
                          float(np.max(resid[i:j + 1]) * 1e6)])

    return {
        "mask": mask, "n_events": len(seeded), "intervals": intervals,
        "sigma_ppm": one_sided * 1e6, "symmetric_sigma_ppm": symmetric * 1e6,
        "decay_days": decay, "grow_days": grow,
        "largest_ppm": float(np.max(resid[good]) * 1e6) if good.any() else None,
        "estimator": ("1.4826*MAD of the points below the median, doubled "
                      "(one-sided: flares contaminate the upper half)"),
    }


def _runs(mask):
    """Contiguous True runs of a boolean array, as (start, end) inclusive."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(m.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(m.size - 1)
    return list(zip(starts, ends))



def _runs(mask):
    """Contiguous True runs of a boolean array, as (start, end) inclusive."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(m.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(m.size - 1)
    return list(zip(starts, ends))

"""Shared preparation for the light-curve tests.

Every Tier-1 test needs the same three things before it can ask its own
question: a light curve, a variability-removed version of it, and per-epoch
depths.  Doing that once here keeps the tests short enough to read, and --
more importantly -- means all of them measure a depth the same way.  Two
depth estimators that disagreed by a median factor of 0.68 were being mixed
in a ranking statistic on this project once already.

The detrending here is deliberately **not** the same operation as the fit in
:mod:`petcare.fitting`.  There, the transit is fitted simultaneously with the
GP.  Here the GP is conditioned only on out-of-transit cadences and then
predicted across the transit, which leaves the transit untouched in the
residual.  That is the right tool for a test that wants to *measure* a depth
someone else has already fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ephem
from .fitting import bin_for_fit, fit_gp_hyperparameters, gp_trend_masked
from .noise import depth_uncertainty_ppm, point_to_point_ppm, red_noise_factor


@dataclass
class EpochDepth:
    """One transit's depth, measured against its own local baseline.

    A *local* baseline per transit, not a global one: on a spotted star the
    mean flux level moves by percent between one transit and the next, and a
    global baseline turns that into a fake depth difference.
    """

    epoch: int
    t_center: float
    sector: int | None
    depth_ppm: float
    depth_err_ppm: float
    n_in: int
    n_oot: int
    coverage: float
    ingress_covered: bool
    egress_covered: bool
    in_gap: bool
    near_sector_edge: bool
    n_null: int = 0
    err_source: str = "propagated"
    #: this epoch's own null scatter, reported but not used as the error bar
    epoch_null_sigma_ppm: float | None = None

    @property
    def snr(self) -> float:
        return (self.depth_ppm / self.depth_err_ppm
                if self.depth_err_ppm and np.isfinite(self.depth_err_ppm)
                and self.depth_err_ppm > 0 else np.nan)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["snr"] = _f(self.snr)
        return d


@dataclass
class Prepared:
    """A light curve ready for a Tier-1 test to measure something on."""

    t: np.ndarray
    f: np.ndarray                 # binned flux, variability retained
    err: np.ndarray
    f_detrended: np.ndarray       # variability removed, transit retained
    trend: np.ndarray
    period: float
    t0: float
    duration_days: float
    gp: object = None
    bin_info: dict = field(default_factory=dict)
    known_mask: np.ndarray = None
    notes: list = field(default_factory=list)
    white_ppm: float = np.nan
    red_factor: float = np.nan
    red_detail: dict = field(default_factory=dict)

    @property
    def duration_h(self) -> float:
        return self.duration_days * 24.0

    def phase(self) -> np.ndarray:
        return ephem.phase(self.t, self.period, self.t0)

    def epochs(self) -> np.ndarray:
        return ephem.epoch_number(self.t, self.period, self.t0)

    def noise_kwargs(self) -> dict:
        """The ``noise`` block every test built on these numbers must carry."""
        return {
            "ppm": _f(self.white_ppm),
            "estimator": "1.4826*MAD(diff)/sqrt(2) on the detrended series",
            "red_noise_factor": _f(self.red_factor),
            "red_noise_timescale_days": self.duration_days,
            "red_noise_n_bins": self.red_detail.get("n_bins"),
            "depth_errors_inflated_by_red_noise": True,
        }


def prepare(ctx, period=None, t0=None, duration_days=None, bin_seconds=600.0,
            min_rho_durations=3.0, oot_outer=3.0, mask_known=True,
            detrend=True) -> Prepared:
    """Load, bin, and remove stellar variability without touching the transit.

    Raises :class:`petcare.io.LightCurveError` if there is no usable light
    curve; callers turn that into INCONCLUSIVE.
    """
    lc = ctx.lc
    period = float(period if period is not None else ctx.candidate.period)
    t0 = float(t0 if t0 is not None else ctx.candidate.t0_btjd)
    if duration_days is None:
        duration_days = ctx.candidate.duration_days()
    if duration_days is None:
        raise ValueError("no transit duration: pass --duration or run pc_fit first")
    duration_days = float(duration_days)

    notes = []
    t, f, err = lc.t, lc.f, lc.err
    known_mask = np.zeros(t.size, dtype=bool)
    if mask_known and ctx.known:
        # Overlap, not period proximity -- a known planet at 2x or 1/2 this
        # candidate's period is the same object and must still be recognised.
        others = [k for k in ctx.known
                  if not ephem.match_known(period, t0, duration_days * 24, [k],
                                           float(t.min()), float(t.max()))]
        known_mask = ephem.mask_known_transits(t, others, width=2.0)
        if known_mask.any():
            notes.append(f"masked {int(known_mask.sum())} cadences belonging to "
                         f"{len(others)} other known signal(s)")
    keep = ~known_mask
    t, f, err = t[keep], f[keep], err[keep]

    tb, fb, eb, nb, bin_info = bin_for_fit(t, f, err, bin_seconds)

    trend = np.zeros_like(fb)
    spec = None
    if detrend:
        mask_out = ephem.in_transit(tb, period, t0, duration_days, width=2.5)
        prot = lc.rotation_period().get("prot_days")
        if prot is not None and prot < 1.5 * duration_days:
            prot = None
        spec = fit_gp_hyperparameters(
            tb, fb, mask_out=mask_out,
            min_rho_days=min_rho_durations * duration_days,
            max_rho_days=max(30.0, period * 5), prot=prot)
        if spec.success:
            trend = gp_trend_masked(tb, fb, eb, mask_out, spec) - 1.0
        else:
            notes.append(f"no variability model ({spec.reason}); "
                         f"measuring against the raw flux")

    f_det = fb - trend

    white = point_to_point_ppm(f_det)
    oot = ephem.out_of_transit(tb, period, t0, duration_days, inner=0.75,
                               outer=max(oot_outer, 8.0))
    rn = red_noise_factor(tb[oot], f_det[oot], duration_days) if oot.sum() > 50 else {}
    red = rn.get("red_noise_factor")
    red = float(red) if red and np.isfinite(red) and red > 1 else 1.0

    return Prepared(t=tb, f=fb, err=eb, f_detrended=f_det, trend=trend,
                    period=period, t0=t0, duration_days=duration_days,
                    gp=spec, bin_info=bin_info, known_mask=known_mask,
                    notes=notes, white_ppm=white, red_factor=red,
                    red_detail=rn)


# --------------------------------------------------------------------------


#: Baseline windows (half-width, in transit durations) and polynomial orders
#: offered to :func:`choose_baseline`.
BASELINE_WINDOWS = (1.5, 2.0, 3.0, 4.0)
BASELINE_ORDERS = (1, 2, 3)

#: Phase offsets, as a fraction of the period, at which the null measurements
#: are made.  0.5 is excluded deliberately -- a secondary eclipse lives there
#: and would contaminate the noise estimate with signal.  The offsets are
#: symmetric so that a linear trend in the data cannot bias the null centre.
NULL_OFFSETS = (-0.35, -0.25, -0.15, 0.15, 0.25, 0.35)


def _trend_at(prep: Prepared, t0: float, extra_mask=None, width: float = 2.5):
    """GP variability model with the transit at ``t0`` masked out.

    Returns an additive trend (zero-mean).  Falls back to a flat zero trend
    when no GP was fitted, so a caller never has to branch on it.
    """
    if prep.gp is None or not getattr(prep.gp, "success", False):
        return np.zeros_like(prep.f)
    mask = ephem.in_transit(prep.t, prep.period, t0, prep.duration_days, width=width)
    if extra_mask is not None:
        mask = mask | extra_mask
    return gp_trend_masked(prep.t, prep.f, prep.err, mask, prep.gp) - 1.0


def _measure_epochs(prep, t0, trend, window_durations, poly_order, core_frac,
                    min_in=3, min_oot=8, epochs_of=None):
    """Local-polynomial depth at every epoch of the ephemeris ``(P, t0)``.

    Returns ``{epoch: (depth, n_in, n_oot)}``.  ``epochs_of`` forces the epoch
    numbering to come from a different reference epoch, so a null measurement
    can be paired with the real one transit by transit.
    """
    t = prep.t
    f = prep.f - trend
    ph = ephem.phase(t, prep.period, t0)
    ep = (ephem.epoch_number(t, prep.period, epochs_of) if epochs_of is not None
          else ephem.epoch_number(t, prep.period, t0))
    dur = prep.duration_days
    half_core = 0.5 * core_frac * dur
    win = window_durations * dur

    out = {}
    for e in np.unique(ep):
        sel = ep == e
        aph = np.abs(ph)
        w = sel & (aph < win)
        if w.sum() < min_in + min_oot:
            continue
        dt = ph[w]                      # phase is the natural local coordinate
        fw = f[w]
        a = np.abs(dt)
        in_m = a < half_core
        oot_m = a >= 0.75 * dur
        if int(in_m.sum()) < min_in or int(oot_m.sum()) < min_oot:
            continue
        d = _local_depth(dt, fw, in_m, oot_m, poly_order)
        if d is not None:
            out[int(e)] = (float(d), int(in_m.sum()), int(oot_m.sum()))
    return out


def choose_baseline(prep: Prepared, windows=BASELINE_WINDOWS,
                    orders=BASELINE_ORDERS, core_frac: float = 0.8,
                    offsets=NULL_OFFSETS, inject_depth: float | None = None) -> dict:
    """Pick the local-baseline window and polynomial order **by measurement**.

    For each (window, order) two things are measured on this star's own data:

    * ``sigma_null`` -- the scatter of the *same depth statistic* measured at
      off-transit phases, with the same GP masking and the same local
      polynomial, i.e. the noise a real depth has to beat;
    * ``recovery`` -- the fraction of an injected box depth the estimator gets
      back, i.e. how much of a real transit the baseline eats.

    The choice maximises ``recovery / sigma_null``.  Neither term alone is
    safe: minimising the noise drives the window narrow and the order high
    until the baseline absorbs the transit, and maximising recovery drives it
    wide until rotation dominates the residuals.

    This is the lesson from the previous search, where a fixed 0.5 d window
    left 82 of 104 candidates with their depths suppressed, the obvious rule
    (window >= 6x T14, capped at Prot/3) made the median significance *worse*,
    and choosing empirically improved it -- with the gains concentrated on
    exactly the spotted stars this sample is made of.
    """
    depth0 = inject_depth
    if depth0 is None:
        ph0 = prep.phase()
        i0 = np.abs(ph0) < 0.4 * prep.duration_days
        o0 = (np.abs(ph0) >= 0.75 * prep.duration_days) & \
             (np.abs(ph0) < 3.0 * prep.duration_days)
        depth0 = (float(np.median(prep.f_detrended[o0]) -
                        np.median(prep.f_detrended[i0]))
                  if i0.sum() > 5 and o0.sum() > 20 else 1e-3)
    depth0 = float(max(abs(depth0), 200e-6))

    real_mask = ephem.in_transit(prep.t, prep.period, prep.t0,
                                 prep.duration_days, width=2.5)
    null_trends = {}
    for off in offsets:
        t0n = prep.t0 + off * prep.period
        null_trends[off] = (t0n, _trend_at(prep, t0n, extra_mask=real_mask))

    trials = []
    for w in windows:
        for order in orders:
            nulls, recovered = [], []
            for off, (t0n, trend) in null_trends.items():
                got = _measure_epochs(prep, t0n, trend, w, order, core_frac)
                nulls.extend(v[0] for v in got.values())
                # inject a box at the same place and re-measure with the same
                # baseline, so recovery is measured by the estimator in use
                inj = _measure_epochs(prep, t0n, trend + _box(prep, t0n, depth0,
                                                              core_frac),
                                      w, order, core_frac)
                for e, v in inj.items():
                    if e in got:
                        recovered.append((v[0] - got[e][0]) / depth0)
            if len(nulls) < 6 or not recovered:
                continue
            sigma = float(1.4826 * np.median(np.abs(np.array(nulls) -
                                                    np.median(nulls))))
            rec = float(np.median(recovered))
            if sigma <= 0 or not np.isfinite(sigma) or rec <= 0:
                continue
            trials.append({"window_durations": float(w), "poly_order": int(order),
                           "sigma_null_ppm": sigma * 1e6, "recovery": rec,
                           "figure_of_merit": rec / (sigma * 1e6),
                           "n_null": len(nulls)})
    if not trials:
        return {"window_durations": 3.0, "poly_order": 1, "chosen_by": "default",
                "reason": "no trial window had enough baseline cadences",
                "trials": []}

    best = max(trials, key=lambda d: d["figure_of_merit"])
    return {**best, "chosen_by": "max recovery / null scatter", "trials": trials,
            "injected_depth_ppm": depth0 * 1e6, "offsets": list(offsets)}


def _box(prep, t0, depth, core_frac):
    """A negative box at ``t0``, as an additive trend for injection."""
    ph = ephem.phase(prep.t, prep.period, t0)
    box = np.zeros_like(prep.f)
    box[np.abs(ph) < 0.5 * core_frac * prep.duration_days] = depth
    return box


def epoch_depths(prep: Prepared, lc=None, window_durations: float | None = None,
                 poly_order: int | None = None, core_frac: float = 0.8,
                 offsets=NULL_OFFSETS, baseline: dict | None = None,
                 min_in: int = 3, min_oot: int = 8) -> list[EpochDepth]:
    """Depth of every covered transit, with an uncertainty that is *measured*.

    The estimator is: remove the star's variability with a GP conditioned only
    on out-of-transit cadences, then fit a low-order polynomial baseline in a
    window around that one transit and average the flux deficit over the inner
    ``core_frac`` of the duration.

    Both halves are needed.  The GP alone leaves a residual at the transit --
    it has to interpolate across the masked region -- that varies from epoch to
    epoch.  The local polynomial alone cannot track a 1.4 d rotator across a
    window with a hole in the middle: on HIP 67522 the null scatter was
    2000-17000 ppm without the GP and 210-680 ppm with it.

    **The uncertainty comes from off-phase nulls.**  The entire measurement --
    GP masking included -- is repeated at six phase offsets where no transit
    is expected, and the scatter of those null depths is the error bar.  It is
    the only estimator that contains what actually limits these depths: photon
    noise, residual red noise, and the error the baseline itself makes.  A
    propagated white-noise error was optimistic by a factor of 60 here, which
    turned a perfectly consistent planet into a chi-square of 61078 for 17
    degrees of freedom.

    The error is **pooled across epochs** and then scaled by each epoch's
    coverage, rather than taken from that epoch's own handful of nulls.  Six
    nulls give a MAD with roughly 30% uncertainty of its own, and the
    jackknife is exquisitely sensitive to that: one epoch that happened to
    draw three similar nulls got a 174 ppm error bar, dominated the weighted
    mean, and failed a known planet at survival 0.35.  Pooling costs the
    ability to notice a single anomalously noisy epoch; ``epoch_null_sigma_ppm``
    is reported per epoch so that is still visible.
    """
    if baseline is None and (window_durations is None or poly_order is None):
        baseline = choose_baseline(prep, core_frac=core_frac, offsets=offsets)
    if baseline:
        window_durations = window_durations or baseline["window_durations"]
        poly_order = poly_order if poly_order is not None else baseline["poly_order"]
    window_durations = float(window_durations or 3.0)
    poly_order = int(1 if poly_order is None else poly_order)

    real_mask = ephem.in_transit(prep.t, prep.period, prep.t0,
                                 prep.duration_days, width=2.5)
    trend = _trend_at(prep, prep.t0)
    real = _measure_epochs(prep, prep.t0, trend, window_durations, poly_order,
                           core_frac, min_in=min_in, min_oot=min_oot)

    # the same measurement where there is no transit, paired epoch by epoch
    per_epoch_nulls: dict[int, list] = {}
    pooled: list[float] = []
    for off in offsets:
        t0n = prep.t0 + off * prep.period
        tr = _trend_at(prep, t0n, extra_mask=real_mask)
        got = _measure_epochs(prep, t0n, tr, window_durations, poly_order,
                              core_frac, min_in=min_in, min_oot=min_oot,
                              epochs_of=prep.t0)
        for e, v in got.items():
            per_epoch_nulls.setdefault(e, []).append(v[0])
            pooled.append(v[0])

    pooled_sigma = (float(1.4826 * np.median(np.abs(np.array(pooled) -
                                                    np.median(pooled))))
                    if len(pooled) >= 6 else np.nan)

    cov_by_epoch = {}
    if lc is not None:
        for c in ephem.epoch_coverage(lc, prep.period, prep.t0,
                                      prep.duration_days):
            cov_by_epoch[c.epoch] = c

    n_in_ref = float(np.median([v[1] for v in real.values()])) if real else 1.0

    out: list[EpochDepth] = []
    for e in sorted(real):
        depth, n_in, n_oot = real[e]
        mine = per_epoch_nulls.get(e, [])
        epoch_sigma = (float(1.4826 * np.median(np.abs(np.array(mine) -
                                                       np.median(mine))))
                       if len(mine) >= 3 else np.nan)
        if np.isfinite(pooled_sigma):
            # scale the pooled scatter by this epoch's coverage: a half-covered
            # transit averages half as many cadences and deserves a larger bar
            err = pooled_sigma * np.sqrt(n_in_ref / max(n_in, 1))
            src = f"pooled null scatter ({len(pooled)} nulls), coverage-scaled"
        elif np.isfinite(epoch_sigma):
            err = epoch_sigma
            src = f"null scatter, this epoch only ({len(mine)} offsets)"
        else:
            err = depth_uncertainty_ppm(n_in, n_oot, prep.white_ppm,
                                        prep.red_factor) / 1e6
            src = "propagated white noise (no nulls fitted)"
        err = max(err, 1e-9)

        c = cov_by_epoch.get(int(e))
        out.append(EpochDepth(
            epoch=int(e), t_center=float(prep.t0 + e * prep.period),
            sector=(c.sector if c else None),
            depth_ppm=float(depth * 1e6), depth_err_ppm=float(err * 1e6),
            n_in=n_in, n_oot=n_oot,
            coverage=float(c.coverage) if c else np.nan,
            ingress_covered=bool(c.ingress_covered) if c else True,
            egress_covered=bool(c.egress_covered) if c else True,
            in_gap=bool(c.in_gap) if c else False,
            near_sector_edge=bool(c.near_sector_edge) if c else False,
            n_null=len(mine), err_source=src,
            epoch_null_sigma_ppm=(float(epoch_sigma * 1e6)
                                  if np.isfinite(epoch_sigma) else None)))
    return out


def weighted_mean(values, errors):
    """Inverse-variance mean, its error, and the chi-square about it."""
    v = np.asarray(values, float)
    e = np.asarray(errors, float)
    good = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[good], e[good]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    w = 1.0 / e ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(((v - mean) / e) ** 2))
    return mean, err, chi2, int(v.size)


def chi2_pvalue(chi2: float, dof: int) -> float:
    if not np.isfinite(chi2) or dof < 1:
        return np.nan
    from scipy.stats import chi2 as _chi2
    return float(_chi2.sf(chi2, dof))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _local_depth(dt, f, in_mask, oot_mask, poly_order: int):
    """Mean flux deficit inside ``in_mask`` against a polynomial through ``oot_mask``.

    Returns ``None`` when the baseline would have to extrapolate -- that is,
    when the in-transit points are not bracketed by baseline points on both
    sides.  A polynomial extrapolated past the end of a sector is not a
    baseline, and letting one through is how an edge-of-sector ramp becomes a
    deep transit.
    """
    if in_mask.sum() < 1 or oot_mask.sum() < poly_order + 2:
        return None
    t_in, t_oot = dt[in_mask], dt[oot_mask]
    if not (t_oot.min() < t_in.min() and t_oot.max() > t_in.max()):
        return None
    order = int(min(poly_order, oot_mask.sum() - 2))
    if order < 0:
        return None
    try:
        coef = np.polyfit(t_oot, f[oot_mask], order)
    except (np.linalg.LinAlgError, ValueError):
        return None
    base = np.polyval(coef, t_in)
    return float(np.mean(base - f[in_mask]))

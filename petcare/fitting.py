"""Transit model, GP noise model, and the joint fit.

The one non-negotiable here: **the transit and the stellar variability are fit
simultaneously**.  Flattening first distorts real signal on rapidly-rotating
young stars, and on this sample the detrending filter was caught manufacturing
a candidate outright -- masking a 29.6 d signal during detrending halved its
depth.

Two guards against the opposite failure, the GP eating the transit:

1. **GP hyperparameters are estimated on transit-masked data first**, then
   allowed to move only within a bounded neighbourhood of that estimate.  With
   the GP fully free, the six highest-evidence candidates in the 2026 search
   were exactly the six where the GP landed on the wrong timescale.
2. **The GP correlation length is bounded below at a few transit durations.**
   A GP that can vary on the transit timescale will absorb the transit
   regardless of where its period prior sits -- it is the *flexibility*, not
   the period, that does the damage.  Bounding the GP period alone changed the
   pass rate on known planets by 4 percentage points; this bound is the one
   that matters.

Everything is written to degrade rather than crash: a failed optimisation
returns ``success=False`` and the caller turns that into INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .stellar import a_over_rs, duration_t14, rho_circ

# --------------------------------------------------------------------------
# transit model


def transit_model(t, t0, period, rp_rs, a_rs, b, u=(0.42, 0.22),
                  exp_time_days=None, supersample=7):
    """Batman quadratic-limb-darkened transit, normalised to 1 out of transit.

    ``exp_time_days`` switches on supersampling, which matters here because
    the fits run on binned data: a 10-minute bin smears a 3-hour ingress
    noticeably, and ignoring it biases the fitted duration long and the impact
    parameter high.
    """
    import batman

    a_rs = max(float(a_rs), 1.001)
    b = float(np.clip(b, 0.0, 1.0 + float(rp_rs)))
    inc = np.degrees(np.arccos(np.clip(b / a_rs, 0.0, 0.999999)))

    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = float(t0), float(period), float(rp_rs), a_rs, inc
    p.ecc, p.w = 0.0, 90.0
    p.u, p.limb_dark = [float(u[0]), float(u[1])], "quadratic"

    kw = {}
    if exp_time_days and exp_time_days > 0:
        kw = dict(supersample_factor=int(supersample), exp_time=float(exp_time_days))
    return batman.TransitModel(p, np.asarray(t, float), **kw).light_curve(p)


# --------------------------------------------------------------------------
# binning for speed


def bin_for_fit(t, f, err, bin_seconds: float, inflate: bool = False):
    """Bin a light curve onto a regular grid, propagating errors.

    20-second TESS data on a 10-sector star is ~10^6 cadences; a GP likelihood
    is linear in that but a multi-start optimisation over it is not worth the
    wall clock at 30k targets.  Binning to 10 minutes keeps ~18 points across
    a 3-hour transit and is handled exactly by batman's supersampling, so no
    information the fit actually uses is lost.

    ``inflate`` scales the binned errors up to the binned series' own measured
    point-to-point scatter.  It defaults to **off**, and the reason is worth
    stating: the GP's diagonal should carry the *white* noise only, because
    the kernel is what models the correlated part.  Inflating the diagonal to
    a measured scatter that contains real short-timescale astrophysical signal
    hands that signal to the GP to absorb -- on HIP 67522 it raised the
    diagonal from 283 to 539 ppm and the GP promptly used the difference.

    The honest way to get this right is the two-pass scheme in ``pc_fit``:
    fit with the propagated errors, measure the scatter of the residual, and
    refit only if the residual is genuinely larger than the errors claimed.
    That direction of correction is safe; the other is not.

    ``info`` always reports both numbers so the comparison is visible whether
    or not anything was scaled.

    Returns ``(t, f, err, n_per_bin, info)``.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    err = np.asarray(err, float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f, err = t[good], f[good], err[good]
    if t.size == 0:
        return t, f, err, np.array([], dtype=int)

    width = bin_seconds / 86400.0
    dt = np.median(np.diff(np.sort(t)))
    if not np.isfinite(width) or width <= 0 or width <= dt * 1.01:
        err, info = _inflate_errors(f, err, inflate)
        info["binned"] = False
        return t, f, err, np.ones(t.size, dtype=int), info

    idx = np.floor((t - t.min()) / width).astype(int)
    order = np.argsort(idx, kind="stable")
    idx, t, f, err = idx[order], t[order], f[order], err[order]
    edges = np.flatnonzero(np.diff(idx)) + 1
    groups = np.split(np.arange(t.size), edges)

    tb = np.array([t[g].mean() for g in groups])
    fb = np.array([f[g].mean() for g in groups])
    nb = np.array([g.size for g in groups], dtype=int)
    eb = np.empty(tb.size)
    for i, g in enumerate(groups):
        e = err[g]
        if np.isfinite(e).all() and e.size:
            eb[i] = np.sqrt(np.sum(e ** 2)) / e.size
        else:
            eb[i] = np.nan
    if not np.isfinite(eb).all():
        fallback = np.nanstd(np.diff(f)) / np.sqrt(2)
        eb = np.where(np.isfinite(eb), eb, fallback / np.sqrt(np.maximum(nb, 1)))

    eb, info = _inflate_errors(fb, eb, inflate)
    info.update({"binned": True, "bin_seconds": float(bin_seconds),
                 "n_in": int(t.size), "n_out": int(tb.size),
                 "median_points_per_bin": float(np.median(nb))})
    return tb, fb, eb, nb, info


def _inflate_errors(f, err, inflate: bool):
    """Scale ``err`` up to the series' measured point-to-point scatter.

    The measured value is the honest one: it contains whatever correlated
    noise survives at this sampling, which a propagated photon error does not.
    """
    measured = float(1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2)) if f.size > 2 else np.nan
    median_err = float(np.nanmedian(err)) if err.size else np.nan
    info = {"measured_scatter_ppm": measured * 1e6 if np.isfinite(measured) else None,
            "propagated_err_ppm": median_err * 1e6 if np.isfinite(median_err) else None,
            "error_inflation": 1.0}
    if not inflate or not (np.isfinite(measured) and np.isfinite(median_err)) \
            or median_err <= 0 or measured <= median_err:
        return err, info
    factor = measured / median_err
    info["error_inflation"] = float(factor)
    return err * factor, info


# --------------------------------------------------------------------------
# the variability model
#
# The kernel choice is not cosmetic.  A Matern-3/2 fitted to a 1.4 d rotator
# with 2% spot modulation drives its correlation length down to whatever floor
# it is given and its amplitude up to several times the data's own scatter,
# because a short-correlation-length kernel is the only way a Matern can
# imitate a periodic signal.  What it then has is exactly the freedom needed
# to swallow an isolated few-hour dip: on HIP 67522 b it returned 2271 ppm for
# an 8000 ppm transit.
#
# A quasi-periodic RotationTerm with a floor on Q0 cannot do that.  High Q0
# means each component is close to an undamped sinusoid, so the model is
# obliged to repeat at the rotation period and cannot spend its amplitude on
# one isolated event.  That -- not where the GP period sits -- is what stops
# the GP eating the transit; bounding a flexible kernel's period changed the
# pass rate on known planets by four percentage points, and the flexibility
# bound is the one that matters.


@dataclass
class GPSpec:
    """A fitted variability model: which kernel, its values, and its bounds.

    ``values`` and ``bounds`` are in the fit's internal (log, where positive)
    parameterisation, so the joint fit can append them to its own vector
    without knowing which kernel it has.
    """

    kind: str = "matern"             # 'rotation' | 'matern'
    names: tuple = ()
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    bounds: list = field(default_factory=list)
    success: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def n_free(self) -> int:
        return len(self.names)

    def kernel(self, x=None):
        from celerite2 import terms
        v = np.asarray(self.values if x is None else x, float)
        if self.kind == "rotation":
            sigma, period, q0 = np.exp(v[0]), np.exp(v[1]), np.exp(v[2])
            return terms.RotationTerm(sigma=sigma, period=period, Q0=q0,
                                      dQ=self.detail.get("dQ", 1.0),
                                      f=self.detail.get("f", 0.5))
        sigma, rho = np.exp(v[0]), np.exp(v[1])
        return terms.Matern32Term(sigma=sigma, rho=rho)

    def amplitude(self, x=None) -> float:
        """The process standard deviation, ``sqrt(k(0))``.

        This is the number compared with the transit depth, so it has to mean
        the same thing for both kernels -- a RotationTerm's ``sigma`` argument
        is already that by construction, but taking it from the kernel keeps
        the comparison honest if the parameterisation ever changes.
        """
        if self.kind == "local polynomial":
            # a polynomial baseline has no amplitude of its own: it is exactly
            # as flexible as its order allows over its window, and the
            # GP-amplitude-to-depth diagnostic does not apply to it
            return float("nan")
        v = np.asarray(self.values if x is None else x, float)
        return float(np.exp(v[0]))

    def timescale(self, x=None) -> float:
        """The shortest timescale the kernel can vary on, in days.

        For a rotation kernel that is the *half*-period component, which is
        what a transit has to hide under.
        """
        if self.kind == "local polynomial":
            # the shortest thing a polynomial of order n can follow across a
            # window of width W is roughly W/n
            w = self.detail.get("window_durations")
            o = max(int(self.detail.get("poly_order", 2)), 1)
            return float(2 * w / o) if w else float("nan")
        v = np.asarray(self.values if x is None else x, float)
        return float(np.exp(v[1]) / 2.0) if self.kind == "rotation" else float(np.exp(v[1]))

    def to_dict(self, x=None) -> dict:
        v = np.asarray(self.values if x is None else x, float)
        d = {"kind": self.kind, "success": self.success, "reason": self.reason,
             "amplitude": self.amplitude(v if v.size else None),
             "min_timescale_days": self.timescale(v if v.size else None)}
        if self.kind == "local polynomial":
            d.update(self.detail)
            d["min_timescale_durations"] = d.pop("min_timescale_days", None)
            return d
        if self.kind == "rotation":
            d.update({"period_days": float(np.exp(v[1])), "Q0": float(np.exp(v[2]))})
        else:
            d.update({"rho_days": float(np.exp(v[1]))})
        d.update(self.detail)
        return d


def make_gp(spec: GPSpec, x, t, yerr, jitter: float = 0.0):
    import celerite2
    gp = celerite2.GaussianProcess(spec.kernel(x), mean=0.0)
    gp.compute(np.asarray(t, float),
               diag=np.asarray(yerr, float) ** 2 + jitter ** 2, quiet=True)
    return gp


#: Floor on the rotation kernel's quality factor.  Q0 = 2 gives Q1 = 3.5,
#: already well into the regime where a component rings for several cycles and
#: cannot localise its amplitude on a single transit.
Q0_MIN = 2.0

#: How far the GP's rotation period may move from the measured Prot.
PROT_TOL = 1.3

#: Factor by which the fitted a/R* may depart from the value the star's
#: density implies.  3 covers e ~ 0.5 or a factor-of-two radius error.
A_RS_TOLERANCE = 3.0


def fit_gp_hyperparameters(t, f, mask_out, min_rho_days: float,
                           max_rho_days: float | None = None,
                           prot: float | None = None,
                           q0_min: float = Q0_MIN,
                           prot_tol: float = PROT_TOL,
                           mask_flares: bool = True,
                           pin_to=None,
                           free: tuple = ("log_sigma",)) -> GPSpec:
    """Estimate the variability model on transit-masked data.

    ``mask_out`` is True for cadences to *exclude* (in transit, in a known
    planet's transit).  Fitting the hyperparameters where the transit is not
    means the GP describes the star and not the candidate.

    With ``prot`` supplied a quality-floored :class:`RotationTerm` is fitted;
    without it, a Matern-3/2 with its correlation length floored at
    ``min_rho_days``.  The rotation kernel is strongly preferred on this
    sample -- see the note at the top of this section.

    ``mask_flares`` runs the fit twice: once to get a variability model, then
    again with the flares that model's residual reveals masked out.  This is
    not optional in practice.  On a heavily flaring M dwarf the unmasked fit drives the GP
    amplitude to its upper bound and the quality factor to its floor trying to
    explain 137 flares -- precisely the maximally-flexible configuration that
    then swallows the transit, taking a real 2490 ppm signal down to 0.94 ppm.
    The previous pipeline ran with no flare masking at all on exactly these
    stars.
    """
    spec = _fit_gp_once(t, f, mask_out, min_rho_days, max_rho_days, prot,
                        q0_min, prot_tol, pin_to=pin_to, free=free)
    if not (mask_flares and spec.success):
        return spec

    from .flares import MAX_MASKED_FRACTION, detect_flares

    # A nominal diagonal is enough here: the flare detector only needs a
    # variability-removed series to threshold on, not a calibrated one.
    trend = gp_trend_masked(t, f, np.full(np.size(t), 1e-4), mask_out, spec)
    fl = detect_flares(t, np.asarray(f, float) - trend + 1.0)
    if fl["n_events"] == 0 or fl["mask"].mean() > MAX_MASKED_FRACTION:
        spec.detail["flare_masking"] = {
            "applied": False, "n_events": fl["n_events"],
            "masked_fraction": float(fl["mask"].mean()),
            "reason": ("no flares found" if fl["n_events"] == 0 else
                       "would mask too much of the light curve")}
        return spec

    refit = _fit_gp_once(t, f, np.asarray(mask_out, bool) | fl["mask"],
                         min_rho_days, max_rho_days, prot, q0_min, prot_tol,
                         pin_to=pin_to, free=free)
    if not refit.success:
        return spec
    refit.detail["flare_masking"] = {
        "applied": True, "n_events": fl["n_events"],
        "masked_fraction": float(fl["mask"].mean()),
        "amplitude_before": spec.amplitude(),
        "amplitude_after": refit.amplitude()}
    return refit


def _fit_gp_once(t, f, mask_out, min_rho_days, max_rho_days, prot,
                 q0_min, prot_tol, pin_to=None, free=("log_sigma",)) -> GPSpec:
    keep = ~np.asarray(mask_out, bool)
    tk, fk = np.asarray(t, float)[keep], np.asarray(f, float)[keep]
    if tk.size < 50:
        return GPSpec(success=False, reason="fewer than 50 unmasked cadences")

    scatter = 1.4826 * np.median(np.abs(np.diff(fk))) / np.sqrt(2)
    spread = float(np.std(fk))
    sigma0 = max(spread, scatter * 2, 1e-5)
    yerr = np.full(tk.size, max(scatter, 1e-6))
    y = fk - np.median(fk)

    use_rotation = bool(prot and np.isfinite(prot) and prot > 0)
    if use_rotation:
        spec = GPSpec(kind="rotation", names=("log_sigma", "log_period", "log_Q0"),
                      detail={"dQ": 1.0, "f": 0.5, "prot_input": float(prot),
                              "q0_min": float(q0_min), "prot_tol": float(prot_tol)})
        x0 = np.array([np.log(sigma0), np.log(prot), np.log(max(q0_min * 2, 5.0))])
        bounds = [
            # the GP's amplitude cannot exceed the data's own spread by much:
            # a variability model with more variance than the data has is
            # mis-specified, whatever the likelihood says
            (np.log(max(scatter, 1e-6)), np.log(max(spread * 1.5, scatter * 10))),
            (np.log(prot / prot_tol), np.log(prot * prot_tol)),
            (np.log(q0_min), np.log(1e3)),
        ]
    else:
        hi_rho = max_rho_days or max(30.0, min_rho_days * 50)
        spec = GPSpec(kind="matern", names=("log_sigma", "log_rho"),
                      detail={"min_rho_days": float(min_rho_days),
                              "max_rho_days": float(hi_rho),
                              "note": "no rotation period available; "
                                      "a Matern kernel is the weaker choice here"})
        x0 = np.array([np.log(sigma0),
                       np.log(float(np.clip(min_rho_days * 3, min_rho_days, hi_rho)))])
        bounds = [(np.log(max(scatter, 1e-6)), np.log(max(spread * 1.5, scatter * 10))),
                  (np.log(min_rho_days), np.log(hi_rho))]

    if pin_to is not None:
        # Hold every hyperparameter but ``free`` at a value fitted elsewhere.
        # Used to give each sector its own amplitude while the shape of the
        # kernel -- the rotation period, the quality factor, the correlation
        # length -- stays at the global value.  Those are the parameters that
        # decide whether a kernel *can* reach into a transit, and a per-sector
        # fit of them on a quarter of the data is how a real planet gets
        # explained away.
        pin_to = np.asarray(pin_to, float)
        for i, name in enumerate(spec.names):
            if name in free or i >= pin_to.size:
                continue
            v = float(np.clip(pin_to[i], *bounds[i]))
            bounds[i] = (v, v)
            x0[i] = v
        spec.detail["pinned"] = [n for i, n in enumerate(spec.names)
                                 if n not in free and i < pin_to.size]

    spec.bounds = bounds

    def nll(theta):
        try:
            gp = make_gp(spec, theta, tk, yerr)
            ll = gp.log_likelihood(y)
        except Exception:  # noqa: BLE001 - a bad kernel is just a bad point
            return 1e10
        return -ll if np.isfinite(ll) else 1e10

    x0 = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x0, bounds)])
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=400))
    if not np.isfinite(res.fun) or res.fun >= 1e9:
        return GPSpec(kind=spec.kind, names=spec.names, bounds=bounds,
                      success=False, reason="hyperparameter fit did not converge",
                      detail=spec.detail)

    spec.values = np.asarray(res.x, float)
    spec.success = True
    spec.detail.update({"nll": float(res.fun), "n_used": int(tk.size),
                        "scatter": float(scatter), "data_spread": spread})
    return spec


# --------------------------------------------------------------------------
# period / epoch refinement


def box_snr(t, f, period, t0, dur_days, min_in=6, min_out=20):
    """Depth SNR of a box at (period, t0), with a local baseline."""
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    intr = np.abs(ph) < 0.5 * dur_days
    oot = (np.abs(ph) >= 0.75 * dur_days) & (np.abs(ph) < 3.0 * dur_days)
    n_in, n_out = int(intr.sum()), int(oot.sum())
    if n_in < min_in or n_out < min_out:
        return -np.inf, n_in
    f_out, f_in = f[oot], f[intr]
    depth = np.median(f_out) - np.median(f_in)
    scatter = 1.4826 * np.median(np.abs(f_out - np.median(f_out)))
    if scatter <= 0:
        return -np.inf, n_in
    return float(depth / (scatter * np.sqrt(1.0 / n_in + 1.0 / n_out))), n_in


def gp_trend_masked(t, f, err, mask_out, spec: "GPSpec"):
    """GP prediction at every ``t``, conditioned only on the unmasked points.

    Used to remove stellar variability before a *search* (period refinement,
    rotation, secondary eclipse), where there is not yet a transit model to
    subtract.  Conditioning on the out-of-transit points only is what keeps the
    prediction from following the transit down: the GP never sees it.

    The mask can be in slightly the wrong place -- at this stage the ephemeris
    is exactly what is being refined -- and that is tolerable, because the
    kernel is constrained (quality-floored rotation, or a Matern with a
    correlation-length floor) so it cannot localise onto a single transit even
    where one is left unmasked.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    err = np.asarray(err, float)
    keep = ~np.asarray(mask_out, bool)
    if keep.sum() < 50 or not spec.success:
        return np.full(t.size, np.nanmedian(f))
    mu = float(np.median(f[keep]))
    try:
        pred = gp_predict_masked(spec, t, f - mu, err, keep)
    except Exception:  # noqa: BLE001 - fall back to a flat trend, never crash
        return np.full(t.size, mu)
    return np.asarray(pred, float) + mu


def _box_snr_grid(t, f, periods, t0_grid_rel, t0_anchor, dur_days):
    """Box SNR on a (period, t0-offset) grid.  Vectorised over cadences."""
    half = 0.5 * dur_days
    out = np.full((periods.size, t0_grid_rel.size), -np.inf)
    n_in_grid = np.zeros_like(out, dtype=int)
    for i, p in enumerate(periods):
        # phase of every cadence for this trial period, once
        base = (t - t0_anchor + 0.5 * p) % p - 0.5 * p
        for j, dt in enumerate(t0_grid_rel):
            ph = base - dt
            ph = np.where(ph < -0.5 * p, ph + p, np.where(ph > 0.5 * p, ph - p, ph))
            aph = np.abs(ph)
            intr = aph < half
            oot = (aph >= 0.75 * dur_days) & (aph < 3.0 * dur_days)
            n_in, n_out = int(intr.sum()), int(oot.sum())
            if n_in < 6 or n_out < 20:
                continue
            f_out = f[oot]
            depth = np.median(f_out) - np.median(f[intr])
            scatter = 1.4826 * np.median(np.abs(f_out - np.median(f_out)))
            if scatter <= 0:
                continue
            out[i, j] = depth / (scatter * np.sqrt(1.0 / n_in + 1.0 / n_out))
            n_in_grid[i, j] = n_in
    return out, n_in_grid


def _blocks(t, gap_days: float = 10.0):
    """Contiguous observing blocks, split at gaps longer than ``gap_days``."""
    order = np.argsort(t)
    ts = t[order]
    cuts = np.flatnonzero(np.diff(ts) > gap_days) + 1
    out, start = [], 0
    for c in list(cuts) + [ts.size]:
        if c - start >= 50:
            out.append((float(ts[start]), float(ts[c - 1])))
        start = c
    return out


def coarse_phase_offsets(t, f, period, t0, dur_days,
                         window_frac: float = 0.25, n_steps: int = 241,
                         min_snr: float = 6.0) -> list:
    """Per-block phase offset of the transit, searched over a wide window.

    Each observing block (a sector, or a run of adjacent sectors) contains
    several transits, so the fold *within* a block is well detected even when
    the ephemeris is badly wrong globally.  Measuring the offset block by
    block and fitting a line through them recovers a period error far larger
    than any fine grid search can, because the fine grid has to keep the fold
    coherent across the whole baseline and a block does not.

    This is what a mismatched catalogue ephemeris looks like, and it is
    common: a period from one paper and an epoch from another put one known planet's
    transits at +1.5 h in sector 1, -0.9 h in sector 27 and -2.1 h in sector
    95, which no amount of fine-grid refinement around the catalogue value
    can reconcile.
    """
    out = []
    half = min(window_frac * period, 20 * dur_days)
    offsets = np.linspace(-half, half, n_steps)
    for t_lo, t_hi in _blocks(t):
        sel = (t >= t_lo) & (t <= t_hi)
        if sel.sum() < 200 or (t_hi - t_lo) < 1.5 * period:
            continue
        tb, fb = t[sel], f[sel]
        snrs = np.array([box_snr(tb, fb, period, t0 + d, dur_days)[0]
                         for d in offsets])
        if not np.isfinite(snrs).any():
            continue
        j = int(np.nanargmax(snrs))
        if snrs[j] < min_snr:
            continue
        out.append({"t_mid": 0.5 * (t_lo + t_hi), "t_start": t_lo,
                    "t_stop": t_hi, "offset_days": float(offsets[j]),
                    "snr": float(snrs[j]),
                    "railed": bool(j in (0, n_steps - 1))})
    return out


def linear_from_offsets(blocks, period, t0):
    """Fit a line through the per-block offsets: a corrected period and epoch.

    The offsets are phases, so they wrap.  They are unwrapped against the
    running fit before the line is drawn, which matters as soon as the total
    drift exceeds half a period.
    """
    if len(blocks) < 2:
        return None
    x = np.array([b["t_mid"] for b in blocks]) - t0
    y = np.array([b["offset_days"] for b in blocks])
    w = np.array([max(b["snr"], 1.0) for b in blocks])

    # unwrap: each successive offset is taken to the branch nearest the
    # previous one, so a drift through a whole period is followed rather than
    # folded back on itself
    for i in range(1, y.size):
        while y[i] - y[i - 1] > 0.5 * period:
            y[i] -= period
        while y[i] - y[i - 1] < -0.5 * period:
            y[i] += period

    sw = np.sum(w)
    sx, sxx = np.sum(w * x), np.sum(w * x * x)
    sy, sxy = np.sum(w * y), np.sum(w * x * y)
    det = sw * sxx - sx * sx
    if det == 0:
        return None
    slope = (sw * sxy - sx * sy) / det        # d(offset)/d(time)
    intercept = (sxx * sy - sx * sxy) / det   # offset at t = t0

    # offset growing by `slope` per day means the period is wrong by
    # slope * period per cycle
    return {"period": float(period * (1.0 + slope)),
            "t0_btjd": float(t0 + intercept),
            "slope_days_per_day": float(slope),
            "n_blocks": len(blocks),
            "offsets_days": [float(v) for v in y],
            "block_snr": [float(v) for v in w]}


def refine_ephemeris(t, f, period, t0, dur_days, max_drift_durations: float = 3.0,
                     max_tol_frac: float = 0.01, n_t0: int = 21,
                     max_period_steps: int = 601) -> dict:
    """Grid-refine the period and epoch around the input values.

    A catalogue ephemeris from a short-baseline detection drifts badly across a
    multi-year TESS baseline, and a drifted ephemeris looks exactly like a
    shallow or inconsistent transit -- 2 of 27 candidates in one run hit
    this.

    The search window is set in **transit durations of drift across the
    baseline**, not as a fixed fraction of the period.  A fixed fraction is
    wrong in both directions: +/-0.4% of a 7 d period over a 2550 d baseline
    lets the fold wander by ten days, so the grid simply finds the most
    convincing patch of stellar variability, while the same 0.4% on a single
    27 d sector is far too tight to fix anything.  Tying the window to drift
    keeps the search inside the region where a fold still means something.

    ``f`` should already have stellar variability removed (see
    :func:`gp_trend_masked`).  Searching on raw flux from a 3%-variable young
    star finds the variability.

    The epoch anchor sits near the middle of the data so the period sweep
    spreads the drift symmetrically instead of piling it at one end.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0) and dur_days > 0):
        return {"success": False, "reason": "no usable input ephemeris"}

    baseline = float(np.ptp(t))
    if baseline < 2 * period:
        return {"success": False,
                "reason": f"baseline {baseline:.1f} d is under two periods"}

    # Stage 1: per-block offsets, to catch a period or epoch error far larger
    # than the fine grid below can reach.  Skipped when the data are one
    # block, where there is nothing to fit a line through.
    coarse = None
    blocks = coarse_phase_offsets(t, f, period, t0, dur_days)
    if len(blocks) >= 2:
        fitted = linear_from_offsets(blocks, period, t0)
        if fitted is not None:
            drift = abs(fitted["slope_days_per_day"]) * baseline
            if drift > 0.5 * dur_days:
                # the blocks genuinely disagree with the input ephemeris;
                # re-centre the fine search on what they say
                coarse = fitted
                period, t0 = fitted["period"], fitted["t0_btjd"]

    tol = min(max_drift_durations * dur_days * period / baseline,
              max_tol_frac * period)
    step = max(0.125 * dur_days * period / baseline, tol / max_period_steps)
    n_period = int(np.clip(2 * tol / step + 1, 5, max_period_steps))
    periods = np.linspace(period - tol, period + tol, n_period)

    n_cycle = np.round((np.median(t) - t0) / period)
    t0_anchor = float(t0 + n_cycle * period)
    t0_rel = np.linspace(-0.6 * dur_days, 0.6 * dur_days, n_t0)

    grid, n_in_grid = _box_snr_grid(t, f, periods, t0_rel, t0_anchor, dur_days)
    if not np.isfinite(grid).any():
        return {"success": False, "reason": "no trial ephemeris had enough in/out cadences"}

    i, j = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    best_p = float(periods[i])
    best_t0 = float(t0_anchor + t0_rel[j])
    best_snr = float(grid[i, j])

    snr_in, _ = box_snr(t, f, period, t0_anchor, dur_days)

    # report the epoch nearest the input t0 so the number is comparable with
    # what the caller handed in
    n_back = np.round((best_t0 - t0) / best_p)
    t0_out = best_t0 - n_back * best_p
    railed = bool(i in (0, n_period - 1) and n_period > 5)
    return {
        "success": True, "period": best_p, "t0_btjd": float(t0_out),
        "coarse_block_fit": coarse,
        "coarse_blocks": blocks,
        "t0_anchor": best_t0, "box_snr": best_snr,
        "box_snr_at_input": float(snr_in) if np.isfinite(snr_in) else None,
        "n_in_transit": int(n_in_grid[i, j]),
        "period_grid_step": float(periods[1] - periods[0]) if n_period > 1 else 0.0,
        "period_search_halfwidth": float(tol),
        "period_search_halfwidth_frac": float(tol / period),
        "max_drift_durations": float(max_drift_durations),
        "baseline_days": baseline,
        "n_period_steps": int(n_period),
        "railed": railed,
    }


# --------------------------------------------------------------------------
# the joint fit


@dataclass
class FitResult:
    success: bool = False
    reason: str = ""
    period: float = np.nan
    t0: float = np.nan
    rp_rs: float = np.nan
    a_rs: float = np.nan
    b: float = np.nan
    depth_ppm: float = np.nan
    t14_h: float = np.nan
    gp: GPSpec = field(default_factory=GPSpec)
    logl: float = np.nan
    logl_no_transit: float = np.nan
    delta_logl: float = np.nan
    errors: dict = field(default_factory=dict)
    u: tuple = (0.42, 0.22)
    exp_time_days: float = np.nan
    n_points: int = 0
    n_starts: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def gp_amplitude(self) -> float:
        return self.gp.amplitude() if self.gp.success else np.nan

    @property
    def gp_sigma_over_depth(self) -> float:
        """GP amplitude divided by transit depth.

        The available discriminator for "did the GP eat the transit": median
        2.6 across fits that recovered a known planet, 10.8 across fits that
        did not.  Reported in every fit result, and the basis of that test's
        self-check.
        """
        d = self.depth_ppm / 1e6
        amp = self.gp_amplitude
        return float(amp / d) if d and np.isfinite(d) and d > 0 and np.isfinite(amp) else np.nan

    @property
    def gp_timescale_over_duration(self) -> float:
        """Shortest GP timescale in units of the transit duration.

        Below ~1 the variability model can vary within a single transit, which
        is the geometry in which it can absorb one.
        """
        t14 = self.t14_h / 24.0
        return float(self.gp.timescale() / t14) if self.gp.success and t14 > 0 else np.nan


N_TRANSIT_PARAMS = 5   # t0 offset, period offset, log rp, log a, b


def _unpack(theta, period0, t0_0):
    """Split the fit vector into transit parameters and the GP's own block."""
    t0_off, dper, log_rp, log_a, b = theta[:N_TRANSIT_PARAMS]
    return (t0_0 + t0_off, period0 + dper, np.exp(log_rp), np.exp(log_a), b,
            np.asarray(theta[N_TRANSIT_PARAMS:], float))


def joint_fit(t, f, err, period, t0, dur_days, u=(0.42, 0.22),
              exp_time_days=None, rho_star=None, gp_prior: GPSpec | None = None,
              min_rho_durations: float = 3.0, n_starts: int = 4,
              period_halfwidth: float | None = None,
              depth_guess: float | None = None,
              max_drift_durations: float = 1.0,
              gp_free: bool = False) -> FitResult:
    """Simultaneous GP + transit maximum-likelihood fit.

    ``gp_prior`` comes from :func:`fit_gp_hyperparameters`, fitted on
    transit-masked data, and by default its hyperparameters are **held
    fixed** here while the transit parameters are fitted.  That is the
    README's prescription and it is not conservatism: on DS Tuc A the
    hyperparameter likelihood is flat to four decimal places along a
    (amplitude, Q0) degeneracy, so a joint fit that is allowed to move along
    it buys GP flexibility for free, and it pays for that flexibility with the
    transit -- a real 3400 ppm transit came back as 27 ppm with an 8x-too-long
    duration, every GP parameter sitting on a bound.

    ``gp_free=True`` restores the joint fit within a bounded neighbourhood of
    the masked estimate.  It is worth having for comparison, and its results
    should be treated as an upper bound on how much the GP can absorb.

    The cost of fixing them is that the transit uncertainties do not include
    the GP hyperparameter uncertainty.  ``errors['method']`` records this.

    ``rho_star`` seeds ``a/R*`` when the stellar density is known, which keeps
    a multi-start optimisation out of the absurd corners of parameter space
    (several 2026 candidates optimised to b ~ 1.2 with 8% depths).

    ``depth_guess`` (fractional) should be measured on a **variability-removed**
    series.  A starting depth measured on raw flux from a 2% variable star is
    noise: the optimiser starts the transit at essentially zero amplitude and
    the GP happily keeps it there.  That omission alone turned a clean 8000 ppm
    transit into a 97 ppm non-detection in testing.

    ``max_drift_durations`` bounds how far the period may move, expressed as
    the transit drift it produces across the whole baseline.  A fixed
    fractional bound is wrong at both ends of this sample's baseline range.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    err = np.asarray(err, float)
    if not np.isfinite(err).all():
        err = np.where(np.isfinite(err),
                       err, 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2))

    out = FitResult(exp_time_days=float(exp_time_days or np.nan),
                    n_points=int(t.size), u=tuple(u))

    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    intr = np.abs(ph) < 0.5 * dur_days
    oot = (np.abs(ph) >= 0.75 * dur_days) & (np.abs(ph) < 3.0 * dur_days)
    if intr.sum() < 5 or oot.sum() < 20:
        out.reason = (f"too few cadences: {int(intr.sum())} in transit, "
                      f"{int(oot.sum())} in the local baseline")
        return out

    if depth_guess is not None and np.isfinite(depth_guess) and depth_guess > 0:
        depth0 = float(depth_guess)
    else:
        depth0 = max(float(np.median(f[oot]) - np.median(f[intr])), 30e-6)
    rp0 = float(np.sqrt(depth0))

    # a/R*, and the range it is allowed to move in.  When the stellar density
    # is known the range is a factor of A_RS_TOLERANCE around it, which spans
    # e ~ 0.5 or a factor-of-two radius error -- generous for a young star and
    # still far too tight for the optimiser to escape into the degenerate
    # corner.  Left unbounded it does exactly that: a known planet ran away to
    # a/R* = 294, b = 0.99 and a 0.03-hour transit, which fits a real 2490 ppm
    # signal as 2 ppm of nothing.  Several 2026 candidates optimised the same
    # way, to b ~ 1.2 with depths of 8%.
    if rho_star and rho_star > 0:
        a0 = a_over_rs(period, rho_star)
        a_lo = max(2.0, a0 / A_RS_TOLERANCE)
        a_hi = min(300.0, a0 * A_RS_TOLERANCE)
        a_bound_source = (f"stellar density {rho_star:.3f} g/cm3, "
                          f"x/{A_RS_TOLERANCE:g}")
    else:
        s_ = np.sin(np.pi * dur_days / period)    # invert the duration at b = 0
        a0 = float((1 + rp0) / s_) if 0 < s_ < 1 else 15.0
        a_lo, a_hi = 2.0, 300.0
        a_bound_source = "unbounded (no stellar density available)"
    a0 = float(np.clip(a0, a_lo, a_hi))

    # ---- the variability block -------------------------------------------
    if gp_prior is None or not gp_prior.success:
        gp_prior = fit_gp_hyperparameters(
            t, f, mask_out=np.abs(ph) < 1.25 * dur_days,
            min_rho_days=min_rho_durations * dur_days)
    if not gp_prior.success:
        out.reason = f"no variability model: {gp_prior.reason}"
        return out

    spec = gp_prior
    gp_x0 = np.asarray(spec.values, float)
    if is_segmented(spec):
        # One kernel per segment, all of them frozen.  Freeing them would give
        # the optimiser a flexibility knob per sector, which is exactly the
        # trade this arrangement exists to refuse.
        gp_bounds = []
        gp_free = False
    elif gp_free:
        # a bounded neighbourhood of the masked-data estimate, intersected
        # with the kernel's own hard bounds
        span = np.array([np.log(2.0)] + [np.log(1.4)] * (spec.n_free - 1))
        gp_bounds = [(max(lo, v - s_), min(hi, v + s_))
                     for v, s_, (lo, hi) in zip(gp_x0, span, spec.bounds)]
        gp_bounds = [(lo, hi) if hi > lo else (lo, lo + 1e-6) for lo, hi in gp_bounds]
    else:
        # a degenerate interval of zero width: L-BFGS-B holds the parameter,
        # and _railed() will not report a fixed parameter as railed
        gp_bounds = [(v, v) for v in gp_x0]

    baseline = max(float(np.ptp(t)), period)
    p_half = period_halfwidth if period_halfwidth is not None else min(
        max(max_drift_durations * dur_days * period / baseline, 1e-6),
        0.01 * period)
    t0_half = 0.5 * dur_days

    def nll(theta):
        t0_i, per_i, rp_i, a_i, b_i, gp_i = _unpack(theta, period, t0)
        if per_i <= 0 or a_i <= 1.0:
            return 1e10
        try:
            model = transit_model(t, t0_i, per_i, rp_i, a_i, b_i, u=u,
                                  exp_time_days=exp_time_days)
            ll = gp_log_likelihood(spec, gp_i, t, f - model, err)
        except Exception:  # noqa: BLE001 - infeasible point, not a crash
            return 1e10
        return -ll if np.isfinite(ll) else 1e10

    bounds = [(-t0_half, t0_half), (-p_half, p_half),
              (np.log(0.002), np.log(0.6)), (np.log(a_lo), np.log(a_hi)),
              (0.0, 1.0)] + gp_bounds

    rng = np.random.default_rng(20260921)
    starts = [np.concatenate([[0.0, 0.0, np.log(rp0), np.log(a0), 0.3], gp_x0])]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.concatenate([[
            rng.uniform(-0.3, 0.3) * dur_days, 0.0,
            np.log(rp0 * rng.uniform(0.6, 1.6)),
            np.log(np.clip(a0 * rng.uniform(0.6, 1.5), a_lo, a_hi)),
            rng.uniform(0.0, 0.85)], gp_x0 * rng.uniform(0.97, 1.03, gp_x0.size)]))

    best = None
    for theta0 in starts:
        theta0 = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(theta0, bounds)])
        try:
            res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=600, maxfun=4000))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(res.fun) and res.fun < 1e9 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        out.reason = "every optimisation start failed"
        return out

    t0_f, per_f, rp_f, a_f, b_f, gp_f = _unpack(best.x, period, t0)
    fitted = (spec if is_segmented(spec) else
              GPSpec(kind=spec.kind, names=spec.names, values=gp_f,
                     bounds=spec.bounds, success=True, detail=dict(spec.detail)))

    out.success = True
    out.period, out.t0 = float(per_f), float(t0_f)
    out.rp_rs, out.a_rs, out.b = float(rp_f), float(a_f), float(b_f)
    out.gp = fitted
    out.logl = float(-best.fun)
    out.n_starts = len(starts)

    # Depth as the model's actual minimum, not rp^2: limb darkening and a
    # non-zero impact parameter make those differ by tens of percent, and
    # carrying two values for one quantity is how two depth estimators came to
    # disagree by a median factor of 0.68 in an earlier analysis.
    fine = np.linspace(t0_f - 0.6 * dur_days, t0_f + 0.6 * dur_days, 601)
    mm = transit_model(fine, t0_f, per_f, rp_f, a_f, b_f, u=u, exp_time_days=None)
    out.depth_ppm = float((1.0 - mm.min()) * 1e6)
    out.t14_h = float(duration_t14(per_f, a_f, rp_f, b_f) * 24.0)

    # evidence for a transit at all, with the GP refitted without one
    try:
        def nll_flat(gp_i):
            try:
                ll = gp_log_likelihood(spec, gp_i, t, f - 1.0, err)
            except Exception:  # noqa: BLE001
                return 1e10
            return -ll if np.isfinite(ll) else 1e10

        if gp_f.size == 0:
            # nothing to optimise: the hyperparameters are frozen, so the
            # no-transit likelihood is a single evaluation
            out.logl_no_transit = float(-nll_flat(gp_f))
        else:
            r0 = minimize(nll_flat, gp_f, method="L-BFGS-B", bounds=gp_bounds,
                          options=dict(maxiter=300))
            out.logl_no_transit = float(-r0.fun)
        out.delta_logl = float(out.logl - out.logl_no_transit)
    except Exception:  # noqa: BLE001
        pass

    out.errors = _hessian_errors(nll, best.x, bounds, period, t0, spec)
    out.detail = {
        "gp_prior": spec.to_dict(gp_x0),
        "gp_fitted": fitted.to_dict(),
        "gp_bounds_log": [list(map(float, bb)) for bb in gp_bounds],
        "period_halfwidth": float(p_half),
        "a_rs_bounds": [float(a_lo), float(a_hi)],
        "a_rs_bound_source": a_bound_source,
        "start_depth_ppm": float(depth0 * 1e6),
        "n_in_transit": int(intr.sum()),
        "n_local_baseline": int(oot.sum()),
        "converged": bool(best.success),
        "railed": _railed(best.x, bounds, spec),
    }
    return out


def _param_names(spec: GPSpec) -> tuple:
    return ("t0", "period", "rp_rs", "a_rs", "b") + tuple(spec.names)


def _railed(x, bounds, spec: GPSpec, tol=1e-3) -> list:
    """Which parameters sat on a bound -- a fit at a wall is not a measurement.

    Parameters held fixed (zero-width bounds) are not railed; they were never
    free to move.
    """
    out = []
    for name, v, (lo, hi) in zip(_param_names(spec), x, bounds):
        span = hi - lo
        if span <= 0:
            continue
        if v - lo < tol * span or hi - v < tol * span:
            out.append(name)
    return out


def _hessian_errors(nll, x, bounds, period0, t0_0, spec: GPSpec) -> dict:
    """1-sigma parameter errors, by whichever method actually works.

    Point estimates with no uncertainties were a real gap in the previous
    chain, so this tries hard and then says plainly what it managed:

    1. the full numerical Hessian, inverted -- **marginal** errors, the ones
       you want, because they carry the rp/b and t0/period covariances;
    2. failing that (it is routinely not positive definite once the GP block
       is included), a likelihood-profile bisection per parameter for
       ``delta lnL = 1/2`` -- **conditional** errors, which ignore covariance
       and are therefore lower limits.

    Step sizes for the Hessian are a fixed fraction of each parameter's
    *bound span*, not of its value: two of the parameters are offsets sitting
    near zero at the optimum, so a value-relative step is meaningless for
    them, and for the period offset a step that looks tiny in days is an
    enormous phase shift once multiplied by a few hundred cycles.
    """
    n = len(x)
    f0 = nll(x)
    if not np.isfinite(f0) or f0 >= 1e9:
        return {"method": "none", "reason": "optimum is not finite"}

    def clipped(v):
        return np.array([np.clip(vi, lo, hi) for vi, (lo, hi) in zip(v, bounds)])

    free = np.array([hi > lo for (lo, hi) in bounds])
    steps = np.array([max(1e-3 * (hi - lo), 1e-12) for (lo, hi) in bounds])
    sig = None
    cov = None
    try:
        # A fixed parameter contributes a zero row and column, which makes the
        # Hessian singular; invert the free block and pad the result back out.
        idx = np.flatnonzero(free)
        H = np.zeros((n, n))
        for i in range(n):
            if not free[i]:
                continue
            for j in range(i, n):
                if not free[j]:
                    continue
                if i == j:
                    xp, xm = x.copy(), x.copy()
                    xp[i] += steps[i]
                    xm[i] -= steps[i]
                    H[i, i] = (nll(clipped(xp)) - 2 * f0 + nll(clipped(xm))) / steps[i] ** 2
                else:
                    xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                    xpp[i] += steps[i]; xpp[j] += steps[j]
                    xpm[i] += steps[i]; xpm[j] -= steps[j]
                    xmp[i] -= steps[i]; xmp[j] += steps[j]
                    xmm[i] -= steps[i]; xmm[j] -= steps[j]
                    H[i, j] = H[j, i] = (
                        nll(clipped(xpp)) - nll(clipped(xpm))
                        - nll(clipped(xmp)) + nll(clipped(xmm))
                    ) / (4 * steps[i] * steps[j])
        sub = np.linalg.inv(H[np.ix_(idx, idx)])
        cand = np.zeros((n, n))
        cand[np.ix_(idx, idx)] = sub
        var = np.diag(cand)
        if np.isfinite(var).all() and (var[idx] > 0).all():
            sig, cov = np.sqrt(np.clip(var, 0, None)), cand
            method = ("laplace, full Hessian (marginal over the free parameters"
                      + ("; GP hyperparameters held fixed, so their "
                         "uncertainty is not included)" if not free.all() else ")"))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    if sig is None:
        sig = _profile_errors(nll, x, bounds, f0)
        method = ("likelihood profile, one parameter at a time "
                  "(conditional: covariance is not included, so these are "
                  "lower limits)")
        if sig is None:
            return {"method": "none",
                    "reason": "Hessian is not positive definite and the "
                              "likelihood profile did not bracket 1 sigma"}

    _, _, rp_f, a_f, _, _ = _unpack(x, period0, t0_0)
    out = {
        "method": method,
        "t0": float(sig[0]), "period": float(sig[1]),
        "rp_rs": float(sig[2] * rp_f), "a_rs": float(sig[3] * a_f),
        "b": float(sig[4]),
        # d(depth)/d(log rp) = 2 rp^2, to first order and ignoring the limb
        # darkening and impact-parameter terms; adequate beside a depth that
        # is itself quoted to three figures
        "depth_ppm": float(2 * sig[2] * rp_f ** 2 * 1e6),
    }
    if cov is not None:
        var = np.diag(cov)
        out["correlation_t0_period"] = (
            float(cov[0, 1] / np.sqrt(var[0] * var[1])) if var[0] * var[1] > 0 else None)
        out["correlation_rp_b"] = (
            float(cov[2, 4] / np.sqrt(var[2] * var[4])) if var[2] * var[4] > 0 else None)
    return out


def _profile_errors(nll, x, bounds, f0, target: float = 0.5, max_iter: int = 30):
    """Per-parameter 1-sigma from ``delta(-lnL) = 1/2``, by bisection.

    Robust where a finite-difference Hessian is not: it only ever evaluates
    the likelihood at points where it is finite, and it cannot be defeated by
    a nearly-singular cross term.
    """
    n = len(x)
    sig = np.zeros(n)
    for i in range(n):
        lo, hi = bounds[i]
        span = hi - lo
        if span <= 0:
            sig[i] = 0.0          # held fixed
            continue

        def d(delta):
            y = x.copy()
            y[i] = np.clip(x[i] + delta, lo, hi)
            v = nll(y)
            return (v - f0) if np.isfinite(v) and v < 1e9 else np.inf

        # bracket: grow until delta(-lnL) exceeds the target
        step = 1e-4 * span
        while step < span and d(step) < target and d(-step) < target:
            step *= 2.0
        if step >= span:
            sig[i] = np.nan
            continue

        left_ok = d(step) >= target
        sign = 1.0 if left_ok else -1.0
        a_, b_ = 0.0, sign * step
        for _ in range(max_iter):
            mid = 0.5 * (a_ + b_)
            if d(mid) < target:
                a_ = mid
            else:
                b_ = mid
        sig[i] = abs(0.5 * (a_ + b_))

    if not np.isfinite(sig).all():
        # a single unconstrained nuisance parameter should not lose every error
        median = np.nanmedian(sig) if np.isfinite(sig).any() else np.nan
        if not np.isfinite(median):
            return None
        sig = np.where(np.isfinite(sig), sig, np.nan)
    return sig


# --------------------------------------------------------------------------
# prediction / detrending


def gp_predict(t, f, err, fit: FitResult, t_out=None):
    """``(transit_model, gp_mean)`` evaluated on ``t_out`` (default ``t``).

    The GP is predicted from ``f - transit_model``, never from ``f`` -- that is
    what keeps it from absorbing the transit.  Subtracting the returned
    ``gp_mean`` from the flux leaves a detrended series with the transit
    intact, which is what every downstream per-epoch test wants.
    """
    t = np.asarray(t, float)
    model = transit_model(t, fit.t0, fit.period, fit.rp_rs, fit.a_rs, fit.b,
                          u=fit.u, exp_time_days=fit.exp_time_days)
    resid = np.asarray(f, float) - model
    gp_mean = gp_predict_masked(fit.gp, t, resid, err,
                                np.ones(t.size, dtype=bool), t_out=t_out)
    if t_out is not None:
        model = transit_model(t_out, fit.t0, fit.period, fit.rp_rs, fit.a_rs,
                              fit.b, u=fit.u, exp_time_days=fit.exp_time_days)
    return model, np.asarray(gp_mean, float)


def detrend(t, f, err, fit: FitResult):
    """Flux with the GP removed and the transit retained, normalised to 1."""
    _, gp_mean = gp_predict(t, f, err, fit)
    return np.asarray(f, float) - gp_mean


def transit_implied_density(fit: FitResult) -> float:
    return rho_circ(fit.period, fit.a_rs)


# --------------------------------------------------------------------------
# the local-baseline alternative
#
# A global GP is the right model for a star whose variability is stationary.
# On the most active stars in this sample it is not, and the failure is not
# subtle: on one young M dwarf the GP-marginalised likelihood prefers *no transit* to the
# real one by 617 in log likelihood, because a 3.1% kernel amplitude against a
# 0.25% transit lets the GP explain the transits itself.  The fold at the same
# ephemeris shows a clean 2400 ppm, 4-hour dip.
#
# This fit replaces the global GP with a low-order polynomial baseline in a
# window around each transit, fitted **simultaneously** with a shared transit
# model rather than before it.  That is the same shape as the notch filter,
# which was the best-performing search arm on this sample by a wide margin
# (67% known-planet recovery against 56% for the GP-simultaneous arm), and it
# has the property the GP lacks here: a quadratic over half a day cannot
# absorb a three-hour dip, whatever the star is doing.
#
# The baseline coefficients are linear given the transit, so they are solved
# exactly inside the objective and never enter the optimiser.  Only the five
# transit parameters are searched, which is why this is faster than the GP fit
# despite fitting more parameters.


#: The local polynomial baseline has to be narrow compared with the rotation
#: period, or it is being asked to follow the star rather than to level a
#: window.  Measured on a simulated 1.4 d rotator with a 4.15 h transit: at
#: Prot/5.4 the fit recovers 4372 of 5027 ppm with the right duration; at
#: Prot/2.7 it recovers 3 ppm with every parameter on a bound.  The default of
#: 3 durations exceeds Prot/3 for any star rotating in under a day with a
#: three-hour transit, which is a large part of this sample.
LOCAL_WINDOW_PROT_FRACTION = 5.0

#: ...but not so narrow that no out-of-transit baseline is left.  Below this
#: there are fewer cadences outside the transit than inside it.
LOCAL_WINDOW_MIN_DURATIONS = 1.25


def local_window_durations(requested: float, dur_days: float, prot=None):
    """Cap a local baseline half-window at a fraction of the rotation period.

    Returns ``(window_in_durations, note_or_None)``.  With no rotation period
    the request is returned unchanged -- there is nothing to cap against.
    """
    if not (prot and np.isfinite(prot) and prot > 0 and dur_days > 0):
        return float(requested), None
    cap = (prot / LOCAL_WINDOW_PROT_FRACTION) / dur_days
    if cap >= requested:
        return float(requested), None
    win = max(cap, LOCAL_WINDOW_MIN_DURATIONS)
    note = (f"local baseline window narrowed from {requested:g} to {win:.2f} "
            f"transit durations: Prot = {prot:.3f} d, and a polynomial over "
            f"more than Prot/{LOCAL_WINDOW_PROT_FRACTION:g} follows the star "
            f"instead of levelling the window")
    if win > cap:
        note += (f" (floored at {LOCAL_WINDOW_MIN_DURATIONS:g} durations, so "
                 f"it still spans Prot/{prot / (win * dur_days):.1f}: the "
                 f"local fit is marginal on this star)")
    return float(win), note


def local_joint_fit(t, f, err, period, t0, dur_days, u=(0.42, 0.22),
                    exp_time_days=None, rho_star=None,
                    window_durations: float = 3.0, poly_order: int = 2,
                    n_starts: int = 4, depth_guess: float | None = None,
                    max_drift_durations: float = 1.0,
                    period_halfwidth: float | None = None,
                    prot: float | None = None) -> FitResult:
    """Transit + per-transit polynomial baselines, fitted together.

    ``window_durations`` is the half-width of each window in transit
    durations; ``poly_order`` the baseline order.  Both should come from
    :func:`petcare.prep.choose_baseline`, which picks them by measuring
    injected-depth recovery against null scatter on this star's own data
    rather than by rule -- a fixed window suppressed the depths of 82 of 104
    candidates in the previous search.

    ``prot`` caps the window at a fraction of the rotation period; see
    :func:`local_window_durations`.  Pass it whenever it is known: on a fast
    rotator the default window is wide enough for the baseline to follow the
    star, and then the transit is what is left over.
    """
    window_durations, window_note = local_window_durations(
        window_durations, dur_days, prot)
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    err = np.asarray(err, float)
    if not np.isfinite(err).all():
        err = np.where(np.isfinite(err),
                       err, 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2))

    out = FitResult(exp_time_days=float(exp_time_days or np.nan),
                    n_points=int(t.size), u=tuple(u))
    out.gp = GPSpec(kind="local polynomial", success=True,
                    detail={"window_durations": float(window_durations),
                            "poly_order": int(poly_order)})

    win = window_durations * dur_days
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    ep = np.round((t - t0) / period).astype(int)

    # windows, as index arrays, with the design matrix for each precomputed
    windows = []
    for e in np.unique(ep):
        sel = np.flatnonzero((ep == e) & (np.abs(ph) < win))
        if sel.size < poly_order + 8:
            continue
        in_n = int((np.abs(ph[sel]) < 0.5 * dur_days).sum())
        oot_n = int((np.abs(ph[sel]) >= 0.75 * dur_days).sum())
        if in_n < 3 or oot_n < 8:
            continue
        # scaled local time, so the Vandermonde matrix stays conditioned
        x = (t[sel] - (t0 + e * period)) / win
        V = np.vander(x, poly_order + 1)
        windows.append((sel, V, 1.0 / err[sel] ** 2))

    if len(windows) < 2:
        out.reason = (f"only {len(windows)} window(s) with enough in- and "
                      f"out-of-transit coverage")
        return out

    if depth_guess is not None and np.isfinite(depth_guess) and depth_guess > 0:
        depth0 = float(depth_guess)
    else:
        intr = np.abs(ph) < 0.5 * dur_days
        oot = (np.abs(ph) >= 0.75 * dur_days) & (np.abs(ph) < 3.0 * dur_days)
        depth0 = max(float(np.median(f[oot]) - np.median(f[intr])), 30e-6)
    rp0 = float(np.sqrt(depth0))

    if rho_star and rho_star > 0:
        a0 = a_over_rs(period, rho_star)
        a_lo = max(2.0, a0 / A_RS_TOLERANCE)
        a_hi = min(300.0, a0 * A_RS_TOLERANCE)
        a_bound_source = f"stellar density {rho_star:.3f} g/cm3"
    else:
        s_ = np.sin(np.pi * dur_days / period)
        a0 = float((1 + rp0) / s_) if 0 < s_ < 1 else 15.0
        a_lo, a_hi = 2.0, 300.0
        a_bound_source = "unbounded (no stellar density available)"
    a0 = float(np.clip(a0, a_lo, a_hi))

    baseline = max(float(np.ptp(t)), period)
    p_half = period_halfwidth if period_halfwidth is not None else min(
        max(max_drift_durations * dur_days * period / baseline, 1e-6),
        0.01 * period)

    def chi2(theta):
        t0_off, dper, log_rp, log_a, b = theta
        per_i, rp_i, a_i = period + dper, np.exp(log_rp), np.exp(log_a)
        if per_i <= 0 or a_i <= 1.0:
            return 1e12
        try:
            model = transit_model(t, t0 + t0_off, per_i, rp_i, a_i, b, u=u,
                                  exp_time_days=exp_time_days)
        except Exception:  # noqa: BLE001 - infeasible geometry
            return 1e12
        total = 0.0
        for sel, V, w in windows:
            # flux = transit * polynomial, so the design matrix is the
            # polynomial scaled by the transit model: linear in the
            # coefficients, solved exactly rather than searched
            A = V * model[sel][:, None]
            y = f[sel]
            sw = np.sqrt(w)
            try:
                coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
            except np.linalg.LinAlgError:
                return 1e12
            total += float(np.sum(w * (y - A @ coef) ** 2))
        return total if np.isfinite(total) else 1e12

    bounds = [(-0.5 * dur_days, 0.5 * dur_days), (-p_half, p_half),
              (np.log(0.002), np.log(0.6)), (np.log(a_lo), np.log(a_hi)),
              (0.0, 0.98)]

    rng = np.random.default_rng(20260921)
    starts = [np.array([0.0, 0.0, np.log(rp0), np.log(a0), 0.3])]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.array([
            rng.uniform(-0.3, 0.3) * dur_days, 0.0,
            np.log(rp0 * rng.uniform(0.6, 1.6)),
            np.log(np.clip(a0 * rng.uniform(0.6, 1.5), a_lo, a_hi)),
            rng.uniform(0.0, 0.85)]))

    best = None
    for theta0 in starts:
        theta0 = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(theta0, bounds)])
        try:
            res = minimize(chi2, theta0, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=600, maxfun=4000))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(res.fun) and res.fun < 1e11 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        out.reason = "every optimisation start failed"
        return out

    t0_off, dper, log_rp, log_a, b_f = best.x
    per_f, rp_f, a_f = period + dper, float(np.exp(log_rp)), float(np.exp(log_a))
    t0_f = t0 + t0_off

    out.success = True
    out.period, out.t0 = float(per_f), float(t0_f)
    out.rp_rs, out.a_rs, out.b = rp_f, a_f, float(b_f)
    out.logl = float(-0.5 * best.fun)
    out.n_starts = len(starts)

    fine = np.linspace(t0_f - 0.6 * dur_days, t0_f + 0.6 * dur_days, 601)
    mm = transit_model(fine, t0_f, per_f, rp_f, a_f, b_f, u=u, exp_time_days=None)
    out.depth_ppm = float((1.0 - mm.min()) * 1e6)
    out.t14_h = float(duration_t14(per_f, a_f, rp_f, b_f) * 24.0)

    # the same windows and baselines with no transit at all
    flat = chi2(np.array([t0_off, dper, np.log(0.002), log_a, b_f]))
    out.logl_no_transit = float(-0.5 * flat)
    out.delta_logl = float(out.logl - out.logl_no_transit)

    out.errors = _hessian_errors(lambda th: 0.5 * chi2(th), best.x, bounds,
                                 period, t0, out.gp)
    n_in = sum(int((np.abs(ph[sel]) < 0.5 * dur_days).sum()) for sel, _, _ in windows)
    out.detail = {
        "method": "local polynomial baselines, fitted simultaneously",
        "n_windows": len(windows),
        "window_durations": float(window_durations),
        "window_note": window_note,
        "poly_order": int(poly_order),
        "n_baseline_coefficients": len(windows) * (poly_order + 1),
        "a_rs_bounds": [float(a_lo), float(a_hi)],
        "a_rs_bound_source": a_bound_source,
        "period_halfwidth": float(p_half),
        "start_depth_ppm": float(depth0 * 1e6),
        "n_in_transit": n_in,
        "chi2": float(best.fun),
        "converged": bool(best.success),
        "railed": _railed(best.x, bounds, out.gp),
    }
    return out


def local_baseline_trend(t, f, err, fit: FitResult, window_durations=3.0,
                         poly_order=2):
    """The fitted per-window baselines, evaluated at ``t`` (1.0 outside them).

    Used for plotting and for handing a detrended series to a later test.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    err = np.asarray(err, float)
    dur_days = fit.t14_h / 24.0
    win = window_durations * dur_days
    ph = ((t - fit.t0 + 0.5 * fit.period) % fit.period) - 0.5 * fit.period
    ep = np.round((t - fit.t0) / fit.period).astype(int)
    model = transit_model(t, fit.t0, fit.period, fit.rp_rs, fit.a_rs, fit.b,
                          u=fit.u, exp_time_days=fit.exp_time_days)

    trend = np.ones_like(f)
    for e in np.unique(ep):
        sel = np.flatnonzero((ep == e) & (np.abs(ph) < win))
        if sel.size < poly_order + 8:
            continue
        x = (t[sel] - (fit.t0 + e * fit.period)) / win
        V = np.vander(x, poly_order + 1)
        A = V * model[sel][:, None]
        w = 1.0 / np.maximum(err[sel], 1e-12) ** 2
        sw = np.sqrt(w)
        try:
            coef, *_ = np.linalg.lstsq(A * sw[:, None], f[sel] * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        trend[sel] = V @ coef
    return model, trend


# ---------------------------------------------------------------------------
# One variability model per observing segment
# ---------------------------------------------------------------------------
#
# A single set of hyperparameters shared across sectors years apart describes
# no epoch of the star.  Spot patterns on a 10 Myr star evolve on weeks: the
# amplitude in one sector is routinely a factor of two from the next, and the
# fit has to settle on a compromise -- too much flexibility where the star was
# quiet, too little where it was active.  Too much flexibility is the
# dangerous direction, because that is where the GP reaches down into a
# transit.
#
# Splitting the kernel by sector and keeping one transit shared across all of
# them costs nothing in the transit parameters (they are still constrained by
# every sector at once) and gives each sector the variability model it
# actually needs.  The transit has to survive in all of them, which is the
# point: this buys a better duration and depth, not a lower bar for existence.

#: Unmasked cadences a segment needs before it gets its own kernel.  Below
#: this the hyperparameters are fitted on too little of a rotation cycle to
#: mean anything, and the segment is merged into its neighbour.  200 binned
#: 10-minute cadences is about 1.4 days.
SEGMENT_MIN_POINTS = 200

#: Time gap that separates two observing blocks when no sector labels exist.
SEGMENT_GAP_DAYS = 10.0


def labels_at(t_new, t_ref, labels_ref):
    """Carry per-point labels from one time array onto another.

    Used to put sector numbers onto a binned series: each new time takes the
    label of the nearest original cadence.  Sectors are contiguous in time, so
    nearest-neighbour is exact everywhere except inside a bin that straddles a
    sector boundary, and there either answer is defensible.
    """
    t_new = np.asarray(t_new, float)
    t_ref = np.asarray(t_ref, float)
    labels_ref = np.asarray(labels_ref)
    if t_ref.size == 0:
        return np.full(t_new.size, -1, dtype=labels_ref.dtype)
    order = np.argsort(t_ref)
    ts, ls = t_ref[order], labels_ref[order]
    idx = np.clip(np.searchsorted(ts, t_new), 0, ts.size - 1)
    left = np.clip(idx - 1, 0, ts.size - 1)
    take_left = np.abs(t_new - ts[left]) <= np.abs(t_new - ts[idx])
    return np.where(take_left, ls[left], ls[idx])


def block_labels(t, gap_days: float = SEGMENT_GAP_DAYS):
    """Contiguous-observing-block labels, for data with no sector column."""
    t = np.asarray(t, float)
    order = np.argsort(t)
    lab = np.zeros(t.size, dtype=int)
    k = 0
    for i in range(1, order.size):
        if t[order[i]] - t[order[i - 1]] > gap_days:
            k += 1
        lab[order[i]] = k
    return lab


def group_segments(t, labels, usable, min_points: int = SEGMENT_MIN_POINTS):
    """Merge short segments into their neighbours, in time order.

    Returns ``(index, ids)``: ``index`` is a per-point group number (-1 for
    points whose label was negative, i.e. unknown sector), and ``ids`` is the
    list of original labels that went into each group, so the record can say
    "sectors 7+8" rather than "segment 1".
    """
    t = np.asarray(t, float)
    labels = np.asarray(labels)
    usable = np.asarray(usable, bool)

    known = labels >= 0 if np.issubdtype(labels.dtype, np.integer) else np.ones(t.size, bool)
    uniq = [lb for lb in dict.fromkeys(labels[known][np.argsort(t[known])])]

    groups, current = [], []
    for lb in uniq:
        current.append(lb)
        n = int((np.isin(labels, current) & usable).sum())
        if n >= min_points:
            groups.append(current)
            current = []
    if current:                      # tail too short: fold into the last group
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    index = np.full(t.size, -1, dtype=int)
    for i, g in enumerate(groups):
        index[np.isin(labels, g)] = i
    return index, [list(map(_plain, g)) for g in groups]


def _plain(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


@dataclass
class SegmentedGP:
    """A list of :class:`GPSpec`, one per observing segment.

    Duck-types :class:`GPSpec` everywhere the rest of the package touches a
    variability model, with ``n_free = 0``: the hyperparameters are always
    held fixed in the joint fit.  Freeing them per segment would hand the
    optimiser one flexibility knob per sector, which is the failure this whole
    arrangement exists to avoid.
    """

    specs: list = field(default_factory=list)
    index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ids: list = field(default_factory=list)
    success: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    names: tuple = ()
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    bounds: list = field(default_factory=list)

    @property
    def n_free(self) -> int:
        return 0

    @property
    def n_segments(self) -> int:
        return len(self.specs)

    @property
    def kind(self) -> str:
        kinds = sorted({s.kind for s in self.specs})
        return f"{'/'.join(kinds)} per segment" if kinds else "segmented"

    def _weights(self):
        return np.array([float((self.index == i).sum()) for i in range(len(self.specs))])

    def amplitude(self, x=None) -> float:
        """The amplitude of the *most* variable segment.

        Not the mean: this number exists to be compared with the transit
        depth, and the question it answers is whether there is a segment in
        which the variability could swallow the signal.
        """
        a = [s.amplitude() for s in self.specs if s.success]
        a = [v for v in a if np.isfinite(v)]
        return float(max(a)) if a else float("nan")

    def timescale(self, x=None) -> float:
        """The shortest timescale any segment allows -- likewise the worst case."""
        ts = [s.timescale() for s in self.specs if s.success]
        ts = [v for v in ts if np.isfinite(v)]
        return float(min(ts)) if ts else float("nan")

    def to_dict(self, x=None) -> dict:
        d = {"kind": self.kind, "success": self.success, "reason": self.reason,
             "n_segments": self.n_segments,
             "amplitude": self.amplitude(), "min_timescale_days": self.timescale(),
             "segments": [dict(ids=self.ids[i] if i < len(self.ids) else None,
                               n_points=int((self.index == i).sum()),
                               **s.to_dict())
                          for i, s in enumerate(self.specs)]}
        d.update(self.detail)
        return d

    def kernel(self, x=None):
        raise TypeError("a segmented model has one kernel per segment; "
                        "use gp_log_likelihood or gp_trend_masked")

    def index_for(self, t):
        """Segment number for arbitrary times.

        ``index`` is sized for the array the model was fitted on; anything
        else (an unbinned series, a plotting grid) is assigned by the time
        span each segment covers, and times outside every span take the
        nearest one.  Segments are separated by weeks at least, so this is
        unambiguous in practice.
        """
        t = np.asarray(t, float)
        if t.size == self.index.size:
            return self.index
        spans = self.detail.get("time_spans") or []
        if not spans:
            return np.zeros(t.size, dtype=int)
        out = np.full(t.size, -1, dtype=int)
        for i, (lo, hi) in enumerate(spans):
            out[(t >= lo) & (t <= hi)] = i
        if (out < 0).any():
            mid = np.array([0.5 * (lo + hi) for lo, hi in spans])
            out[out < 0] = np.argmin(np.abs(t[out < 0, None] - mid[None, :]), axis=1)
        return out


def fit_gp_hyperparameters_segmented(
        t, f, mask_out, labels, min_rho_days: float,
        max_rho_days: float | None = None, prot: float | None = None,
        q0_min: float = Q0_MIN, prot_tol: float = PROT_TOL,
        mask_flares: bool = True, min_points: int = SEGMENT_MIN_POINTS,
        fallback: GPSpec | None = None,
        free: str = "amplitude") -> SegmentedGP:
    """Fit one variability model per segment of ``labels``.

    ``free`` decides how much of the kernel is allowed to differ between
    segments:

    ``"amplitude"`` (default)
        only the amplitude, with the rotation period, quality factor or
        correlation length pinned to ``fallback``'s global values.  This is
        the part that actually changes between sectors -- a young star's spot
        coverage evolves on weeks, its rotation and the coherence of its spot
        pattern do not -- and it is the safe part.
    ``"all"``
        every hyperparameter refitted per segment.  Measured on the known
        planets, this loses real transits: with every segment free, HIP 67522 c
        came back at 3 ppm against a true 3600 ppm, because a quarter of the
        data is not enough to pin a quality factor and the segment that
        happened to fit a flexible one absorbed its transits.  Kept for
        comparison, not recommended.

    ``fallback`` (the global fit) is used for any segment whose own fit fails,
    so one bad sector degrades to the old behaviour there rather than losing
    the whole model.  A segmented model with no successfully fitted segment of
    its own is returned unsuccessful, and the caller should keep the global
    one.
    """
    if free == "amplitude" and not (fallback is not None and fallback.success):
        return SegmentedGP(success=False,
                           reason="amplitude-only segmentation needs a global "
                                  "model to pin the kernel shape to")
    pin_to = (np.asarray(fallback.values, float)
              if free == "amplitude" else None)
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    keep = ~np.asarray(mask_out, bool)

    index, ids = group_segments(t, labels, keep, min_points=min_points)
    n_seg = len(ids)
    if n_seg < 2:
        return SegmentedGP(success=False,
                           reason=f"only {n_seg} segment(s) long enough "
                                  f"({min_points} unmasked cadences each)")

    specs, n_own, notes = [], 0, []
    for i in range(n_seg):
        m = index == i
        sub = fit_gp_hyperparameters(
            t[m], f[m], mask_out=np.asarray(mask_out, bool)[m],
            min_rho_days=min_rho_days, max_rho_days=max_rho_days, prot=prot,
            q0_min=q0_min, prot_tol=prot_tol, mask_flares=mask_flares,
            pin_to=pin_to)
        if sub.success:
            n_own += 1
        elif fallback is not None and fallback.success:
            notes.append(f"segment {ids[i]}: {sub.reason}; used the global model")
            sub = GPSpec(kind=fallback.kind, names=fallback.names,
                         values=np.array(fallback.values, float),
                         bounds=list(fallback.bounds), success=True,
                         reason="global model (this segment's own fit failed)",
                         detail=dict(fallback.detail))
        else:
            return SegmentedGP(success=False,
                               reason=f"segment {ids[i]} failed ({sub.reason}) "
                                      f"and there is no global model to fall back on")
        specs.append(sub)

    unassigned = int((index < 0).sum())
    seg = SegmentedGP(specs=specs, index=index, ids=ids, success=n_own > 0,
                      reason=f"{n_own} of {n_seg} segments fitted on their own data")
    seg.detail = {"n_segments": n_seg, "segment_ids": ids,
                  "free_per_segment": free,
                  "n_fitted_independently": n_own,
                  "n_points_unassigned": unassigned,
                  "min_points_per_segment": int(min_points),
                  "amplitude_per_segment": [float(s.amplitude()) for s in specs],
                  "amplitude_spread": _spread([s.amplitude() for s in specs]),
                  "time_spans": [[float(t[index == i].min()),
                                  float(t[index == i].max())]
                                 for i in range(n_seg)],
                  "notes": notes}
    if unassigned:
        # points whose sector is unknown: give them the nearest segment rather
        # than dropping them from the likelihood entirely
        seg.index = np.where(index >= 0, index,
                             labels_at(t, t[index >= 0], index[index >= 0])
                             if (index >= 0).any() else 0)
    return seg


def _spread(values) -> float:
    """max/min of the finite positive entries -- how unequal the segments are."""
    v = [float(x) for x in values if np.isfinite(x) and x > 0]
    return float(max(v) / min(v)) if len(v) >= 2 else float("nan")


def is_segmented(spec) -> bool:
    return isinstance(spec, SegmentedGP)


def gp_log_likelihood(spec, x, t, resid, err, jitter: float = 0.0) -> float:
    """Log likelihood of ``resid`` under ``spec``, segmented or not.

    Segments are independent by construction -- they are separated by weeks to
    years, far beyond any kernel timescale here -- so the total is the sum.
    """
    if not is_segmented(spec):
        return float(make_gp(spec, x, t, err, jitter).log_likelihood(resid))
    idx = spec.index_for(t)
    total = 0.0
    for i, sub in enumerate(spec.specs):
        m = idx == i
        if m.sum() < 3:
            continue
        total += float(make_gp(sub, None, t[m], err[m], jitter).log_likelihood(resid[m]))
    return total


def gp_predict_masked(spec, t, resid, err, keep, t_out=None):
    """GP mean at ``t_out``, conditioned on ``resid[keep]``, segment by segment."""
    t = np.asarray(t, float)
    out_t = t if t_out is None else np.asarray(t_out, float)
    pred = np.zeros(out_t.size)
    if not is_segmented(spec):
        gp = make_gp(spec, None, t[keep], err[keep])
        return np.asarray(gp.predict(resid[keep], t=out_t), float)
    # each output time is predicted by the segment it falls in
    in_index = spec.index_for(t)
    out_index = in_index if t_out is None else spec.index_for(out_t)
    for i, sub in enumerate(spec.specs):
        m_in = (in_index == i) & keep
        m_out = out_index == i
        if not m_out.any():
            continue
        if m_in.sum() < 10:
            pred[m_out] = 0.0
            continue
        gp = make_gp(sub, None, t[m_in], err[m_in])
        pred[m_out] = np.asarray(gp.predict(resid[m_in], t=out_t[m_out]), float)
    return pred

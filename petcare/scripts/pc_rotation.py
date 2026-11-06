"""pc_rotation -- is the candidate period a harmonic of the rotation period?

**This test flags and never fails.**  That is the whole point of it.  Using
rotation proximity as a veto in the previous search killed 25 real-looking
signals, one of which was flagged in a published paper, and for HIP 67522 c --
a real planet -- the eight strongest periodogram peaks are all 1x, 2x or 3x
the rotation period.  Proximity to Prot is a reason to look harder, never a
reason to discard.

What it reports:

* **Prot**, measured two ways.  The Lomb-Scargle peak and the autocorrelation
  peak, per sector, plus the extraction's own values.  The two disagree often
  and in a specific way -- LS lands on a harmonic, and its argmax can rail at
  whatever upper period limit it was given -- so both are reported with their
  sector-to-sector scatter, and the ACF is preferred.
* **How close the candidate is** to Prot and to its low harmonics and
  sub-harmonics, in units of the width over which a fold stays coherent, not
  in per cent.  A 1% period difference is enormous on a 3000 d baseline and
  negligible on a single sector.
* **Whether the transit depth tracks rotation phase**, which is the mechanism
  that actually matters: a signal manufactured by spot modulation has a depth
  that depends on where in the rotation cycle each epoch fell.

Two specific traps are guarded here.  TESS has a ~13.7 d orbit whose
systematics produce rotation-like periodicity, so a "Prot" near 13.7 d or its
harmonics is called out as suspect rather than reported as a stellar property.
And a Lomb-Scargle argmax that sits at the edge of its own search range is a
non-detection, not a measurement.

    python -m petcare.scripts.pc_rotation --tic 166527623 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, prep as pcprep
from ..plotting import (C_BAD, C_BINNED, C_MODEL, C_OK, annotate_status,
                        figure, save)
from ..result import Result

TEST = "rotation"

#: Per-sector fractional scatter above which the extraction's own Prot is a
#: non-detection rather than a measurement, and this test measures its own.
#: PROVISIONAL: 0.25 is the stability HIP 67522's extraction achieved (0.2%
#: across five sectors) rounded up generously; candidates that fail
#: it fail by 200-700%, so nothing sits near the boundary.
EXTRACTION_SCATTER_MAX = 0.25
VERSION = "0.2.1"

#: TARS v2 rotation periods (a parquet table cut to TICID /
#: adopted_period), preferred over our own measurement when the star is in it.
#: The extraction's ACF is under 0.2 d for 46% of the survey's
#: stars, piled at 0.10-0.15 d by its lag floor; TARS gives > 1 d for 7,382 of
#: them (one young multi-planet host: ACF 0.107 d, TARS 2.937 d, LS 2.93 d).
#: This test only -- the GP kernels still take io.rotation_period(); swapping TARS in there made
#: the rotation kernel absorb two of that star's known planets.
TARS_ENV = "PETCARE_TARS"
#: An extraction ACF period below this, disagreeing with the extraction LS by
#: more than ACF_LS_AGREE, is the ACF's lag floor and is replaced by the LS.
ACF_FLOOR_DAYS = 0.2
ACF_LS_AGREE = 0.2

#: The TESS spacecraft orbit, in days.  Systematics on this period and its
#: harmonics masquerade as rotation and as transits.
TESS_ORBIT_DAYS = 13.7
TESS_ORBIT_TOL = 0.06          # fractional

#: Harmonics of Prot tested against the candidate period.
HARMONICS = (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0)

#: Proximity, in coherence widths, inside which the candidate is called a
#: harmonic of the rotation period.  The coherence width is
#: ``P * T14 / baseline`` -- the period change that shifts the fold by one
#: duration across the data, which is the only scale on which "close" means
#: anything.
COHERENCE_FLAG = 3.0

#: ...or within this fraction of a harmonic, whichever is looser.
#:
#: The coherence width alone is the wrong scale, and it failed in the one place
#: it mattered.  It answers "are these two *ephemerides* distinguishable?", which
#: over an eight-year baseline is a question about parts per million: on one
#: star a candidate at 0.70583 d sits 0.04% from 2 x Prot (0.70556 d), which
#: is 94 coherence widths, so the test called it "not near a low harmonic".  But
#: rotation is not a clock.  An ACF period is good to about a percent, and
#: differential rotation on a young star spreads the spot period by several
#: percent more, so nothing about the rotation is known to 0.04%.  The question
#: being asked is whether the candidate could be the rotation signal, and on that
#: question 0.04% is a match.
#:
#: 3% is the tolerance our search itself uses for its rot_harmonic flag,
#: which keeps the two in agreement.  This remains a FLAG: proximity to rotation
#: is not evidence against a planet, and pc_phasecurve is the test that decides
#: whether the dip is the rotation.
HARMONIC_FRAC = 0.03


def add_arguments(ap):
    g = ap.add_argument_group("rotation")
    g.add_argument("--bin", type=float, default=1800.0,
                   help="bin for the periodogram; rotation is a slow signal "
                        "and half-hour sampling is ample")
    g.add_argument("--p-min", type=float, default=0.1)
    g.add_argument("--p-max", type=float, default=30.0,
                   help="upper period limit for the periodogram; a peak at "
                        "this value is reported as railed, not as a detection")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    try:
        lc = ctx.lc
    except Exception as exc:  # noqa: BLE001
        return r.inconclusive(f"no usable light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)

    from ..fitting import bin_for_fit
    tb, fb, eb, nb, bin_info = bin_for_fit(lc.t, lc.f, lc.err,
                                           float(getattr(a, "bin", 1800.0)))
    r.set_noise(*_noise(fb))

    p_min = float(getattr(a, "p_min", 0.1))
    p_max = float(getattr(a, "p_max", 30.0))

    # ---- rotation from the extraction, per sector -------------------------
    from_extraction = lc.rotation_period()

    # ---- our own measurement, per sector and combined ---------------------
    ls = _lomb_scargle(tb, fb, p_min, p_max)
    acf = _acf_period(tb, fb, p_min, min(p_max, max(1.0, lc.baseline_days / 4)))

    per_sector = []
    for s in lc.sectors:
        if s.sector is None:
            continue
        m = lc.sector_id == s.sector
        if m.sum() < 100:
            continue
        t_s, f_s = lc.t[m], lc.f[m]
        ts, fs, es, ns, _ = bin_for_fit(t_s, f_s, lc.err[m],
                                        float(getattr(a, "bin", 1800.0)))
        span = float(np.ptp(ts))
        one = _lomb_scargle(ts, fs, p_min, min(p_max, max(1.0, span / 2)))
        per_sector.append({"sector": s.sector, "baseline_days": span,
                           "ls_period": one.get("period"),
                           "ls_power": one.get("power"),
                           "ls_railed": one.get("railed"),
                           "extraction_acf": s.prot_acf_days,
                           "extraction_ls": s.prot_ls_days})

    # preferred value: the extraction's ACF, which was stable to 0.2% across
    # five sectors on HIP 67522 where its LS jumped to a harmonic on one
    prot = from_extraction.get("prot_days")
    prot_source = {"user": "supplied with --prot",
                   "ls_measured": "Lomb-Scargle (per sector, this run)"}.get(
        from_extraction.get("source"), f"extraction {from_extraction.get('source')}")
    user_prot = from_extraction.get("source") == "user"
    ls_ext = [v for v in (from_extraction.get("ls_days") or []) if v and np.isfinite(v)]
    acf_floor = bool(prot is not None and from_extraction.get("source") == "acf"
                     and prot < ACF_FLOOR_DAYS and ls_ext
                     and abs(float(np.median(ls_ext)) / prot - 1) > ACF_LS_AGREE)
    if acf_floor:
        prot, prot_source = float(np.median(ls_ext)), (
            f"extraction ls, replacing an extraction ACF of {prot:.3f} d "
            f"(below {ACF_FLOOR_DAYS:g} d: the ACF's lag floor)")

    # ...but only when it agrees with itself.  On the faint end of this sample
    # the per-sector ACF scatters by 200-700%, which is a non-detection wearing
    # a number.  Taking it anyway made this test INCONCLUSIVE on 21 of 22
    # candidates -- the test was effectively absent on exactly the population
    # it exists for.  When it is that unstable, measure Prot here instead, on
    # the stitched curve, where there is far more baseline to work with.
    ext_scatter = from_extraction.get("sector_scatter_frac")
    extraction_unstable = (ext_scatter is not None
                           and np.isfinite(ext_scatter)
                           and ext_scatter > EXTRACTION_SCATTER_MAX)
    if prot is not None and extraction_unstable:
        fallback = None
        if acf.get("period"):
            fallback = (acf["period"], "autocorrelation (this run)")
        elif ls.get("period") and not ls.get("railed"):
            fallback = (ls["period"], "Lomb-Scargle (this run)")
        if fallback:
            prot, prot_source = fallback
            prot_source += (f", after the extraction's per-sector Prot "
                            f"disagreed with itself by {ext_scatter:.0%}")
        else:
            prot = None

    if prot is None and acf.get("period"):
        prot, prot_source = acf["period"], "autocorrelation (this run)"
    if prot is None and ls.get("period") and not ls.get("railed"):
        prot, prot_source = ls["period"], "Lomb-Scargle (this run)"

    tars = _tars_period(ctx.target.tic)
    ours = (prot, prot_source)
    if tars is not None and not user_prot:
        prot, prot_source = tars, "TARS v2 adopted_period"

    spurious = _tess_orbit_check(prot)

    r.metrics.update({
        "prot_days": prot, "prot_source": prot_source,
        "from_extraction": from_extraction,
        "lomb_scargle": ls, "autocorrelation": acf,
        "per_sector": per_sector,
        "prot_sector_scatter_frac": from_extraction.get("sector_scatter_frac"),
        "tess_orbit_alias": spurious,
        "tars_days": tars, "ours_days": ours[0], "ours_source": ours[1],
        "acf_lag_floor_replaced": acf_floor,
    })

    if prot is None:
        r.set_self_check(False, "no rotation period could be measured")
        return r.inconclusive(
            "no rotation period: neither the extraction nor this run found "
            "one. Not a statement about the candidate.")

    checks, ok = [f"Prot = {prot:.4f} d from {prot_source}"], True
    scatter = from_extraction.get("sector_scatter_frac")
    if scatter is not None:
        checks.append(f"extraction sector-to-sector scatter {scatter:.1%}")
    # The stability check applies to the value actually used.  An unstable
    # extraction that has already been replaced above is reported, not held
    # against the number that replaced it.
    if scatter is not None and prot_source.startswith("extraction") \
            and scatter > EXTRACTION_SCATTER_MAX:
        checks.append("Prot is not stable between sectors")
        ok = False
    if ls.get("railed"):
        checks.append(f"the Lomb-Scargle peak sits at the search limit "
                      f"({p_max} d) and is not a measurement")
    if spurious["is_alias"]:
        checks.append(spurious["note"])
        # TARS models the TESS-orbit systematics itself (its periods show a
        # deficit at 13.7 d, not a pile-up), so near-alias is reported, not failed
        if not prot_source.startswith("TARS"):
            ok = False
    r.set_self_check(ok, "; ".join(checks))

    # ---- proximity of the candidate to Prot and its harmonics -------------
    period = ctx.candidate.period
    prox = []
    closest = None
    if period:
        dur_days = ctx.candidate.duration_days() or 0.05 * period ** (1 / 3)
        coh = period * dur_days / max(lc.baseline_days, period)
        for h in HARMONICS:
            target = prot * h
            d = abs(period - target)
            entry = {"harmonic": float(h), "target_period": float(target),
                     "difference_days": float(d),
                     "difference_frac": float(d / period),
                     "difference_coherence_widths": float(d / coh) if coh > 0 else None}
            prox.append(entry)
            if closest is None or d < closest["difference_days"]:
                closest = entry
        r.metrics["coherence_width_days"] = float(coh)
        r.metrics["proximity"] = prox
        r.metrics["closest_harmonic"] = closest

    # ---- does the depth track rotation phase? -----------------------------
    depth_vs_phase = _depth_vs_rotation_phase(ctx, lc, prot)
    r.metrics["depth_vs_rotation_phase"] = depth_vs_phase

    # ---- verdict (flag only) ----------------------------------------------
    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif period is None:
        r.passed(f"Prot = {prot:.4f} d ({prot_source}); no candidate period "
                 f"given, so nothing to compare it with")
    elif closest and (closest["difference_frac"] < HARMONIC_FRAC or (
            closest["difference_coherence_widths"] is not None
            and closest["difference_coherence_widths"] < COHERENCE_FLAG)):
        h = closest["harmonic"]
        r.metrics["is_rotation_harmonic"] = True
        r.flag(f"the candidate period {period:.5f} d is {h:g}x the rotation "
               f"period {prot:.4f} d to within "
               f"{closest['difference_frac']:.2%} -- inside what a spot period "
               f"can be measured to. FLAG ONLY -- rotation "
               f"proximity is not evidence against a planet; using it as a "
               f"veto previously killed 25 signals including a published one. "
               f"Check the per-epoch depths and the transit shape instead.")
    elif depth_vs_phase.get("sigma") is not None and depth_vs_phase["sigma"] > 3:
        r.flag(f"the candidate period is not a rotation harmonic (closest is "
               f"{closest['harmonic']:g}x at "
               f"{closest['difference_coherence_widths']:.0f} coherence widths), "
               f"but the per-epoch depth correlates with rotation phase at "
               f"{depth_vs_phase['sigma']:.1f} sigma -- look for spot crossings")
    else:
        r.passed(f"Prot = {prot:.4f} d ({prot_source}); the candidate period "
                 f"{period:.5f} d is not near a low harmonic "
                 f"(closest {closest['harmonic']:g}x, "
                 f"{closest['difference_frac']:.1%} away)")

    if ctx.make_plots:
        try:
            _plot(ctx, r, tb, fb, prot, ls, acf, period)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


_TARS = None


def _tars_period(tic):
    """TARS v2 adopted period for a TIC, or None (no table, not in it)."""
    global _TARS
    import os
    if _TARS is None:
        path = os.environ.get(TARS_ENV)
        _TARS = {}
        if path and os.path.exists(path):
            import pandas as pd
            t = pd.read_parquet(path, columns=["TICID", "adopted_period"])
            _TARS = dict(zip(t.TICID.astype("int64"), t.adopted_period.astype(float)))
    try:
        v = _TARS.get(int(tic))
    except (TypeError, ValueError):
        return None
    return float(v) if v is not None and np.isfinite(v) and v > 0 else None


def _noise(f):
    from ..noise import point_to_point_ppm
    return point_to_point_ppm(f), "1.4826*MAD(diff)/sqrt(2)"


def _lomb_scargle(t, f, p_min, p_max, n_freq=20000) -> dict:
    """Lomb-Scargle peak, with an explicit railing check.

    A peak at the edge of the search range is a non-detection dressed as a
    measurement: the argmax had nowhere else to go.  Reported as ``railed``
    rather than silently returned as Prot.
    """
    from astropy.timeseries import LombScargle

    t = np.asarray(t, float)
    f = np.asarray(f, float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size < 50 or p_max <= p_min:
        return {"period": None, "reason": "too few points"}
    freq = np.linspace(1.0 / p_max, 1.0 / p_min, n_freq)
    try:
        power = LombScargle(t, f).power(freq)
    except Exception as exc:  # noqa: BLE001
        return {"period": None, "reason": f"{type(exc).__name__}: {exc}"}
    i = int(np.argmax(power))
    period = float(1.0 / freq[i])
    railed = bool(i <= 2 or i >= n_freq - 3)
    # the strongest peaks, so harmonic structure is visible rather than implied
    order = np.argsort(power)[::-1]
    peaks, seen = [], []
    for j in order:
        pj = 1.0 / freq[j]
        if any(abs(pj - q) < 0.02 * q for q in seen):
            continue
        seen.append(pj)
        peaks.append({"period": float(pj), "power": float(power[j])})
        if len(peaks) >= 5:
            break
    return {"period": period, "power": float(power[i]), "railed": railed,
            "p_min": p_min, "p_max": p_max, "top_peaks": peaks,
            "note": ("the peak is at the search limit and is not a measurement"
                     if railed else "")}


def _acf_period(t, f, p_min, p_max) -> dict:
    """Autocorrelation peak on a regularly-resampled series."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good]
    if t.size < 100 or p_max <= p_min:
        return {"period": None, "reason": "too few points"}
    dt = float(np.median(np.diff(np.sort(t))))
    if not np.isfinite(dt) or dt <= 0:
        return {"period": None, "reason": "no usable cadence"}

    grid = np.arange(t.min(), t.max(), dt)
    if grid.size > 200000:
        grid = np.linspace(t.min(), t.max(), 200000)
        dt = grid[1] - grid[0]
    y = np.interp(grid, t, f) - np.mean(f)
    # zero out the long gaps between sectors so they cannot create a peak
    y[np.interp(grid, t, np.arange(t.size)) % 1 > 0.999] = 0.0

    ac = np.correlate(y, y, mode="full")[y.size - 1:]
    if ac[0] <= 0:
        return {"period": None, "reason": "zero variance"}
    ac = ac / ac[0]
    lags = np.arange(ac.size) * dt

    sel = (lags >= p_min) & (lags <= p_max)
    if sel.sum() < 10:
        return {"period": None, "reason": "period range does not fit the baseline"}
    idx = np.flatnonzero(sel)
    # first local maximum above a modest height, not the global argmax: the
    # global maximum of an ACF often lands on a multiple of the true period
    best = None
    a = ac[idx]
    for j in range(1, a.size - 1):
        if a[j] > a[j - 1] and a[j] > a[j + 1] and a[j] > 0.1:
            best = (float(lags[idx[j]]), float(a[j]))
            break
    if best is None:
        # The global maximum of a falling ACF is just the shortest lag, p_min.
        return {"period": None, "reason": "no local maximum above 0.1"}
    return {"period": best[0], "strength": best[1], "p_min": p_min, "p_max": p_max}


def _tess_orbit_check(prot) -> dict:
    """Is this 'rotation period' really the spacecraft orbit?"""
    if not prot:
        return {"is_alias": False, "note": ""}
    for mult in (0.25, 1 / 3, 0.5, 1.0, 2.0):
        target = TESS_ORBIT_DAYS * mult
        if abs(prot - target) < TESS_ORBIT_TOL * target:
            return {"is_alias": True, "multiple": float(mult),
                    "orbit_days": TESS_ORBIT_DAYS,
                    "note": (f"Prot = {prot:.3f} d is {mult:g}x the ~13.7 d TESS "
                             f"orbit; this is very likely a systematic, not the "
                             f"star")}
    return {"is_alias": False, "note": ""}


def _depth_vs_rotation_phase(ctx, lc, prot) -> dict:
    """Does the per-epoch transit depth depend on rotation phase?

    The mechanism that matters: a signal manufactured by spot modulation has
    epoch depths that track where in the rotation cycle each transit fell.
    A real planet's should not (spot *crossings* add scatter, but not a
    coherent sinusoid in rotation phase).
    """
    if not (ctx.candidate.period and ctx.candidate.t0_btjd is not None
            and ctx.candidate.duration_h and prot):
        return {"sigma": None, "reason": "needs an ephemeris and a Prot"}
    try:
        p = pcprep.prepare(ctx, bin_seconds=600.0)
        eds = pcprep.epoch_depths(p, lc=lc)
    except Exception as exc:  # noqa: BLE001
        return {"sigma": None, "reason": f"{type(exc).__name__}: {exc}"}
    if len(eds) < 6:
        return {"sigma": None, "reason": f"only {len(eds)} epochs"}

    phase = np.array([(e.t_center / prot) % 1.0 for e in eds])
    d = np.array([e.depth_ppm for e in eds])
    w = 1.0 / np.array([e.depth_err_ppm for e in eds]) ** 2

    x = 2 * np.pi * phase
    A = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x)])
    W = np.diag(w)
    try:
        cov = np.linalg.inv(A.T @ W @ A)
        coef = cov @ (A.T @ W @ d)
    except np.linalg.LinAlgError:
        return {"sigma": None, "reason": "singular design matrix"}
    amp = float(np.hypot(coef[1], coef[2]))
    amp_err = float(np.sqrt(abs(cov[1, 1]) + abs(cov[2, 2])) / np.sqrt(2))
    return {"amplitude_ppm": amp, "amplitude_err_ppm": amp_err,
            "sigma": float(amp / amp_err) if amp_err > 0 else None,
            "n_epochs": len(eds),
            "note": "sinusoid in rotation phase fitted to the per-epoch depths"}


def _plot(ctx, r, t, f, prot, ls, acf, period):
    from astropy.timeseries import LombScargle

    fig, axes = figure(2, 1, figsize=(10, 6.5))

    ax = axes[0]
    ax.plot(t, (f - 1) * 1e6, ".", ms=1.0, color="#4c4c4c", alpha=0.4,
            rasterized=True)
    ax.set_xlabel("BTJD")
    ax.set_ylabel("flux - 1 [ppm]")
    ax.set_title(f"light curve (Prot = {prot:.4f} d)" if prot else "light curve")

    ax = axes[1]
    p_lo = ls.get("p_min", 0.1)
    p_hi = ls.get("p_max", 30.0)
    freq = np.linspace(1 / p_hi, 1 / p_lo, 8000)
    good = np.isfinite(t) & np.isfinite(f)
    power = LombScargle(t[good], f[good]).power(freq)
    ax.plot(1 / freq, power, "-", color=C_BINNED, lw=0.9)
    if prot:
        ax.axvline(prot, color=C_OK, lw=1.4, label=f"Prot {prot:.3f} d")
        for h in (0.5, 2.0, 3.0):
            ax.axvline(prot * h, color=C_OK, lw=0.7, ls=":", alpha=0.7)
    if period:
        ax.axvline(period, color=C_MODEL, lw=1.4, label=f"candidate {period:.4f} d")
    ax.axvline(TESS_ORBIT_DAYS, color=C_BAD, lw=1.0, ls="--",
               label=f"TESS orbit {TESS_ORBIT_DAYS} d")
    ax.set_xscale("log")
    ax.set_xlabel("period [d]")
    ax.set_ylabel("LS power")
    ax.set_title("Lomb-Scargle periodogram")
    ax.legend(ncol=3, loc="upper left")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("rotation")))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Measure the rotation period and report how close the candidate "
              "is to it. Flags only -- never fails a candidate."))
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

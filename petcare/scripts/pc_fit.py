"""pc_fit -- simultaneous GP + transit fit.

The backbone of PetCare.  Refines the ephemeris, fits the transit and the
star's variability together, and reports P, t0, depth, T14, b, a/R* with
uncertainties and the ephemeris drift it had to apply.  Other tests consume
its JSON via ``--use-fit``; none of them import it.

Run it::

    python -m petcare.scripts.pc_fit --tic 166527623 \\
        --period 6.9596 --t0 1604.02 --duration 4.85 --out out

Two plots are mandatory and both are written:

* **the GP over the raw data**, so a human can see the GP tracking stellar
  variability rather than the transit;
* **the phase fold with the model**, so the thing being fitted is visible.

The self-check is the GP-amplitude-to-depth ratio.  Calibrated on 71 known
planets, that ratio has a median of 2.6 where the fit recovered the planet and
10.8 where it did not; a large value means the GP has enough freedom to have
absorbed the signal and the fit should not be trusted on its own.
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, ephem
from ..fitting import (A_RS_TOLERANCE, FitResult, bin_for_fit, block_labels,
                       fit_gp_hyperparameters,
                       fit_gp_hyperparameters_segmented, gp_predict,
                       gp_trend_masked, joint_fit, labels_at,
                       local_baseline_trend, local_joint_fit, refine_ephemeris,
                       transit_model)
from ..noise import noise_block, point_to_point_ppm, red_noise_factor
from ..plotting import (C_BINNED, C_MODEL, annotate_status, figure, plot_fold,
                        plot_series, save)
from ..result import Result
from ..stellar import duration_t14, reconcile_density, rho_circ

TEST = "fit"
VERSION = "0.2.1"

#: GP amplitude / transit depth thresholds.  On 71 known planets this ratio
#: had a median of 2.6 where the GP+transit fit recovered the planet and 10.8
#: where it did not, so it is the available discriminator for "did the GP eat
#: the transit".  PROVISIONAL: set from that published pair of medians, not yet
#: from a PetCare calibration run.  See calibration/README.md -- rule 7 says a
#: test with an uncalibrated threshold is not finished.
GP_SIGMA_OVER_DEPTH_FLAG = 8.0
GP_SIGMA_OVER_DEPTH_INCONCLUSIVE = 20.0

#: Ephemeris drift, in durations at the far end of the baseline, above which
#: the refinement is large enough that anything folded on the input ephemeris
#: was measuring the wrong cadences.
DRIFT_FLAG_DURATIONS = 0.25


def add_arguments(ap):
    g = ap.add_argument_group("fit")
    g.add_argument("--fit-bin", type=float, default=600.0,
                   help="bin the light curve to this many seconds before "
                        "fitting (batman supersamples to match); 0 disables")
    g.add_argument("--no-refine", action="store_true",
                   help="fit at the input ephemeris without refining it")
    g.add_argument("--refine-drift", type=float, default=3.0,
                   help="period search half-width, expressed as the transit "
                        "drift it allows across the whole baseline, in "
                        "durations. Scales correctly from one sector to ten years.")
    g.add_argument("--min-rho-durations", type=float, default=3.0,
                   help="GP correlation length floor, in transit durations. "
                        "The bound that stops the GP absorbing the transit.")
    g.add_argument("--starts", type=int, default=4,
                   help="optimiser restarts")
    g.add_argument("--no-rotation-kernel", action="store_true",
                   help="use a Matern-3/2 kernel even when Prot is known "
                        "(weaker on spotted stars; see petcare/fitting.py)")
    g.add_argument("--variability", choices=("auto", "gp", "local"),
                   default="auto",
                   help="how stellar variability is modelled alongside the "
                        "transit. 'gp': one global Gaussian process. 'local': "
                        "a polynomial baseline per transit window, fitted "
                        "simultaneously (the notch form). 'auto': run both "
                        "and keep whichever recovers the model-independent "
                        "box depth -- on the most active stars a global GP "
                        "explains the transits itself and returns nothing.")
    g.add_argument("--gp-segments", choices=("sector", "block", "none"),
                   default="none",
                   help="fit a separate GP per observing segment, with one "
                        "transit shared across all of them. 'sector': one per "
                        "TESS sector. 'block': one per contiguous observing "
                        "block. 'none' (default): one global model. The "
                        "physical argument for splitting is good -- a 10 Myr "
                        "star's spot amplitude changes between sectors -- but "
                        "measured on the nine validation planets it is a "
                        "wash: depths move by 10-25%%, toward the literature "
                        "value about as often as away. Default off until a "
                        "larger sample says otherwise.")
    g.add_argument("--segment-min-points", type=int, default=200,
                   help="cadences a segment needs before it gets its own "
                        "kernel; shorter segments merge into their neighbour")
    g.add_argument("--segment-free", choices=("amplitude", "all"),
                   default="amplitude",
                   help="which hyperparameters may differ between segments. "
                        "'amplitude': only the amplitude, with the kernel's "
                        "shape pinned to the global fit -- the part that "
                        "really changes between sectors. 'all': everything, "
                        "which on the known planets loses real transits "
                        "(HIP 67522 c came back at 3 ppm against 3600).")
    g.add_argument("--local-window", type=float, default=3.0,
                   help="local baseline half-window, in transit durations")
    g.add_argument("--local-order", type=int, default=2,
                   help="local baseline polynomial order")
    g.add_argument("--free-gp", action="store_true",
                   help="refit the GP hyperparameters jointly with the "
                        "transit, within a bounded neighbourhood of their "
                        "transit-masked values. Off by default: the "
                        "hyperparameter likelihood is nearly degenerate on "
                        "these stars and a joint fit spends the transit on it.")
    g.add_argument("--mask-known", action="store_true", default=True,
                   help="mask other known signals on this star while fitting")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 are required"):
        return r

    try:
        lc = ctx.lc
    except Exception as exc:  # noqa: BLE001 - a missing light curve is normal
        return r.inconclusive(f"no usable light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)

    period0 = float(ctx.candidate.period)
    t0_0 = float(ctx.candidate.t0_btjd)
    star = ctx.star
    rho_star, e_rho_star = star.density()

    # duration: given, else from the star's density at b = 0.5
    dur_days = ctx.candidate.duration_days()
    dur_source = "input"
    if dur_days is None:
        if rho_star:
            from ..stellar import a_over_rs
            dur_days = duration_t14(period0, a_over_rs(period0, rho_star), 0.05, b=0.5)
            dur_source = "stellar density, b=0.5"
        else:
            dur_days = 0.05 * period0 ** (1 / 3)
            dur_source = "crude P^(1/3) scaling (no stellar density)"

    # ------------------------------------------------------------------ data
    t, f, err = lc.t, lc.f, lc.err
    known_mask = np.zeros(t.size, dtype=bool)
    if getattr(a, "mask_known", True) and ctx.known:
        # Overlap, not period proximity: a known planet at 2x or 1/2 the
        # candidate's period must still be recognised and masked.
        others = [k for k in ctx.known
                  if not ephem.match_known(period0, t0_0, dur_days * 24, [k],
                                           float(t.min()), float(t.max()))]
        known_mask = ephem.mask_known_transits(t, others, width=2.0)
        if known_mask.any():
            ctx.notes.append(
                f"masked {int(known_mask.sum())} cadences in "
                f"{len(others)} other known signal(s)")

    keep = ~known_mask
    t, f, err = t[keep], f[keep], err[keep]
    if not cli.require(r, t.size > 100, f"only {t.size} cadences after masking"):
        return r

    bin_s = float(getattr(a, "fit_bin", 600.0) or 0.0)
    tb, fb, eb, nb, bin_info = bin_for_fit(t, f, err, bin_s)
    exp_days = bin_s / 86400.0 if bin_info.get("binned") else lc.exp_time_days
    r.provenance["fit_bin_s"] = bin_s if bin_info.get("binned") else lc.median_cadence_s
    r.provenance["n_points_fitted"] = int(tb.size)
    r.provenance["binning"] = bin_info

    # The GP's timescale floor and the local window are both set in transit
    # durations, and the duration is only the search's guess until the fit has
    # measured it.  The search routinely reports 0.3-0.5 h for dips the fit
    # makes 2-3 h, which left the GP free to vary inside the fitted transit:
    # 31% of the 804 fits refuted by jointgp alone (2026-10-28) had a GP
    # faster than their own T14, against 8% of survivors, and on fast rotators
    # that GP carves a rotation minimum into a transit-shaped dip.  So when the
    # fitted T14 exceeds the seed by more than REFIT_DURATION_RATIO, the whole
    # block runs once more seeded with the fitted duration.  Only upward: a
    # shorter re-seed lowers the floor and frees the GP, and on the known
    # planets that turned one planet's 785 ppm fit into 6900 (literature 714).
    seed_dur = dur_days
    for fit_pass in (1, 2):
        # -------------------------------------------------------------- GP prior
        # Hyperparameters first, on transit-masked data, so that the variability
        # model describes the star rather than the candidate.
        min_rho = float(getattr(a, "min_rho_durations", 3.0)) * dur_days
        in_any = ephem.in_transit(tb, period0, t0_0, dur_days, width=2.5)
        if ctx.known:
            in_any |= ephem.mask_known_transits(tb, ctx.known, width=2.0)

        rot = lc.rotation_period()
        prot = rot.get("prot_days")
        if getattr(a, "no_rotation_kernel", False):
            prot = None
        if prot is not None and prot < 1.5 * dur_days:
            # A rotation period barely longer than the transit leaves the kernel
            # free to vary within the transit anyway; fall back rather than
            # pretend the constraint is doing something.
            ctx.notes.append(f"Prot={prot:.3f} d is under 1.5 durations; "
                             f"using a Matern kernel instead of a rotation kernel")
            prot = None

        gp_prior = fit_gp_hyperparameters(tb, fb, mask_out=in_any,
                                          min_rho_days=min_rho,
                                          max_rho_days=max(30.0, period0 * 5),
                                          prot=prot)

        # One kernel per segment, transit shared.  The global fit above is kept
        # as the fallback for any segment that cannot fit its own.
        seg_mode = getattr(a, "gp_segments", "sector")
        gp_fit_prior = gp_prior
        if seg_mode != "none":
            if seg_mode == "sector" and getattr(lc, "sector_id", None) is not None \
                    and np.size(lc.sector_id) == np.size(keep):
                seg_labels = labels_at(tb, t, np.asarray(lc.sector_id)[keep])
            else:
                seg_labels = block_labels(tb)
                if seg_mode == "sector":
                    ctx.notes.append("no per-point sector labels; segmented the "
                                     "GP by contiguous observing block instead")
            segmented = fit_gp_hyperparameters_segmented(
                tb, fb, mask_out=in_any, labels=seg_labels, min_rho_days=min_rho,
                max_rho_days=max(30.0, period0 * 5), prot=prot,
                min_points=int(getattr(a, "segment_min_points", 200)),
                fallback=gp_prior,
                free=str(getattr(a, "segment_free", "amplitude")))
            if segmented.success:
                ctx.notes.append(
                    f"variability fitted per segment: {segmented.n_segments} "
                    f"segments {segmented.ids}, amplitudes spanning a factor of "
                    f"{segmented.detail.get('amplitude_spread', float('nan')):.1f}")
                # Only the fit gets the segmented model.  The period search below
                # keeps the global one: a search detrended segment by segment lost
                # two known planets outright in testing, because a per-segment
                # model removes slightly different things from each sector and the
                # box SNR of a transit split across them drops.
                gp_fit_prior = segmented
            else:
                r.provenance.setdefault("notes", []).append(
                    f"kept one global variability model: {segmented.reason}")
        r.provenance["gp_segments"] = seg_mode

        # ------------------------------------------------------- ephemeris refine
        # The search runs on a variability-removed series.  On a star with 3%
        # spot modulation and a 4000 ppm transit, a box search on the raw flux
        # finds the spots: the first version of this test moved HIP 67522 b's
        # period by 0.3% and came back with a third of its true depth.
        refined = {"success": False, "reason": "refinement disabled"}
        period, t0 = period0, t0_0
        if gp_prior.success:
            f_search = fb - gp_trend_masked(tb, fb, eb, in_any, gp_prior) + 1.0
        else:
            f_search = fb
            ctx.notes.append("GP hyperparameter fit failed; "
                             "the period search ran on undetrended flux")
        if not getattr(a, "no_refine", False):
            refined = refine_ephemeris(
                tb, f_search, period0, t0_0, dur_days,
                max_drift_durations=float(getattr(a, "refine_drift", 3.0)))
            if refined.get("success"):
                period, t0 = refined["period"], refined["t0_btjd"]

        drift = ephem.ephemeris_drift(period0, t0_0, period, t0,
                                      t_ref=float(tb.max()), duration_h=dur_days * 24)

        # ------------------------------------------------------------- the fit
        u1, u2, ld_source = star.limb_darkening()

        # Starting depth measured on the variability-removed series.  See the
        # note in joint_fit: a depth seeded from raw flux on an active star is
        # noise, and the optimiser never recovers from it.
        depth_guess = None
        if refined.get("success") or gp_prior.success:
            ph_g = ephem.phase(tb, period, t0)
            gi = np.abs(ph_g) < 0.4 * dur_days
            go = (np.abs(ph_g) >= 0.75 * dur_days) & (np.abs(ph_g) < 3.0 * dur_days)
            if gi.sum() >= 5 and go.sum() >= 20:
                src = f_search if gp_prior.success else fb
                depth_guess = float(np.median(src[go]) - np.median(src[gi]))
                r.provenance["depth_guess_ppm"] = depth_guess * 1e6

        fit_kwargs = dict(
            u=(u1, u2), exp_time_days=exp_days, rho_star=rho_star,
            gp_prior=gp_fit_prior,
            min_rho_durations=float(getattr(a, "min_rho_durations", 3.0)),
            n_starts=int(getattr(a, "starts", 4)), depth_guess=depth_guess,
            max_drift_durations=float(getattr(a, "refine_drift", 3.0)) / 3.0,
            gp_free=bool(getattr(a, "free_gp", False)))
        mode = getattr(a, "variability", "auto")
        fit_gp = (joint_fit(tb, fb, eb, period, t0, dur_days, **fit_kwargs)
                  if mode in ("auto", "gp") else None)
        fit_local = None
        if mode in ("auto", "local"):
            fit_local = local_joint_fit(
                tb, fb, eb, period, t0, dur_days, u=(u1, u2),
                exp_time_days=exp_days, rho_star=rho_star,
                prot=(rot.get("prot_days") if rot else None),
                window_durations=float(getattr(a, "local_window", 3.0)),
                poly_order=int(getattr(a, "local_order", 2)),
                n_starts=int(getattr(a, "starts", 4)), depth_guess=depth_guess,
                max_drift_durations=float(getattr(a, "refine_drift", 3.0)) / 3.0)
            if fit_local is not None and fit_local.detail.get("window_note"):
                ctx.notes.append(fit_local.detail["window_note"])

        fit, variability_choice = _choose_fit(mode, fit_gp, fit_local, depth_guess)
        t14_fit = getattr(fit, "t14_h", None)
        if fit_pass == 2 or fit is None or not t14_fit or not np.isfinite(t14_fit) \
                or (t14_fit / 24.0) / dur_days < REFIT_DURATION_RATIO:
            break
        ctx.notes.append(f"fitted T14 {t14_fit:.2f} h against the {dur_days * 24:.2f} h "
                         f"seed ({dur_source}); re-ran the variability model and the "
                         f"fit seeded with the fitted duration")
        dur_days, dur_source = t14_fit / 24.0, f"pc_fit first pass (seed was {dur_source})"
    r.provenance["fit_passes"] = fit_pass

    # Second pass, only if the residual is genuinely noisier than the errors
    # claimed.  Correcting in this direction is safe; pre-inflating the
    # diagonal is not, because the GP simply absorbs whatever slack it is
    # given (see the note in fitting.bin_for_fit).
    err_rescale = 1.0
    if fit is not None and fit.success and fit.gp.kind != "local polynomial":
        res0 = fb - transit_model(tb, fit.t0, fit.period, fit.rp_rs, fit.a_rs,
                                  fit.b, u=fit.u, exp_time_days=exp_days)
        _, gp0 = gp_predict(tb, fb, eb, fit)
        meas = point_to_point_ppm(res0 - gp0 + 1.0) / 1e6
        claimed = float(np.median(eb))
        if np.isfinite(meas) and claimed > 0 and meas > 1.1 * claimed:
            err_rescale = float(meas / claimed)
            fit2 = joint_fit(tb, fb, eb * err_rescale, period, t0, dur_days,
                             **fit_kwargs)
            if fit2.success:
                fit = fit2
                eb = eb * err_rescale
            ctx.notes.append(
                f"residual scatter is {err_rescale:.2f}x the propagated error; "
                f"refitted with rescaled errors")
    r.provenance["error_rescale"] = err_rescale
    if not cli.require(r, fit is not None and fit.success,
                       f"fit did not converge: "
                       f"{fit.reason if fit is not None else 'no fit was run'}"):
        r.metrics["refine"] = refined
        return r

    # ------------------------------------------------------------- coverage
    cov = ephem.epoch_coverage(lc, fit.period, fit.t0, fit.t14_h / 24.0)
    cov_sum = ephem.coverage_summary(cov)

    # ---------------------------------------------------------------- noise
    if fit.gp.kind == "local polynomial":
        model_t, trend = local_baseline_trend(
            tb, fb, eb, fit,
            window_durations=float(fit.detail.get("window_durations")
                                   or getattr(a, "local_window", 3.0)),
            poly_order=int(getattr(a, "local_order", 2)))
        resid_full = fb - model_t * trend
        gp_mean = trend - 1.0
    else:
        resid = fb - (transit_model(tb, fit.t0, fit.period, fit.rp_rs, fit.a_rs,
                                    fit.b, u=fit.u, exp_time_days=exp_days))
        _, gp_mean = gp_predict(tb, fb, eb, fit)
        resid_full = resid - gp_mean
    # White noise from the full residual; red noise from the *masked*-GP
    # series instead.  A GP conditioned on all the data is a conditional mean
    # that has already removed structure on every timescale it can reach, so
    # its residual bins down better than white by construction and any
    # red-noise factor measured from it is meaningless (it came out at 0.25 on
    # a target whose depth was in fact recovered to 2%).  The masked-GP series
    # is the one the per-epoch tests actually measure depths against, and its
    # out-of-transit part is where a depth uncertainty's inflation belongs.
    nb_block = noise_block(tb, resid_full + 1.0)
    rn = {}
    if gp_prior.success:
        oot_mask = ephem.out_of_transit(tb, fit.period, fit.t0,
                                        fit.t14_h / 24.0, inner=0.75, outer=8.0)
        if oot_mask.sum() > 50:
            rn = red_noise_factor(tb[oot_mask], f_search[oot_mask],
                                  fit.t14_h / 24.0)
    r.set_noise(nb_block["white_ppm"], nb_block["estimator"],
                red_noise_factor=rn.get("red_noise_factor"),
                red_noise_timescale_days=fit.t14_h / 24.0,
                red_noise_binned_rms_ppm=rn.get("binned_rms_ppm"),
                red_noise_n_bins=rn.get("n_bins"),
                red_noise_measured_on="out-of-transit, masked-GP detrended",
                gp_amplitude_ppm=fit.gp_amplitude * 1e6,
                raw_scatter_ppm=point_to_point_ppm(fb))

    # ---------------------------------------------------------- self-check
    is_local = fit.gp.kind == "local polynomial"
    ratio = fit.gp_sigma_over_depth
    checks, ok, gp_dominated = [], True, False
    if is_local:
        # A polynomial baseline has no amplitude, so the GP-to-depth
        # diagnostic does not apply.  Its equivalent is whether the baseline
        # could have absorbed the transit, which the window and order settle:
        # a quadratic over 6 durations cannot.
        checks.append(f"local baseline: order "
                      f"{fit.detail.get('poly_order')} over "
                      f"{fit.detail.get('window_durations')} durations, "
                      f"{fit.detail.get('n_windows')} windows")
        if fit.depth_ppm <= 0:
            checks.append("fitted depth is not positive")
            ok = False
    elif not np.isfinite(ratio):
        checks.append("GP/depth ratio undefined (depth is zero or negative)")
        ok = False
    else:
        checks.append(f"GP amplitude / depth = {ratio:.1f}")
        if ratio > GP_SIGMA_OVER_DEPTH_INCONCLUSIVE:
            ok, gp_dominated = False, True
    rnf = r.noise.get("red_noise_factor")
    if isinstance(rnf, (int, float)) and np.isfinite(rnf):
        checks.append(f"red noise x{rnf:.1f} white at the transit timescale")
    if fit.detail.get("railed"):
        checks.append("parameters at a bound: " + ", ".join(fit.detail["railed"]))
        # Same rule as _valid(): rp_rs or b on a bound means the depth or the
        # geometry is unconstrained, and the fit is not a measurement.  a/R*
        # on a *density-derived* bound means the duration disagrees with a
        # provisional catalogue density, which is a flag and pc_duration's
        # subject, not grounds for throwing the depth away.
        blocking = [pp for pp in fit.detail["railed"] if pp in ("rp_rs", "b")]
        if "a_rs" in fit.detail["railed"] and str(
                fit.detail.get("a_rs_bound_source") or "").startswith("unbounded"):
            blocking.append("a_rs")
        if blocking:
            ok = False
    if cov_sum["n_epochs_usable"] < 2:
        checks.append(f"only {cov_sum['n_epochs_usable']} usable epoch(s)")
        ok = False
    if fit.errors.get("method") == "none":
        checks.append(f"no uncertainties: {fit.errors.get('reason')}")
    r.set_self_check(ok, "; ".join(checks))

    # ---------------------------------------------------------------- metrics
    rho_t = rho_circ(fit.period, fit.a_rs)
    r.metrics = {
        "variability_model": variability_choice,
        "period": fit.period,
        "period_err": fit.errors.get("period"),
        "t0_btjd": fit.t0,
        "t0_err": fit.errors.get("t0"),
        "depth_ppm": fit.depth_ppm,
        "depth_err_ppm": fit.errors.get("depth_ppm"),
        "duration_h": fit.t14_h,
        "rp_rs": fit.rp_rs,
        "rp_rs_err": fit.errors.get("rp_rs"),
        "a_rs": fit.a_rs,
        "a_rs_err": fit.errors.get("a_rs"),
        "b": fit.b,
        "b_err": fit.errors.get("b"),
        "delta_logl_vs_no_transit": fit.delta_logl,
        "gp": fit.gp.to_dict(),
        "gp_amplitude_ppm": fit.gp_amplitude * 1e6,
        "gp_sigma_over_depth": ratio,
        "gp_timescale_over_duration": fit.gp_timescale_over_duration,
        "rotation": rot,
        "rho_transit_gcc": rho_t,
        "rho_star_gcc": rho_star,
        "density_reconciliation": reconcile_density(rho_star, rho_t) if rho_star else None,
        "limb_darkening": {"u1": u1, "u2": u2, "source": ld_source},
        "errors": fit.errors,
        "refine": refined,
        "ephemeris_drift": drift,
        "duration_seed": {"days": dur_days, "source": dur_source, "first_seed_days": seed_dur},
        "gp_prior": gp_prior.to_dict() if gp_prior.success else
                    {"success": False, "reason": gp_prior.reason},
        "coverage": cov_sum,
        "fit_detail": fit.detail,
    }
    if star.radius and fit.rp_rs:
        from ..stellar import planet_radius_earth
        r.metrics["rp_earth"] = planet_radius_earth(fit.rp_rs, star.radius)
        r.metrics["rp_earth_note"] = "from the light curve's depth, which is already de-diluted"

    # ---------------------------------------------------------------- status
    if not ok:
        if gp_dominated:
            r.inconclusive(
                f"GP amplitude is {ratio:.0f}x the depth: the fit cannot be "
                f"separated from stellar variability (P={fit.period:.5f} d, "
                f"depth={fit.depth_ppm:.0f} ppm as fitted)")
        else:
            r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif not is_local and np.isfinite(ratio) and ratio > GP_SIGMA_OVER_DEPTH_FLAG:
        r.flag(f"fit converged (P={fit.period:.6f} d, depth={fit.depth_ppm:.0f} ppm, "
               f"T14={fit.t14_h:.2f} h) but the GP amplitude is {ratio:.0f}x the "
               f"depth; check the GP panel before trusting the depth")
    elif "a_rs" in (fit.detail.get("railed") or []):
        # kept as a measurement, but say so plainly: the fitted duration wants
        # an a/R* the catalogue density does not allow.  pc_duration is the
        # test that quantifies it.
        r.flag(f"fit converged (P={fit.period:.6f} d, depth={fit.depth_ppm:.0f} ppm, "
               f"T14={fit.t14_h:.2f} h) but a/R* = {fit.a_rs:.1f} sits on the "
               f"bound set by the stellar density "
               f"({fit.detail.get('a_rs_bound_source')}): the duration and the "
               f"catalogue density disagree by more than a factor of "
               f"{A_RS_TOLERANCE:g}. The depth is still a measurement; see "
               f"pc_duration for the eccentricity or radius error that would "
               f"reconcile it")
    else:
        drift_d = drift.get("shift_at_tref_durations")
        if drift_d is not None and abs(drift_d) > DRIFT_FLAG_DURATIONS:
            r.flag(f"fit converged but the ephemeris moved {abs(drift_d):.2f} "
                   f"durations over the baseline (P {period0:.5f} -> "
                   f"{fit.period:.5f} d); anything folded on the input "
                   f"ephemeris was off-transit")
        elif refined.get("railed"):
            r.flag(f"refined period sits at the edge of the +/-"
                   f"{getattr(a, 'refine_tol', 0.004):.1%} search window; "
                   f"widen --refine-tol and re-run")
        else:
            r.passed(f"P={fit.period:.6f} d, t0={fit.t0:.5f} BTJD, "
                     f"depth={fit.depth_ppm:.0f} ppm, T14={fit.t14_h:.2f} h, "
                     f"b={fit.b:.2f}, {cov_sum['n_epochs_usable']} usable epochs")

    # ----------------------------------------------------------------- plots
    if ctx.make_plots:
        try:
            _plot(ctx, r, lc, tb, fb, eb, fit, exp_days, known_mask.any(),
                  local_window=float(fit.detail.get("window_durations")
                                     or getattr(a, "local_window", 3.0)),
                  local_order=int(getattr(a, "local_order", 2)))
        except Exception as exc:  # noqa: BLE001 - a failed plot is not a failed test
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, lc, tb, fb, eb, fit: FitResult, exp_days, had_known_mask,
          local_window=3.0, local_order=2):
    """The two mandatory panels, plus the per-epoch stack."""
    if fit.gp.kind == "local polynomial":
        model, trend = local_baseline_trend(tb, fb, eb, fit,
                                            window_durations=local_window,
                                            poly_order=local_order)
        gp_mean = trend - 1.0
    else:
        model, gp_mean = gp_predict(tb, fb, eb, fit)

    fig, axes = figure(3, 1, figsize=(11, 9.5),
                       gridspec_kw={"height_ratios": [1.15, 1, 1]})

    # 1. the GP over the raw data -- is it tracking the star or the transit?
    ax = axes[0]
    boundaries = [s.t_start for s in lc.sectors[1:]]
    plot_series(ax, tb, fb, model=model + gp_mean, gp=gp_mean + 1.0,
                sector_boundaries=boundaries,
                title=("variability model over the data -- it must track the "
                       "star, not the transit  [" + fit.gp.kind + "]"))
    intr = ephem.in_transit(tb, fit.period, fit.t0, fit.t14_h / 24.0)
    if intr.any():
        ax.plot(tb[intr], fb[intr], ".", ms=2.5, color=C_MODEL, alpha=0.9,
                label="in transit")
    ax.legend(loc="upper right", ncol=4)

    # 2. the phase fold with the model
    ax = axes[1]
    det = fb - gp_mean
    ph = ephem.phase(tb, fit.period, fit.t0)
    win = min(3.0 * fit.t14_h / 24.0, 0.45 * fit.period)
    sel = np.abs(ph) < win
    fine = np.linspace(-win, win, 601)
    mfine = transit_model(fine + fit.t0, fit.t0, fit.period, fit.rp_rs,
                          fit.a_rs, fit.b, u=fit.u, exp_time_days=exp_days)
    plot_fold(ax, ph[sel], det[sel], duration_days=fit.t14_h / 24.0,
              model_phase=fine, model_flux=mfine,
              label_data="detrended",
              title="phase fold, variability model removed")
    ax.legend(loc="lower right", ncol=3)

    # 3. per-epoch stack, where nearly every artefact in this sample is visible
    ax = axes[2]
    cov = ephem.epoch_coverage(lc, fit.period, fit.t0, fit.t14_h / 24.0)
    usable = [c for c in cov if c.usable][:14]
    epn = ephem.epoch_number(tb, fit.period, fit.t0)
    offset = max(3.0 * fit.depth_ppm, 300.0)
    for i, c in enumerate(usable):
        m = (epn == c.epoch) & sel
        if m.sum() < 3:
            continue
        y = (det[m] - 1.0) * 1e6 + i * offset
        ax.plot(ph[m] * 24, y, ".", ms=1.8, alpha=0.5, color="#4c4c4c",
                rasterized=True)
        from ..plotting import bin_series
        xb, yb, _, _ = bin_series(ph[m] * 24, y, max(fit.t14_h / 8, 0.15), min_points=2)
        ax.plot(xb, yb, "-", lw=1.1, color=C_BINNED)
        ax.plot(fine * 24, (mfine - 1) * 1e6 + i * offset, "-", lw=0.9,
                color=C_MODEL, alpha=0.8)
        ax.text(win * 24 * 1.02, i * offset, f"e{c.epoch} s{c.sector}",
                fontsize=6, va="center")
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel(f"per-epoch stack [ppm, offset {offset:.0f}]")
    ax.set_title(f"individual epochs ({len(usable)} of "
                 f"{sum(1 for c in cov if c.usable)} usable shown)")
    ax.set_xlim(-win * 24 * 1.05, win * 24 * 1.28)

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("fit")))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Simultaneous GP + transit fit: refined ephemeris, depth, "
              "duration, impact parameter, and the drift it had to apply."))
    args = ap.parse_args(argv)
    ctx = cli.Context.from_args(args)
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        r.error(exc)
    return cli.finish(r, ctx)


#: Factor by which the GP fit's depth may fall short of the local fit's
#: before the local fit is used instead.  A GP and a local polynomial should
#: not disagree about a transit depth by more than this; when they do, it is
#: the GP that has absorbed it, because a quadratic over a few durations
#: cannot.
#:
#: PROVISIONAL, set from the nine known planets on the four validation stars.
#: At 2 the switch fires exactly where the GP is drastically wrong -- one planet
#: (GP 5 ppm against 2272) and another (887 against 4595) -- and leaves
#: the marginal cases with the GP, where the two fits bracket the literature
#: value from either side and neither is clearly better.  At 1.5 it also
#: flipped DS Tuc A b, which gained nothing: the GP gave Rp 4.9 R-earth and
#: the local fit 6.3, against a literature 5.7.
DEPTH_DISAGREEMENT = 2.0


def _valid(fit) -> tuple:
    """Is this fit a measurement at all?  Returns (ok, why not)."""
    if fit is None:
        return False, "not run"
    if not fit.success:
        return False, fit.reason or "did not converge"
    if not (fit.depth_ppm > 0):
        return False, f"depth is {fit.depth_ppm:.1f} ppm"
    if fit.delta_logl is not None and np.isfinite(fit.delta_logl) \
            and fit.delta_logl <= 0:
        return False, (f"the model with a transit is no better than the one "
                       f"without (delta lnL = {fit.delta_logl:.1f})")
    # A shape parameter on a bound is not a measurement of that shape.  t0 and
    # the period are allowed to rail -- that means the ephemeris needs more
    # room, which the drift flag reports separately.
    railed = list(fit.detail.get("railed") or [])
    bad = [p for p in railed if p == "rp_rs"]

    # a/R* is different, and treating it like rp_rs was wrong.  When the
    # stellar density is known, a/R*'s bounds *come from that density* (a
    # factor of A_RS_TOLERANCE around it), so railing there does not mean the
    # fit is unconstrained -- it means the duration disagrees with the
    # catalogue density, on a sample whose densities are provisional.  That
    # is design rule 6, and pc_duration is the test that reports it, with the
    # eccentricity and the stellar-radius error that would reconcile it.
    #
    # Rejecting on it deleted four candidates whose local fits were
    # unambiguous: one at 9255 ppm with delta lnL = 742 against a
    # box depth of 10384, thrown away because a/R* sat on a bound set by a
    # work-in-progress stellar radius.  With no density the bounds are the
    # generic 2-300, and railing there is a real runaway (a known planet once
    # optimised to a/R* = 294), so it still disqualifies.
    if "a_rs" in railed:
        src = str(fit.detail.get("a_rs_bound_source") or "")
        if src.startswith("unbounded"):
            bad.append("a_rs")
    if bad:
        return False, "at a bound: " + ", ".join(bad)
    return True, ""


#: Re-seed the duration and fit again when the fitted T14 exceeds the seed by
#: more than this factor (see the two-pass loop in run).
REFIT_DURATION_RATIO = 1.25


def _choose_fit(mode, fit_gp, fit_local, depth_guess):
    """Pick the variability model **by measurement**, not by preference.

    The two fits arbitrate against each other.  A global GP and a local
    polynomial baseline should agree about a transit depth; when they do not,
    it is the GP that has absorbed the transit, because a quadratic over a few
    transit durations cannot follow a three-hour dip whatever the star is
    doing.  So the GP is kept whenever it reproduces the local fit's depth,
    and replaced when it falls a factor of
    :data:`DEPTH_DISAGREEMENT` short of it.

    An earlier version arbitrated against a box depth measured on the
    GP-detrended series, which fails for the obvious reason: that reference is
    itself GP-suppressed.  It kept a 887 ppm GP fit of a known planet over a
    4595 ppm local one, against a literature ~5000 ppm.

    Either fit is disqualified first if it is not a measurement -- no
    convergence, a non-positive depth, no preference for a transit over no
    transit, or a shape parameter sitting on a bound.
    """
    if mode == "gp":
        return fit_gp, {"mode": "gp", "chosen": "gp", "reason": "requested"}
    if mode == "local":
        return fit_local, {"mode": "local", "chosen": "local",
                           "reason": "requested"}

    ok_gp, why_gp = _valid(fit_gp)
    ok_local, why_local = _valid(fit_local)

    def summarise(fit, ok, why):
        if fit is None:
            return {"ran": False, "reason": why}
        d = {"ran": True, "usable": ok, "depth_ppm": fit.depth_ppm,
             "t14_h": fit.t14_h, "delta_logl": fit.delta_logl,
             "railed": fit.detail.get("railed")}
        if not ok:
            d["rejected_because"] = why
        if fit.gp.kind != "local polynomial":
            d["gp_over_depth"] = fit.gp_sigma_over_depth
        return d

    detail = {"mode": "auto",
              "box_depth_ppm": (depth_guess * 1e6) if depth_guess else None,
              "depth_disagreement_factor": DEPTH_DISAGREEMENT,
              "gp": summarise(fit_gp, ok_gp, why_gp),
              "local": summarise(fit_local, ok_local, why_local)}

    if not ok_gp and not ok_local:
        detail.update(chosen=None,
                      reason=f"neither fit is a measurement (GP: {why_gp}; "
                             f"local: {why_local})")
        return (fit_gp if fit_gp is not None else fit_local), detail
    if not ok_local:
        detail.update(chosen="gp", reason=f"the local fit is unusable ({why_local})")
        return fit_gp, detail
    if not ok_gp:
        detail.update(chosen="local", reason=f"the GP fit is unusable ({why_gp})")
        return fit_local, detail

    ratio = fit_gp.depth_ppm / fit_local.depth_ppm
    detail["gp_over_local_depth"] = float(ratio)
    if ratio >= 1.0 / DEPTH_DISAGREEMENT:
        detail.update(chosen="gp",
                      reason=f"the two models agree on the depth "
                             f"({fit_gp.depth_ppm:.0f} vs "
                             f"{fit_local.depth_ppm:.0f} ppm); kept the GP")
        return fit_gp, detail
    detail.update(chosen="local",
                  reason=f"the GP fit returns {fit_gp.depth_ppm:.0f} ppm "
                         f"against the local baseline fit's "
                         f"{fit_local.depth_ppm:.0f} ppm -- a factor of "
                         f"{1 / ratio:.1f}. A polynomial baseline cannot "
                         f"absorb a transit, so the GP has.")
    return fit_local, detail


if __name__ == "__main__":
    sys.exit(main())

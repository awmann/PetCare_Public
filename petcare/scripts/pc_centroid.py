"""pc_centroid -- difference imaging: which star is the transit signal on?

v0.2.  The image work is in ``petcare.diffimage``; this module turns it into
a verdict.  What changed from v0.1, and why, is in that module's docstring.
In short:

* every pixel gets its own quadratic baseline across every transit window,
  so a young star's variability no longer swamps the depth image;
* the SPOC TESS PRF, registered to the out-of-transit image with every
  catalogued neighbour in the scene, replaces a clipped flux-weighted
  centroid;
* the depth image is fitted with a PRF at a free position (offset measured
  from the *target's* registered position, not the blended light centroid)
  and with a PRF at each catalogued star's position in turn, so the
  chi-square names the star;
* noise is measured from depth images built at null phases, and position
  errors by injecting the signal into those nulls.

What v0.1 learned and this keeps:

* **Deduplicate sectors.**  MAST returns the same sector more than once;
  double-counting halves the apparent error.
* **Does the image contain the transit?**  An offset from an image without
  the signal is meaningless (v0.1 once reported a confident 3.6 sigma offset
  from such an image).  Here that is the PRF-fit depth at the target against
  the null images.
* **"On target" needs the alternatives excluded.**  A small offset measured
  imprecisely is a non-detection, not a clean bill of health: one
  NEB confirmed from the ground, was reported on target by an early
  version.  PASS requires every neighbour that could produce the depth to be
  excluded by the star-by-star fit.

Calibration: the
benchmark planets and the confirmed off-target cases; see the constants.

    python -m petcare.scripts.pc_centroid --tic 410214986 --use-fit
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse

from .. import catalogs, cli, diffimage, ephem
from ..cache import cache_dir, ensure_certificates
from ..plotting import annotate_status, figure, save
from ..result import Result

TEST = "centroid"
VERSION = "0.2.3"

#: Cutout size in pixels.
DEFAULT_CUTOUT = 15

#: The depth image must show the signal at the target's PRF at least this
#: significantly (combined over sectors, against the null images) before any
#: position it gives means anything.
SNR_MIN = 5.0

#: Systematic floor on each sector's position, in pixels, added in quadrature.
#: The SPOC PRF is a commissioning model and the scene is a catalogue; this is
#: what the known planets need to come out on target.
SYS_FLOOR_PIX = 0.03

#: A sector contributes a position only if its depth image detects the
#: signal this well on its own.
SECTOR_SNR_MIN = 4.0
#: Registration error, as a fraction of the shift the scene needed.
REG_ERR_FRAC = 0.5

#: Offset significance (floor included) for FAIL and FLAG.
OFFSET_SIGMA_FAIL, OFFSET_SIGMA_FLAG = 5.0, 3.0
#: ...and for a FAIL that names a star able to produce the depth.  The known
#: planets reach 2.6 sigma at most.
OFFSET_SIGMA_FAIL_STAR = 4.0
#: With no catalogued star there that could produce the depth, an offset must
#: be this large (pixels) as well as OFFSET_SIGMA_FAIL to FAIL; between
#: OFFSET_NEAR_PIX and this it is a FLAG.  Kill when the
#: offset points at a star that can make the depth, or when it is about a
#: pixel off and the answer is obvious; otherwise flag.
OFFSET_FAR_PIX, OFFSET_NEAR_PIX = 1.0, 0.5
#: Chi-square margin by which a neighbour must beat the target to be named
#: the source (FAIL) or suspected (FLAG), and by which a viable neighbour must
#: lose to the target to count as excluded.  Chi-squares are scaled by each
#: sector's reduced chi-square when that exceeds 1.  These are floors: the
#: thresholds actually used are raised to the look-elsewhere null (below).
DCHI2_FAIL, DCHI2_FLAG, DCHI2_EXCLUDE = 25.0, 9.0, 9.0
#: The null for the margin: the measured signal moved onto the target and
#: added to each null image (diffimage._null_margins).  A FAIL needs a margin
#: beyond every null and NULL_K_FAIL robust sigma above their median, a FLAG
#: NULL_K_FLAG.  With ~100 scene stars tried, the best one beats the target by
#: chance more often than a fixed 25 allows when the PRF is imperfect.
NULL_K_FAIL, NULL_K_FLAG = 5.0, 3.0
#: Roll test (diffimage._roll_test): with sectors at roll angles this far
#: apart, an offset that fits a constant *pixel* offset at under ROLL_PIX_FRAC
#: of the sky-frame chi-square, while the sky frame is poor (chi2/dof above
#: ROLL_SKY_RCHI2), turns with the spacecraft: a PRF systematic, not a star.
ROLL_MIN_SPREAD_DEG, ROLL_PIX_FRAC, ROLL_SKY_RCHI2 = 30.0, 0.5, 4.0

#: A FAIL resting on a single sector needs at least this many transits in
#: it.  A run on our shortlist put 35 of 50 FAILs on one sector with
#: 1-5 transits of a 13-39 d candidate, at 40-70 arcsec offsets naming stars
#: that would need impossible eclipses; with so few events a single
#: systematic in one window is the whole depth image.  Below this the answer
#: is INCONCLUSIVE whatever the offset: a FLAG drops a candidate from the
#: shortlist, and evidence this thin should not do that either.
MIN_TRANSITS_SINGLE_SECTOR = 10

#: Masks for other known signals, in units of their T14.
KNOWN_MASK_WIDTH = 2.0


def add_arguments(ap):
    g = ap.add_argument_group("centroid")
    g.add_argument("--cutout-size", type=int, default=DEFAULT_CUTOUT)
    g.add_argument("--pixel-sectors", type=int, nargs="+", default=None,
                   help="restrict the pixel analysis to these sectors "
                        "(default: all available). Distinct from --sectors, "
                        "which selects light-curve sectors.")
    g.add_argument("--max-sectors", type=int, default=10,
                   help="use at most this many sectors (the most recent)")
    g.add_argument("--bootstrap", type=int, default=0,
                   help="ignored since v0.2: errors come from null images")
    g.add_argument("--no-download", action="store_true",
                   help="use only cutouts already in the cache")
    g.add_argument("--inject", default=None,
                   help="injection-recovery: a JSON file {\"star\": \"target\" or a Gaia DR3 "
                        "id, \"frac\": fractional eclipse of that star}; the box-shaped "
                        "dip is put into the pixels at the candidate ephemeris")
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

    star = ctx.star
    ra, dec = ctx.target.ra or star.ra, ctx.target.dec or star.dec
    if not cli.require(r, ra is not None and dec is not None,
                       "no coordinates for the target"):
        return r

    period = float(ctx.candidate.period)
    t0 = float(ctx.candidate.t0_btjd)
    dur = ctx.candidate.duration_days()
    depth_lc = ctx.candidate.depth_ppm
    if not cli.require(r, depth_lc and depth_lc > 0,
                       "no candidate depth: pass --depth (ppm) or run pc_fit first; without it "
                       "no neighbour can be ruled in or out, so a PASS would be empty"):
        return r

    try:
        paths, dl_note = _cutouts(ra, dec, int(getattr(a, "cutout_size", DEFAULT_CUTOUT)),
                                  getattr(a, "pixel_sectors", None),
                                  bool(getattr(a, "no_download", False)))
    except Exception as exc:  # noqa: BLE001 - an archive outage is not a verdict
        return r.inconclusive(f"could not obtain TESScut cutouts: {exc}")
    if not cli.require(r, bool(paths), f"no pixel data for {ctx.target.name}: {dl_note}"):
        r.set_self_check(False, "no cutouts")
        return r
    paths = paths[-int(getattr(a, "max_sectors", 10)):]
    r.set_provenance(lightcurve=None, cutouts=[str(p) for p in paths],
                     n_cutouts=len(paths), download_note=dl_note)

    # ---- the scene ----------------------------------------------------------
    stars, scene_src, ra_s, dec_s = _scene(ctx, ra, dec, t0, depth_lc)
    if not stars:
        return r.inconclusive("no catalogue match for the target, so there is no scene to "
                              "register the images on")

    # ---- other known signals on this star -----------------------------------
    others = [k for k in (ctx.known or [])
              if not ephem.match_known(period, t0, dur * 24, [k], t0 - 1e4, t0 + 1e4)]

    def exclude(t):
        return (ephem.mask_known_transits(t, others, width=KNOWN_MASK_WIDTH)
                if others else np.zeros(t.size, bool))

    prot = _prot(ctx)
    inject = None
    if getattr(a, "inject", None):
        spec = json.loads(Path(a.inject).read_text())
        inject = _injector(spec, period, t0, dur)
        r.metrics["injected"] = spec
    results = []
    for p in paths:
        sec = _sector_of(p)
        try:
            res = diffimage.analyse_sector(p, ra_s, dec_s, period, t0, dur, stars,
                                           exclude_fn=exclude, sector=sec, prot=prot,
                                           inject=inject)
        except Exception as exc:  # noqa: BLE001 - one bad file is not fatal
            res = diffimage.SectorResult(sec, False, f"{type(exc).__name__}: {exc}")
        results.append(res)
    good = [s for s in results if s.ok]
    r.metrics["skipped_sectors"] = [{"sector": s.sector, "why": s.why} for s in results if not s.ok]
    r.metrics["scene"] = {"source": scene_src, "n_stars": len(stars),
                          "known_signals_masked": [getattr(k, "label", None) for k in others]}
    if not good:
        r.set_self_check(False, "no usable sector")
        return r.inconclusive("no usable sector: " + "; ".join(
            f"S{s.sector}: {s.why}" for s in results[:4]))

    comb = diffimage.combine_sectors(good, sys_floor_px=SYS_FLOOR_PIX,
                                     pixel_arcsec=catalogs.TESS_PIXEL_ARCSEC,
                                     snr_min=SECTOR_SNR_MIN, reg_frac=REG_ERR_FRAC)
    hyps = comb["hypotheses"]
    tgt = next(h for h in hyps if h["is_target"])
    best = hyps[0]
    positioned = "offset_arcsec" in comb
    off = comb.get("offset_arcsec", np.nan)
    sig = comb.get("sigma_arcsec", np.nan)
    osig = comb.get("offset_sigma", 0.0)
    off_px = off / catalogs.TESS_PIXEL_ARCSEC
    margin = tgt["chi2"] - best["chi2"]          # >= 0; how much better the best star is
    depths = [s.metrics["target_depth_ppm"] for s in good
              if s.metrics.get("target_depth_ppm") is not None]
    img_depth = float(np.median(depths)) if depths else None

    r.set_noise(None, "positions in arcsec, from signal injected into null-phase depth images",
                combined_sigma_arcsec=sig, sys_floor_pix=SYS_FLOOR_PIX)
    r.metrics.update({
        "method": "per-pixel local baselines; SPOC PRF scene; null-phase noise",
        "n_sectors": comb["n_sectors"], "sectors": [s.sector for s in good],
        "n_sectors_positioned": comb["n_sectors_positioned"], "birge": comb.get("birge"),
        "baseline_half_width_days": diffimage.oot_outer_days(dur, prot), "prot_days": _f(prot),
        "n_transits": int(sum(s.n_transits for s in good)),
        "depth_snr": comb["depth_snr"],
        "combined_offset_arcsec": off, "combined_sigma_arcsec": sig,
        "combined_snr": osig, "offset_in_pixels": off_px,
        "combined_dra_arcsec": comb.get("dra_arcsec"), "combined_ddec_arcsec": comb.get("ddec_arcsec"),
        "per_sector_sigma": comb.get("per_sector_sigma"),
        "hypotheses": hyps[:8],
        "best_source": best["id"], "best_margin_dchi2": margin,
        "image_depth_at_target_ppm": img_depth, "lightcurve_depth_ppm": _f(depth_lc),
        "per_sector": [{"sector": s.sector, **{k: v for k, v in s.metrics.items()
                                               if k != "hypotheses"}} for s in good],
    })

    detected = comb["depth_snr"] >= SNR_MIN and positioned
    # a FAIL needs agreement between sectors, or one sector with many transits
    robust = (comb["n_sectors_positioned"] >= 2
              or comb.get("max_transits_positioned", 0) >= MIN_TRANSITS_SINGLE_SECTOR)
    beyond_region = (comb["depth_snr"] >= SNR_MIN and comb.get("n_sectors_at_bound", 0) > 0
                     and comb["n_sectors_at_bound"] >= comb["n_sectors_positioned"])
    r.metrics.update({"robust_for_fail": robust, "n_sectors_at_bound": comb.get("n_sectors_at_bound"),
                      "n_sectors_detected": comb.get("n_sectors_detected")})

    nulls = np.asarray(comb.get("null_margins") or [], float)
    if nulls.size >= 6:
        med = float(np.median(nulls))
        s = float(1.4826 * np.median(np.abs(nulls - med)))
        dchi2_fail = max(DCHI2_FAIL, float(nulls.max()), med + NULL_K_FAIL * s)
        dchi2_flag = max(DCHI2_FLAG, med + NULL_K_FLAG * s)
    else:
        med = s = None
        dchi2_fail, dchi2_flag = DCHI2_FAIL, DCHI2_FLAG
    roll_ok = comb.get("roll_dof", 0) > 0 and comb.get("roll_spread_deg", 0) >= ROLL_MIN_SPREAD_DEG
    rotates = bool(roll_ok and comb["chi2_sky_between"] > ROLL_SKY_RCHI2 * comb["roll_dof"]
                   and comb["chi2_pix_between"] < ROLL_PIX_FRAC * comb["chi2_sky_between"])
    r.metrics.update({
        "null_margin_median": med, "null_margin_scatter": s,
        "null_margin_max": float(nulls.max()) if nulls.size else None, "n_null_margins": int(nulls.size),
        "dchi2_fail_used": dchi2_fail, "dchi2_flag_used": dchi2_flag,
        "roll_spread_deg": comb.get("roll_spread_deg"), "roll_testable": roll_ok,
        "chi2_sky_between": comb.get("chi2_sky_between"),
        "chi2_pix_between": comb.get("chi2_pix_between"), "roll_dof": comb.get("roll_dof"),
        "offset_pa_deg_per_sector": comb.get("offset_pa_deg_per_sector"),
        "offset_turns_with_spacecraft": rotates,
    })
    star_ok = not best["is_target"] and _physical(best)
    unphysical = not best["is_target"] and (best.get("implied_depth") or 0) < 0
    r.set_self_check(detected, f"{comb['n_sectors']} sector(s), {r.metrics['n_transits']} "
                               f"transits; the depth image holds the signal at "
                               f"{comb['depth_snr']:.1f} sigma against null phases")

    name = (lambda h: "the target" if h["is_target"] else
            f"{'TIC' if len(str(h['id'])) < 12 else 'Gaia DR3'} {h['id']} "
            f"({h['sep_arcsec']:.1f} arcsec, T = {h['tmag']:.1f})")
    where = (f"offset {off:.1f} +/- {sig:.1f} arcsec ({osig:.1f} sigma, {off_px:.2f} px) "
             f"over {comb.get('n_sectors_positioned')} sector(s)" if positioned else "")

    thin_note = (f"Only one sector places it, with {comb.get('max_transits_positioned')} "
                 f"transit(s) (fewer than {MIN_TRANSITS_SINGLE_SECTOR}): one systematic in one "
                 f"window can make this, so it is not evidence either way. Look at the depth "
                 f"image in 06.")
    # The free fit searches +/- FREE_BOX_PX; a source beyond that pins it at
    # the edge, but the star-by-star fit still names it.  Injection-recovery:
    # eclipses put on stars 64-70 arcsec out
    # came back FLAG at delta-chi2 = 600-350000 because of this.
    out_of_reach = (beyond_region and star_ok and margin >= dchi2_fail
                    and best["sep_arcsec"] > (diffimage.FREE_BOX_PX - 0.5) * catalogs.TESS_PIXEL_ARCSEC
                    and (comb.get("n_sectors_detected", 0) >= 2
                         or r.metrics["n_transits"] >= MIN_TRANSITS_SINGLE_SECTOR))
    if out_of_reach:
        r.fail(f"off target: the signal is on {name(best)}, which fits the depth image better "
               f"than the target by delta-chi2 = {margin:.0f} (chance alone reaches "
               f"{dchi2_fail:.0f} at most) and needs a {_pct(best['implied_depth'])} eclipse. "
               f"It lies beyond the {diffimage.FREE_BOX_PX:.0f}-pixel reach of the free-position "
               f"fit, which runs to the edge of its box toward it in "
               f"{comb['n_sectors_at_bound']} of {comb['n_sectors_detected']} detecting sector(s).")
    elif beyond_region and not robust and comb.get("n_sectors_at_bound", 0) < 2:
        r.inconclusive(f"the transit is detected in the pixels ({comb['depth_snr']:.1f} sigma) but "
                       f"its fitted centre runs to the edge of the search box in the one sector "
                       f"that detects it. " + thin_note)
    elif beyond_region:
        r.flag(f"the transit is detected in the pixels ({comb['depth_snr']:.1f} sigma), but in "
               f"{comb['n_sectors_at_bound']} of {comb['n_sectors_detected']} detecting sector(s) "
               f"the fitted centre runs to the edge of the {2 * diffimage.FREE_BOX_PX:.0f}-pixel "
               f"search box: the dimming is either centred well away from the target or not "
               f"PRF-shaped. Look at the depth image."
               + (f" Where it could be placed: {where}." if positioned else ""))
    elif not detected:
        r.inconclusive(f"the transit is not detected well enough in the pixels to place it: "
                       f"{comb['depth_snr']:.1f} sigma over all sectors against null phases "
                       f"(floor {SNR_MIN:g}), and {comb['n_sectors_positioned']} sector(s) "
                       f"reach {SECTOR_SNR_MIN:g} sigma on their own. A position from an image "
                       f"without the signal means nothing.")
    elif unphysical and (osig >= OFFSET_SIGMA_FLAG or margin >= dchi2_flag):
        r.inconclusive(f"the depth image is best fitted by {name(best)} getting BRIGHTER in "
                       f"transit (implied depth {_pct(best['implied_depth'])}), which no "
                       f"eclipse does; {where}. The image holds structure the scene does not "
                       f"model (a flare, a neighbour's variability, a systematic): look at 06.")
    elif rotates and (osig >= OFFSET_SIGMA_FLAG or (star_ok and margin >= dchi2_flag)):
        r.flag(f"possibly off target, but the offset turns with the spacecraft: {where}. Over "
               f"sectors {comb['roll_spread_deg']:.0f} deg apart in roll it fits one offset in "
               f"the pixel frame (chi2 {comb['chi2_pix_between']:.1f}) far better than one on "
               f"the sky (chi2 {comb['chi2_sky_between']:.1f}, {comb['roll_dof']} dof): a PRF "
               f"systematic moves with the detector, a contaminating star does not."
               + (f" {name(best)} fits best, by delta-chi2 = {margin:.0f}." if star_ok else ""))
    elif robust and osig >= OFFSET_SIGMA_FAIL_STAR and star_ok and margin >= dchi2_fail:
        r.fail(f"off target: the signal is on {name(best)}, which fits the depth image better "
               f"than the target by delta-chi2 = {margin:.0f} (chance alone reaches "
               f"{dchi2_fail:.0f} at most) and needs a {_pct(best['implied_depth'])} "
               f"eclipse; {where}.")
    elif robust and osig >= OFFSET_SIGMA_FAIL and off_px >= OFFSET_FAR_PIX:
        r.fail(f"off target: {where}. "
               + (f"Of the catalogued stars, {name(best)} fits best (it would need a "
                  f"{_pct(best.get('implied_depth'))} eclipse), but by delta-chi2 = "
                  f"{margin:.0f}, within the {dchi2_fail:.0f} chance can give, so the source "
                  f"is not named." if star_ok else
                  f"Of the catalogued stars, {name(best)} fits best but would need a "
                  f"{_pct(best.get('implied_depth'))} eclipse; the source is more likely "
                  f"uncatalogued or brighter than catalogued." if not best["is_target"]
                  else "No catalogued star within the fit region explains it; the source may "
                       "lie further out or be uncatalogued."))
    elif not robust and (osig >= OFFSET_SIGMA_FLAG or (star_ok and margin >= dchi2_flag)):
        r.inconclusive(f"thin evidence of an offset: {where}"
                       + (f"; {name(best)} fits better than the target by delta-chi2 = {margin:.0f}"
                          if not best["is_target"] else "") + ". " + thin_note)
    elif osig >= OFFSET_SIGMA_FLAG or (star_ok and margin >= dchi2_flag):
        why = ""
        if osig >= OFFSET_SIGMA_FAIL and off_px >= OFFSET_NEAR_PIX and not star_ok:
            why = (f" Significant, but under {OFFSET_FAR_PIX:g} px with no catalogued star there "
                   f"that could produce the depth, so it is a flag, not a kill.")
        elif star_ok and margin >= dchi2_flag and margin < dchi2_fail:
            why = (f" The margin is within what chance gives when the signal is on the target "
                   f"({dchi2_fail:.0f}), so it is a flag, not a kill.")
        r.flag(f"possibly off target: {where}; "
               + (f"{name(best)} fits better than the target by delta-chi2 = {margin:.0f}."
                  if not best["is_target"] else "the target is still the best catalogued star.")
               + why)
    else:
        open_ = [h for h in hyps if not h["is_target"] and h["viable"]
                 and h["chi2"] - tgt["chi2"] < DCHI2_EXCLUDE]
        beyond = 3 * sig + off
        outside = [s for s in stars if not s.is_target and s.viable and s.sep > beyond
                   and s.id not in {h["id"] for h in hyps}]
        unfit = [s for s in stars if not s.is_target and s.viable and s.sep <= beyond
                 and s.id not in {h["id"] for h in hyps}]
        r.metrics["neighbours_not_excluded"] = [h["id"] for h in open_] + [s.id for s in unfit]
        if open_ or unfit:
            c = min(open_, key=lambda h: h["sep_arcsec"]) if open_ else None
            who = (f"{name(c)} fits within delta-chi2 = {c['chi2'] - tgt['chi2']:.1f} of the target"
                   if c else f"{len(unfit)} neighbour(s) lie outside the fit region but inside "
                             f"the {beyond:.0f} arcsec the offset allows")
            r.inconclusive(f"consistent with the target ({where}), but not exclusive: {who}, "
                           f"and could produce this depth.")
        else:
            r.passed(f"on target: {where}; every catalogued neighbour that could produce this "
                     f"depth fits the depth image worse than the target by delta-chi2 >= "
                     f"{DCHI2_EXCLUDE:g}" + (f" (or lies beyond the {beyond:.0f} arcsec the "
                                            f"offset allows)" if outside else "") + ".")

    if ctx.make_plots:
        try:
            _plot(ctx, r, good, comb)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


def _scene(ctx, ra, dec, t0, depth_ppm):
    """Every Gaia star near the target, with pc_neighbours' viability where it ran."""
    path = ctx.out / ctx.target.slug / (
        f"neighbours_c{ctx.candidate.index}.json" if ctx.candidate.index else "neighbours.json")
    nm, src = None, "Gaia DR3 cone"
    if path.exists():
        try:
            nm = json.loads(path.read_text()).get("metrics") or None
            src = "Gaia DR3 cone, viability from pc_neighbours"
        except (OSError, json.JSONDecodeError):
            nm = None
    pos = (nm or {}).get("target_position_at_epoch")
    if pos:
        ra, dec = float(pos[0]), float(pos[1])
    try:
        stars = diffimage.stars_from_gaia(ra, dec, catalogs.btjd_to_jyear(t0), depth_ppm,
                                          neighbours=nm)
    except Exception:  # noqa: BLE001
        stars = None
    return stars, src, ra, dec


def _prot(ctx):
    """Rotation period from pc_rotation, if it ran; it sets the baseline window."""
    path = ctx.out / ctx.target.slug / (
        f"rotation_c{ctx.candidate.index}.json" if ctx.candidate.index else "rotation.json")
    try:
        return float((json.loads(path.read_text()).get("metrics") or {}).get("prot_days"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _injector(spec, period, t0, dur):
    """A diffimage ``inject`` callback: dims one scene star by ``spec['frac']`` in transit.

    The star's flux and position come from registering the scene on the cube's
    median image first, so the dip has the PRF and the flux the analysis will
    itself measure; the registration is then reset for the analysis proper.
    """
    def f(t, F, scene):
        O = np.nanmedian(F, axis=0)
        ys, xs = np.indices(O.shape)
        cx, cy = scene.xy0[scene.it]
        reg = (np.hypot(xs - cx, ys - cy) <= diffimage.SCENE_RADIUS) & np.isfinite(O)
        scene.fit_oot(O, np.sqrt(np.clip(np.abs(O), 1, None)), reg)
        i = scene.it if spec["star"] == "target" else next(
            k for k, s in enumerate(scene.stars) if s.id == str(spec["star"]))
        model = float(spec["frac"]) * scene.flux[i] * scene.prf(xs, ys, *scene.xy[i])
        ph = (t - t0 + 0.5 * period) % period - 0.5 * period
        G = F.copy()
        G[np.abs(ph) < 0.5 * dur] -= model
        scene.shift = np.zeros(2)
        return G
    return f


def _physical(h):
    d = h.get("implied_depth")
    return d is not None and 0 < d <= 1.0


def _pct(d):
    return f"{100 * d:.1f}%" if d is not None else "unknown-depth"


def _cutouts(ra, dec, size, want_sectors, no_download):
    """Download (or reuse) TESScut cutouts, **deduplicated by sector**.

    MAST returns the same sector more than once when several products cover
    it.  Every duplicated sector double-counts its epochs and halves the
    apparent error, which is one of the bugs that hid behind v0.1's confident
    wrong answers.
    """
    ensure_certificates()
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord

    d = cache_dir("tesscut") / f"ra{ra:.6f}_dec{dec:.6f}_{size}x{size}"
    d.mkdir(parents=True, exist_ok=True)
    have = sorted(d.glob("*.fits"))

    note = f"{len(have)} cutout(s) already cached"
    if not have and not no_download:
        coord = SkyCoord(ra, dec, unit="deg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tesscut.download_cutouts(coordinates=coord, size=size,
                                     sector=None, path=str(d), inflate=True)
        have = sorted(d.rglob("*.fits"))
        note = f"downloaded {len(have)} cutout(s)"

    by_sector = {}
    for p in have:
        sec = _sector_of(p)
        if want_sectors and sec not in set(want_sectors):
            continue
        # keep the largest file for a sector: the others are partial products
        if sec not in by_sector or p.stat().st_size > by_sector[sec].stat().st_size:
            by_sector[sec] = p
    n_dupes = len(have) - len(by_sector)
    if n_dupes > 0:
        note += f"; dropped {n_dupes} duplicate sector product(s)"
    return [by_sector[k] for k in sorted(by_sector, key=lambda x: (x is None, x))], note


def _sector_of(path):
    for tok in Path(path).name.split("-"):
        if tok.startswith("s") and tok[1:].isdigit():
            return int(tok[1:])
    return None


def _plot(ctx, r, good, comb):
    """Per sector: out-of-transit scene, depth image in sigma, and the residual."""
    rows = sorted(good, key=lambda s: -s.metrics["depth_snr"])[:4]
    fig, axes = figure(len(rows), 3, figsize=(10, 3.3 * len(rows)), squeeze=False)
    best = comb["hypotheses"][0]
    ids = [h["id"] for h in comb["hypotheses"]]
    for i, s in enumerate(rows):
        im = s.images
        D, O, sig = im["D"], im["O"], im["sig"]
        tx, ty = im["target"]
        fx, fy = im["free"]
        ys, xs = np.indices(D.shape)
        S = D / sig
        resid = (D - im["model_t"]) / sig
        lim = max(3.0, float(np.nanpercentile(np.abs(S), 99.5)))
        panels = [(np.arcsinh(O / max(1.0, np.nanpercentile(O, 60))), "out of transit (scene)",
                   "viridis", None),
                  (S, "depth image / noise (sigma)", "RdBu_r", lim),
                  (resid, "depth image minus target model (sigma)", "RdBu_r", lim)]
        for j, (img, title, cmap, v) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(img, origin="lower", cmap=cmap, vmin=-v if v else None, vmax=v)
            xy = im["xy"]
            for k, (x, y) in enumerate(xy):
                stinfo = im["stars"][k]
                if stinfo["is_target"]:
                    continue
                if stinfo["ratio"] < 0.002 and stinfo["sep"] > 25:
                    continue
                col = "orange" if stinfo["viable"] else "0.75"
                ms = max(3, 11 - 1.3 * max(0, stinfo["tmag"] - im["stars"][0]["tmag"]))
                if stinfo["id"] == best["id"] and not best["is_target"]:
                    col, ms = "red", max(ms, 10)
                ax.plot(x, y, "o", mfc="none", mec=col, ms=ms,
                        mew=2 if col == "red" else 1.2)
            ax.plot(tx, ty, "+", color="white" if j == 0 else "k", ms=13, mew=2)
            if j > 0:
                ax.plot(fx, fy, "x", color="lime", ms=10, mew=2)
                C = np.asarray(im["cov_px"])
                if np.all(np.isfinite(C)):
                    w, vecs = np.linalg.eigh(C)
                    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                    ax.add_patch(Ellipse((fx, fy), 2 * np.sqrt(max(w[1], 0)) * 3,
                                         2 * np.sqrt(max(w[0], 0)) * 3, angle=ang,
                                         fill=False, color="lime", lw=1))
            if j == 0:
                _compass(ax, np.array(s.metrics["jacobian"]), tx, ty)
            ax.set_xlim(tx - 4.5, tx + 4.5)
            ax.set_ylim(ty - 4.5, ty + 4.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"S{s.sector}: {title}", fontsize=8)
        m = s.metrics
        axes[i][1].text(0.02, 0.02,
                        f"{m['n_transits']} transits, {m['depth_snr']:.0f} sigma\n"
                        f"offset {np.hypot(m['dra_arcsec'], m['ddec_arcsec']):.1f}″",
                        transform=axes[i][1].transAxes, fontsize=7, va="bottom",
                        bbox=dict(fc="white", alpha=0.7, lw=0))
    hy = comb["hypotheses"][:4]
    txt = "  ".join(f"{'target' if h['is_target'] else h['id'][-6:]}: Δχ²="
                    f"{h['dchi2_vs_target']:+.0f}" for h in hy)
    pos = (f"combined offset {comb['offset_arcsec']:.1f} ± {comb['sigma_arcsec']:.1f}″ "
           f"({comb['offset_sigma']:.1f}σ)" if "offset_arcsec" in comb
           else "no sector detects the signal well enough to place it")
    fig.supxlabel(f"{pos}.  {txt}\n+ target, x signal centre (3σ ellipse), o neighbours "
                  f"(orange: could produce the depth; red: fits best; grey: could not). "
                  f"Pixel frame; arrows show N and E.", fontsize=7)
    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("centroid")))


def _compass(ax, J, x, y, length=1.0):
    """N and E arrows in the pixel frame, from the WCS jacobian (pixel -> arcsec).

    Drawn from the corner the arrows point away from, so both stay on the panel.
    """
    try:
        Ji = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return
    dirs = []
    for vec in ((0, 1), (1, 0)):
        d = Ji @ np.array(vec, float)
        dirs.append(d / np.hypot(*d) * length)
    mean = np.mean(dirs, axis=0)
    x0 = x - 3.0 * np.sign(mean[0] or 1)
    y0 = y - 3.0 * np.sign(mean[1] or 1)
    for d, lab in zip(dirs, "NE"):
        ax.annotate("", xy=(x0 + d[0], y0 + d[1]), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
        ax.text(x0 + 1.35 * d[0], y0 + 1.35 * d[1], lab, color="white", fontsize=7,
                ha="center", va="center")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Difference imaging: which star is the transit signal on?"))
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

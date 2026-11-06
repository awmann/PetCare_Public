"""pc_neighbours -- which nearby star could reproduce this depth?

A cone search of Gaia DR3 and the TIC around the target, turned into the only
question that matters for a blend: **how deep would that star have to eclipse
to produce the depth we see?**

For each neighbour, ``d_required = d_observed / f``, where ``f`` is that star's
share of the flux inside the photometric aperture.  Then:

* ``d_required > 1`` -- impossible.  The neighbour is too faint to produce
  this depth however totally it eclipses, and is excluded outright.
* ``d_required > 0.5`` -- needs a near-total eclipse.  Constrained, but
  possible; a fully-eclipsing binary of similar components does reach this.
* otherwise -- a viable contaminating source, and the report names it.

Two numbers a human actually needs come out of this: the **closest** viable
neighbour and the **brightest** one, and the separation a ground-based image
would have to resolve to rule them out.

Positions are propagated from Gaia's 2016.0 epoch to the epoch of the
observation.  A nearby high-proper-motion star can move 2.5 arcseconds over a nine-year baseline -- more
than a TESS pixel -- so a neighbour list built at the catalogue epoch puts the
contaminating star in the wrong place.

The flux fractions come from the extraction's own aperture mask weighted by a
**Moffat** standing in for the PRF.  It was a Gaussian first, and that was
wrong in the one regime this test exists for: a Gaussian has no wings, so a
bright star two or three pixels outside the aperture contributes essentially
nothing and is "excluded", when in reality it contributes several percent and
is the false positive.  On one candidate the Gaussian required a 397% eclipse
from the star pc_centroid then found the signal on at 70.8 sigma; the Moffat
requires 20%.  The approximation is still stated in the result and in
``reason`` -- the real PRF is asymmetric and field-dependent -- so anything
near a decision threshold belongs in pc_centroid.  Without an aperture mask
the test falls back to a total-flux fraction and says so.

    python -m petcare.scripts.pc_neighbours --tic 410214986 --depth 2936
"""

from __future__ import annotations

import sys

import numpy as np

from .. import catalogs, cli
from ..plotting import C_ALT, C_BAD, C_MODEL, annotate_status, figure, save
from ..result import Result

TEST = "neighbours"
VERSION = "0.1.3"

#: Cone radius.  Wide enough to reach every star whose PRF wings can land in a
#: TESS aperture; the depth requirement removes the irrelevant ones.
DEFAULT_RADIUS_ARCSEC = 150.0

#: Required eclipse depth above which a neighbour is excluded outright.
IMPOSSIBLE_DEPTH = 1.0

#: ...and above which it needs a near-total eclipse.
CONSTRAINED_DEPTH = 0.5

#: Separation below which no ground-based image will resolve the neighbour,
#: so the question has to be settled from space or by difference imaging.
UNRESOLVABLE_ARCSEC = 1.0
#: ...and beyond it, how much fainter than the target a neighbour can be and
#: still be separated in seeing-limited imaging, per arcsec past that limit.
#: A rough contrast curve for ~1.5 arcsec seeing (LCO-class): 2 mag at 2
#: arcsec, 6 mag at 4.  The 2026-10-27 audit found 7.6 arcsec / 4.5 mag
#: neighbour marked resolvable with no magnitude check at all.
GROUND_CONTRAST_MAG_PER_ARCSEC = 2.0
#: Wide-field surveys (ATLAS 1.86 arcsec pixels, ZTF ~2 arcsec seeing) blend
#: anything inside SURVEY_BLEND_ARCSEC, and a neighbour inside
#: SURVEY_CONFUSION_ARCSEC of a target at least SURVEY_CONFUSION_DMAG brighter
#: is lost in its wings: DEATHSTAR's fitter swapped the two on one target
#: (4 arcsec) and its photometry went bimodal.
SURVEY_BLEND_ARCSEC, SURVEY_CONFUSION_ARCSEC, SURVEY_CONFUSION_DMAG = 6.0, 12.0, 1.0


def add_arguments(ap):
    g = ap.add_argument_group("neighbours")
    g.add_argument("--radius", type=float, default=DEFAULT_RADIUS_ARCSEC,
                   help="cone radius in arcseconds")
    g.add_argument("--no-aperture", action="store_true",
                   help="ignore the aperture entirely and use a total-flux "
                        "fraction, which overstates every neighbour")
    g.add_argument("--aperture-radius", type=float, default=2.5,
                   help="radius in pixels of the circular aperture assumed "
                        "when the extraction does not carry its own mask "
                        "(the 20 s SPOC products do not). The assumption is "
                        "checked against the extraction's own dilution.")
    g.add_argument("--epoch", type=float, default=None,
                   help="Julian year to propagate positions to; default is "
                        "the midpoint of the light curve, or 2021.0")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    star = ctx.star
    ra, dec = ctx.target.ra or star.ra, ctx.target.dec or star.dec
    if not cli.require(r, ra is not None and dec is not None,
                       f"no coordinates for {ctx.target.name}: pass --ra/--dec "
                       f"or make sure the target is in the catalogue"):
        return r

    depth = ctx.candidate.depth_ppm
    if not cli.require(r, depth and depth > 0,
                       "no candidate depth: pass --depth (ppm) or run pc_fit "
                       "first; without it there is nothing to attribute"):
        return r
    depth_frac = float(depth) / 1e6

    # ---- aperture and epoch ----------------------------------------------
    aperture, ap_source, epoch = None, "none", getattr(a, "epoch", None)
    tmag_target = star.tmag
    try:
        lc = ctx.lc
        ctx.attach_lc_provenance(r, lc)
        if epoch is None:
            epoch = catalogs.btjd_to_jyear(float(np.median(lc.t)))
        if not getattr(a, "no_aperture", False):
            masks = [s.aperture_mask for s in lc.sectors
                     if s.aperture_mask is not None]
            ids = {s.aperture_id for s in lc.sectors if s.aperture_id}
            if masks:
                aperture = masks[0]
                ap_source = f"extraction aperture {'/'.join(sorted(ids))}"
            else:
                rad = float(getattr(a, "aperture_radius", 2.5))
                aperture = catalogs.circular_aperture(rad)
                ap_source = (f"assumed circular aperture, r = {rad:g} px "
                             f"(the extraction reports "
                             f"{'/'.join(sorted(ids)) or 'no'} aperture and "
                             f"carries no mask)")
    except Exception as exc:  # noqa: BLE001 - this test can run without one
        r.provenance.setdefault("notes", []).append(
            f"no light curve, so no aperture mask: {exc}")
        r.set_provenance(lightcurve=None)
    if epoch is None:
        epoch = 2021.0

    # The depth is measured on flux already corrected for dilution, but a neighbour
    # supplies the dip as it was in the aperture, which is shallower by the target's share.
    aperture_dip = depth_frac
    try:
        dil_lc, _ = lc.dilution()
        if dil_lc and 0 < dil_lc < 1 and "sap" not in str(lc.flux_column).lower():
            aperture_dip = depth_frac * dil_lc
    except Exception:  # noqa: BLE001 - no light curve, nothing to scale by
        pass

    # ---- the cone ---------------------------------------------------------
    radius = float(getattr(a, "radius", DEFAULT_RADIUS_ARCSEC))
    try:
        gaia = catalogs.gaia_cone(ra, dec, radius)
    except Exception as exc:  # noqa: BLE001 - an archive outage is not a verdict
        gaia = []
        r.provenance.setdefault("notes", []).append(f"Gaia cone failed: {exc}")
    try:
        tic = catalogs.tic_cone(ra, dec, radius)
    except Exception as exc:  # noqa: BLE001
        tic = []
        r.provenance.setdefault("notes", []).append(f"TIC cone failed: {exc}")

    if not cli.require(r, bool(gaia or tic),
                       f"both catalogue cones failed or returned nothing at "
                       f"{ra:.5f}, {dec:.5f}"):
        r.set_self_check(False, "no catalogue rows")
        return r

    # ---- build the neighbour list ----------------------------------------
    tmag_by_gaia = {}
    for row in tic:
        gid = row.get("GAIA")
        if gid and row.get("Tmag") is not None:
            try:
                tmag_by_gaia[int(gid)] = float(row["Tmag"])
            except (TypeError, ValueError):
                pass
    if tmag_target is None:
        # the target's own TIC row, by nearest position
        best = min(tic, key=lambda x: (float(x["ra"]) - ra) ** 2 +
                   (float(x["dec"]) - dec) ** 2, default=None)
        if best is not None and best.get("Tmag") is not None:
            tmag_target = float(best["Tmag"])

    if not cli.require(r, tmag_target is not None,
                       "no T magnitude for the target, so flux ratios cannot "
                       "be computed"):
        r.set_self_check(False, "no target Tmag")
        return r

    # Identify the target's own row and propagate *it* to the same epoch, so
    # separations are measured between two positions at one epoch.  Skipping
    # this put DS Tuc A 1.0 arcsec from itself -- its proper motion over five
    # years -- and the test then treated the target as its own brightest
    # contaminant, halving the computed flux share.
    target_row, target_reason = _find_target_row(
        gaia, ctx.target.gaia_dr3, star.gaia_dr3, ra=ra, dec=dec)
    if target_row is not None:
        ra_e, dec_e = catalogs.propagate(
            float(target_row["ra"]), float(target_row["dec"]),
            target_row.get("pmra"), target_row.get("pmdec"), to_epoch=epoch)
    else:
        ra_e, dec_e = ra, dec
    target_sid = target_row.get("source_id") if target_row else None

    neighbours = []
    approximated_tmag = 0
    for row in (gaia or []):
        sid = row.get("source_id")
        if target_sid is not None and sid == target_sid:
            continue          # the target itself
        n_ra, n_dec = catalogs.propagate(
            float(row["ra"]), float(row["dec"]), row.get("pmra"),
            row.get("pmdec"), to_epoch=epoch)
        sep, dra, ddec = catalogs.separation_arcsec(ra_e, dec_e, n_ra, n_dec)
        sep = float(sep)
        if sep < 0.2 and target_sid is None:
            continue          # no Gaia id to match on; fall back to position
        tmag = tmag_by_gaia.get(int(sid)) if sid else None
        tmag_approx = tmag is None
        if tmag_approx:
            tmag = catalogs.gaia_g_to_tmag(row.get("phot_g_mean_mag"),
                                           row.get("bp_rp"))
            approximated_tmag += 1
        if tmag is None or not np.isfinite(tmag):
            continue

        ratio = catalogs.flux_ratio(tmag, tmag_target)
        if aperture is not None:
            f_n = catalogs.prf_fraction_in_aperture(
                float(dra) / catalogs.TESS_PIXEL_ARCSEC,
                float(ddec) / catalogs.TESS_PIXEL_ARCSEC, aperture)
            f_t = catalogs.prf_fraction_in_aperture(0.0, 0.0, aperture)
        else:
            f_n, f_t = 1.0, 1.0

        # the neighbour's share of the flux in the aperture
        num = ratio * f_n
        share = num / (f_t + num) if (f_t + num) > 0 else 0.0
        d_req = aperture_dip / share if share > 0 else np.inf

        neighbours.append({
            "gaia_dr3": int(sid) if sid else None,
            "sep_arcsec": sep, "dra_arcsec": float(dra), "ddec_arcsec": float(ddec),
            "position_angle_deg": float(np.degrees(np.arctan2(dra, ddec)) % 360),
            "g_mag": row.get("phot_g_mean_mag"), "bp_rp": row.get("bp_rp"),
            "tmag": float(tmag), "tmag_is_approximate": bool(tmag_approx),
            "flux_ratio_to_target": float(ratio),
            "prf_fraction_in_aperture": float(f_n),
            "flux_share_in_aperture": float(share),
            "required_eclipse_depth": float(d_req) if np.isfinite(d_req) else None,
            "verdict": _verdict(d_req),
            "parallax": row.get("parallax"), "ruwe": row.get("ruwe"),
            "resolvable_from_ground": _resolvable(sep, _dmag(tmag, tmag_target)),
            "resolvable_by_survey": _survey_separable(sep, _dmag(tmag, tmag_target)),
        })

    neighbours.sort(key=lambda x: x["sep_arcsec"])

    # Cross-check the aperture assumption against the dilution the extraction
    # measured for itself.  This is the self-check that makes an assumed
    # aperture usable: if the target's computed flux share does not reproduce
    # the extraction's own number, the aperture is wrong and every required
    # depth above is wrong with it.
    dil_check = {"extraction_dilution": None, "computed_target_share": None}
    try:
        dil, dil_src = lc.dilution()
    except Exception:  # noqa: BLE001
        dil, dil_src = None, None
    if aperture is not None:
        f_t = catalogs.prf_fraction_in_aperture(0.0, 0.0, aperture)
        total_neighbour = sum(
            catalogs.flux_ratio(n["tmag"], tmag_target) *
            n["prf_fraction_in_aperture"] for n in neighbours)
        share = f_t / (f_t + total_neighbour) if (f_t + total_neighbour) > 0 else None
        dil_check = {
            "extraction_dilution": _f(dil), "extraction_source": dil_src,
            "computed_target_share": _f(share),
            "ratio": _f(share / dil) if (share and dil) else None,
            "note": "the target's share of the aperture flux, computed here, "
                    "against the value the extraction measured; a large "
                    "disagreement means the assumed aperture is wrong",
        }
    r.metrics["dilution_cross_check"] = dil_check
    r.metrics["aperture_dip_ppm"] = aperture_dip * 1e6
    viable = [n for n in neighbours if n["verdict"] != "excluded"]
    easy = [n for n in viable if n["verdict"] == "viable"]

    closest = viable[0] if viable else None
    brightest = min(viable, key=lambda n: n["tmag"]) if viable else None

    r.set_noise(None, "not applicable: this is a catalogue test",
                depth_used_ppm=float(depth))
    r.metrics.update({
        "n_gaia_in_cone": len(gaia), "n_tic_in_cone": len(tic),
        "cone_radius_arcsec": radius,
        "epoch_jyear": float(epoch),
        "target_gaia_dr3": int(target_sid) if target_sid else None,
        "target_identified_by": target_reason,
        "target_position_at_epoch": [float(ra_e), float(dec_e)],
        "target_proper_motion_arcsec": _f(
            float(catalogs.separation_arcsec(ra, dec, ra_e, dec_e)[0])),
        "target_tmag": float(tmag_target),
        "aperture_source": ap_source,
        "flux_fraction_method": (
            "Moffat PRF (FWHM = 1.5 px, beta = 2.5) over the extraction "
            "aperture mask"
            if aperture is not None else
            "total flux ratio, no aperture weighting"),
        "n_neighbours": len(neighbours),
        "n_viable": len(viable), "n_viable_without_total_eclipse": len(easy),
        "n_excluded": len(neighbours) - len(viable),
        "closest_viable": closest, "brightest_viable": brightest,
        "neighbours": neighbours[:60],
        "neighbours_truncated": len(neighbours) > 60,
        "n_tmag_approximated_from_gaia_g": approximated_tmag,
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{len(gaia)} Gaia and {len(tic)} TIC sources within "
                  f"{radius:.0f} arcsec"], True
    checks.append(f"target row {target_reason}")
    if target_sid is None:
        checks.append("the target's own Gaia row was not identified, so it "
                      "may be counted among its own neighbours")
        ok = False
    if aperture is None:
        checks.append("no aperture: flux fractions are total-flux ratios and "
                      "overestimate every neighbour's contribution")
    else:
        checks.append(f"{ap_source}, {int((np.asarray(aperture) > 0).sum())} px")
    ratio = dil_check.get("ratio")
    if ratio is not None:
        checks.append(f"computed target flux share is {ratio:.2f}x the "
                      f"extraction's dilution "
                      f"({dil_check['computed_target_share']:.3f} vs "
                      f"{dil_check['extraction_dilution']:.3f})")
        if not (0.8 < ratio < 1.25):
            checks.append("the aperture model does not reproduce the "
                          "extraction's dilution, so the required depths "
                          "below are unreliable")
            ok = False
    if approximated_tmag:
        checks.append(f"{approximated_tmag} neighbour T magnitudes converted "
                      f"from Gaia G")
    if not gaia:
        checks.append("no Gaia rows: the neighbour list is TIC-only and will "
                      "miss close pairs the TIC does not resolve")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    approx_note = ("flux fractions use a Moffat PRF approximation (FWHM 1.5 "
                   "px, beta 2.5); the real PRF is asymmetric and varies "
                   "across a camera, so check anything near a decision "
                   "threshold against pc_centroid")

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif not viable:
        r.passed(f"no neighbour within {radius:.0f} arcsec can produce a "
                 f"{depth:.0f} ppm dip: the faintest required eclipse depth "
                 f"among {len(neighbours)} stars is "
                 f"{_min_required(neighbours):.0%}, i.e. all are excluded. "
                 f"{approx_note}")
    else:
        unres = [n for n in viable if not n["resolvable_from_ground"]]
        bits = [f"{len(viable)} neighbour(s) could produce {depth:.0f} ppm "
                f"({len(easy)} without needing a near-total eclipse)"]
        if closest:
            bits.append(f"closest is Gaia DR3 {closest['gaia_dr3']} at "
                        f"{closest['sep_arcsec']:.1f} arcsec, T = "
                        f"{closest['tmag']:.1f}, needing a "
                        f"{closest['required_eclipse_depth']:.0%} eclipse")
        if brightest and brightest is not closest:
            bits.append(f"brightest is Gaia DR3 {brightest['gaia_dr3']} at "
                        f"{brightest['sep_arcsec']:.1f} arcsec, T = "
                        f"{brightest['tmag']:.1f}")
        if unres:
            bits.append(f"{len(unres)} of them are too close for their brightness "
                        f"to be resolved from the ground (inside "
                        f"{UNRESOLVABLE_ARCSEC:.0f} arcsec, or too faint beside the "
                        f"target)")
        r.flag("; ".join(bits) + f". {approx_note}")

    if ctx.make_plots:
        try:
            _plot(ctx, r, ra, dec, neighbours, aperture, radius)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _find_target_row(gaia, *gaia_ids, ra=None, dec=None):
    """The target's own row in the cone, and how it was identified.

    By ``source_id`` when one is known -- that is unambiguous.  Otherwise by
    position at the *catalogue* epoch, which may be 2016.0 (Gaia) or 2000.0 (TIC:
    DS Tuc A moves 1.7 arcsec between them), and, among anything within an
    arcsecond, the brightest.
    """
    ids = [int(g) for g in gaia_ids if g]
    if ids:
        for row in (gaia or []):
            if row.get("source_id") and int(row["source_id"]) in ids:
                return row, f"matched on Gaia DR3 {row['source_id']}"
    if ra is None or dec is None or not gaia:
        return None, "not identified"
    near = []
    for row in gaia:
        sep = min(float(catalogs.separation_arcsec(ra, dec, *catalogs.propagate(
            float(row["ra"]), float(row["dec"]), row.get("pmra"), row.get("pmdec"),
            to_epoch=ep))[0]) for ep in (catalogs.GAIA_EPOCH, 2000.0))
        if sep < 1.5:
            near.append((sep, row))
    if not near:
        return None, "no Gaia source within 1.5 arcsec of the catalogue position"
    row = min(near, key=lambda sr: sr[1].get("phot_g_mean_mag") or 99)[1]
    return row, (f"brightest of {len(near)} Gaia source(s) within 1.5 arcsec "
                 f"of the catalogue position")


def _dmag(tmag, tmag_target):
    """Neighbour minus target magnitude; 0 (equal brightness) when unknown."""
    try:
        d = float(tmag) - float(tmag_target)
    except (TypeError, ValueError):
        return 0.0
    return d if np.isfinite(d) else 0.0


def _resolvable(sep, dmag):
    """Could seeing-limited imaging separate a neighbour ``dmag`` fainter?"""
    return bool(sep > UNRESOLVABLE_ARCSEC
                and dmag <= GROUND_CONTRAST_MAG_PER_ARCSEC * (sep - UNRESOLVABLE_ARCSEC))


def _survey_separable(sep, dmag):
    """Could ATLAS/ZTF photometry keep it apart from the target?"""
    return bool(sep >= SURVEY_BLEND_ARCSEC
                and not (sep < SURVEY_CONFUSION_ARCSEC and dmag >= SURVEY_CONFUSION_DMAG))


def _verdict(d_req) -> str:
    if not np.isfinite(d_req) or d_req > IMPOSSIBLE_DEPTH:
        return "excluded"
    if d_req > CONSTRAINED_DEPTH:
        return "needs a near-total eclipse"
    return "viable"


def _min_required(neighbours):
    vals = [n["required_eclipse_depth"] for n in neighbours
            if n["required_eclipse_depth"] is not None]
    return min(vals) if vals else float("inf")


def _plot(ctx, r, ra, dec, neighbours, aperture, radius):
    fig, axes = figure(1, 2, figsize=(11, 5))

    ax = axes[0]
    mask = np.asarray(aperture, float) if aperture is not None else None
    if mask is not None and mask.ndim == 2 and mask.size:
        ny, nx = mask.shape
        # Place the mask so the target (the lit pixels' centre) is at the
        # origin.  Pixel axes are drawn as if aligned with RA/Dec, which TESS
        # pixels are not -- the orientation is schematic, the offset is not.
        cy, cx = catalogs.aperture_centre(mask)
        px = catalogs.TESS_PIXEL_ARCSEC
        ax.imshow(mask > 0, origin="lower", cmap="Greys", alpha=0.35,
                  extent=[(cx + 0.5) * px, -(nx - 0.5 - cx) * px,
                          -(cy + 0.5) * px, (ny - 0.5 - cy) * px])
    colours = {"viable": C_BAD, "needs a near-total eclipse": "#b8860b",
               "excluded": "#bbbbbb"}
    for n in neighbours:
        size = max(4, 90 * 10 ** (-0.4 * (n["tmag"] - r.metrics["target_tmag"])))
        ax.scatter(n["dra_arcsec"], n["ddec_arcsec"], s=min(size, 400),
                   facecolor="none", edgecolor=colours[n["verdict"]], lw=1.1)
    ax.plot(0, 0, "*", ms=16, color=C_MODEL, label="target")
    for label, colour in colours.items():
        ax.scatter([], [], facecolor="none", edgecolor=colour, label=label)
    lim = min(radius, 120)
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("dRA [arcsec]  (east left)")
    ax.set_ylabel("dDec [arcsec]")
    ax.set_title(f"neighbours and the aperture (grey; pixel axes, orientation "
                 f"schematic) -- epoch {r.metrics['epoch_jyear']:.1f}", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5)

    ax = axes[1]
    if neighbours:
        sep = np.array([n["sep_arcsec"] for n in neighbours])
        req = np.array([n["required_eclipse_depth"] if n["required_eclipse_depth"]
                        else 5.0 for n in neighbours])
        col = [colours[n["verdict"]] for n in neighbours]
        ax.scatter(sep, req, c=col, s=26, edgecolor="k", lw=0.3)
        ax.axhline(IMPOSSIBLE_DEPTH, color=C_BAD, ls="--", lw=1.0,
                   label="impossible (>100% eclipse)")
        ax.axhline(CONSTRAINED_DEPTH, color="#b8860b", ls=":", lw=1.0,
                   label="needs a near-total eclipse")
        ax.axvline(UNRESOLVABLE_ARCSEC, color=C_ALT, ls="-.", lw=1.0,
                   label=f"{UNRESOLVABLE_ARCSEC:.0f} arcsec: unresolvable from the ground")
        ax.set_yscale("log")
        ax.set_xscale("log")
    ax.set_xlabel("separation [arcsec]")
    ax.set_ylabel("eclipse depth this star would need")
    ax.set_title("what each neighbour would have to do")
    ax.legend(loc="lower right", fontsize=6.5)

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("neighbours")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Which nearby star could reproduce this depth? Cone-searches "
              "Gaia DR3 and the TIC and reports the eclipse each would need."))
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

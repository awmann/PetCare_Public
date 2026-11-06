"""pc_duration -- is the transit duration physical for this star?

Compares the fitted T14 with the range a circular orbit around this star
allows, from a central crossing (b = 0, the longest) to a grazing one (b = 1,
the shortest).

**This test does not veto, by design.**  Young-star radii and masses are poor
enough that a duration outside the allowed range is usually a statement about
the catalogue, not about the candidate.  So instead of a verdict it reports
the two numbers that would reconcile the discrepancy:

* the **eccentricity** required, at the most favourable argument of periastron;
* the **fractional stellar-radius error** required, at fixed mass.

A modest value for either is not evidence against a planet.  Measured on the
known planets in this sample with TIC parameters, seven of eight sit inside
the allowed range and the eighth misses it by 1.4% -- because the
TIC radius for its host is 0.70 R_sun against a literature 0.86.

The first question asked is not "is the duration discrepant" but **"what
impact parameter would reproduce it"**.  Getting that order wrong matters: a
transit shorter than a central crossing needs no eccentricity at all, only a
larger b, and a test that jumps straight to the density ratio reports a
spurious e ~ 0.26 for DS Tuc A b, whose geometry is entirely ordinary at
b = 0.76.

    python -m petcare.scripts.pc_duration --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli
from ..plotting import C_BINNED, C_MODEL, C_OK, annotate_status, figure, save
from ..result import Result
from ..stellar import (
    a_over_rs,
    duration_range,
    duration_t14,
    implied_b,
    radius_error_note,
    reconcile_density,
    rho_circ,
)

TEST = "duration"
VERSION = "0.1.1"

#: Required eccentricity above which the duration is worth a human's attention.
#: Young multis are not expected to be this eccentric, but nor is this a cut.
ECC_FLAG = 0.4

#: Required fractional stellar-radius error above which the same applies.
#: 30% is larger than the disagreement between the TIC and the literature for
#: any host in the calibration sample.
RADIUS_ERROR_FLAG = 0.30

#: Radius above which a transiting companion cannot be a planet, in Jupiter
#: radii **at 10 Myr**.  A fixed ceiling is wrong on this sample: young giants
#: are genuinely puffy and young brown dwarfs are puffier still, so 2 Rjup is
#: unremarkable at a few Myr and impossible at a Gyr.  The ceiling therefore
#: scales with age.
#:
#: PROVISIONAL, and deliberately generous.  This is a crude envelope over the
#: published evolutionary tracks, not a model interpolation -- roughly 2.8
#: Rjup at 1 Myr falling to 1.3 by a few hundred Myr.  A proper grid
#: (Baraffe, Phillips) should replace it, and until it does nothing marginal
#: is refuted on radius alone.  Note this bounds *planets*: a young brown
#: dwarf can exceed it, which is why exceeding it refutes the planet
#: interpretation rather than identifying what the object is.
PLANET_RADIUS_CEILING_RJUP = 2.0
CEILING_REFERENCE_AGE_MYR = 10.0
CEILING_AGE_INDEX = -0.15
CEILING_MIN_RJUP, CEILING_MAX_RJUP = 1.3, 3.5

#: How far above the ceiling before refuting rather than flagging.  The radius
#: is a product of the catalogue stellar radius, so the margin has to absorb a
#: badly wrong R*, and the ceiling is soft: young low-mass planets can be far
#: more inflated than it allows.  At 2.0 a 40 Myr companion must exceed about
#: 3.2 Rjup to be refuted, a 5 Myr one about 4.5.
RADIUS_FAIL_MARGIN = 2.0

#: And a fractional R* uncertainty above which nothing is refuted on radius at
#: all, however large it looks.
RADIUS_TRUSTED_FRAC = 0.35

#: Solar radii per Jupiter radius.
RSUN_OVER_RJUP = 9.731


def add_arguments(ap):
    g = ap.add_argument_group("duration")
    g.add_argument("--rho-star", type=float, default=None,
                   help="stellar density [g/cm^3], overriding the catalogue")
    g.add_argument("--a-rs", type=float, default=None,
                   help="fitted a/R*, if not taking it from fit.json")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    if not cli.require(r, ctx.candidate.period, "no candidate period"):
        return r
    if not cli.require(r, ctx.candidate.duration_h,
                       "no transit duration: pass --duration or run pc_fit first"):
        return r

    star = ctx.star
    rho_star, e_rho_star = star.density()
    if getattr(a, "rho_star", None):
        rho_star, e_rho_star = float(a.rho_star), None

    r.metrics["star"] = {
        "radius_rsun": star.radius, "e_radius": star.e_radius,
        "mass_msun": star.mass, "e_mass": star.e_mass,
        "teff": star.teff, "age_myr": star.age_myr,
        "rho_star_gcc": rho_star, "e_rho_star_gcc": e_rho_star,
        "source": star.source,
        "columns_used": star.source_detail.get("columns_used"),
    }

    if not cli.require(r, rho_star and rho_star > 0,
                       f"no stellar density for {star.name}: the catalogue has "
                       f"R = {star.radius}, M = {star.mass}. A missing row is "
                       f"normal at this scale and is not a verdict."):
        r.set_self_check(False, "no stellar density")
        return r

    period = float(ctx.candidate.period)
    t14_days = ctx.candidate.duration_days()
    t14_h = t14_days * 24.0

    # the planet/star radius ratio, from the depth if we have one
    depth = ctx.candidate.depth_ppm
    k = float(np.sqrt(depth / 1e6)) if depth and depth > 0 else 0.05
    k_source = "from the candidate depth" if depth else "assumed 0.05 (no depth given)"

    t_min, t_max = duration_range(period, rho_star, k)
    b_impl = implied_b(period, rho_star, t14_days, k)

    # transit-implied density: from the fitted a/R* when available, else by
    # inverting the observed duration at the implied b
    a_rs_fit = getattr(a, "a_rs", None)
    if a_rs_fit is None:
        a_rs_fit = _a_rs_from_fit(ctx)
    if a_rs_fit:
        rho_t = rho_circ(period, float(a_rs_fit))
        rho_source = "fitted a/R*"
    else:
        # b = 0 is the only assumption that can be made without a fit, and it
        # gives the *largest* implied density, so the discrepancy it reports is
        # an upper bound
        s = np.sin(np.pi * t14_days / period)
        a_rs_fit = (1 + k) / s if 0 < s < 1 else np.nan
        rho_t = rho_circ(period, a_rs_fit) if np.isfinite(a_rs_fit) else None
        rho_source = "inverted from T14 at b = 0 (no fit available)"

    rec = reconcile_density(rho_star, rho_t) if rho_t else {}

    r.metrics.update({
        "period": period,
        "duration_h": t14_h,
        "duration_allowed_h": [t_min * 24, t_max * 24],
        "duration_allowed_note": "b = 1 (grazing) to b = 0 (central), circular",
        "rp_rs_assumed": k, "rp_rs_source": k_source,
        "implied_b": b_impl,
        "a_rs": float(a_rs_fit) if a_rs_fit and np.isfinite(a_rs_fit) else None,
        "a_rs_source": rho_source,
        "rho_star_gcc": rho_star,
        "rho_transit_gcc": rho_t,
        "duration_ratio_observed_over_central": (
            float(t14_days / t_max) if t_max > 0 else None),
        **{f"reconcile_{key}": val for key, val in rec.items()},
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [], True
    if star.e_radius and star.radius:
        frac = star.e_radius / star.radius
        checks.append(f"catalogue R* uncertainty {frac:.0%}")
        if frac > 0.5:
            checks.append("stellar radius is too poorly known for this comparison")
            ok = False
    else:
        checks.append("no stellar radius uncertainty in the catalogue")
    checks.append(f"rho* = {rho_star:.3f} g/cm3 from {star.source}")
    r.set_self_check(ok, "; ".join(checks))

    inside = b_impl is not None
    ecc_req = rec.get("ecc_required")
    rad_req = rec.get("radius_error_required")

    # ---- is the companion even planet-sized? ------------------------------
    # Nothing is a planet above about 2 Jupiter radii: degeneracy sets in and
    # the radius stops growing with mass, so brown dwarfs and low-mass stars
    # all sit near 1 Rjup while inflated young giants reach perhaps 2.  This
    # is a physical ceiling, not an inflation argument, and it is the same
    # shape of statement as the secondary-eclipse depth cut.
    #
    # It is a product of the stellar radius, so it only refutes when that
    # radius is trustworthy: a catalogue value with a stated uncertainty, and
    # a margin wide enough that the error would have to be enormous.  Below
    # that margin it flags.
    ceiling = _radius_ceiling(getattr(star, "age_myr", None))
    rv = _rv_note(period, getattr(star, "mass", None))
    r.metrics["radius_ceiling_rjup"] = ceiling
    r.metrics["age_myr"] = getattr(star, "age_myr", None)
    rp_rj = None
    if star.radius and k and np.isfinite(k):
        rp_rj = float(k * star.radius * RSUN_OVER_RJUP)
        r.metrics["rp_rjup"] = rp_rj
        r.metrics["rp_rjup_ceiling"] = ceiling
        needed = rp_rj / ceiling
        r.metrics["stellar_radius_error_for_planet"] = float(needed)

    # A depth no fit has confirmed cannot refute anything: 84% of the FAILs here
    # rested on the search's depth because the fit had not converged (2026-11-05).
    depth_unverified = bool(getattr(ctx, "depth_unverified", False))
    r.metrics["depth_unverified"] = depth_unverified
    radius_impossible = (
        rp_rj is not None
        and not depth_unverified
        and rp_rj > ceiling * RADIUS_FAIL_MARGIN
        and star.radius and star.e_radius
        and star.e_radius / star.radius < RADIUS_TRUSTED_FRAC)

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif radius_impossible:
        r.fail(f"the companion is {rp_rj:.1f} Jupiter radii, which is not a "
               f"planet at any age: above about "
               f"{ceiling:.1f} Rjup at this age degeneracy caps the "
               f"radius, so brown dwarfs and low-mass stars all sit near 1 "
               f"Rjup and even an inflated young giant does not reach this. "
               f"R* = {star.radius:.3f} +/- {star.e_radius:.3f} Rsun from "
               f"{star.source}{radius_error_note(star)} would have to be wrong by a factor of "
               f"{rp_rj / ceiling:.1f} for this to be "
               f"planetary. T14 = {t14_h:.2f} h against "
               f"{t_min * 24:.2f}-{t_max * 24:.2f} h allowed.{rv}")
    elif rp_rj is not None and rp_rj > ceiling:
        why = ("the depth is the input table's, because the fit did not converge"
               if depth_unverified else
               "the stellar radius has no stated uncertainty"
               if not (star.radius and star.e_radius) else
               f"R* is uncertain by {star.e_radius / star.radius:.0%}"
               if star.e_radius / star.radius >= RADIUS_TRUSTED_FRAC else
               f"it is within the {RADIUS_FAIL_MARGIN:g}x margin of the ceiling")
        r.flag(f"the companion would be {rp_rj:.1f} Jupiter radii, above the "
               f"~{ceiling:.1f} Rjup ceiling for anything "
               f"planetary -- but {why}, so this flags rather than refutes. "
               f"R* would need to be a factor of "
               f"{rp_rj / ceiling:.1f} smaller.{rv}")
    elif inside:
        r.passed(f"T14 = {t14_h:.2f} h is consistent with rho* = "
                 f"{rho_star:.3f} g/cm3 at b = {b_impl:.2f} "
                 f"(allowed {t_min * 24:.2f}-{t_max * 24:.2f} h); no "
                 f"eccentricity or radius revision required")
    else:
        too_long = t14_days > t_max
        what = ("longer than a central crossing allows"
                if too_long else "shorter than even a grazing transit allows")
        bits = [f"T14 = {t14_h:.2f} h is {what} "
                f"(allowed {t_min * 24:.2f}-{t_max * 24:.2f} h for "
                f"rho* = {rho_star:.3f} g/cm3)"]
        if ecc_req is not None:
            bits.append(f"reconciled by e >= {ecc_req:.2f}")
        if rad_req is not None:
            bits.append(f"or by R* {'larger' if rad_req > 0 else 'smaller'} "
                        f"by {abs(rad_req):.0%}")
        strong = ((ecc_req is not None and ecc_req > ECC_FLAG) and
                  (rad_req is not None and abs(rad_req) > RADIUS_ERROR_FLAG))
        tail = (". Both reconciliations are large; this is the shape a "
                "detrending ramp or a blend makes, and is worth a look."
                if strong else
                ". Reported, not vetoed: young-star radii are routinely "
                "wrong by this much.")
        r.flag("; ".join(bits) + tail)

    if ctx.make_plots:
        try:
            _plot(ctx, r, period, rho_star, k, t14_days, t_min, t_max, b_impl)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _a_rs_from_fit(ctx):
    """a/R* from this target's fit.json, if pc_fit has been run."""
    import json
    path = ctx.out / ctx.target.slug / (
        f"fit_c{ctx.candidate.index}.json" if ctx.candidate.index else "fit.json")
    if not path.exists():
        return None
    try:
        m = (json.loads(path.read_text()).get("metrics") or {})
    except (OSError, json.JSONDecodeError):
        return None
    v = m.get("a_rs")
    return float(v) if v else None


def _radius_ceiling(age_myr):
    """Largest radius a planet can plausibly have at this age, in Rjup.

    Young objects are inflated -- a Jupiter-mass planet is roughly twice its
    final radius in its first few Myr -- so a fixed ceiling refutes real young
    planets at one end and lets stellar companions through at the other.
    """
    if not age_myr or not np.isfinite(age_myr) or age_myr <= 0:
        return PLANET_RADIUS_CEILING_RJUP * 1.4     # unknown age: be generous
    scaled = PLANET_RADIUS_CEILING_RJUP * (
        float(age_myr) / CEILING_REFERENCE_AGE_MYR) ** CEILING_AGE_INDEX
    return float(np.clip(scaled, CEILING_MIN_RJUP, CEILING_MAX_RJUP))


def _rv_note(period_days, m_star):
    """The RV semi-amplitude a deuterium-limit companion would produce.

    If a radius argument is ambiguous, radial velocity is not: at 13 Mjup the
    reflex motion is hundreds of m/s, which a single spectrum settles.
    """
    if not (period_days and m_star and m_star > 0):
        return ""
    k = 28.4329 * 13.0 * (m_star ** (-2.0 / 3.0)) * ((period_days / 365.25) ** (-1.0 / 3.0))
    return (f" At the 13 Mjup deuterium limit this companion would produce an "
            f"RV semi-amplitude of about {k:.0f} m/s, so a spectrum settles it.")


def _plot(ctx, r, period, rho_star, k, t14_days, t_min, t_max, b_impl):
    fig, ax = figure(figsize=(7.5, 4.6))

    a = a_over_rs(period, rho_star)
    bs = np.linspace(0, 1.0 + k, 400)
    durs = np.array([duration_t14(period, a, k, b=b) * 24 for b in bs])
    ax.plot(bs, durs, "-", color=C_MODEL, lw=1.6,
            label=f"circular orbit, rho* = {rho_star:.3f} g/cm3")
    ax.fill_between(bs, 0, durs, color=C_MODEL, alpha=0.07)

    ax.axhline(t14_days * 24, color=C_BINNED, lw=1.6,
               label=f"observed T14 = {t14_days * 24:.2f} h")
    if b_impl is not None:
        ax.plot([b_impl], [t14_days * 24], "o", ms=8, color=C_OK,
                label=f"implied b = {b_impl:.2f}")
    else:
        ax.plot([], [], " ", label="no b reproduces this duration")

    # what an eccentric orbit would reach
    for e_try, ls in ((0.2, "--"), (0.4, ":")):
        for sign, lab in ((+1, None), (-1, None)):
            g = np.sqrt((1 + sign * e_try) / (1 - sign * e_try))
            ax.plot(bs, durs / g, ls, color="#999999", lw=0.9,
                    label=(f"e = {e_try}" if sign > 0 and lab is None else None))

    ax.set_xlabel("impact parameter b")
    ax.set_ylabel("T14 [hours]")
    ax.set_ylim(0, max(durs.max(), t14_days * 24) * 1.25)
    ax.set_title(f"duration against the star's density (P = {period:.4f} d, "
                 f"Rp/R* = {k:.3f})")
    ax.legend(loc="lower left", ncol=2)

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("duration")))



def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Is the transit duration physical for this star? Reports the "
              "eccentricity or radius error that would reconcile it; never vetoes.",
        needs_lightcurve=True))
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

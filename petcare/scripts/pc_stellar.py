"""pc_stellar -- the stellar parameters the physical tests depend on.

Not a vetting test in its own right: it collects the parameters, records where
each one came from, and says plainly how much of the rest of the pipeline they
can support.  It exists as a separate script because "the duration test
vetoed this candidate" and "the TIC radius for this star is 20% too small"
are different statements, and the second one has to be visible.

What it reports:

* radius, mass, effective temperature, log g and the **derived** density, with
  the catalogue column each came from;
* the limb-darkening coefficients the fits will use, and their provenance;
* the planet radius the candidate depth implies (the light curve is already
  de-diluted, so the depth is not corrected again here);
* the age, which is what this whole survey is selected on.

Density is computed from M/R^3 rather than taken from a catalogue ``rho``
column even when one exists.  The TIC's density is itself derived, and
carrying two derivations of one quantity is exactly how two depth estimators
came to disagree by a median factor of 0.68 on this project.

    python -m petcare.scripts.pc_stellar --tic 166527623 --depth 5279
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli
from ..result import Result
from ..stellar import a_over_rs, duration_range, planet_radius_earth

TEST = "stellar"
VERSION = "0.1.1"

#: Fractional radius uncertainty above which the physical tests cannot say
#: much.  A 30% radius error moves the allowed duration range by a factor of
#: two, which is wider than the range itself.
RADIUS_ERROR_FLAG = 0.30

#: Ages above the survey's upper bound.  Reported rather than acted on: the
#: age is the selection criterion, so a candidate on a star that turns out to
#: be older is still a candidate, just not a young one.
AGE_MAX_MYR = 50.0


def add_arguments(ap):
    g = ap.add_argument_group("stellar")
    g.add_argument("--dilution", type=float, default=None,
                   help="target flux fraction in the aperture, overriding the "
                        "extraction's value (1.0 = uncontaminated)")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    star = ctx.star
    rho, e_rho = star.density()
    u1, u2, ld_source = star.limb_darkening()

    # dilution, from the light curve if there is one
    dil, dil_source = getattr(a, "dilution", None), "--dilution"
    if dil is None:
        try:
            lc = ctx.lc
            ctx.attach_lc_provenance(r, lc)
            dil, dil_source = lc.dilution()
        except Exception as exc:  # noqa: BLE001 - this test runs without one
            r.set_provenance(lightcurve=None)
            r.provenance.setdefault("notes", []).append(f"no light curve: {exc}")
            dil, dil_source = None, None
    if dil is not None:
        r.set_dilution(factor=dil, source=dil_source or "unknown", applied=True,
                       note="the light curve is already de-diluted; "
                            "do not divide depths by this factor")

    r.set_noise(None, "not applicable: this is a catalogue test")

    r.metrics.update({
        "name": star.name,
        "tic": star.tic, "gaia_dr3": star.gaia_dr3,
        "ra": star.ra, "dec": star.dec, "tmag": star.tmag,
        "radius_rsun": star.radius, "e_radius_rsun": star.e_radius,
        "mass_msun": star.mass, "e_mass_msun": star.e_mass,
        "teff_k": star.teff, "e_teff_k": star.e_teff,
        "logg": star.logg, "e_logg": star.e_logg,
        "density_gcc": rho, "e_density_gcc": e_rho,
        "density_source": ("M/R^3" if star.mass and star.radius
                           else ("catalogue rho" if star.rho else None)),
        "age_myr": star.age_myr, "age_lo_myr": star.e_age_lo,
        "age_hi_myr": star.e_age_hi,
        "distance_pc": star.dist_pc, "ruwe": star.ruwe,
        "contratio": star.contratio, "vsini": star.vsini,
        "prot_days": star.prot_days,
        "limb_darkening": {"u1": u1, "u2": u2, "source": ld_source},
        "parameter_source": star.source,
        "columns_used": star.source_detail.get("columns_used"),
        "source_detail": {k: v for k, v in star.source_detail.items()
                          if k != "columns_used"},
    })

    # ---- what the candidate implies, if there is one ----------------------
    depth = ctx.candidate.depth_ppm
    if depth and depth > 0 and star.radius:
        k = float(np.sqrt(depth / 1e6))
        rp = planet_radius_earth(k, star.radius)
        entry = {"rp_rs": k, "rp_earth": rp}
        if dil and dil > 0:
            entry["dilution_factor"] = float(dil)
        r.metrics["implied_planet"] = entry

    if ctx.candidate.period and rho:
        t_min, t_max = duration_range(ctx.candidate.period, rho,
                                      r.metrics.get("implied_planet", {})
                                      .get("rp_rs", 0.05))
        r.metrics["allowed_duration_h"] = [t_min * 24, t_max * 24]
        r.metrics["a_over_rs_circular"] = a_over_rs(ctx.candidate.period, rho)

    # ---- self-check -------------------------------------------------------
    checks, ok = [], True
    missing = [n for n, v in (("radius", star.radius), ("mass", star.mass),
                              ("Teff", star.teff)) if v is None]
    if missing:
        checks.append("missing: " + ", ".join(missing))
    frac = (star.e_radius / star.radius
            if star.e_radius and star.radius else None)
    if frac is not None:
        checks.append(f"radius uncertainty {frac:.0%}")
    else:
        checks.append("no radius uncertainty in the catalogue")
    if rho is None:
        ok = False
        checks.append("no density: every physical test will be INCONCLUSIVE")
    r.set_self_check(ok, "; ".join(checks))

    # ---- the verdict ------------------------------------------------------
    supports = []
    if rho is not None:
        supports.append("duration/density")
    if star.radius:
        supports.append("planet radius")
    if star.teff:
        supports.append("limb darkening")

    if rho is None:
        r.inconclusive(
            f"no stellar density for {star.name}: the catalogue gives "
            f"R = {star.radius}, M = {star.mass} ({star.source}). "
            f"pc_duration and any radius will be INCONCLUSIVE until this "
            f"star gets parameters. A missing row is normal at this scale.")
    elif frac is not None and frac > RADIUS_ERROR_FLAG:
        r.flag(f"{star.describe()}; the radius is uncertain by {frac:.0%}, "
               f"which moves the allowed transit duration by more than the "
               f"allowed range itself. The physical tests will run but should "
               f"not be read as constraints.")
    elif star.age_myr is not None and star.age_myr > AGE_MAX_MYR:
        r.flag(f"{star.describe()}; the age is {star.age_myr:.0f} Myr, above "
               f"the {AGE_MAX_MYR:.0f} Myr the survey selects on. The "
               f"parameters are fine; the star may not belong in the sample.")
    else:
        r.passed(f"{star.describe()}; supports " + ", ".join(supports)
                 + (f"; radius good to {frac:.0%}" if frac is not None else ""))
    return r


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Collect the stellar parameters, say where each came from, and "
              "say what the rest of the pipeline can do with them."))
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

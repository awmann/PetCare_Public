"""pc_companion -- what could this object be, at this age?

A transit depth gives a radius, and a radius alone does not distinguish a
planet from a brown dwarf: degeneracy caps both near a Jupiter radius once
they are old, and both are inflated when they are young.  Age breaks the
degeneracy.  This test asks the Sonora Diamondback evolutionary grid (Morley
et al. 2024) what masses could produce the measured radius at the measured
age, and reports the answer as a range rather than a number.

Three outcomes:

**PASS** -- every consistent mass is below the deuterium-burning limit.  The
companion is planetary, and the test says at what mass.

**FLAG** -- the consistent range reaches into the brown-dwarf regime.  This is
not a refutation: a transiting brown dwarf around a young star is a real and
interesting object, and at these radii the stellar-radius uncertainty usually
spans the boundary anyway.  The range is reported so a human can see how much
of it is planetary.

**FAIL** -- the radius exceeds anything substellar at this age, by more than
the measurement uncertainty.  Nothing below the hydrogen-burning limit gets
that big, so the companion is a star and the candidate is refuted.  This
needs no stellar model grid: the substellar upper envelope is the bound.

The uncertainty that matters is almost never the transit depth.  It is the
stellar radius, which multiplies straight through -- so a 20% error on R*
is a 20% error on the companion and is propagated as such.  Where the
catalogue gives no R* uncertainty, nothing is refuted.

    python -m petcare.scripts.pc_companion --tic 410214986 --use-fit \\
        --sonora /path/to/diamondback/evolution
"""

from __future__ import annotations

import sys

import numpy as np

from .. import cli, models
from ..plotting import C_ALT, C_BAD, C_MODEL, C_OK, annotate_status, figure, save
from ..result import Result
from ..stellar import radius_error_note

TEST = "companion"
VERSION = "0.1.1"

#: How far above the substellar envelope, in sigma of the measured radius,
#: before the companion is refuted as stellar.  PROVISIONAL: 2 leaves room
#: for a badly-quoted stellar radius without letting an obvious star through.
STELLAR_SIGMA = 2.0

#: And a floor on the fractional radius uncertainty, so a catalogue that
#: quotes an implausibly small R* error cannot drive a refutation.
MIN_RADIUS_FRAC_ERR = 0.10

#: Above this fractional R* uncertainty nothing is refuted at all -- the
#: radius is not measured well enough to argue from.
MAX_TRUSTED_FRAC_ERR = 0.35

#: And how far above the substellar maximum (after the sigma margin) before
#: refuting rather than flagging.  The envelope is not a hard wall: young,
#: low-mass planets can be far more inflated than hot-start grids allow, and
#: young-star radii are often off by more than their quoted errors, so at
#: these ages radius barely constrains mass.  At 2.0 a 5 Myr companion must exceed about
#: 5.9 Rjup to be refuted and a 40 Myr one about 3.9 -- sizes only a star
#: reaches.  Everything between the envelope and that flags.
RADIUS_FAIL_MARGIN = 2.0

RSUN_PER_RJUP = 9.731

#: Impact parameter above which the radius ratio is degenerate with geometry
#: and only sqrt(depth) is meaningful.
GRAZING_B = 0.95


def add_arguments(ap):
    g = ap.add_argument_group("companion")
    g.add_argument("--sonora", default=None,
                   help=f"Sonora Diamondback evolution tables; else "
                        f"${models.ENV_PATH}")
    g.add_argument("--fetch-models", action="store_true",
                   help="download the 0.8 MB evolution tables from Zenodo "
                        "into the --sonora directory if they are not there")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args
    star = ctx.star

    grid = models.load(getattr(a, "sonora", None))
    if grid is None and getattr(a, "fetch_models", False):
        dest = getattr(a, "sonora", None) or models.default_grid_path()
        try:
            models.fetch(dest)
            grid = models.load(dest)
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"could not fetch the Sonora grid: {exc}")
    if not cli.require(
            r, grid is not None,
            f"no Sonora Diamondback grid: pass --sonora or set "
            f"${models.ENV_PATH}, or re-run with --fetch-models. The grid is "
            f"Morley et al. 2024, doi:10.5281/zenodo.{models.ZENODO_RECORD}, "
            f"and is not shipped with this package."):
        return r

    age_myr = star.age_myr
    if not cli.require(r, age_myr and np.isfinite(age_myr) and age_myr > 0,
                       "no stellar age: pass --age, or give the candidate "
                       "table an age_myr column. Radius alone cannot "
                       "separate a planet from a brown dwarf; age is what "
                       "breaks the degeneracy."):
        return r
    if not cli.require(r, star.radius and star.radius > 0,
                       "no stellar radius, so there is no companion radius "
                       "to interpret"):
        return r

    # The radius ratio is taken from the depth, not from the fit, and that is
    # deliberate.  On a grazing geometry rp_rs is degenerate with the impact
    # parameter -- one candidate fits at b = 1.0 railed with rp_rs = 0.277
    # against a sqrt(depth) of 0.177, a factor of 1.6 in radius and 8.4 versus
    # 4.6 Rjup.  sqrt(depth) is the geometry-independent *lower* bound: a
    # grazing companion is larger than this, never smaller.  Refuting on the
    # lower bound is unimpeachable; refuting on the fit's preferred value
    # would be refuting on an unconstrained parameter.
    import json
    fit = ctx.out / ctx.target.slug / (
        f"fit_c{ctx.candidate.index}.json" if ctx.candidate.index else "fit.json")
    depth_ppm = ctx.candidate.depth_ppm
    k_fit = b_fit = railed = None
    if fit.exists():
        try:
            m = (json.loads(fit.read_text()).get("metrics") or {})
            k_fit, b_fit = m.get("rp_rs"), m.get("b")
            railed = (m.get("fit_detail") or {}).get("railed")
            depth_ppm = m.get("depth_ppm") or depth_ppm
        except (OSError, json.JSONDecodeError):
            pass
    k = float(np.sqrt(float(depth_ppm) / 1e6)) if depth_ppm and depth_ppm > 0 else None
    grazing = bool((b_fit is not None and b_fit > GRAZING_B)
                   or (railed and "b" in railed))
    if not cli.require(r, k and np.isfinite(k) and k > 0,
                       "no transit depth to take a radius from: run pc_fit "
                       "first, or pass --depth"):
        return r

    # ---- the companion radius, and what actually limits it ----------------
    rp_rjup = float(k) * float(star.radius) * RSUN_PER_RJUP
    frac_star = (star.e_radius / star.radius
                 if star.e_radius and star.radius else None)
    frac = max(frac_star or MIN_RADIUS_FRAC_ERR, MIN_RADIUS_FRAC_ERR)
    sigma_rjup = rp_rjup * frac

    age_gyr = float(age_myr) / 1000.0
    lo, hi, n_masses, slice_age = models.masses_consistent(
        age_gyr, rp_rjup, sigma_rjup, grid)
    r_max, m_at_max, _ = models.max_substellar_radius(age_gyr, grid)
    r_min, m_at_min = models.min_substellar_radius(age_gyr, grid)

    r.set_noise(sigma_rjup * 1e3, "stellar-radius uncertainty, in milli-Rjup",
                radius_frac_err=frac,
                radius_err_source=(
                    "assumed (none in catalogue)"
                    if star.source_detail.get("e_radius_assumed_frac") else
                    "catalogue" if frac_star else
                    f"floor of {MIN_RADIUS_FRAC_ERR:.0%}"))
    r.metrics.update({
        "rp_rjup": rp_rjup, "rp_rjup_err": sigma_rjup,
        "rp_rs_used": float(k), "rp_rs_source": "sqrt(depth), geometry-free",
        "rp_rs_from_fit": k_fit, "impact_parameter": b_fit, "grazing": grazing,
        "rp_rjup_if_fit_geometry": (float(k_fit) * float(star.radius)
                                    * RSUN_PER_RJUP) if k_fit else None,
        "rp_earth": rp_rjup * 11.209,
        "age_myr": float(age_myr), "grid_age_gyr": slice_age,
        "mass_range_mjup": [lo, hi], "n_grid_masses_consistent": n_masses,
        "max_substellar_radius_rjup": r_max,
        "max_substellar_at_mjup": m_at_max,
        "min_grid_radius_rjup": r_min, "min_grid_at_mjup": m_at_min,
        "deuterium_limit_mjup": models.DEUTERIUM_LIMIT_MJUP,
        "grid": "Sonora Diamondback (Morley+2024)",
    })
    if lo is not None and ctx.candidate.period and star.mass:
        r.metrics["rv_semi_amplitude_ms"] = {
            "at_min_mass": models.rv_semi_amplitude(
                lo, ctx.candidate.period, star.mass),
            "at_max_mass": models.rv_semi_amplitude(
                hi, ctx.candidate.period, star.mass)}

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"Rp = {rp_rjup:.2f} +/- {sigma_rjup:.2f} Rjup at "
                  f"{age_myr:.0f} Myr (grid slice {slice_age * 1000:.0f} Myr)"], True
    if grazing:
        checks.append(f"grazing fit (b = {b_fit:.2f}"
                      f"{', railed' if railed and 'b' in railed else ''}), so "
                      f"the radius is a lower bound -- the true companion is "
                      f"larger than this, never smaller")
    if frac_star is None:
        checks.append(f"no R* uncertainty in the catalogue; using a "
                      f"{MIN_RADIUS_FRAC_ERR:.0%} floor, and not refuting")
    elif frac_star > MAX_TRUSTED_FRAC_ERR:
        checks.append(f"R* is uncertain by {frac_star:.0%}, too poorly known "
                      f"to argue a mass from")
        ok = False
    if abs(slice_age - age_gyr) > 0.5 * age_gyr:
        checks.append(f"nearest grid age is {slice_age * 1000:.0f} Myr against "
                      f"the star's {age_myr:.0f} Myr")
        ok = False
    r.set_self_check(ok, "; ".join(checks))

    # A depth no converged fit has confirmed cannot refute anything:
    # 423 of 833 FAILs here rested on an unconverged fit's depth.
    depth_unverified = bool(getattr(ctx, "depth_unverified", False))
    r.metrics["depth_unverified"] = depth_unverified
    refutable = (ok and not depth_unverified and frac_star is not None
                 and frac_star <= MAX_TRUSTED_FRAC_ERR)
    rv = ""
    if lo is not None and (r.metrics.get("rv_semi_amplitude_ms") or {}).get("at_max_mass"):
        k_rv = r.metrics["rv_semi_amplitude_ms"]["at_max_mass"]
        if np.isfinite(k_rv):
            rv = (f" At the top of that range the RV semi-amplitude is about "
                  f"{k_rv:.0f} m/s, which a spectrum settles.")

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif (rp_rjup - STELLAR_SIGMA * sigma_rjup > r_max * RADIUS_FAIL_MARGIN
          and refutable):
        bound = (" -- and that is a lower bound, since the grazing fit leaves "
                 "the true radius larger" if grazing else "")
        r.fail(f"too large to be substellar at this age: Rp = {rp_rjup:.2f} "
               f"+/- {sigma_rjup:.2f} Rjup against a maximum of {r_max:.2f} "
               f"Rjup for anything below the hydrogen-burning limit at "
               f"{age_myr:.0f} Myr (reached at {m_at_max:.0f} Mjup in the "
               f"Sonora Diamondback grid). The companion is a star. R* = "
               f"{star.radius:.3f} +/- {star.e_radius:.3f} Rsun from "
               f"{star.source}{radius_error_note(star)} would have to be "
               f"{rp_rjup / r_max:.1f}x smaller "
               f"for this to be substellar{bound}.")
    elif rp_rjup - STELLAR_SIGMA * sigma_rjup > r_max:
        why = ("the depth is from a fit that did not converge" if depth_unverified else
               "R* has no stated uncertainty" if frac_star is None else
               f"R* is uncertain by {frac_star:.0%}" if not refutable else
               f"it is within the {RADIUS_FAIL_MARGIN:g}x margin that young "
               f"radii need")
        r.flag(f"larger than anything substellar at this age -- Rp = "
               f"{rp_rjup:.2f} +/- {sigma_rjup:.2f} Rjup against a maximum of "
               f"{r_max:.2f} -- but {why}, so this flags rather than "
               f"refutes.{rv}")
    elif lo is None and rp_rjup + STELLAR_SIGMA * sigma_rjup < r_min:
        # Smaller than the lightest object the grid carries, which is already
        # sub-Saturn.  Nothing substellar is this small, so it is a planet --
        # the grid simply does not extend down to it, and that is not an
        # ambiguity.
        r.passed(f"planetary, and below the grid: Rp = {rp_rjup:.2f} +/- "
                 f"{sigma_rjup:.2f} Rjup is smaller than the lightest object "
                 f"Sonora Diamondback carries ({r_min:.2f} Rjup at "
                 f"{m_at_min:.1f} Mjup, {age_myr:.0f} Myr). Nothing substellar "
                 f"is this small; a mass cannot be inferred from a radius here "
                 f"and none is claimed.")
    elif lo is None:
        r.flag(f"Rp = {rp_rjup:.2f} +/- {sigma_rjup:.2f} Rjup matches no mass "
               f"in the grid at {age_myr:.0f} Myr, where substellar radii run "
               f"{models.radius_at(age_gyr, 1.0, grid):.2f}-{r_max:.2f} Rjup. "
               f"Either the radius is below the grid's lightest object or the "
               f"age is outside where this grid applies.")
    elif hi > models.DEUTERIUM_LIMIT_MJUP:
        planetary = lo < models.DEUTERIUM_LIMIT_MJUP
        r.flag(f"consistent with a brown dwarf: the grid allows "
               f"{lo:.1f}-{hi:.1f} Mjup at {age_myr:.0f} Myr for Rp = "
               f"{rp_rjup:.2f} +/- {sigma_rjup:.2f} Rjup"
               + (f", which straddles the {models.DEUTERIUM_LIMIT_MJUP:.0f} "
                  f"Mjup deuterium limit -- part of the range is still "
                  f"planetary" if planetary else
                  f", entirely above the {models.DEUTERIUM_LIMIT_MJUP:.0f} "
                  f"Mjup deuterium limit") + f".{rv} A transiting brown dwarf "
               f"around a young star is a real object, so this is a flag, not "
               f"a refutation.")
    else:
        r.passed(f"consistent with a planet: the grid allows {lo:.1f}-{hi:.1f} "
                 f"Mjup at {age_myr:.0f} Myr for Rp = {rp_rjup:.2f} +/- "
                 f"{sigma_rjup:.2f} Rjup, all of it below the "
                 f"{models.DEUTERIUM_LIMIT_MJUP:.0f} Mjup deuterium limit.")

    if ctx.make_plots:
        try:
            _plot(ctx, r, grid, age_myr, rp_rjup, sigma_rjup, lo, hi)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r, grid, age_myr, rp, sigma, lo, hi):
    ages, masses, radii, _ = grid
    fig, axes = figure(1, 2, figsize=(11, 4.4))

    # ---- radius against age, for a ladder of masses -----------------------
    ax = axes[0]
    ladder = [1, 2, 5, 10, models.DEUTERIUM_LIMIT_MJUP, 20, 40, 70]
    for m in ladder:
        sel_a, sel_r = [], []
        for age in np.unique(ages):
            s = np.isclose(ages, age)
            if masses[s].min() <= m <= masses[s].max():
                o = np.argsort(masses[s])
                sel_a.append(age * 1000)
                sel_r.append(np.interp(m, masses[s][o], radii[s][o]))
        if len(sel_a) > 2:
            brown = m >= models.DEUTERIUM_LIMIT_MJUP
            ax.plot(sel_a, sel_r, "-", lw=1.2, alpha=0.85,
                    color=C_ALT if brown else C_OK)
            ax.annotate(f"{m:g}", (sel_a[-1], sel_r[-1]), fontsize=7,
                        color=C_ALT if brown else C_OK,
                        xytext=(3, -2), textcoords="offset points")
    ax.errorbar([age_myr], [rp], yerr=[sigma], fmt="o", ms=7, color=C_BAD,
                capsize=3, zorder=5, label="this candidate")
    ax.set_xscale("log")
    ax.set_xlabel("age [Myr]")
    ax.set_ylabel("radius [$R_{\\rm Jup}$]")
    ax.set_title("Sonora Diamondback tracks, labelled in $M_{\\rm Jup}$\n"
                 "below the deuterium limit, and above it")
    ax.legend(loc="upper right", fontsize=8)

    # ---- radius against mass at this age ----------------------------------
    ax = axes[1]
    slice_age = np.unique(ages)[np.argmin(np.abs(np.unique(ages) - age_myr / 1000))]
    s = np.isclose(ages, slice_age)
    o = np.argsort(masses[s])
    ax.plot(masses[s][o], radii[s][o], "-", color=C_MODEL, lw=1.6,
            label=f"grid at {slice_age * 1000:.0f} Myr")
    ax.axhspan(rp - sigma, rp + sigma, color=C_BAD, alpha=0.16,
               label=f"measured {rp:.2f} $\\pm$ {sigma:.2f}")
    ax.axhline(rp, color=C_BAD, lw=1.2)
    ax.axvline(models.DEUTERIUM_LIMIT_MJUP, color="k", ls="--", lw=0.9,
               alpha=0.6)
    ax.annotate("deuterium limit", (models.DEUTERIUM_LIMIT_MJUP, ax.get_ylim()[1]),
                rotation=90, fontsize=7, va="top", ha="right",
                xytext=(-3, -4), textcoords="offset points")
    if lo is not None:
        ax.axvspan(lo, hi, color=C_OK, alpha=0.18,
                   label=f"allowed {lo:.1f}-{hi:.1f} $M_{{\\rm Jup}}$")
    ax.set_xscale("log")
    ax.set_xlabel("companion mass [$M_{\\rm Jup}$]")
    ax.set_ylabel("radius [$R_{\\rm Jup}$]")
    ax.set_title("what mass could give this radius, at this age")
    ax.legend(loc="upper left", fontsize=8)

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("companion")))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Planet, brown dwarf or star? Asks the Sonora Diamondback "
              "evolutionary grid what masses match the measured radius at "
              "the star's age."))
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

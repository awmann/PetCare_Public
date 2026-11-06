"""pc_astrometry -- is there an unresolved companion?

One Gaia DR3 query, high value.  Gaia's astrometric solution assumes a single
star moving in a straight line; a companion inside roughly an arcsecond makes
the photocentre wobble, and the solution degrades in a characteristic way.
This catches companions **no centroid shift and no seeing-limited image will
ever resolve** -- which is precisely the regime where a blend is hardest to
rule out and most likely to matter.

What is read:

* ``ruwe`` -- the renormalised unit weight error.  Above ~1.4 the single-star
  solution does not fit, and binarity is the usual reason.
* ``astrometric_excess_noise`` and its significance -- the scatter the model
  cannot explain, in mas.  The significance matters as much as the value: a
  large excess noise with a significance below 2 is not a detection.
* ``ipd_frac_multi_peak`` and ``ipd_gof_harmonic_amplitude`` -- image-parameter
  diagnostics sensitive to a resolved-but-blended second peak, reaching in to
  ~0.15 arcsec where RUWE is already saturated.
* ``non_single_star`` -- Gaia's own binary flag, when the solution converged.
* ``rv_amplitude_robust`` and the RV scatter -- a spectroscopic companion.

**This flags, it does not fail.**  Young stars are photometrically and
astrometrically noisy for reasons that have nothing to do with companions --
spots move the photocentre, discs scatter light, and the target sample is
selected for youth.  An elevated RUWE says "there may be a second star here",
which changes what follow-up is worth doing; it does not refute a transit.

    python -m petcare.scripts.pc_astrometry --tic 410214986
"""

from __future__ import annotations

import sys

import numpy as np

from .. import catalogs, cli
from ..plotting import C_BAD, C_MODEL, C_OK, annotate_status, figure, save
from ..result import Result

TEST = "astrometry"
VERSION = "0.1.0"

#: The conventional RUWE threshold for a poor single-star fit.
RUWE_FLAG = 1.4

#: ...and the value above which the solution is badly wrong rather than noisy.
RUWE_STRONG = 2.0

#: Excess noise (mas) and its significance, both needed.  A large excess noise
#: at low significance is a poorly-constrained solution, not a companion.
EXCESS_NOISE_MAS_FLAG = 0.5
EXCESS_NOISE_SIG_FLAG = 2.0

#: Fraction of windows with multiple peaks, above which a second source is
#: being resolved inside the Gaia window.  Gaia publishes
#: ``ipd_frac_multi_peak`` as an integer **percent**, not a fraction, and
#: reading it as a fraction flags every star in the sample at "100%".
IPD_MULTI_PEAK_FLAG = 0.02
IPD_PERCENT_COLUMNS = ("ipd_frac_multi_peak", "ipd_frac_odd_win")


def add_arguments(ap):
    g = ap.add_argument_group("astrometry")
    g.add_argument("--search-radius", type=float, default=5.0,
                   help="arcseconds, when the Gaia source must be found by "
                        "position rather than by source_id")
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    a = ctx.args

    star = ctx.star
    gaia_id = ctx.target.gaia_dr3 or star.gaia_dr3
    row, how = None, None

    if gaia_id:
        try:
            row = catalogs.gaia_source(int(gaia_id))
            how = f"source_id {int(gaia_id)}"
        except Exception as exc:  # noqa: BLE001 - an outage is not a verdict
            r.provenance.setdefault("notes", []).append(
                f"Gaia source query failed: {exc}")

    if row is None:
        ra, dec = ctx.target.ra or star.ra, ctx.target.dec or star.dec
        if not cli.require(r, ra is not None and dec is not None,
                           "no Gaia source_id and no coordinates to search by"):
            return r
        radius = float(getattr(a, "search_radius", 5.0))
        try:
            cone = catalogs.gaia_cone(ra, dec, radius)
        except Exception as exc:  # noqa: BLE001
            return r.inconclusive(f"Gaia query failed: {exc}")
        if not cli.require(r, bool(cone),
                           f"no Gaia DR3 source within {radius:.0f} arcsec of "
                           f"{ra:.5f}, {dec:.5f}"):
            r.set_self_check(False, "no Gaia source")
            return r
        row = min(cone, key=lambda x: x.get("phot_g_mean_mag") or 99)
        how = (f"brightest of {len(cone)} Gaia source(s) within "
               f"{radius:.0f} arcsec")

    r.set_provenance(lightcurve=None, gaia_source_id=row.get("source_id"),
                     identified_by=how)
    r.set_noise(None, "not applicable: this is a catalogue test")

    ruwe = _f(row.get("ruwe"))
    aen = _f(row.get("astrometric_excess_noise"))
    aen_sig = _f(row.get("astrometric_excess_noise_sig"))
    multi = _percent(row.get("ipd_frac_multi_peak"))
    harm = _f(row.get("ipd_gof_harmonic_amplitude"))
    nss = row.get("non_single_star")
    rv_amp = _f(row.get("rv_amplitude_robust"))
    rv_err = _f(row.get("radial_velocity_error"))
    n_rv = row.get("rv_nb_transits")
    plx = _f(row.get("parallax"))
    plx_err = _f(row.get("parallax_error"))

    r.metrics.update({
        "gaia_dr3": row.get("source_id"), "identified_by": how,
        "g_mag": _f(row.get("phot_g_mean_mag")), "bp_rp": _f(row.get("bp_rp")),
        "ruwe": ruwe,
        "astrometric_excess_noise_mas": aen,
        "astrometric_excess_noise_sig": aen_sig,
        "astrometric_gof_al": _f(row.get("astrometric_gof_al")),
        "astrometric_n_good_obs_al": row.get("astrometric_n_good_obs_al"),
        "ipd_frac_multi_peak": multi,
        "ipd_frac_odd_win": _percent(row.get("ipd_frac_odd_win")),
        "ipd_note": "Gaia publishes these two as integer percentages; "
                    "they are reported here as fractions",
        "ipd_gof_harmonic_amplitude": harm,
        "duplicated_source": row.get("duplicated_source"),
        "non_single_star": nss,
        "radial_velocity": _f(row.get("radial_velocity")),
        "radial_velocity_error": rv_err,
        "rv_nb_transits": n_rv,
        "rv_amplitude_robust": rv_amp,
        "vbroad": _f(row.get("vbroad")),
        "parallax": plx, "parallax_error": plx_err,
        "distance_pc": _f(1000.0 / plx) if plx and plx > 0 else None,
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [], True
    n_obs = row.get("astrometric_n_good_obs_al")
    if n_obs is not None:
        checks.append(f"{n_obs} good along-scan observations")
        if int(n_obs) < 50:
            checks.append("too few astrometric observations for RUWE to mean "
                          "anything")
            ok = False
    if ruwe is None:
        checks.append("no RUWE in the Gaia row")
        ok = False
    g = _f(row.get("phot_g_mean_mag"))
    if g is not None and g < 4:
        checks.append(f"G = {g:.1f} is bright enough that Gaia's astrometric "
                      f"diagnostics are saturated and unreliable")
        ok = False
    if row.get("duplicated_source"):
        checks.append("duplicated_source is set: Gaia had trouble with this "
                      "source, which is itself a hint of multiplicity")
    r.set_self_check(ok, "; ".join(checks) or "Gaia row present")

    # ---- the verdict ------------------------------------------------------
    hits = []
    if ruwe is not None and ruwe > RUWE_FLAG:
        hits.append(f"RUWE = {ruwe:.2f}")
    if (aen is not None and aen > EXCESS_NOISE_MAS_FLAG and
            aen_sig is not None and aen_sig > EXCESS_NOISE_SIG_FLAG):
        hits.append(f"excess noise {aen:.2f} mas at {aen_sig:.0f} sigma")
    if multi is not None and multi > IPD_MULTI_PEAK_FLAG:
        hits.append(f"{multi:.0%} of windows show multiple peaks")
    if nss:
        hits.append(f"Gaia non_single_star = {nss}")
    if rv_amp is not None and rv_amp > 10:
        hits.append(f"robust RV amplitude {rv_amp:.1f} km/s")

    context = (f"RUWE = {ruwe:.2f}" if ruwe is not None else "no RUWE") + \
              (f", excess noise {aen:.2f} mas" if aen is not None else "") + \
              (f" ({aen_sig:.0f} sigma)" if aen_sig is not None else "")

    if not ok:
        r.inconclusive("self-check failed: " + r.self_check["detail"])
    elif not hits:
        r.passed(f"no astrometric sign of a companion: {context}. "
                 f"This constrains companions inside ~1 arcsec, which imaging "
                 f"and centroids cannot reach; it says nothing about wider ones.")
    elif ruwe is not None and ruwe > RUWE_STRONG:
        r.flag("the single-star astrometric solution fails badly: "
               + "; ".join(hits) + ". An unresolved companion is the usual "
               "cause. FLAG ONLY -- young stars are astrometrically noisy for "
               "reasons unrelated to multiplicity, and this does not refute a "
               "transit; it changes what follow-up is worth doing.")
    else:
        r.flag("possible unresolved companion: " + "; ".join(hits) +
               ". FLAG ONLY -- see the note in the test docstring.")

    if ctx.make_plots:
        try:
            _plot(ctx, r)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


def _plot(ctx, r):
    """A single panel: where this star sits against the flag thresholds.

    Deliberately simple -- there is no image to show, and a bar chart of five
    numbers against their thresholds is what a human actually wants to see.
    """
    m = r.metrics
    items = [
        ("RUWE", m.get("ruwe"), RUWE_FLAG, RUWE_STRONG),
        ("excess noise [mas]", m.get("astrometric_excess_noise_mas"),
         EXCESS_NOISE_MAS_FLAG, 2 * EXCESS_NOISE_MAS_FLAG),
        ("excess noise sig", m.get("astrometric_excess_noise_sig"),
         EXCESS_NOISE_SIG_FLAG, 5 * EXCESS_NOISE_SIG_FLAG),
        ("IPD multi-peak frac", m.get("ipd_frac_multi_peak"),
         IPD_MULTI_PEAK_FLAG, 4 * IPD_MULTI_PEAK_FLAG),
        ("IPD harmonic amp", m.get("ipd_gof_harmonic_amplitude"), 0.1, 0.2),
    ]
    items = [(n, v, f1, f2) for n, v, f1, f2 in items if v is not None]
    if not items:
        return

    fig, ax = figure(figsize=(7.5, 3.6))
    y = np.arange(len(items))
    for i, (name, val, flag, strong) in enumerate(items):
        scale = max(val, strong) * 1.2
        ax.barh(i, val / scale, color=(C_BAD if val > flag else C_OK),
                height=0.5, alpha=0.75)
        ax.plot([flag / scale], [i], "|", ms=22, color=C_MODEL, mew=2)
        ax.text(min(val / scale, 0.95) + 0.02, i, f"{val:.3g}", va="center",
                fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, *_ in items], fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.plot([], [], "|", color=C_MODEL, ms=12, mew=2, label="flag threshold")
    ax.legend(loc="lower right")
    ax.set_title(f"Gaia DR3 {m.get('gaia_dr3')} astrometric diagnostics")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("astrometry")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _percent(x):
    """Gaia's integer-percent IPD columns, as a fraction."""
    v = _f(x)
    return None if v is None else v / 100.0


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Gaia DR3 astrometric signs of an unresolved companion inside "
              "~1 arcsec. Flags only.",
        needs_lightcurve=False, needs_candidate=False))
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

"""pc_crossmatch -- does the depth survive a different extraction?

Measures the same candidate in every independent light curve that exists for
the star -- SPOC, QLP, TESS-SPOC FFI, the local tessquicklook
extraction, and Kepler or K2 where the fields overlap -- with **one depth
estimator applied identically to all of them**.

**A depth that changes with the pipeline is an aperture effect.**  This is
what decisively refuted a 34-day candidate in the young-planet search: Kepler
measured 43 +/- 34 ppm where TESS claimed 1230 ppm.  Nothing else in this
package can make that comparison, because everything else works on one
extraction.

Two things make it a fair test rather than a collection of numbers:

* every extraction is measured with the **same** local-polynomial estimator
  and the same in-transit window, so a difference cannot come from the
  measurement;
* every uncertainty is measured from **off-phase nulls in that extraction**,
  so a noisy pipeline gets the large error bar it deserves instead of being
  treated as a disagreement.

The dilution each pipeline reports is carried alongside, because SPOC's
``CROWDSAP``-corrected PDCSAP and an uncorrected aperture sum *should* differ,
and by a predictable factor.  A disagreement that the dilution explains is not
a disagreement.

    python -m petcare.scripts.pc_crossmatch --tic 410214986 --use-fit
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from .. import cli, ephem, io as pcio
from ..cache import cache_dir, ensure_certificates
from ..plotting import (C_BINNED, C_MODEL, C_OK, annotate_status,
                        figure, save)
from ..prep import chi2_pvalue, weighted_mean
from ..result import Result

TEST = "crossmatch"
VERSION = "0.1.1"

#: Phase offsets used for the null (uncertainty) measurements.
NULL_OFFSETS = (-0.35, -0.25, -0.15, 0.15, 0.25, 0.35)

#: Chi-square p-value below which the extractions disagree.
DISAGREE_P = 1e-3

#: Fractional depth spread below which a disagreement is not worth acting on
#: even if it is formally significant.
DISAGREE_FRACTION = 0.3

#: Minimum independent extractions for the comparison to mean anything.
MIN_PIPELINES = 2

#: Per-transit local baseline (0.1.1).  0.1.0 folded every transit together
#: against one out-of-transit median, so on an active star each pipeline's depth
#: was whatever rotation phase its transits sampled: DS Tuc A c (812 ppm) came out
#: 229-5325 ppm across four pipelines and FLAGged; with a quadratic per transit
#: they agree at 631-1165 ppm (chi2 1.6 for 3 dof).  The window is +/- WINDOW_T14
#: durations -- small against Prot, or the baseline eats the transit.
WINDOW_T14 = 2.0
IN_T14 = 0.35
OOT_T14 = 0.6
MIN_IN = 3
MIN_OOT_SIDE = 4


def add_arguments(ap):
    g = ap.add_argument_group("crossmatch")
    g.add_argument("--authors", nargs="+",
                   default=["SPOC", "QLP", "TESS-SPOC", "TASOC"],
                   help="MAST light-curve authors to fetch")
    g.add_argument("--missions", nargs="+", default=["TESS", "Kepler", "K2"],
                   help="missions to search")
    g.add_argument("--max-products", type=int, default=12,
                   help="cap the number of files downloaded per author")
    g.add_argument("--no-download", action="store_true",
                   help="use only what is already in the cache")
    g.add_argument("--no-local", action="store_true",
                   help="skip the local (tessquicklook) extraction")
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
    if not cli.require(r, ctx.target.tic is not None,
                       "pc_crossmatch needs a TIC to search MAST by"):
        return r

    period = float(ctx.candidate.period)
    t0 = float(ctx.candidate.t0_btjd)
    dur = ctx.candidate.duration_days()

    measurements, notes = [], []

    # ---- the local extraction --------------------------------------------
    if not getattr(a, "no_local", False):
        try:
            lc = ctx.lc
            dil, dil_src = lc.dilution()
            m = _measure(lc.t, lc.f, period, t0, dur, ctx.known)
            if m:
                m.update({"pipeline": "local (tessquicklook)",
                          "mission": "TESS", "n_sectors": lc.n_sectors,
                          "sectors": [s.sector for s in lc.sectors],
                          "dilution": dil, "dilution_source": dil_src,
                          "files": [Path(p).name for p in lc.paths]})
                measurements.append(m)
        except Exception as exc:  # noqa: BLE001 - absence is normal
            notes.append(f"no local light curve: {exc}")

    # ---- everything MAST has ---------------------------------------------
    try:
        products = _search(int(ctx.target.tic), getattr(a, "missions", None),
                           getattr(a, "authors", None))
    except Exception as exc:  # noqa: BLE001 - an archive outage is not a verdict
        products = {}
        notes.append(f"MAST search failed: {exc}")

    for (mission, author), rows in sorted(products.items()):
        try:
            paths = _download(rows, int(getattr(a, "max_products", 12)),
                              bool(getattr(a, "no_download", False)))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{author}: download failed: {exc}")
            continue
        if not paths:
            continue
        try:
            lc = pcio.load_lightcurve(paths)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{author}: could not read products: {exc}")
            continue
        m = _measure(lc.t, lc.f, period, t0, dur, ctx.known)
        if not m:
            notes.append(f"{author}: no usable transit coverage")
            continue
        dil, dil_src = lc.dilution()
        m.update({"pipeline": author, "mission": mission,
                  "n_sectors": lc.n_sectors,
                  "sectors": [s.sector for s in lc.sectors],
                  "dilution": dil, "dilution_source": dil_src,
                  "files": [Path(p).name for p in lc.paths]})
        measurements.append(m)

    r.metrics["measurements"] = measurements
    r.metrics["notes"] = notes
    r.set_provenance(lightcurve=";".join(m["pipeline"] for m in measurements),
                     n_pipelines=len(measurements))

    if not cli.require(r, len(measurements) >= MIN_PIPELINES,
                       f"only {len(measurements)} independent extraction(s) "
                       f"available; {MIN_PIPELINES} are needed for a "
                       f"comparison" +
                       (". " + "; ".join(notes[:2]) if notes else "")):
        r.set_self_check(False, f"{len(measurements)} extractions")
        return r

    depths = np.array([m["depth_ppm"] for m in measurements])
    errs = np.array([m["depth_err_ppm"] for m in measurements])
    mean, mean_err, chi2, n = weighted_mean(depths, errs)
    dof = max(n - 1, 1)
    p_value = chi2_pvalue(chi2, dof)
    spread = float((depths.max() - depths.min()) / abs(mean)) if mean else np.nan

    r.set_noise(float(np.median(errs)),
                "off-phase null scatter, measured separately in each extraction",
                n_null_offsets=len(NULL_OFFSETS))

    r.metrics.update({
        "n_pipelines": len(measurements),
        "weighted_mean_depth_ppm": _f(mean),
        "weighted_mean_err_ppm": _f(mean_err),
        "chi2": _f(chi2), "dof": int(dof), "chi2_p_value": _f(p_value),
        "depth_spread_fraction": _f(spread),
        "deepest": max(measurements, key=lambda m: m["depth_ppm"])["pipeline"],
        "shallowest": min(measurements, key=lambda m: m["depth_ppm"])["pipeline"],
        "missions": sorted({m["mission"] for m in measurements}),
        "has_independent_mission": len({m["mission"] for m in measurements}) > 1,
    })

    # ---- self-check -------------------------------------------------------
    checks, ok = [f"{len(measurements)} extractions: "
                  + ", ".join(m["pipeline"] for m in measurements)], True
    dils = {m["pipeline"]: m.get("dilution") for m in measurements
            if m.get("dilution") is not None}
    if dils:
        checks.append("dilutions " + ", ".join(f"{k}={v:.3f}"
                                               for k, v in dils.items()))
        vals = list(dils.values())
        if max(vals) - min(vals) > 0.1:
            checks.append("the extractions do not agree about how much of the "
                          "aperture flux is the target, so some of the depth "
                          "difference below is expected and not a disagreement")
    if any(m["n_in"] < 20 for m in measurements):
        checks.append("at least one extraction has fewer than 20 in-transit "
                      "cadences")
    r.set_self_check(ok, "; ".join(checks))

    table = "; ".join(f"{m['pipeline']} {m['depth_ppm']:.0f}+/-"
                      f"{m['depth_err_ppm']:.0f}" for m in measurements)

    if p_value is not None and p_value < DISAGREE_P and spread > DISAGREE_FRACTION:
        r.flag(f"the extractions disagree about the depth: {table} ppm "
               f"(chi2 = {chi2:.1f}/{dof}, p = {p_value:.1e}, spread "
               f"{spread:.0%}). A depth that changes with the pipeline is an "
               f"aperture effect, not a planet -- check the dilution each one "
               f"applied before concluding anything.")
    elif p_value is not None and p_value < DISAGREE_P:
        r.flag(f"formally inconsistent depths but only a {spread:.0%} spread: "
               f"{table} ppm (p = {p_value:.1e}). Probably differing dilution "
               f"corrections rather than a real disagreement.")
    else:
        r.passed(f"all {len(measurements)} extractions agree: {table} ppm, "
                 f"weighted mean {mean:.0f} +/- {mean_err:.0f} ppm "
                 f"(chi2 = {chi2:.1f}/{dof}, p = {p_value:.2f})"
                 + (" -- including an independent mission"
                    if r.metrics["has_independent_mission"] else ""))

    if ctx.make_plots:
        try:
            _plot(ctx, r, measurements, period, t0, dur)
        except Exception as exc:  # noqa: BLE001
            r.provenance.setdefault("notes", []).append(f"plotting failed: {exc!r}")
    return r


# --------------------------------------------------------------------------


def _search(tic, missions, authors) -> dict:
    """MAST light-curve products, grouped by (mission, author)."""
    ensure_certificates()
    import lightkurve as lk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = lk.search_lightcurve(f"TIC {tic}")
    if res is None or len(res) == 0:
        return {}

    want_authors = {a.upper() for a in (authors or [])}
    want_missions = {m.upper() for m in (missions or [])}

    out = {}
    for row in res:
        author = _scalar(row.author)
        mission = _scalar(row.mission).split()[0] if row.mission is not None else "?"
        if want_authors and author.upper() not in want_authors:
            continue
        if want_missions and mission.upper() not in want_missions:
            continue
        out.setdefault((mission, author), []).append(row)
    return out


def _scalar(v) -> str:
    """One-element astropy columns stringify as "['SPOC']"; unwrap them.

    A single-row SearchResult returns a length-1 column rather than a value,
    and ``str()`` on it gives the list repr.  Matching that against an author
    name silently matches nothing, which looks exactly like "this star has
    only one extraction".
    """
    try:
        if hasattr(v, "__len__") and not isinstance(v, str):
            v = list(v)[0] if len(v) else ""
    except (TypeError, IndexError):
        pass
    return str(v).strip().strip("[]'\" ")


def _download(rows, max_products, no_download):
    """Download (or reuse) the products, into the shared cache."""

    d = cache_dir("mast_lightcurves")
    paths = []
    for row in rows[:max_products]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if no_download:
                    continue
                obj = row.download(download_dir=str(d))
            if obj is None:
                continue
            path = getattr(obj, "filename", None) or getattr(
                getattr(obj, "meta", {}), "get", lambda *_: None)("FILENAME")
            if path and Path(path).exists():
                paths.append(str(path))
        except Exception:  # noqa: BLE001 - one bad product is not fatal
            continue
    return paths


def _measure(t, f, period, t0, dur, known=()) -> dict | None:
    """Depth and its null-measured uncertainty, identically for every input."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    good = np.isfinite(t) & np.isfinite(f)
    # the other known signals are masked, matched by overlap as in prep
    others = [k for k in (known or []) if good.any() and not ephem.match_known(
        period, t0, dur * 24, [k], float(t[good].min()), float(t[good].max()))]
    if others:
        good &= ~ephem.mask_known_transits(t, others, width=2.0)
    t, f = t[good], f[good]
    if t.size < 200:
        return None

    real = _epoch_depths(t, f, period, t0, dur)
    if real.size == 0:
        return None
    # the null error is the scatter of single-transit depths at phases with no
    # transit, divided by sqrt(N transits): what this estimator does here by chance
    nulls = np.concatenate([_epoch_depths(t, f, period, t0 + off * period, dur)
                            for off in NULL_OFFSETS])
    if nulls.size < 3:
        return None
    sd = float(1.4826 * np.median(np.abs(nulls - np.median(nulls))))
    err = sd / np.sqrt(real.size)

    ph = ephem.phase(t, period, t0)
    return {"depth_ppm": float(np.mean(real)), "depth_err_ppm": max(err, 1.0),
            "n_points": int(t.size),
            "n_in": int((np.abs(ph) < 0.5 * dur).sum()),
            "n_transits": int(real.size),
            "epoch_depth_ppm": [float(x) for x in real],
            "null_epoch_sd_ppm": sd,
            "baseline_days": float(np.ptp(t)),
            "n_nulls": int(nulls.size)}


def _epoch_depths(t, f, period, t0, dur):
    """Per-transit depth [ppm] against a quadratic through that transit's own
    shoulders; transits without enough data on both sides are skipped."""
    out = []
    for e in np.unique(ephem.epoch_number(t, period, t0)):
        x = t - float(ephem.transit_time(e, period, t0))
        w = np.abs(x) < WINDOW_T14 * dur
        oot = w & (np.abs(x) > OOT_T14 * dur)
        inn = w & (np.abs(x) < IN_T14 * dur)
        if inn.sum() < MIN_IN or (oot & (x < 0)).sum() < MIN_OOT_SIDE \
                or (oot & (x > 0)).sum() < MIN_OOT_SIDE:
            continue
        c = np.polyfit(x[oot], f[oot], 2)
        base = np.polyval(c, x[inn])
        out.append(float(-np.mean(f[inn] / base - 1.0) * 1e6))
    return np.array(out)


def _plot(ctx, r, measurements, period, t0, dur):
    fig, axes = figure(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    y = np.arange(len(measurements))
    d = [m["depth_ppm"] for m in measurements]
    e = [m["depth_err_ppm"] for m in measurements]
    ax.errorbar(d, y, xerr=e, fmt="o", ms=6, color=C_BINNED)
    mean = r.metrics["weighted_mean_depth_ppm"]
    if mean is not None:
        ax.axvline(mean, color=C_OK, lw=1.2, label=f"weighted mean {mean:.0f} ppm")
        me = r.metrics.get("weighted_mean_err_ppm")
        if me:
            ax.axvspan(mean - me, mean + me, color=C_OK, alpha=0.15)
    exp = ctx.candidate.depth_ppm
    if exp:
        ax.axvline(exp, color=C_MODEL, ls=":", lw=1.1,
                   label=f"input depth {exp:.0f} ppm")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m['pipeline']}\n{m['mission']}, "
                        f"{m['n_transits']} transits" for m in measurements],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("depth [ppm]")
    p = r.metrics.get("chi2_p_value")
    ax.set_title("depth by extraction"
                 + (f"  (chi2 p = {p:.2g})" if p is not None else ""))
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.text(0.5, 0.5,
            "\n".join(f"{m['pipeline']}: {m['depth_ppm']:.0f} +/- "
                      f"{m['depth_err_ppm']:.0f} ppm"
                      + (f", dilution {m['dilution']:.3f}"
                         if m.get("dilution") else "")
                      for m in measurements),
            ha="center", va="center", transform=ax.transAxes, fontsize=8,
            family="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("the same estimator on every extraction")

    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("crossmatch")))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(
        TEST, "Measure the same candidate in every independent extraction "
              "with one estimator. A depth that changes with the pipeline is "
              "an aperture effect."))
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

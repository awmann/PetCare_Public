"""pc_pixels -- the light curve of every pixel around the transit.

A grid of per-pixel light curves, folded on the candidate and normalised pixel
by pixel, in the style of LATTE's pixel-level plot (Eisner et al. 2020; used
for vetting in e.g. Khan et al. 2026, arXiv:2607.08132, their Fig. 8).  A real
transit on the target is deepest -- in flux lost -- on the target's pixels and
fades with distance like the PSF.  A blended eclipsing binary shows up as a
dip that is strongest somewhere else, often as a deep *fractional* dip in a
faint pixel next to the target.

This is a diagnostic for a human first.  It repeats what pc_centroid measures
(the location of the missing flux) in a form that is easy to read by eye, so it
only ever PASSes, FLAGs or is INCONCLUSIVE; refuting stays with pc_centroid.

**PASS** -- the flux deficit peaks on the target pixel or one next to it.
**FLAG** -- it peaks two or more pixels away, significantly above the target's,
at the same place in at least two sectors (one sector alone is too easily a
residual of the star's variability).
**INCONCLUSIVE** -- no pixel shows the dip at 3 sigma, or no usable cutout.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from .. import catalogs, cli, ephem
from ..tesscut import cutout_wcs
from ..plotting import annotate_status, save
from . import pc_centroid as cen

TEST = "pixels"
VERSION = "0.1.1"

#: Half-width of the grid, in pixels: 3 gives 7 x 7 panels around the target.
BOX = 3
#: Half-width of the plotted grid: 5 x 5 panels (a readable plot
#: size).  The test still searches all of BOX; the plot
#: widens back to BOX when the peak pixel lies outside PLOT_BOX.
PLOT_BOX = 2
#: The folded window, in transit durations either side of mid-transit.
WINDOW_T14 = 2.5
#: Baseline points are those beyond this many durations from mid-transit.
OOT_T14 = 0.75
#: Detection threshold for a pixel's flux deficit.
PIX_SNR = 3.0
#: How much more the off-target peak must lose than the target, in sigma.
OFF_SIGMA = 2.0
MAX_SECTORS = 3
TRANSIT_COLOUR = "#e8710a"     # LATTE's in-transit orange


def add_arguments(ap):
    g = ap.add_argument_group("pixels")
    g.add_argument("--cutout-size", type=int, default=cen.DEFAULT_CUTOUT)
    g.add_argument("--no-download", action="store_true")
    return ap


def run(ctx: cli.Context):
    r = ctx.result(TEST, VERSION)
    a = ctx.args
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r
    if not cli.require(r, ctx.candidate.duration_h, "no transit duration"):
        return r
    star = ctx.star
    ra, dec = ctx.target.ra or star.ra, ctx.target.dec or star.dec
    if not cli.require(r, ra is not None and dec is not None, "no coordinates"):
        return r
    period, t0 = float(ctx.candidate.period), float(ctx.candidate.t0_btjd)
    dur = ctx.candidate.duration_days()

    try:
        paths, note = cen._cutouts(ra, dec, int(getattr(a, "cutout_size", cen.DEFAULT_CUTOUT)),
                                   None, bool(getattr(a, "no_download", False)))
    except Exception as exc:  # noqa: BLE001 - an archive outage is not a verdict
        return r.inconclusive(f"could not obtain TESScut cutouts: {exc}")
    if not cli.require(r, bool(paths), f"no pixel data: {note}"):
        r.set_self_check(False, "no cutouts")
        return r
    r.set_provenance(lightcurve=None, cutouts=[str(p) for p in paths], download_note=note)

    sectors, skipped = [], []
    for p in paths:
        try:
            s = _one_sector(p, ra, dec, period, t0, dur)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{Path(p).name}: {type(exc).__name__}: {exc}")
            continue
        (skipped.append(f"{Path(p).name}: {s}") if isinstance(s, str) else sectors.append(s))
    r.metrics["skipped_cutouts"] = skipped
    if not cli.require(r, bool(sectors), "no usable sector: " + "; ".join(skipped[:3])):
        r.set_self_check(False, "no usable sector")
        return r
    # the sector with the most in-transit cadences is the one plotted
    sectors.sort(key=lambda s: -s["n_in"])
    best = sectors[0]
    r.metrics["sectors"] = [{k: s[k] for k in ("sector", "n_in", "n_transits", "peak_dxdy",
                                               "peak_snr", "target_snr", "verdict")}
                            for s in sectors[:MAX_SECTORS]]
    r.set_self_check(True, f"{len(sectors)} usable sector(s); plotted s{best['sector']} "
                           f"({best['n_transits']} transits, {best['n_in']} in-transit cadences)")

    used = sectors[:MAX_SECTORS]
    verdicts = [s["verdict"] for s in used]
    det = [s for s in used if s["verdict"] != "none"]
    if not det:
        r.inconclusive(f"no pixel shows the dip at {PIX_SNR:g} sigma in any of {len(used)} "
                       f"sector(s): too shallow to see pixel by pixel. Read the grid by eye, "
                       f"but rely on pc_centroid.")
    elif all(v == "on" for v in (s["verdict"] for s in det)) and _centroid_off(ctx):
        # "on" here means the peak is on the target's pixel or one of its eight
        # neighbours -- up to 30 arcsec.  Next to a centroid FAIL/FLAG that reads
        # as a contradiction it is not (one case: 7.6 arcsec, both "on").
        r.inconclusive(f"the flux lost in transit peaks within 1 pixel of the target in "
                       f"{len(det)} sector(s), which cannot separate the target from the "
                       f"source pc_centroid places off it ({_centroid_off(ctx)}): a pixel is "
                       f"21 arcsec. pc_centroid's sub-pixel measurement decides.")
    elif all(v == "on" for v in (s["verdict"] for s in det)):
        r.passed(f"the flux lost in transit peaks on the target (within 1 pixel) in "
                 f"{len(det)} sector(s) where it is detected; in s{best['sector']} the target "
                 f"pixel loses flux at {best['target_snr']:.1f} sigma.  A pixel is 21 arcsec: a "
                 f"blend closer than that looks the same here, and only pc_centroid's "
                 f"sub-pixel measurement can tell.")
    else:
        off = [s for s in det if s["verdict"] == "off"]
        s = off[0]
        same = [o for o in off if max(abs(o["peak_dxdy"][0] - s["peak_dxdy"][0]),
                                      abs(o["peak_dxdy"][1] - s["peak_dxdy"][1])) <= 1]
        if len(same) < 2:
            on = [x for x in det if x["verdict"] == "on"]
            msg = (f"the flux lost in transit peaks on the target in {len(on)} sector(s)"
                   if on else "only one sector localises the dip")
            r.inconclusive(
                f"{msg}, but peaks {s['peak_dxdy'][0]:+d},{s['peak_dxdy'][1]:+d} pixels away in "
                f"s{s['sector']} alone ({s['peak_snr']:.1f} sigma against the target pixel's "
                f"{s['target_snr']:.1f}). One sector is not enough: residual stellar variability "
                f"moves the peak around. Read the grid by eye.")
            _plot_safe(ctx, r, best, ra, dec, dur)
            return r
        r.flag(f"the flux lost in transit peaks {s['peak_dxdy'][0]:+d},{s['peak_dxdy'][1]:+d} "
               f"pixels from the target in s{s['sector']} ({s['peak_snr']:.1f} sigma against "
               f"the target pixel's {s['target_snr']:.1f}); {verdicts.count('off')} of "
               f"{len(det)} detected sector(s) peak off target. Look for a neighbour there; "
               f"pc_centroid decides.")
    _plot_safe(ctx, r, best, ra, dec, dur)
    return r


def _beta(t, y, dur):
    """Red-noise factor: scatter of residuals binned on a quarter-duration scale
    against what white noise would give.  Faint pixels carry correlated
    systematics, and a white-noise error made a corner pixel '38 sigma'."""
    w = max(dur / 4, 10 / 1440)
    k = np.floor((t - t.min()) / w).astype(int)
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=y) / cnt
    ok = cnt >= 3
    if ok.sum() < 5:
        return 1.0
    expected = np.std(y) / np.sqrt(np.median(cnt[ok]))
    return float(max(1.0, np.std(means[ok]) / expected)) if expected > 0 else 1.0


def _plot_safe(ctx, r, best, ra, dec, dur):
    try:
        gaia = catalogs.gaia_cone(ra, dec, 150.0)
    except Exception:  # noqa: BLE001
        gaia = []
    _plot(ctx, r, best, gaia, dur)


def _centroid_off(ctx):
    """pc_centroid's status when it says FAIL or FLAG for this candidate, else None."""
    import json
    path = ctx.out / ctx.target.slug / (
        f"centroid_c{ctx.candidate.index}.json" if ctx.candidate.index else "centroid.json")
    try:
        st = json.loads(path.read_text()).get("status")
    except (OSError, ValueError):
        return None
    return f"centroid {st}" if st in ("FAIL", "FLAG") else None


def _one_sector(path, ra, dec, period, t0, dur):
    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            hdr, d = hdul[1].header, hdul[1].data
            t = np.asarray(d["TIME"], float)
            flux = np.asarray(d["FLUX"], float)
            quality = np.asarray(d["QUALITY"], int)
            w = cutout_wcs(hdul)
    good = (quality == 0) & np.isfinite(t)
    t, flux = t[good], flux[good]
    if t.size < 20:
        return "fewer than 20 good cadences"
    tx, ty = (float(v) for v in w.all_world2pix(ra, dec, 0))
    ny, nx = flux.shape[1:]
    if not (0 <= tx < nx and 0 <= ty < ny):
        return "target outside this cutout"
    ph = ephem.phase(t, period, t0)
    win = np.abs(ph) < WINDOW_T14 * dur
    intr = np.abs(ph) < 0.5 * dur
    if intr.sum() < 10:
        return f"only {int(intr.sum())} in-transit cadences"
    # additive background first (as in pc_centroid), from the faintest quarter
    med = np.nanmedian(flux, axis=0)
    faint = med <= np.nanpercentile(med[np.isfinite(med)], 25)
    if faint.sum() >= 4:
        bg = np.array([np.nanmedian(f[faint]) for f in flux])
        flux = flux - np.where(np.isfinite(bg), bg, np.nanmedian(bg))[:, None, None]
        med = np.nanmedian(flux, axis=0)
    epoch = np.round((t - t0) / period).astype(int)
    ix, iy = int(round(tx)), int(round(ty))
    grid = {}
    for dy in range(-BOX, BOX + 1):
        for dx in range(-BOX, BOX + 1):
            x, y = ix + dx, iy + dy
            if not (0 <= x < nx and 0 <= y < ny):
                continue
            f = flux[:, y, x]
            rel = np.full(t.size, np.nan)
            # each transit normalised by a line through its own baseline
            for e in np.unique(epoch[win]):
                m = win & (epoch == e)
                b = m & (np.abs(ph) >= OOT_T14 * dur) & np.isfinite(f)
                if b.sum() < 4 or not (m & intr).any():
                    continue
                # quadratic: a young star's rotation curves the baseline
                # across a window several durations wide
                c = np.polyfit(ph[b], f[b], 2 if b.sum() >= 8 else 1)
                base = np.polyval(c, ph[m])
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel[m] = np.where(base > 0, f[m] / base - 1, np.nan)
            ok = np.isfinite(rel)
            i_in, i_out = ok & intr, ok & win & (np.abs(ph) >= OOT_T14 * dur)
            if i_in.sum() < 5 or i_out.sum() < 5:
                continue
            depth = -np.mean(rel[i_in])
            sig = np.std(rel[i_out]) / np.sqrt(i_in.sum()) * _beta(t[i_out], rel[i_out], dur)
            grid[(dx, dy)] = dict(ph=ph[ok], rel=rel[ok], depth=depth, sig=sig,
                                  deficit=depth * med[y, x], deficit_sig=sig * abs(med[y, x]),
                                  med=med[y, x])
    if (0, 0) not in grid:
        return "no measurable target pixel"
    snr = {k: g["deficit"] / g["deficit_sig"] if g["deficit_sig"] > 0 else 0 for k, g in grid.items()}
    peak = max(snr, key=snr.get)
    tgt = grid[(0, 0)]
    if snr[peak] < PIX_SNR:
        verdict = "none"
    elif max(abs(peak[0]), abs(peak[1])) <= 1:
        verdict = "on"
    else:
        g = grid[peak]
        diff = (g["deficit"] - tgt["deficit"]) / np.hypot(g["deficit_sig"], tgt["deficit_sig"])
        verdict = "off" if diff > OFF_SIGMA else "on"
    return dict(sector=cen._sector_of(path), grid=grid, wcs=w, tx=tx, ty=ty, ix=ix, iy=iy,
                n_in=int(intr.sum()), n_transits=int(len(np.unique(epoch[intr]))),
                peak_dxdy=[int(peak[0]), int(peak[1])], peak_snr=float(snr[peak]),
                target_snr=float(snr[(0, 0)]), verdict=verdict)


def _plot(ctx, r, s, gaia, dur):
    import matplotlib.pyplot as plt
    box = BOX if max(abs(v) for v in s["peak_dxdy"]) > PLOT_BOX and s["verdict"] != "none" else PLOT_BOX
    n = 2 * box + 1
    # panels stay the 7 x 7 size; the header and footer keep their height in inches
    w, h = 13 * (n + 0.3) / 7.3, 12.0 * n / 7 + 1.6
    # not plotting.figure(): its constrained layout would override the margins
    fig, axes = plt.subplots(n, n, figsize=(w, h), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.04, hspace=0.04, bottom=0.68 / h,
                                              top=1 - 0.95 / h, left=0.02, right=0.98))
    tgt = s["grid"][(0, 0)]
    scale = max(3 * abs(tgt["depth"]), 4 * tgt["sig"] * np.sqrt(10), 1e-4)
    meds = np.array([g["med"] for g in s["grid"].values()])
    lo, hi = np.log10(max(np.nanmin(meds[meds > 0]), 1e-3)), np.log10(np.nanmax(meds))
    hw = WINDOW_T14 * dur * 24
    bins = np.linspace(-hw, hw, 41)
    # neighbours, mapped into pixel offsets from the target pixel
    stars = {}
    for row in gaia or []:
        try:
            px, py = (float(v) for v in s["wcs"].all_world2pix(row["ra"], row["dec"], 0))
        except Exception:  # noqa: BLE001
            continue
        k = (int(round(px)) - s["ix"], int(round(py)) - s["iy"])
        g = row.get("phot_g_mean_mag")
        if g is not None and abs(k[0]) <= box and abs(k[1]) <= box:
            stars.setdefault(k, []).append(float(g))
    for j in range(n):            # rows: top of the figure is +y
        for i in range(n):
            dx, dy = i - box, box - j
            ax = axes[j][i]
            ax.set_xticks([])
            ax.set_yticks([])
            g = s["grid"].get((dx, dy))
            if g is None:
                ax.set_facecolor("0.85")
                continue
            shade = 0.97 - 0.25 * (np.log10(max(g["med"], 1e-3)) - lo) / max(hi - lo, 1e-6)
            ax.set_facecolor(str(np.clip(shade, 0.7, 0.98)))
            x, y = g["ph"] * 24, g["rel"]
            intr = np.abs(x) < 0.5 * dur * 24
            ax.plot(x[~intr], y[~intr], ".", color="0.3", ms=1, alpha=0.35)
            ax.plot(x[intr], y[intr], ".", color=TRANSIT_COLOUR, ms=1.5, alpha=0.6)
            idx = np.digitize(x, bins)
            bx = [x[idx == b].mean() for b in range(1, len(bins)) if (idx == b).sum() >= 3]
            by = [np.median(y[idx == b]) for b in range(1, len(bins)) if (idx == b).sum() >= 3]
            ax.plot(bx, by, "-", color="k", lw=1)
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_ylim(-2.2 * scale, 1.2 * scale)
            ax.set_xlim(-hw, hw)
            snr = g["deficit"] / g["deficit_sig"] if g["deficit_sig"] > 0 else 0
            ax.text(0.03, 0.04, f"{g['depth'] * 1e6:.0f} ppm\n{snr:.1f}σ", fontsize=6,
                    transform=ax.transAxes, va="bottom",
                    color=("#b3242b" if snr >= PIX_SNR else "0.3"))
            if (dx, dy) in stars:
                ax.text(0.97, 0.95, "★ G" + ",".join(f"{m:.1f}" for m in sorted(stars[(dx, dy)])[:2]),
                        fontsize=6, ha="right", va="top", transform=ax.transAxes, color="#1b3a6b")
            if (dx, dy) == (0, 0):
                for sp in ax.spines.values():
                    sp.set_color("#b3242b")
                    sp.set_linewidth(2.2)
            if tuple(s["peak_dxdy"]) == (dx, dy) and s["verdict"] != "none":
                for sp in ax.spines.values():
                    sp.set_linestyle("--")
                    sp.set_color("#1b3a6b" if (dx, dy) != (0, 0) else "#b3242b")
                    sp.set_linewidth(2.2)
    fig.text(0.5, 0.004,
             f"Sector {s['sector']}: every pixel's light curve, folded on the candidate "
             f"({s['n_transits']} transits), each normalised to its own baseline.  Red frame = target "
             f"pixel; dashed frame = pixel losing the most flux; orange = in transit; ★ = Gaia star "
             f"(G mag) in that pixel; darker panel = brighter pixel.  Same y-scale in every panel "
             f"(-{2.2 * scale * 1e3:.1f} to +{1.2 * scale * 1e3:.1f} ppt).  Each panel: depth and "
             f"significance of the flux lost.",
             ha="center", va="bottom", fontsize=8.5, wrap=True)
    annotate_status(fig, r)
    r.add_plot(save(fig, ctx.plot_path("pixel_lightcurves")))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(TEST, "Per-pixel light curves around the transit"))
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

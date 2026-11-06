"""Shared figure style and plotting helpers.

Plots are part of the output, not a debugging afterthought: every test that
can be checked visually writes a figure and records its path in the JSON.
The reason is specific -- difference imaging once returned a confident 3.6
sigma centroid offset from images that contained none of the transit signal,
and a human looking at the difference image would have seen that immediately.

Figures are written with a non-interactive backend and closed, so a batch of
30k targets does not accumulate open windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A muted, colour-blind-safe set; the transit model is always the same colour
# across every figure in this package so a reader can scan a report quickly.
C_DATA = "#4c4c4c"
C_BINNED = "#1b3a6b"
C_MODEL = "#c1440e"
C_GP = "#2e7d5b"
C_ALT = "#7b5aa6"
C_BAD = "#b3242b"
C_OK = "#2e7d5b"

RC = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
}

#: Standing preference for any corner plot this package ever grows:
#: ``fill_contours=True, plot_datapoints=False``.
CORNER_KWARGS = dict(fill_contours=True, plot_datapoints=False,
                     show_titles=True, title_fmt=".4f")


def figure(nrows=1, ncols=1, figsize=(8, 5), **kwargs):
    """A styled figure.  Always paired with :func:`save`."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 layout="constrained", **kwargs)
    return fig, axes


def save(fig, path, close: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig.savefig(path)
    if close:
        plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
# binning


def bin_series(x, y, bin_width, yerr=None, min_points: int = 1):
    """Bin ``y(x)`` into fixed-width bins.

    Returns ``(x_binned, y_binned, yerr_binned, n_per_bin)``.  The error is the
    standard error of the mean within the bin when there are enough points,
    falling back to the propagated ``yerr`` otherwise -- a bin of two points
    has no meaningful internal scatter and pretending otherwise produces
    wildly optimistic error bars at the edges of a fold.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    e = np.asarray(yerr, float)[good] if yerr is not None else None
    if x.size == 0 or not np.isfinite(bin_width) or bin_width <= 0:
        return (np.array([]),) * 3 + (np.array([], dtype=int),)

    idx = np.floor((x - x.min()) / bin_width).astype(int)
    xb, yb, eb, nb = [], [], [], []
    for i in np.unique(idx):
        sel = idx == i
        n = int(sel.sum())
        if n < min_points:
            continue
        xb.append(np.mean(x[sel]))
        yb.append(np.mean(y[sel]))
        if n >= 4:
            eb.append(np.std(y[sel], ddof=1) / np.sqrt(n))
        elif e is not None and np.isfinite(e[sel]).all():
            eb.append(np.sqrt(np.sum(e[sel] ** 2)) / n)
        else:
            eb.append(np.nan)
        nb.append(n)
    return (np.array(xb), np.array(yb), np.array(eb), np.array(nb, dtype=int))


# --------------------------------------------------------------------------
# reusable panels


def plot_fold(ax, phase_days, flux, duration_days=None, bin_minutes=None,
              model_phase=None, model_flux=None, label_data="data",
              ppm=True, title=None):
    """Phase-folded light curve with a binned overlay and an optional model.

    ``phase_days`` is signed phase in days (see :func:`petcare.ephem.phase`).
    Flux is plotted in ppm deviation from 1 by default, because every
    threshold in this package is quoted in ppm.
    """
    y = (np.asarray(flux, float) - 1.0) * (1e6 if ppm else 1.0)
    ax.plot(phase_days * 24, y, ".", ms=1.5, color=C_DATA, alpha=0.30,
            rasterized=True, label=label_data)

    if bin_minutes is None and duration_days:
        bin_minutes = max(duration_days * 24 * 60 / 12.0, 2.0)
    if bin_minutes:
        xb, yb, eb, _ = bin_series(phase_days * 24, y, bin_minutes / 60.0, min_points=3)
        if xb.size:
            ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=3.2, color=C_BINNED,
                        ecolor=C_BINNED, elinewidth=0.8, capsize=0,
                        label=f"binned ({bin_minutes:.0f} min)", zorder=3)

    if model_phase is not None and model_flux is not None:
        mp = np.asarray(model_phase, float)
        order = np.argsort(mp)
        mf = (np.asarray(model_flux, float) - 1.0) * (1e6 if ppm else 1.0)
        ax.plot(mp[order] * 24, mf[order], "-", color=C_MODEL, lw=1.6,
                label="model", zorder=4)

    if duration_days:
        for s in (-0.5, 0.5):
            ax.axvline(s * duration_days * 24, color=C_MODEL, ls=":", lw=0.8, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux - 1 [ppm]" if ppm else "normalised flux")
    if title:
        ax.set_title(title)
    return ax


def plot_series(ax, t, f, model=None, gp=None, mask=None, ppm=False,
                title=None, sector_boundaries=None):
    """Raw time series with an optional GP and transit model drawn over it.

    This panel exists so a human can confirm the GP is tracking the star's
    variability and *not* eating the transit -- the single most common way a
    GP+transit fit goes wrong on these targets.
    """
    t = np.asarray(t, float)
    y = (np.asarray(f, float) - 1.0) * 1e6 if ppm else np.asarray(f, float)
    ax.plot(t, y, ".", ms=1.2, color=C_DATA, alpha=0.35, rasterized=True, label="data")

    # Break the model lines at observing gaps.  Drawn straight through, a
    # two-year gap becomes a diagonal line across the whole panel and the
    # variability model inside each sector is invisible behind it -- which
    # defeats the one thing this panel exists to show.
    def _gapped(values):
        v = np.array(values, float)
        if t.size > 2:
            d = np.diff(t)
            step = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0
            if step > 0:
                v[:-1][d > max(50 * step, 0.5)] = np.nan
        return v

    if gp is not None:
        g = (np.asarray(gp, float) - 1.0) * 1e6 if ppm else np.asarray(gp, float)
        ax.plot(t, _gapped(g), "-", color=C_GP, lw=1.0, alpha=0.9,
                label="GP (variability)")
    if model is not None:
        m = (np.asarray(model, float) - 1.0) * 1e6 if ppm else np.asarray(model, float)
        ax.plot(t, _gapped(m), "-", color=C_MODEL, lw=1.0, alpha=0.9,
                label="GP + transit")

    # ...and scale to the data, not to whatever an outlier reached.  On a
    # Tmag 14 target a handful of bad cadences run to flux -0.2 and squash
    # every real feature into a line.
    finite = np.isfinite(y)
    if finite.sum() > 20:
        lo, hi = np.percentile(y[finite], [0.2, 99.8])
        pad = 0.15 * (hi - lo) or 1.0
        ax.set_ylim(lo - pad, hi + pad)
    if mask is not None and np.any(mask):
        ax.plot(np.asarray(t)[mask], y[mask], ".", ms=2.0, color=C_MODEL,
                alpha=0.8, label="in transit")
    for b in (sector_boundaries or []):
        ax.axvline(b, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.set_xlabel("BTJD")
    ax.set_ylabel("flux - 1 [ppm]" if ppm else "normalised flux")
    if title:
        ax.set_title(title)
    return ax


def annotate_status(fig, result) -> None:
    """Stamp the status, reason and noise level on a figure.

    A figure that leaves the pipeline without its verdict attached gets
    misread later; this keeps the two together in one file.
    """
    colour = {"PASS": C_OK, "FLAG": "#b8860b", "FAIL": C_BAD,
              "INCONCLUSIVE": "#666666", "ERROR": C_BAD}.get(result.status, "#666666")
    noise = result.noise.get("white_ppm")
    rn = result.noise.get("red_noise_factor")
    bits = [f"{result.status}: {result.reason}"]
    if isinstance(noise, (int, float)):
        n = f"noise {noise:.0f} ppm"
        if isinstance(rn, (int, float)):
            n += f" x{rn:.1f} red"
        bits.append(n)
    # Wrap: a one-line suptitle with a full reason in it runs off both edges
    # of the figure, which is how several plots shipped with their verdict
    # unreadable.
    import textwrap
    width = max(int(fig.get_figwidth() * 16), 60)
    body = "\n".join(textwrap.wrap("   |   ".join(bits), width=width,
                                   max_lines=3, placeholder=" ..."))
    fig.suptitle(f"{result.target.name}  |  {result.test}\n{body}",
                 fontsize=9, color=colour)

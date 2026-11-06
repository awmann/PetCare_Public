"""Difference imaging for TESS cutouts: per-pixel local baselines and a PRF scene.

The engine behind ``pc_centroid`` (v0.2).  Two things the v0.1 test got wrong
on young stars, and what replaces them:

* **The star's variability.**  v0.1 divided every pixel by a 1-day running
  median of the *whole cutout's* summed flux.  On one star (2.4% spot
  amplitude, 0.71 d period, 2 h transit) that left the cutout's own light
  curve showing the transit at only 5-11 sigma, and the difference image was
  mostly variability.  Here every pixel gets its own quadratic baseline across
  every transit window, fitted to that window's out-of-transit cadences, which
  is what ``pc_pixels`` and ``pc_uniqueness`` already do.  Depth-image SNR on
  the same star: 24-28.

* **The centroid estimator.**  v0.1 took a flux-weighted centroid of the
  difference image inside a 3-pixel circle centred on the target, after
  clipping negative pixels to zero.  That pulls every answer toward the circle's
  centre, i.e. toward "on target", and threw away the PSF shape.  Here a PRF
  model of every catalogued star near the target is registered to the
  out-of-transit image, and the depth image is then (a) fitted with one PRF at
  a free position, measured against the *target's* registered position rather
  than against the blended light centroid -- which is what produced a false
  1.6 arcsec offset on DS Tuc A, 5 arcsec from DS Tuc B -- and (b) fitted
  with a PRF at each star's position in turn, so the chi-square says which
  star the dimming belongs to.

Noise comes from the data, not from a formula: the same depth image is built
at a set of *null phases* where nothing transits, and the pixel-by-pixel
scatter across those nulls is the noise, red noise and residual variability
included.  Position errors come from injecting the best-fit signal into each
null image and refitting.  That works with three transits as well as with
eighty, where a bootstrap over epochs does not (with three epochs it reported
DS Tuc A b to 0.12 arcsec).

The PRF is an elliptical Moffat integrated over the pixel, fitted to each
sector's own out-of-transit image.  It is not the TESS PRF, and the
calibration's systematic floor (``pc_centroid.SYS_FLOOR_PIX``) is what that
costs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: Transit core used for the depth: inside +/- this many T14 of mid-transit.
#: Excludes most of ingress and egress, where the depth is partial.
IN_HALF = 0.45
#: Out-of-transit baseline window, in T14 from mid-transit.
OOT_INNER, OOT_OUTER = 0.75, 2.5
#: ...shrunk toward this many T14 when the star rotates fast: the window may
#: not exceed a fifth of the rotation period, or a quadratic cannot follow the
#: spots (HIP 67522, Prot 1.4 d: 4.7 sigma at 2.5 T14, 8.8 at 1.5).
OOT_OUTER_MIN = 1.5
#: Minimum out-of-transit baseline half-width in days, so 30-minute FFIs and
#: short transits still get enough baseline cadences on each side.
OOT_MIN_DAYS = 0.15
#: Detector saturation in e-/s per pixel, roughly.  A saturated star bleeds
#: along its column and no PRF describes it.
SATURATION_E_S = 8.0e4
#: Radius of the scene and of the depth-image fits, in pixels.
SCENE_RADIUS, FIT_RADIUS = 4.5, 3.5
#: Neighbours putting more than this fraction of the target's flux into the
#: fit region get their own free flux in the out-of-transit scene fit (at
#: most MAX_FREE of them); the rest ride on the target's at their catalogue
#: ratio.
FREE_FLUX_RATIO = 0.05
MAX_FREE = 8
#: Half-width of the box the free-position fit may search, px.
FREE_BOX_PX = 3.0
#: Prior width on the registration shift, px.
SHIFT_PRIOR_PX = 0.2
#: Largest believable registration shift between the WCS and the image, px.
MAX_SHIFT_PX = 0.75
#: Null phases per sector.
N_NULL = 24


@dataclass
class Star:
    id: str
    dra: float            # arcsec east of the target
    ddec: float           # arcsec north
    tmag: float
    ratio: float          # flux relative to the target, TESS band
    is_target: bool = False
    viable: bool = True   # pc_neighbours: could it produce this depth at all?
    sep: float = 0.0
    ra: float | None = None   # the target's own position at the scene's epoch
    dec: float | None = None


@dataclass
class SectorResult:
    sector: int | None
    ok: bool
    why: str = ""
    n_transits: int = 0
    n_null: int = 0
    metrics: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# the stars in the scene


def stars_from_gaia(ra, dec, epoch_jyear, depth_ppm=None, radius=150.0, neighbours=None):
    """Scene stars from a Gaia DR3 cone, propagated to ``epoch_jyear``.

    The scene must be complete: pc_neighbours keeps only the neighbours it
    judges relevant, and a G = 11.7 star 3.6 px from one target that it
    left out dragged the registration a pixel and a half.  So every star comes
    from the cone, and ``neighbours`` (pc_neighbours' metrics, when it ran)
    only supplies which of them could produce the depth and the target's id.

    Without pc_neighbours, the target is the brightest source within 2 arcsec,
    and a neighbour is viable when a total eclipse of it could produce
    ``depth_ppm`` (no aperture weighting -- generous on purpose).
    """
    from . import catalogs
    rows = catalogs.gaia_cone(ra, dec, radius)
    nb = {}
    tgt_id = None
    if neighbours:
        tgt_id = str(neighbours.get("target_gaia_dr3") or "") or None
        for n in neighbours.get("neighbours") or []:
            if n.get("gaia_dr3") is not None:
                nb[str(n["gaia_dr3"])] = n
    pts = []
    for r in rows:
        g = r.get("phot_g_mean_mag")
        if g is None or not np.isfinite(g):
            continue
        r_ra, r_dec = catalogs.propagate(r["ra"], r["dec"], r.get("pmra"), r.get("pmdec"),
                                         to_epoch=epoch_jyear)
        sep, dra, ddec = catalogs.separation_arcsec(ra, dec, r_ra, r_dec)
        # the input position may be a catalogue position at another epoch
        # (TIC positions are J2000; a nearby young star moves 4 arcsec since then)
        sep_any = min(float(sep), *(float(catalogs.separation_arcsec(
            ra, dec, *catalogs.propagate(r["ra"], r["dec"], r.get("pmra"), r.get("pmdec"),
                                         to_epoch=ep))[0]) for ep in (2000.0, 2016.0)))
        sid = str(r.get("source_id"))
        tm = nb[sid]["tmag"] if sid in nb and nb[sid].get("tmag") is not None else \
            catalogs.gaia_g_to_tmag(g, r.get("bp_rp"))
        pts.append((sep_any, float(dra), float(ddec), float(tm), sid, r_ra, r_dec))
    tgt = next((p for p in pts if p[4] == tgt_id), None)
    if tgt is None:
        close = [p for p in pts if p[0] < 2.0]
        if not close:
            return None
        tgt = min(close, key=lambda p: p[3])
    stars = [Star(id="target", dra=0.0, ddec=0.0, tmag=tgt[3], ratio=1.0, is_target=True,
                  ra=tgt[5], dec=tgt[6])]
    for sep, dra, ddec, tm, sid, _, _ in pts:
        if sid == tgt[4]:
            continue
        ratio = 10 ** (-0.4 * (tm - tgt[3]))
        sep_t = float(np.hypot(dra - tgt[1], ddec - tgt[2]))
        if not _in_scene(sep_t, ratio):
            continue
        if sid in nb:
            req = nb[sid].get("required_eclipse_depth")
            viable = nb[sid].get("verdict") != "excluded" and req is not None and 0 < req <= 1
        else:
            viable = bool(depth_ppm) and depth_ppm * 1e-6 * (1 + ratio) / ratio <= 1
        stars.append(Star(id=sid, dra=dra - tgt[1], ddec=ddec - tgt[2], tmag=tm, ratio=ratio,
                          viable=viable, sep=sep_t))
    return stars


def _in_scene(sep, ratio):
    # bright stars well outside the fit still shape the registration region
    return (sep <= 130 and ratio >= (0.0005 if sep < 30 else 0.002)) or (sep <= 150 and ratio >= 0.05)


# --------------------------------------------------------------------------
# PRF

#: SPOC PRF models on MAST; sectors 1-3 have their own set.
PRF_URL = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles"
#: Gaussian blur (pixels) tried on top of the SPOC PRF; the best is chosen by
#: the out-of-transit fit.  The commissioning PRFs are sharper than most
#: sectors' real images, and jitter and focus vary.
BLURS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8)
_PRF_FILES: dict = {}


def _prf_files(cam, ccd, sector):
    """The 25 SPOC PRF images for one CCD, downloaded once into the cache."""
    import requests
    from astropy.io import fits
    from .cache import cache_dir, with_retries
    epoch = "start_s0001" if sector is not None and sector < 4 else "start_s0004"
    key = (epoch, int(cam), int(ccd))
    if key in _PRF_FILES:
        return _PRF_FILES[key]
    d = cache_dir("tess_prf") / epoch / f"cam{cam}_ccd{ccd}"
    d.mkdir(parents=True, exist_ok=True)
    have = sorted(d.glob("*.fits"))
    if len(have) < 25:
        url = f"{PRF_URL}/{epoch}/cam{cam}_ccd{ccd}/"
        html = with_retries(lambda: requests.get(url, timeout=60).text, what="TESS PRF listing")
        import re
        names = sorted(set(n for n in re.findall(r'href="([^"]+\.fits)"', html) if "phot" not in n))
        for n in names:
            f = d / n
            if not f.exists():
                r = with_retries(lambda: requests.get(url + n, timeout=120), what=f"PRF {n}")
                r.raise_for_status()
                tmp = f.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(f)
        have = sorted(d.glob("*.fits"))
    out = []
    for f in have:
        row = int(f.name.split("-row")[1][:4])
        col = int(f.name.split("-col")[1][:4])
        out.append((row, col, np.asarray(fits.getdata(f), float)))
    _PRF_FILES[key] = out
    return out


class TessPRF:
    """The SPOC pixel-integrated PRF at one place on the detector.

    Each MAST file holds the response of a 13x13 pixel stamp to a star at 9x9
    sub-pixel offsets, interleaved into a 117x117 array.  Read with the
    sub-pixel index reversed, that array is the pixel response as a function
    of (pixel centre - star position) on a 1/9-pixel grid -- an effective PRF
    -- so a model pixel is one interpolation, with no sub-sampling.
    """

    def __init__(self, cam, ccd, sector, ccd_row, ccd_col):
        files = _prf_files(cam, ccd, sector)
        rows = np.array([f[0] for f in files])
        cols = np.array([f[1] for f in files])
        # bilinear in detector position between the four surrounding models
        ur, uc = np.unique(rows), np.unique(cols)
        r0 = ur[max(0, np.searchsorted(ur, ccd_row) - 1)]
        r1 = ur[min(len(ur) - 1, np.searchsorted(ur, ccd_row))]
        c0 = uc[max(0, np.searchsorted(uc, ccd_col) - 1)]
        c1 = uc[min(len(uc) - 1, np.searchsorted(uc, ccd_col))]
        fr = 0.0 if r1 == r0 else np.clip((ccd_row - r0) / (r1 - r0), 0, 1)
        fc = 0.0 if c1 == c0 else np.clip((ccd_col - c0) / (c1 - c0), 0, 1)

        def get(r, c):
            return next(f[2] for f in files if f[0] == r and f[1] == c)

        prf = ((1 - fr) * (1 - fc) * get(r0, c0) + (1 - fr) * fc * get(r0, c1)
               + fr * (1 - fc) * get(r1, c0) + fr * fc * get(r1, c1))
        self.base = prf / prf[4::9, 4::9].sum()
        self.set_blur(0.0)

    def set_blur(self, blur_px):
        from scipy.ndimage import gaussian_filter, spline_filter
        p = gaussian_filter(self.base, blur_px * 9, mode="constant") if blur_px > 0 else self.base
        p = p / p[4::9, 4::9].sum()
        self.blur = float(blur_px)
        self._coef = spline_filter(p, order=3)
        self._n = p.shape[0]

    def __call__(self, xs, ys, x0, y0):
        from scipy.ndimage import map_coordinates
        c = (self._n - 1) / 2
        u = c + 9 * (np.asarray(ys, float) - y0)
        v = c + 9 * (np.asarray(xs, float) - x0)
        return map_coordinates(self._coef, [u.ravel(), v.ravel()], order=3,
                               prefilter=False, mode="constant", cval=0.0).reshape(np.shape(xs))


class Scene:
    """Catalogued stars at pixel positions, a PRF, and a registration shift."""

    def __init__(self, stars, wcs, ra, dec, prf):
        self.stars = stars
        self.prf_model = prf
        xy = []
        for s in stars:
            r = ra + s.dra / 3600 / np.cos(np.radians(dec))
            d = dec + s.ddec / 3600
            x, y = wcs.all_world2pix(r, d, 0)
            xy.append((float(x), float(y)))
        self.xy0 = np.array(xy)
        self.shift = np.zeros(2)
        self.it = [i for i, s in enumerate(stars) if s.is_target][0]

    @property
    def xy(self):
        return self.xy0 + self.shift

    def prf(self, xs, ys, x, y):
        return self.prf_model(xs, ys, x, y)

    def fit_oot(self, O, sigma, region):
        """Register the scene to the out-of-transit image.

        Free: a global shift, the PRF blur (from BLURS), the target's flux,
        the flux of every neighbour brighter than FREE_FLUX_RATIO that is at
        least a pixel away, and a background plane.  Closer or fainter stars
        follow the target at their catalogue flux ratio: a free flux for a
        star a quarter-pixel away trades off against the shift and moves the
        target's registered position (DS Tuc B, 5 arcsec from DS Tuc A).
        """
        from scipy.optimize import least_squares
        ys, xs = np.indices(O.shape)
        xs, ys = xs[region], ys[region]
        d, w = O[region], 1 / sigma[region]
        it = self.it
        sep_px = np.hypot(*(self.xy0 - self.xy0[it]).T)
        # a free flux only for a neighbour that puts real light into the
        # region, at most MAX_FREE of them: a faint target in a crowded field
        # otherwise had 77 free fluxes for 64 pixels, fitted everything
        # exactly and left the shift unconstrained
        inreg = np.array([s.ratio * self.prf(xs, ys, *self.xy0[i]).sum()
                          for i, s in enumerate(self.stars)])
        cand = [i for i, s in enumerate(self.stars)
                if not s.is_target and inreg[i] >= FREE_FLUX_RATIO and sep_px[i] >= 1.0]
        free = sorted(cand, key=lambda i: -inreg[i])[:MAX_FREE]
        tied = [i for i, s in enumerate(self.stars) if not s.is_target and i not in free]
        cx, cy = self.xy0[it]

        def design(p):
            pos = self.xy0 + p
            col_t = self.prf(xs, ys, *pos[it])
            for i in tied:
                col_t = col_t + self.stars[i].ratio * self.prf(xs, ys, *pos[i])
            cols = [col_t] + [self.prf(xs, ys, *pos[i]) for i in free]
            cols += [np.ones_like(d, float), xs - cx, ys - cy]
            return np.array(cols).T

        def resid(p):
            A = design(p)
            coef, *_ = np.linalg.lstsq(A * w[:, None], d * w, rcond=None)
            # a Gaussian prior on the shift: the TESScut WCS is good to a
            # couple of tenths of a pixel, and a faint target in a crowded
            # field otherwise registers onto whatever is brightest nearby
            return np.concatenate([(A @ coef - d) * w, np.asarray(p) / SHIFT_PRIOR_PX])

        best = None
        for blur in BLURS:
            self.prf_model.set_blur(blur)
            sol = least_squares(resid, [0.0, 0.0], bounds=([-1.5, -1.5], [1.5, 1.5]),
                                x_scale=[0.1, 0.1])
            c2 = float(np.sum(sol.fun[:-2] ** 2))
            if best is None or c2 < best[0]:
                best = (c2, blur, sol)
        c2, blur, sol = best
        self.prf_model.set_blur(blur)
        A = design(sol.x)
        coef, *_ = np.linalg.lstsq(A * w[:, None], d * w, rcond=None)
        self.shift = np.array(sol.x)
        flux = np.zeros(len(self.stars))
        flux[it] = coef[0]
        for k, i in enumerate(free):
            flux[i] = coef[1 + k]
        for i in tied:
            flux[i] = coef[0] * self.stars[i].ratio
        self.flux = flux
        return {"shift_px": self.shift.tolist(), "blur_px": blur,
                "target_flux_e_s": float(flux[it]), "n_free_neighbours": len(free),
                "oot_rchi2": c2 / max(1, d.size - 2 - A.shape[1])}


# --------------------------------------------------------------------------
# depth images


def oot_outer_days(dur, prot=None):
    """Half-width of the baseline window, in days."""
    outer = OOT_OUTER * dur
    if prot and np.isfinite(prot) and prot > 0:
        outer = min(outer, max(OOT_OUTER_MIN * dur, prot / 5))
    return max(outer, 0.5 * dur + OOT_MIN_DAYS)


def _windows(t, period, t0, dur, exclude, prot=None):
    """Per-epoch index arrays for the in-transit core and the baseline."""
    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    cyc = np.round((t - t0 - ph) / period).astype(int)
    outer = oot_outer_days(dur, prot)
    outer = min(outer, 0.45 * period)
    inner = min(OOT_INNER * dur, 0.9 * outer)
    intr = (np.abs(ph) < IN_HALF * dur) & ~exclude
    oot = (np.abs(ph) >= inner) & (np.abs(ph) < outer) & ~exclude
    out = []
    for c in np.unique(cyc[intr]):
        i = np.where(intr & (cyc == c))[0]
        o = np.where(oot & (cyc == c))[0]
        nl, nr = (ph[o] < 0).sum(), (ph[o] > 0).sum()
        if i.size < 2 or nl < 2 or nr < 2 or o.size < 5:
            continue
        out.append((i, o))
    return out


def depth_stack(t, F, period, t0, dur, exclude, prot=None):
    """Per-epoch depth images (baseline minus in-transit, e-/s; positive = dimming).

    Each pixel's baseline is a quadratic in time across that epoch's window
    (linear when the window is thin), so the star's own variability is taken
    out pixel by pixel rather than by one scale factor for the whole cutout.
    Returns (E, var, B): depth images, their per-epoch white-noise variance,
    and the baseline images.
    """
    ny, nx = F.shape[1:]
    flat = F.reshape(F.shape[0], -1)
    good_pix = np.all(np.isfinite(flat), axis=0)
    E, V, B = [], [], []
    for i, o in _windows(t, period, t0, dur, exclude, prot):
        tm = t[i].mean()
        to, ti = t[o] - tm, t[i] - tm
        deg = 2 if o.size >= 8 else 1
        X = np.vander(to, deg + 1, increasing=True)
        Xi = np.vander(ti, deg + 1, increasing=True)
        Y = flat[o][:, good_pix]
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ coef
        dof = max(1, o.size - deg - 1)
        s2 = (res ** 2).sum(0) / dof
        C = np.linalg.inv(X.T @ X)
        m = Xi.mean(0)
        lev = float(m @ C @ m)
        pred = (Xi @ coef).mean(0)
        e = np.full(flat.shape[1], np.nan)
        v = np.full(flat.shape[1], np.nan)
        b = np.full(flat.shape[1], np.nan)
        e[good_pix] = pred - flat[i][:, good_pix].mean(0)
        v[good_pix] = s2 * (1 / i.size + lev)
        b[good_pix] = coef[0]
        E.append(e.reshape(ny, nx))
        V.append(v.reshape(ny, nx))
        B.append(b.reshape(ny, nx))
    return np.array(E), np.array(V), np.array(B)


def combine(E, V):
    """Inverse-variance mean over epochs, one weight per epoch (keeps the PRF shape)."""
    wk = 1 / np.nanmedian(V.reshape(len(V), -1), axis=1)
    wk = np.where(np.isfinite(wk), wk, 0)
    D = np.tensordot(wk, E, axes=1) / wk.sum()
    return D


# --------------------------------------------------------------------------
# fits to the depth image


def _lsq_dimming(A, d, w):
    """Least squares for [PRF amplitude, constant] with the amplitude held >= 0.

    The depth image is positive where flux is lost.  An unconstrained fit is as
    happy with a negative PRF -- a star *brightening* in transit -- and on one
    candidate the free fit locked onto one 66 arcsec away (snr_free -7 to -17 in
    every sector) while the target's own pixels held the dip at 3-10 sigma: v0.2
    reported a 152-sigma offset and failed a candidate that is on target.  A
    brightening cannot be the source of a dip, so it scores as no source at all.
    """
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, d * w, rcond=None)
    if coef[0] < 0:
        c = float(np.sum(d * w * w) / np.sum(w * w))
        coef = np.array([0.0, c])
    return coef


def _fit_free(D, sig, region, scene, start, dimming=True):
    """One PRF at a free position, plus a constant; with ``dimming`` the PRF may
    only dim (_lsq_dimming).  Returns (x, y, amp, chi2)."""
    from scipy.optimize import least_squares
    ys, xs = np.indices(D.shape)
    xs, ys = xs[region], ys[region]
    d, w = D[region], 1 / sig[region]

    def solve(A):
        if dimming:
            return _lsq_dimming(A, d, w)
        return np.linalg.lstsq(A * w[:, None], d * w, rcond=None)[0]

    def resid(p):
        x, y = p
        A = np.array([scene.prf(xs, ys, x, y), np.ones_like(d)]).T
        return (A @ solve(A) - d) * w

    best = None
    # a coarse start grid keeps a noisy image from trapping the fit
    for sx, sy in [start] + [(start[0] + a, start[1] + b)
                             for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]:
        sol = least_squares(resid, [sx, sy], bounds=([start[0] - FREE_BOX_PX, start[1] - FREE_BOX_PX],
                                                     [start[0] + FREE_BOX_PX, start[1] + FREE_BOX_PX]))
        c2 = float(np.sum(sol.fun ** 2))
        if best is None or c2 < best[1]:
            best = (sol, c2)
    sol, c2 = best
    A = np.array([scene.prf(xs, ys, *sol.x), np.ones_like(d)]).T
    coef = solve(A)
    return float(sol.x[0]), float(sol.x[1]), float(coef[0]), c2


def _fit_fixed(D, sig, region, scene, x, y, dimming=False):
    """Amplitude of a PRF at a fixed position, plus a constant; returns (amp, amp_err, chi2).

    ``dimming``: hold the amplitude >= 0 (see _lsq_dimming), for the question
    "is the dip on this star?"; the target's own signed amplitude, used for its
    significance against the nulls, is left free.  The returned amplitude is
    the unconstrained one either way, so a brightening still shows as negative.
    """
    ys, xs = np.indices(D.shape)
    xs, ys = xs[region], ys[region]
    d, w = D[region], 1 / sig[region]
    A = np.array([scene.prf(xs, ys, x, y), np.ones_like(d)]).T
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, d * w, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    c_fit = _lsq_dimming(A, d, w) if dimming else coef
    chi2 = float(np.sum((Aw @ c_fit - d * w) ** 2))
    return float(coef[0]), float(np.sqrt(cov[0, 0])), chi2


def _null_offsets(period, dur, n, prot=None):
    """Phase offsets (days) for the null images: clear of the transit and of phase 0.5."""
    # the real transit's cadences are masked out of every null, so a null's
    # window may overlap it; only its in-transit core must stay clear
    lo = (IN_HALF + 0.6) * dur / period + 0.01
    sec = (IN_HALF + 0.5) * dur / period + 0.01
    cand = np.linspace(0, 1, 4 * n, endpoint=False)
    ok = (cand > lo) & (cand < 1 - lo) & (np.abs(cand - 0.5) > sec)
    cand = cand[ok]
    if cand.size == 0:
        return np.array([])
    pick = cand[np.linspace(0, cand.size - 1, min(n, cand.size)).astype(int)]
    return np.unique(pick) * period


# --------------------------------------------------------------------------
# one sector


def analyse_sector(path, ra, dec, period, t0, dur, stars, exclude_fn=None, n_null=N_NULL,
                   sector=None, prot=None, inject=None):
    """Depth image, scene registration, free-position and per-star fits for one cutout.

    ``inject(t, F, scene)``, if given, returns the background-subtracted flux
    cube with a synthetic signal added; it is for injection-recovery tests and
    sees the scene at its catalogue positions, before registration.
    """
    from astropy.io import fits
    from .tesscut import cutout_camera_ccd, cutout_ccd_origin, cutout_wcs
    t0_star = next(s for s in stars if s.is_target)
    if t0_star.ra is not None:
        ra, dec = t0_star.ra, t0_star.dec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as h:
            d = h[1].data
            t = np.asarray(d["TIME"], float)
            F = np.asarray(d["FLUX"], float)
            q = np.asarray(d["QUALITY"], int)
            wcs = cutout_wcs(h)
            cam, ccd = cutout_camera_ccd(h)
            sector = sector or h[0].header.get("SECTOR")
            origin = cutout_ccd_origin(h)
    good = (q == 0) & np.isfinite(t) & np.isfinite(F).any(axis=(1, 2))
    t, F = t[good], F[good]
    if t.size < 50:
        return SectorResult(sector, False, "fewer than 50 good cadences")
    # Off-detector pixels can be exact zeros instead of NaN, which defeats the edge guard
    # below and leaves a singular fit (LinAlgError).
    F = np.where((F == 0).all(axis=0), np.nan, F)

    med = np.nanmedian(F, axis=0)
    if np.isfinite(med).mean() < 0.5:
        return SectorResult(sector, False, "more than half the cutout is NaN (edge of detector)")
    faint = med <= np.nanpercentile(med[np.isfinite(med)], 25)
    bg = np.nanmedian(F[:, faint], axis=1)
    F = F - np.where(np.isfinite(bg), bg, np.nanmedian(bg))[:, None, None]

    tx, ty = (float(v) for v in wcs.all_world2pix(ra, dec, 0))
    if None in (cam, ccd):
        return SectorResult(sector, False, "cutout header has no camera/CCD for the PRF")
    # astrocut >= 1.4 no longer records where the cutout sits on the CCD; the
    # PRF then comes from the CCD's centre, and the fitted blur absorbs most
    # of the difference
    ccd_row, ccd_col = ((origin[0] + ty, origin[1] + tx) if origin else (1024.0, 1068.0))
    prf = TessPRF(int(cam), int(ccd), sector, ccd_row, ccd_col)
    scene = Scene(stars, wcs, ra, dec, prf)
    it = scene.it
    ny, nx = med.shape
    if not (1 <= tx < nx - 1 and 1 <= ty < ny - 1):
        return SectorResult(sector, False, "target is at or beyond the cutout edge")

    if inject is not None:
        F = inject(t, F, scene)

    ys, xs = np.indices(med.shape)
    rr = np.hypot(xs - tx, ys - ty)
    peak = float(np.nanmax(np.where(rr <= 2, F.mean(0), np.nan)))
    if peak > SATURATION_E_S:
        return SectorResult(sector, False, f"the target saturates ({peak:.0f} e-/s in its "
                                           f"brightest pixel); no PRF describes a bleed column",
                            metrics={"peak_e_s": peak})

    exclude = exclude_fn(t) if exclude_fn else np.zeros(t.size, bool)
    E, V, B = depth_stack(t, F, period, t0, dur, exclude, prot)
    n = len(E)
    if n < 1:
        return SectorResult(sector, False, "no transit with baseline on both sides",
                            n_transits=n)
    D = combine(E, V)
    O = np.nanmean(B, axis=0)

    # ---- nulls: the same construction where nothing transits -------------
    in_any = np.abs((t - t0 + 0.5 * period) % period - 0.5 * period) < 0.6 * dur
    nulls = []
    for off in _null_offsets(period, dur, n_null):
        En, Vn, _ = depth_stack(t, F, period, t0 + off, dur, exclude | in_any, prot)
        if len(En) >= max(1, n // 2):
            nulls.append(combine(En, Vn))
    nulls = np.array(nulls)
    if len(nulls) < 6:
        return SectorResult(sector, False, f"only {len(nulls)} null phase(s) usable; the noise "
                                           f"cannot be measured", n_transits=n)
    sig = 1.4826 * np.nanmedian(np.abs(nulls - np.nanmedian(nulls, 0)), axis=0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmax(sig))

    # ---- register the scene on the out-of-transit image ------------------
    scene_reg = (rr <= SCENE_RADIUS) & np.isfinite(O)
    o_sig = np.sqrt(np.clip(np.abs(O), 1, None) + np.nanmedian(sig) ** 2) + 0.01 * np.abs(O)
    reg = scene.fit_oot(O, o_sig, scene_reg)
    if np.max(np.abs(scene.shift)) > MAX_SHIFT_PX:
        return SectorResult(sector, False, f"the catalogue does not register on this image "
                                           f"(shift {scene.shift[0]:+.2f}, {scene.shift[1]:+.2f} px)",
                            n_transits=n, metrics={"registration": reg})
    txr, tyr = scene.xy[it]

    region = (np.hypot(xs - txr, ys - tyr) <= FIT_RADIUS) & np.isfinite(D) & np.isfinite(sig)

    # ---- does the depth image contain the signal? ------------------------
    a_t, a_t_err, chi2_t = _fit_fixed(D, sig, region, scene, txr, tyr)
    a_null = np.array([_fit_fixed(Nn, sig, region, scene, txr, tyr)[0] for Nn in nulls])
    snr = float(a_t / (1.4826 * np.median(np.abs(a_null - np.median(a_null)))))

    # ---- free-position fit, errors by injection into the nulls -----------
    fx, fy, fa, fchi2 = _fit_free(D, sig, region, scene, (txr, tyr))
    # a free fit pinned at the edge of its box has not found a centre: the
    # signal lies beyond the fit region, or the image holds no PRF-like dip
    at_bound = bool(max(abs(fx - txr), abs(fy - tyr)) > FREE_BOX_PX - 0.05)
    inj, fa_null = [], []
    model = fa * scene.prf(xs, ys, fx, fy)
    for Nn in nulls:
        x2, y2, _, _ = _fit_free(Nn + model, sig, region, scene, (fx, fy))
        inj.append((x2, y2))
        # unconstrained on the nulls: held >= 0 most nulls return exactly 0
        # and their MAD collapses (a known planet: 7.7 sigma against a
        # target-pixel 3.2).  The scatter of the free fit is what is wanted;
        # a real dip gives the same amplitude either way.
        fa_null.append(_fit_free(Nn, sig, region, scene, (txr, tyr), dimming=False)[2])
    # the free fit finds the brightest dip anywhere in the region, so its
    # significance is judged against free fits to the nulls (look-elsewhere
    # included); it is what detects a signal centred on a neighbour, where
    # the amplitude at the target's position is near zero
    fa_null = np.array(fa_null)
    snr_free = float((fa - np.median(fa_null))
                     / (1.4826 * np.median(np.abs(fa_null - np.median(fa_null)))))
    inj = np.array(inj)
    cov_px = np.cov(inj.T)
    off_px = np.array([fx - txr, fy - tyr])

    # pixel -> sky through the local WCS jacobian
    r0 = np.array(wcs.all_pix2world(txr, tyr, 0), float)
    J = np.zeros((2, 2))
    for k, (ddx, ddy) in enumerate(((1e-2, 0), (0, 1e-2))):
        r1 = np.array(wcs.all_pix2world(txr + ddx, tyr + ddy, 0), float)
        J[0, k] = (r1[0] - r0[0]) * np.cos(np.radians(dec)) * 3600 / 1e-2
        J[1, k] = (r1[1] - r0[1]) * 3600 / 1e-2
    off_sky = J @ off_px
    cov_sky = J @ cov_px @ J.T

    # ---- which star? one PRF at each catalogued position ------------------
    hyp = []
    for i, s in enumerate(stars):
        x, y = scene.xy[i]
        if np.hypot(x - txr, y - tyr) > FIT_RADIUS:
            continue
        a, ae, c2 = _fit_fixed(D, sig, region, scene, x, y, dimming=True)
        fl = scene.flux[i]
        hyp.append({"id": s.id, "is_target": s.is_target, "sep_arcsec": s.sep, "tmag": s.tmag,
                    "viable": s.viable, "chi2": c2, "amp_e_s": a, "amp_err": ae,
                    "implied_depth": (a / fl) if fl > 0 else None})
    npix = int(region.sum())
    best = min(hyp, key=lambda h: h["chi2"])
    rchi2 = best["chi2"] / max(1, npix - 2)

    # ---- look-elsewhere: what margin does a neighbour win by chance? ------
    # The signal as found, moved onto the target and added to each null: the
    # star-by-star comparison on those images is what it returns when the
    # target IS the source, PRF mismatch and the number of stars tried
    # included.  Per null, chi2(target) - chi2(star) for every hypothesis.
    on_t = fa * scene.prf(xs, ys, txr, tyr)
    hyp_xy = [scene.xy[i] for i, s in enumerate(stars)
              if np.hypot(*(scene.xy[i] - (txr, tyr))) <= FIT_RADIUS]
    null_dchi2 = []
    for Nn in nulls:
        c2s = np.array([_fit_fixed(Nn + on_t, sig, region, scene, x, y, dimming=True)[2]
                        for x, y in hyp_xy])
        it_h = next(k for k, h in enumerate(hyp) if h["is_target"])
        null_dchi2.append((c2s[it_h] - c2s).tolist())

    m = {
        "n_transits": n, "n_null": len(nulls), "npix": npix,
        "target_pix_catalogue": [float(tx), float(ty)], "target_pix_registered": [float(txr), float(tyr)],
        "prf_position": "detector" if origin else "CCD centre (cutout records no detector position)",
        "registration": reg, "depth_snr": max(snr, snr_free), "snr_at_target": snr,
        "snr_free": snr_free,
        "target_depth_ppm": float(a_t / scene.flux[it] * 1e6) if scene.flux[it] > 0 else None,
        "free_pix": [fx, fy], "free_at_bound": at_bound, "offset_pix": off_px.tolist(), "cov_pix": cov_px.tolist(),
        "dra_arcsec": float(off_sky[0]), "ddec_arcsec": float(off_sky[1]),
        "cov_sky": cov_sky.tolist(), "jacobian": J.tolist(),
        "free_chi2": fchi2, "best_rchi2": float(rchi2),
        "hypotheses": hyp,
    }
    imgs = {"D": D, "O": O, "sig": sig, "xy": scene.xy.copy(),
            "free": (fx, fy), "cov_px": cov_px, "target": (txr, tyr),
            "model_t": a_t * scene.prf(xs, ys, txr, tyr),
            "null_dchi2": {"ids": [h["id"] for h in hyp], "rows": null_dchi2},
            "flux": scene.flux.copy(), "shift_px": scene.shift.copy(), "blur_px": prf.blur,
            "stars": [{"id": s.id, "is_target": s.is_target, "ratio": s.ratio, "sep": s.sep,
                       "tmag": s.tmag, "viable": s.viable} for s in stars]}
    return SectorResult(sector, True, n_transits=n, n_null=len(nulls), metrics=m, images=imgs)


# --------------------------------------------------------------------------
# all sectors


def combine_sectors(results, sys_floor_px=0.0, pixel_arcsec=21.0, snr_min=4.0,
                    reg_frac=0.5):
    """Inverse-covariance mean offset on the sky, and summed per-star chi-square.

    Each sector's position covariance (from injection into the nulls) is
    inflated three ways before it is combined, each for a failure seen on a
    known planet:

    * by the depth-image fit's reduced chi-square when that exceeds 1 -- a
      flare inside a transit window or a systematic leaves
      structure no PRF fits;
    * by ``reg_frac`` times the registration shift the scene needed -- a
      large shift means the catalogue scene matches the image poorly
      (two planets, both ~0.37 px);
    * by ``sys_floor_px``, the PRF model's own floor.

    Only sectors where the depth image detects the signal (``snr_min``)
    contribute a position: an image without the signal gives a position
    anywhere, with an error that knows nothing of that.  Finally, if the
    sectors disagree with each other by more than their errors allow, the
    combined covariance is scaled by the excess (the Birge ratio): one known
    planet's sectors scatter by 12-20 arcsec, while a real contaminant's agree.
    Per-star chi-squares are divided by each sector's reduced chi-square when
    it exceeds 1, so a badly modelled sector cannot dominate the comparison.
    """
    ok = [r for r in results if r.ok]
    if not ok:
        return None
    det = [r for r in ok if r.metrics["depth_snr"] >= snr_min]
    pos = [r for r in det if not r.metrics.get("free_at_bound")]
    out = {"n_sectors": len(ok), "n_sectors_detected": len(det), "n_sectors_positioned": len(pos),
           "n_sectors_at_bound": len(det) - len(pos),
           "max_transits_positioned": max((r.n_transits for r in pos), default=0)}
    if pos:
        Winv, Wx, vs, Cs = np.zeros((2, 2)), np.zeros(2), [], []
        for r in pos:
            m = r.metrics
            J = np.array(m["jacobian"])
            sh = np.abs(np.array(m["registration"]["shift_px"]))
            C = (np.array(m["cov_sky"]) * max(1.0, m["best_rchi2"])
                 + J @ np.diag((reg_frac * sh) ** 2) @ J.T
                 + np.eye(2) * (sys_floor_px * pixel_arcsec) ** 2)
            v = np.array([m["dra_arcsec"], m["ddec_arcsec"]])
            m["cov_sky_total"] = C.tolist()
            Ci = np.linalg.inv(C)
            Winv += Ci
            Wx += Ci @ v
            vs.append(v)
            Cs.append(C)
        C = np.linalg.inv(Winv)
        v = C @ Wx
        chi_between = sum(float((vk - v) @ np.linalg.inv(Ck) @ (vk - v)) for vk, Ck in zip(vs, Cs))
        dof = 2 * (len(pos) - 1)
        birge = max(1.0, chi_between / dof) if dof > 0 else 1.0
        out.update(_roll_test(pos, vs, Cs, chi_between, dof))
        C = C * birge
        out.update({
            "dra_arcsec": float(v[0]), "ddec_arcsec": float(v[1]),
            "offset_arcsec": float(np.hypot(*v)), "cov_sky": C.tolist(),
            "sigma_arcsec": float(np.sqrt(np.trace(C) / 2)),
            "offset_sigma": float(np.sqrt(v @ np.linalg.inv(C) @ v)),
            "birge": birge,
            "per_sector_sigma": [float(np.sqrt(vk @ np.linalg.inv(Ck) @ vk)) for vk, Ck in zip(vs, Cs)],
        })

    chi = {}
    for r in ok:
        scale = max(1.0, r.metrics["best_rchi2"])
        for h in r.metrics["hypotheses"]:
            e = chi.setdefault(h["id"], {k: h[k] for k in ("id", "is_target", "sep_arcsec",
                                                           "tmag", "viable")}
                               | {"chi2": 0.0, "n_sectors": 0, "depths": []})
            e["chi2"] += h["chi2"] / scale
            e["n_sectors"] += 1
            if h["implied_depth"] is not None:
                e["depths"].append(h["implied_depth"])
    nmax = max(e["n_sectors"] for e in chi.values())
    hyps = [e for e in chi.values() if e["n_sectors"] == nmax]
    for e in hyps:
        e["implied_depth"] = float(np.median(e["depths"])) if e["depths"] else None
        del e["depths"]
    tgt = next((e for e in hyps if e["is_target"]), None)
    for e in hyps:
        e["dchi2_vs_target"] = (e["chi2"] - tgt["chi2"]) if tgt else None
    hyps.sort(key=lambda e: e["chi2"])
    snr = np.array([r.metrics["depth_snr"] for r in ok])
    out["depth_snr"] = float(np.sqrt(np.sum(np.clip(snr, 0, None) ** 2)))
    out["hypotheses"] = hyps
    out["null_margins"] = _null_margins(ok, {e["id"] for e in hyps})
    return out


def _roll_test(pos, vs, Cs, chi_sky, dof):
    """Does the offset stay put on the sky, or turn with the spacecraft?

    A real contaminant sits at one sky position whatever the roll angle; a
    PRF-model error sits at one place in the *pixel* frame and so rotates on
    the sky from sector to sector.  Fitting one constant offset in each frame
    says which, when the sectors' roll angles differ enough to tell them apart.
    """
    Js = [np.array(r.metrics["jacobian"]) for r in pos]
    # position angle of the pixel +y axis on the sky, per sector
    pa = np.array([np.degrees(np.arctan2(J[0, 1], J[1, 1])) for J in Js])
    d = np.abs((pa[:, None] - pa[None, :] + 180) % 360 - 180)
    spread = float(d.max()) if len(pa) > 1 else 0.0
    if len(pos) < 2:
        return {"roll_spread_deg": spread, "chi2_sky_between": None, "chi2_pix_between": None,
                "roll_dof": 0}
    ps = [np.linalg.solve(J, vk) for J, vk in zip(Js, vs)]
    Cp = [np.linalg.inv(J) @ Ck @ np.linalg.inv(J).T for J, Ck in zip(Js, Cs)]
    W = sum(np.linalg.inv(c) for c in Cp)
    pbar = np.linalg.solve(W, sum(np.linalg.inv(c) @ p for c, p in zip(Cp, ps)))
    chi_pix = sum(float((p - pbar) @ np.linalg.inv(c) @ (p - pbar)) for p, c in zip(ps, Cp))
    return {"roll_spread_deg": spread, "chi2_sky_between": float(chi_sky),
            "chi2_pix_between": chi_pix, "roll_dof": dof,
            "offset_pa_deg_per_sector": [float(np.degrees(np.arctan2(vk[0], vk[1]))) for vk in vs]}


def _null_margins(ok, ids):
    """Combined look-elsewhere margins: per null index, each star's chi-square
    advantage over the target summed over sectors (scaled as the real ones are),
    then the best non-target star's.  One value per null."""
    rows = []
    n = min(len(r.images.get("null_dchi2", {}).get("rows", [])) for r in ok)
    for k in range(n):
        tot = {}
        for r in ok:
            nd = r.images["null_dchi2"]
            scale = max(1.0, r.metrics["best_rchi2"])
            for sid, val in zip(nd["ids"], nd["rows"][k]):
                tot[sid] = tot.get(sid, 0.0) + val / scale
        tgt = next(h["id"] for h in ok[0].metrics["hypotheses"] if h["is_target"])
        vals = [v for sid, v in tot.items() if sid in ids and sid != tgt]
        if vals:
            rows.append(max(vals))
    return rows

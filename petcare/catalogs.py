"""Catalogue queries: Gaia DR3 at ESA, the TIC at MAST.

Two rules, both from experience rather than preference.

**Query the archive that owns the data.**  Gaia at ESA, the TIC at MAST,
2MASS/WISE at IRSA.  CDS mirrors most of it, but CDS is rate-limited per IP
and an IP it has filtered breaks ordinary interactive use for days.  Nothing
here touches CDS; :func:`petcare.cache.cds_throttle` exists for anything that
ever has to.

**Cache everything.**  A Gaia cone takes ~19 s.  At 30k targets that is a
week of wall clock spent re-asking questions already answered, and it is the
fastest way to get an IP blocked.

Proper motion is propagated from the Gaia epoch to the epoch of the
observation.  It is not a detail: a nearby high-proper-motion star can move 2.5 arcseconds between Gaia
DR3's 2016.0 and a 2025 TESS sector, which is more than a TESS pixel, and a
neighbour list built at the wrong epoch puts the contaminating star in the
wrong place.
"""

from __future__ import annotations

import numpy as np

from .cache import cached, ensure_certificates, with_retries

#: Reference epoch of Gaia DR3 positions, in Julian years.
GAIA_EPOCH = 2016.0

#: TESS pixel scale, arcseconds per pixel.
TESS_PIXEL_ARCSEC = 21.0

#: Gaussian sigma, in pixels, standing in for the TESS PRF core.  A crude
#: approximation with real wings missing -- every flux fraction computed from
#: it is labelled as such, and a fraction near a threshold should be checked
#: against a proper scene model before anyone acts on it.
PRF_SIGMA_PIXELS = 0.65

#: Moffat parameters standing in for the TESS PRF.  A Gaussian was used here
#: first and it is badly wrong in the regime that matters.  The wings are the
#: whole point: a Gaussian at sigma = 0.65 px puts 0.13% of a star's peak flux
#: 2.4 pixels away, and a Moffat matched to the same 1.5 px FWHM puts 2.8%
#: there -- a factor of 20, and the difference between "excluded" and "this is
#: your false positive".
#:
#: The case that forced it: on one candidate, pc_centroid measured the transit
#: signal 49.7 arcsec (2.4 px) off target at 70.8 sigma, pointing at a star
#: 3.1x brighter than the target -- while pc_neighbours "excluded" that same
#: star by requiring a 397% eclipse from it.  Two tests on the same data
#: disagreeing that badly means one of them is wrong, and it was this model.
#: With the Moffat the required depth becomes a possible eclipse.
#:
#: PROVISIONAL: beta and FWHM are the conventional ground-truth-free values
#: for TESS.  The real PRF varies across a camera and is available from MAST
#: per camera/CCD/field position; fitting those is the right long-term answer
#: and would also make this asymmetric, which it really is.
PRF_MOFFAT_BETA = 2.5
PRF_FWHM_PIXELS = 1.5


class CatalogError(Exception):
    """Raised when a catalogue query fails in a way the caller must see."""


# --------------------------------------------------------------------------
# Gaia


_GAIA_COLUMNS = (
    "source_id, ra, dec, parallax, parallax_error, pmra, pmdec, "
    "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe, "
    "astrometric_excess_noise, astrometric_excess_noise_sig, "
    "astrometric_gof_al, astrometric_chi2_al, astrometric_n_good_obs_al, "
    "ipd_gof_harmonic_amplitude, ipd_frac_multi_peak, ipd_frac_odd_win, "
    "duplicated_source, phot_bp_rp_excess_factor, rv_amplitude_robust, "
    "radial_velocity, radial_velocity_error, rv_nb_transits, vbroad, "
    "non_single_star, teff_gspphot"
)

#: Columns to fall back to if the full list is rejected.  The Gaia archive
#: moves columns between tables between releases (``radius_gspphot`` lives in
#: astrophysical_parameters, not gaia_source, and asking for it returns a 400
#: that takes the whole cone with it), so a narrow always-present set is kept
#: as a second chance rather than losing the query.
_GAIA_COLUMNS_MINIMAL = (
    "source_id, ra, dec, parallax, parallax_error, pmra, pmdec, "
    "phot_g_mean_mag, bp_rp, ruwe, astrometric_excess_noise, "
    "astrometric_excess_noise_sig, duplicated_source"
)


#: Rows the Gaia synchronous TAP endpoint returns at most.
SYNC_ROW_CAP = 2000


def gaia_cone(ra: float, dec: float, radius_arcsec: float = 120.0,
              g_limit: float = 20.5):
    """Gaia DR3 sources within ``radius_arcsec``, as a list of dicts.  Cached."""
    ensure_certificates()
    radius_deg = radius_arcsec / 3600.0

    def _select(rad_deg, extra=""):
        from astroquery.gaia import Gaia
        where = (f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
                 f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {rad_deg:.7f})) "
                 f"AND phot_g_mean_mag < {g_limit}")
        last = None
        for cols in (_GAIA_COLUMNS, _GAIA_COLUMNS_MINIMAL):
            try:
                # Synchronous, not async: the async endpoint returns HTTP 500
                # on cones this small often enough to be a nuisance.
                job = Gaia.launch_job(
                    f"SELECT {cols} FROM gaiadr3.gaia_source {where}{extra}")
                return _to_dicts(job.get_results())
            except Exception as exc:  # noqa: BLE001 - retried with fewer columns
                last = exc
        raise last

    def _query():
        # The synchronous endpoint stops at SYNC_ROW_CAP rows, in no useful order,
        # and raises nothing: 107 of 20,811 cached cones were cut at exactly that.
        # In a field that dense, keep everything in a smaller circle, where the
        # aperture is, and only the brightest stars further out, since a faint
        # star beyond that cannot contribute.
        rows = _select(radius_deg)
        if len(rows) < SYNC_ROW_CAP:
            return rows
        inner_deg = radius_deg
        inner = rows
        while len(inner) >= SYNC_ROW_CAP:
            inner_deg /= 2.0
            inner = _select(inner_deg)
        bright = _select(radius_deg, " ORDER BY phot_g_mean_mag ASC")
        seen = {r.get("source_id") for r in inner}
        return inner + [r for r in bright if r.get("source_id") not in seen]

    return cached("gaia_cone", round(ra, 5), round(dec, 5),
                  round(radius_arcsec, 1), g_limit)(
        lambda: with_retries(_query, what=f"Gaia cone at {ra:.4f},{dec:.4f}"))


def gaia_source(source_id: int):
    """One Gaia DR3 row by source_id, or None.  Cached."""
    ensure_certificates()

    def _query():
        from astroquery.gaia import Gaia
        adql = (f"SELECT {_GAIA_COLUMNS} FROM gaiadr3.gaia_source "
                f"WHERE source_id = {int(source_id)}")
        return _to_dicts(Gaia.launch_job(adql).get_results())

    rows = cached("gaia_source", int(source_id))(
        lambda: with_retries(_query, what=f"Gaia source {source_id}"))
    return rows[0] if rows else None


def _to_dicts(table) -> list:
    """astropy Table -> plain dicts, with masked values as None.

    Plain dicts on purpose: they pickle into the cache without dragging an
    astropy version dependency along with them.
    """
    out = []
    for row in table:
        d = {}
        for c in table.colnames:
            v = row[c]
            if v is None or (hasattr(v, "mask") and bool(getattr(v, "mask"))):
                d[c] = None
                continue
            try:
                if np.ma.is_masked(v):
                    d[c] = None
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(v, (np.integer,)):
                d[c] = int(v)
            elif isinstance(v, (np.floating,)):
                d[c] = float(v) if np.isfinite(float(v)) else None
            elif isinstance(v, (np.bool_, bool)):
                d[c] = bool(v)
            else:
                d[c] = str(v)
        out.append(d)
    return out


# --------------------------------------------------------------------------
# TIC


def tic_cone(ra: float, dec: float, radius_arcsec: float = 120.0,
             tmag_limit: float = 19.0):
    """TIC sources within ``radius_arcsec``, as a list of dicts.  Cached."""
    ensure_certificates()

    def _query():
        from astroquery.mast import Catalogs
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(ra, dec, unit="deg")
        tbl = Catalogs.query_region(coord, radius=radius_arcsec * u.arcsec,
                                    catalog="TIC")
        if tbl is None or len(tbl) == 0:
            return []
        keep = [c for c in ("ID", "ra", "dec", "Tmag", "Vmag", "Jmag", "Kmag",
                            "Teff", "rad", "mass", "logg", "plx", "pmRA",
                            "pmDEC", "objType", "contratio", "disposition",
                            "d", "lumclass", "GAIA", "TWOMASS", "wdflag")
                if c in tbl.colnames]
        tbl = tbl[keep]
        if "Tmag" in tbl.colnames:
            tbl = tbl[~np.isnan(np.asarray(tbl["Tmag"], float))]
            tbl = tbl[np.asarray(tbl["Tmag"], float) < tmag_limit]
        return _to_dicts(tbl)

    return cached("tic_cone", round(ra, 5), round(dec, 5),
                  round(radius_arcsec, 1), tmag_limit)(
        lambda: with_retries(_query, what=f"TIC cone at {ra:.4f},{dec:.4f}"))


# --------------------------------------------------------------------------
# geometry


def propagate(ra, dec, pmra, pmdec, from_epoch=GAIA_EPOCH, to_epoch=None):
    """Move a position by its proper motion, in degrees.

    ``pmra`` is mu_alpha* (already including the cos(dec) factor), in mas/yr,
    as Gaia reports it.
    """
    if to_epoch is None or pmra is None or pmdec is None:
        return ra, dec
    dt = float(to_epoch) - float(from_epoch)
    ra_out = ra + (pmra / 3.6e6) * dt / max(np.cos(np.radians(dec)), 1e-6)
    dec_out = dec + (pmdec / 3.6e6) * dt
    return float(ra_out), float(dec_out)


def separation_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation and its components (dRA*, dDec) in arcseconds."""
    dra = (np.asarray(ra2, float) - ra1) * np.cos(np.radians(dec1)) * 3600.0
    ddec = (np.asarray(dec2, float) - dec1) * 3600.0
    return np.hypot(dra, ddec), dra, ddec


def btjd_to_jyear(btjd: float) -> float:
    """BTJD to Julian year, for propagating proper motions to an observation."""
    return 2000.0 + (btjd + 2457000.0 - 2451545.0) / 365.25


# --------------------------------------------------------------------------
# aperture flux fractions


def aperture_centre(aperture_mask) -> tuple[float, float]:
    """(row, col) of the target inside an extraction aperture mask.

    tessquicklook centres its circular apertures on the star's WCS position,
    which is generally NOT the centre of the cutout array: one star's
    sector-98 aperture is pixels (7..8, 7..8) of a 15x15 cutout, so the star is
    at (7.5, 7.5), and its sector-6 two-pixel aperture puts it at (8, 7.5).
    Assuming the array centre put the drawn aperture a pixel off the star and
    biased every flux fraction.  The lit pixels' centroid is the best estimate
    available without the cutout WCS.
    """
    mask = np.asarray(aperture_mask, float)
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        ny, nx = mask.shape
        return (ny - 1) / 2.0, (nx - 1) / 2.0
    return float(ys.mean()), float(xs.mean())


def prf_fraction_in_aperture(dx_pix, dy_pix, aperture_mask,
                             fwhm_pix: float = PRF_FWHM_PIXELS,
                             beta: float = PRF_MOFFAT_BETA,
                             pad_pix: int = 6) -> float:
    """Fraction of a source's flux landing inside ``aperture_mask``.

    The source sits ``(dx_pix, dy_pix)`` from the mask centre and is modelled
    as a circular **Moffat**, normalised over a region large enough to contain
    essentially all of it.  See :data:`PRF_MOFFAT_BETA` for why this is not a
    Gaussian: the wings are what decide whether a bright star outside the
    aperture can produce the signal, and a Gaussian has none.

    Still an approximation, and still stated as one in every result that uses
    it -- the real PRF is asymmetric and varies across a camera.  Treat the
    number as an order of magnitude and check anything near a decision
    threshold against the pixels themselves (``pc_centroid``).
    """
    mask = np.asarray(aperture_mask, float)
    if mask.ndim != 2 or mask.sum() <= 0:
        return float("nan")
    ny, nx = mask.shape
    cy, cx = aperture_centre(mask)

    alpha = float(fwhm_pix) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

    def _moffat(yy, xx, y0, x0):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        return (1.0 + r2 / alpha ** 2) ** (-beta)

    # Normalise on a grid padded well beyond the mask: a Moffat with beta=2.5
    # keeps ~2% of its flux outside 6 pixels, and leaving that out would
    # inflate every fraction.
    gy, gx = np.indices((ny + 2 * pad_pix, nx + 2 * pad_pix))
    y0, x0 = cy + pad_pix + dy_pix, cx + pad_pix + dx_pix
    total = float(np.sum(_moffat(gy, gx, y0, x0)))
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    ys, xs = np.indices(mask.shape)
    inside = float(np.sum(_moffat(ys, xs, cy + dy_pix, cx + dx_pix)
                          * (mask > 0)))
    return float(np.clip(inside / total, 0.0, 1.0))


def flux_ratio(tmag_neighbour, tmag_target) -> float:
    """Neighbour flux divided by target flux, from TESS magnitudes."""
    if tmag_neighbour is None or tmag_target is None:
        return float("nan")
    return float(10 ** (-0.4 * (float(tmag_neighbour) - float(tmag_target))))


def gaia_g_to_tmag(g, bp_rp=None) -> float:
    """Rough G -> T conversion, for neighbours with no TIC row.

    Stassun et al. (2019) relation, valid for -0.2 < BP-RP < 3.5; falls back
    to ``T = G - 0.5`` outside that, which is good to a few tenths and is
    labelled wherever it is used.
    """
    if g is None:
        return float("nan")
    g = float(g)
    if bp_rp is None or not np.isfinite(bp_rp) or not (-0.2 < bp_rp < 3.5):
        return g - 0.5
    x = float(bp_rp)
    return float(g - (0.00522555 * x ** 3 - 0.0891337 * x ** 2
                      + 0.633923 * x - 0.0324473))


def circular_aperture(radius_pixels: float = 2.5, size: int = 15):
    """A circular pixel mask, for extractions that do not carry their own.

    The 20 s SPOC products in this project record ``aperture_id='spoc_sap'``
    and an empty array: the aperture is the SPOC pipeline's and is not
    propagated.  Rather than fall all the way back to a total-flux ratio,
    which overstates every neighbour, a circular stand-in is used -- and the
    assumption is then **tested**, by checking that the target's computed flux
    share reproduces the dilution the extraction itself measured.  If it does
    not, the aperture assumption is wrong and the caller says so.
    """
    n = int(size)
    c = (n - 1) / 2.0
    ys, xs = np.indices((n, n))
    return (((xs - c) ** 2 + (ys - c) ** 2) <= radius_pixels ** 2).astype(float)


# --------------------------------------------------------------------------
# ExoFOP


#: The whole TOI table, ~8000 rows and 4 MB.  Downloaded once and cached:
#: ExoFOP's per-target endpoints time out often enough that pulling the table
#: and looking rows up locally is both faster and kinder to the service.
TOI_URL = ("https://exofop.ipac.caltech.edu/tess/download_toi.php"
           "?toi=&output=csv&sort=toi")

#: TFOPWG dispositions, expanded.
TFOPWG_DISPOSITIONS = {
    "CP": "confirmed planet",
    "KP": "known planet",
    "PC": "planet candidate",
    "APC": "ambiguous planet candidate",
    "FP": "false positive",
    "FA": "false alarm",
}


def toi_table(max_age_days: float = 7.0):
    """The ExoFOP TOI table as a DataFrame.  Cached, refreshed weekly."""
    def _fetch():
        import io as _io
        import urllib.request
        import pandas as pd
        with urllib.request.urlopen(TOI_URL, timeout=300) as fh:
            raw = fh.read()
        return pd.read_csv(_io.BytesIO(raw))

    return cached("exofop", "toi_table", max_age_days=max_age_days)(
        lambda: with_retries(_fetch, attempts=2, base_delay=10.0,
                             what="ExoFOP TOI table"))


def toi_rows(tic: int, max_age_days: float = 7.0) -> list:
    """Every TOI row for one TIC, as plain dicts (possibly empty)."""
    import pandas as pd
    df = toi_table(max_age_days=max_age_days)
    if df is None or "TIC ID" not in df.columns:
        return []
    sub = df[df["TIC ID"].astype("Int64") == int(tic)]
    out = []
    for _, row in sub.iterrows():
        d = {}
        for k, v in row.to_dict().items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
            elif pd.isna(v) if not isinstance(v, (list, tuple, np.ndarray)) else False:
                d[k] = None
            else:
                d[k] = v.item() if isinstance(v, np.generic) else v
        out.append(d)
    return out


# --------------------------------------------------------------------------
# Gaia variability


def gaia_variability(source_ids) -> dict:
    """Gaia DR3 variability classification and EB membership, batched.

    One query for a whole list of source_ids rather than one per target: at
    30k targets the per-source round trip is the entire cost of the test.

    Returns ``{source_id: {...}}``; sources absent from the variability tables
    simply do not appear, which is the common case -- Gaia classifies only a
    minority of sources as variable.
    """
    ids = sorted({int(s) for s in source_ids if s})
    if not ids:
        return {}
    ensure_certificates()

    def _query(chunk):
        from astroquery.gaia import Gaia
        in_list = ",".join(str(i) for i in chunk)
        out = {}
        try:
            t = Gaia.launch_job(
                f"SELECT source_id, best_class_name, best_class_score "
                f"FROM gaiadr3.vari_classifier_result "
                f"WHERE source_id IN ({in_list})").get_results()
            for row in _to_dicts(t):
                out.setdefault(int(row["source_id"]), {}).update(
                    {"class": row.get("best_class_name"),
                     "class_score": row.get("best_class_score")})
        except Exception:  # noqa: BLE001 - one table failing is not fatal
            pass
        try:
            t = Gaia.launch_job(
                f"SELECT source_id, frequency, global_ranking "
                f"FROM gaiadr3.vari_eclipsing_binary "
                f"WHERE source_id IN ({in_list})").get_results()
            for row in _to_dicts(t):
                freq = row.get("frequency")
                out.setdefault(int(row["source_id"]), {}).update(
                    {"eclipsing_binary": True,
                     "eb_frequency_per_day": freq,
                     # Gaia's EB frequency is per *orbit*, so the catalogued
                     # period is 1/frequency and is often the half-period a
                     # periodogram finds; both are reported
                     "eb_period_days": (1.0 / freq) if freq else None,
                     "eb_ranking": row.get("global_ranking")})
        except Exception:  # noqa: BLE001
            pass
        return out

    merged = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        merged.update(
            cached("gaia_vari", tuple(chunk))(
                lambda c=chunk: with_retries(lambda: _query(c),
                                             what="Gaia variability")))
    return merged

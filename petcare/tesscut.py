"""Reading TESScut cutouts, in both of the formats MAST has served.

Until 2026-10-26 the cutout's WCS lived in the PIXELS binary table's header
(``1CTYP5``-style keys, read with ``keysel=["binary"]``), together with the
physical-pixel keys ``1CRV4P``/``2CRV4P`` that give the cutout's corner on the
CCD.  astrocut 1.4 moved the WCS into the APERTURE image extension and dropped
the physical keys.  Read the old way, a new file raises "If no header is
provided, keysel may not be provided either", which silently cost every
pixel-level test those sectors (488 results by the time it was noticed).
"""

from __future__ import annotations

import warnings


def cutout_wcs(hdul):
    """The cutout's celestial WCS, whichever extension holds it."""
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w = WCS(hdul[1].header, keysel=["binary"])
            if w.has_celestial:
                return w
        except Exception:  # noqa: BLE001 - new format: no WCS in the binary table
            pass
        for ext in ("APERTURE", 2):
            try:
                w = WCS(hdul[ext].header)
            except (KeyError, IndexError, ValueError):
                continue
            if w.has_celestial:
                return w
    raise ValueError("no celestial WCS in the cutout's PIXELS or APERTURE extension")


def cutout_ccd_origin(hdul):
    """(row, column) on the CCD of the cutout's pixel (0, 0), or None if unrecorded."""
    h = hdul[1].header
    col, row = h.get("1CRV4P"), h.get("2CRV4P")
    if col is None or row is None:
        return None
    return float(row), float(col)


def cutout_camera_ccd(hdul):
    cam = hdul[0].header.get("CAMERA") or hdul[1].header.get("CAMERA")
    ccd = hdul[0].header.get("CCD") or hdul[1].header.get("CCD")
    return (int(cam) if cam else None), (int(ccd) if ccd else None)

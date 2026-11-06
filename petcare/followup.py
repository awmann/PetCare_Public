"""Clients for the two archives that record what has already been observed.

**ExoFOP-TESS**, per target.  ``target.php?id=<TIC>&json`` returns everything
the community has filed on one star: high-resolution imaging, spectroscopy,
ground-based time series, resolved stellar companions, and the TFOPWG
disposition.  The per-table endpoints (``download_imaging.php`` and friends)
ignore their ``id`` parameter and hand back the whole table -- 9645 rows for
stellar companions -- so they time out and are not used here.

**The LCO science archive**, by position.  A public cone search finds frames
whose footprint contains the target, whether or not anybody filed them on
ExoFOP.  This is the half that works for candidates that are not TOIs, which
is most of ours.

Two quirks of the LCO API, both discovered the hard way:

* ``count`` is a placeholder for spatial queries.  It comes back as 1 whether
  the search matched nothing or ten thousand frames, so only ``results`` can
  be counted.
* the anonymous page limit is 100.  Asking for more is a 400, not a clamp.

Neither archive needs credentials for this metadata.  ExoFOP's follow-up rows
are public; logging in only matters for *downloading* the data products, and
the LCO cone search sees public frames only -- a target's proprietary frames
stay invisible, so absence of LCO data is never evidence of absence.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .cache import cached

EXOFOP_TARGET = "https://exofop.ipac.caltech.edu/tess/target.php"
LCO_FRAMES = "https://archive-api.lco.global/frames/"

#: The anonymous page limit.  Larger is a 400.
LCO_PAGE = 100

TIMEOUT = 60.0


def _get_json(url: str, timeout: float = TIMEOUT):
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8", "replace"))


def exofop_target(tic: int, max_age_days: float = 7.0) -> dict:
    """Everything ExoFOP holds on one TIC.

    Cached for a week: follow-up accumulates over months, and the endpoint is
    slow enough that re-querying it per test run is rude.
    """
    def fetch():
        url = f"{EXOFOP_TARGET}?{urllib.parse.urlencode({'id': int(tic), 'json': ''})}"
        # ExoFOP wants a bare ``&json``, which urlencode writes as ``json=``;
        # it accepts both, but the bare form is what their documentation says.
        return _get_json(url.replace("json=", "json"))

    return cached("exofop", "target", int(tic), max_age_days=max_age_days)(fetch)


def lco_frames(ra: float, dec: float, max_frames: int = 500,
               reduction_level: int | None = 91,
               max_age_days: float = 7.0) -> list:
    """Public LCO frames whose footprint contains this position.

    ``reduction_level=91`` is science-ready data; pass ``None`` to include raw
    frames as well.  Paged because the anonymous limit is 100 per request.
    """
    def fetch():
        out, offset = [], 0
        while len(out) < max_frames:
            params = {"covers": f"POINT({float(ra)} {float(dec)})",
                      "limit": LCO_PAGE, "offset": offset, "public": "true"}
            if reduction_level is not None:
                params["reduction_level"] = int(reduction_level)
            page = _get_json(f"{LCO_FRAMES}?{urllib.parse.urlencode(params)}")
            got = page.get("results") or []
            out.extend(got)
            if len(got) < LCO_PAGE:
                break
            offset += LCO_PAGE
        return out[:max_frames]

    return cached("lco", "frames", round(float(ra), 5), round(float(dec), 5),
                  reduction_level, max_frames, max_age_days=max_age_days)(fetch)


def required_depth_on_companion(depth_ppm: float, dmag: float) -> float | None:
    """Eclipse depth the companion would need to fake this signal, as a fraction.

    If the dip is on a companion contributing a fraction ``f`` of the blended
    light, its own eclipse must be ``depth_observed / f`` deep.  With a
    magnitude difference ``dmag`` the companion-to-target flux ratio is
    ``10 ** (-0.4 * dmag)``, so ``f = ratio / (1 + ratio)`` and the required
    depth is ``depth_observed * (1 + 1 / ratio)``.

    Returns ``None`` when ``dmag`` is missing.  Values above 1 are impossible
    and exclude the companion outright.

    Two caveats travel with every number this returns.  ``dmag`` is measured
    in the imaging filter, not the TESS band, and a companion redder than the
    target is brighter in TESS than the speckle contrast suggests -- so the
    required depth is an over-estimate for red companions.  And only the two
    stars are counted: any third star in the aperture raises the requirement
    further.
    """
    if dmag is None:
        return None
    try:
        dmag = float(dmag)
    except (TypeError, ValueError):
        return None
    ratio = 10.0 ** (-0.4 * dmag)
    if ratio <= 0:
        return None
    return float(depth_ppm / 1e6 * (1.0 + 1.0 / ratio))


def iso_to_btjd(stamp) -> float | None:
    """An archive UTC timestamp as BTJD (BJD - 2457000).

    The barycentric correction is not applied: it reaches 8 minutes, which is
    immaterial against transit windows of hours, and applying it properly
    needs the target's position and a solar-system ephemeris.  Every window
    this feeds is widened to absorb it.
    """
    if not stamp:
        return None
    s = str(stamp).replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = (datetime.fromisoformat(s) if fmt is None
                  else datetime.strptime(s, fmt).replace(tzinfo=timezone.utc))
            break
        except ValueError:
            continue
    else:
        return None
    if dt.tzinfo is None:
        # Both archives report UTC; a stamp without an offset is still UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    unix = dt.timestamp()
    return float(unix / 86400.0 + 2440587.5 - 2457000.0)

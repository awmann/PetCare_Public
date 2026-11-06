"""Substellar evolutionary models, for asking what a companion could be.

Reads the **Sonora Diamondback** evolution tables (Morley et al. 2024,
doi:10.5281/zenodo.12735103).  The grid is not redistributed with this
package -- it is someone else's model output with its own citation -- so it is
read from a path you supply, and :func:`fetch` will download just the 0.8 MB
evolution tables from Zenodo if you have not already.

    export PETCARE_SONORA=~/models/diamondback/evolution

The tables cover 0.5-84 Jupiter masses and 2 Myr-13 Gyr, which is the whole
substellar range: everything from a sub-Saturn to the hydrogen-burning limit.
That span is what makes them the right grid for this question.  Their *upper*
envelope at a given age is the useful bound -- a companion larger than
anything substellar at that age is a star, and saying so needs no stellar
grid at all.

One quirk worth knowing: the radius column is headed ``R/Rsun`` and is
actually in Jupiter radii.  A 0.5 Mjup object at 2 Myr is listed at 1.632,
which is nonsense in solar radii and exactly right in Jovian ones.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import cache

ENV_PATH = "PETCARE_SONORA"
ZENODO_RECORD = 12735103
ZENODO_FILE = "evolution.zip"

#: Which table to read.  ``nc`` is the cloud-free sequence; ``hybrid`` adds
#: the cloud treatment that matters at the L/T transition.  For a radius this
#: choice is a percent-level effect, far below the stellar-radius error that
#: dominates any candidate here.
DEFAULT_TABLE = "nc_m0.0_age"

#: Jupiter masses per solar mass, and the deuterium-burning limit.
MJUP_PER_MSUN = 1047.57
DEUTERIUM_LIMIT_MJUP = 13.0

#: Above this the object is burning hydrogen and is a star, not a brown
#: dwarf.  The Sonora grid stops at 84 Mjup, just short of it.
HYDROGEN_LIMIT_MJUP = 78.0


def default_grid_path() -> Path:
    """Where :func:`fetch` puts the tables when no destination is given."""
    root = Path(os.environ.get(cache.ENV_CACHE, cache.DEFAULT_CACHE)).expanduser()
    return root / "models" / "diamondback"


def grid_path(path=None) -> Path | None:
    for cand in (path, os.environ.get(ENV_PATH)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_dir():
                return p
            if p.is_file():
                return p.parent
    d = default_grid_path()
    return d if d.is_dir() and any(d.iterdir()) else None


def fetch(dest) -> Path:
    """Download the Diamondback evolution tables from Zenodo (0.8 MB)."""
    import json
    import urllib.request
    import zipfile

    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60) as r:
        rec = json.load(r)
    f = next(x for x in rec["files"] if x["key"] == ZENODO_FILE)
    tmp = dest / ZENODO_FILE
    urllib.request.urlretrieve(f["links"]["self"], tmp)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    tmp.unlink(missing_ok=True)
    return dest / "evolution"


def load(path=None, table: str = DEFAULT_TABLE):
    """``(age_gyr, mass_mjup, radius_rjup, teff)`` as flat arrays.

    Returns ``None`` when the grid is not configured, so every caller can
    degrade to saying so rather than guessing.
    """
    root = grid_path(path)
    if root is None:
        return None
    f = root / table
    if not f.exists():
        for alt in root.rglob(table):
            f = alt
            break
        else:
            return None

    rows = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if len(rows) < 100:
        return None
    a = np.asarray(rows, float)
    return a[:, 0], a[:, 1] * MJUP_PER_MSUN, a[:, 5], a[:, 3]


def radius_at(age_gyr, mass_mjup, grid) -> float:
    """Model radius in Rjup, nearest-age-slice with mass interpolation."""
    ages, masses, radii, _ = grid
    slice_age = ages[np.argmin(np.abs(ages - age_gyr))]
    m = np.isclose(ages, slice_age)
    order = np.argsort(masses[m])
    return float(np.interp(mass_mjup, masses[m][order], radii[m][order]))


def masses_consistent(age_gyr, radius_rjup, sigma_rjup, grid):
    """Every grid mass whose radius at this age matches the measurement.

    The radius-mass relation is not monotonic -- it dips through the
    degenerate regime and rises again as the object approaches the
    hydrogen-burning limit -- so this returns the full set of consistent
    masses rather than a single inversion.
    """
    ages, masses, radii, _ = grid
    slice_age = ages[np.argmin(np.abs(ages - age_gyr))]
    m = np.isclose(ages, slice_age)
    ok = np.abs(radii[m] - radius_rjup) <= max(sigma_rjup, 1e-3)
    sel = masses[m][ok]
    return (float(sel.min()), float(sel.max()), int(sel.size),
            float(slice_age)) if sel.size else (None, None, 0, float(slice_age))


def max_substellar_radius(age_gyr, grid) -> tuple:
    """Largest radius anything substellar can have at this age, and its mass."""
    ages, masses, radii, _ = grid
    slice_age = ages[np.argmin(np.abs(ages - age_gyr))]
    m = np.isclose(ages, slice_age)
    i = int(np.argmax(radii[m]))
    return float(radii[m][i]), float(masses[m][i]), float(slice_age)


def min_substellar_radius(age_gyr, grid) -> tuple:
    """Smallest radius in the grid at this age, and its mass.

    Anything below it is smaller than the lightest object the grid carries --
    already sub-Saturn -- and is therefore planetary without needing a mass.
    """
    ages, masses, radii, _ = grid
    slice_age = ages[np.argmin(np.abs(ages - age_gyr))]
    m = np.isclose(ages, slice_age)
    i = int(np.argmin(radii[m]))
    return float(radii[m][i]), float(masses[m][i])


def rv_semi_amplitude(mass_mjup, period_days, m_star_msun, ecc=0.0) -> float:
    """RV semi-amplitude in m/s for a circular-ish orbit, edge-on."""
    if not (period_days and m_star_msun and m_star_msun > 0):
        return float("nan")
    k = (28.4329 * mass_mjup * (m_star_msun ** (-2.0 / 3.0))
         * ((period_days / 365.25) ** (-1.0 / 3.0)))
    return float(k / np.sqrt(max(1.0 - ecc ** 2, 1e-6)))

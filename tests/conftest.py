"""Shared fixtures: synthetic light curves in the formats MAST actually serves."""

import numpy as np
import pytest
from astropy.io import fits

TIC = 123456
PROT = 2.9
PERIOD = 4.2
T0 = 1001.0
T14_H = 3.0
DEPTH = 0.01


def spot_transit_flux(t, seed=1):
    """A spotted star with a box transit and white noise."""
    rng = np.random.default_rng(seed)
    phase = ((t - T0 + PERIOD / 2) % PERIOD) - PERIOD / 2
    in_transit = np.abs(phase) < T14_H / 48.0
    f = (1.0 + 0.02 * np.sin(2 * np.pi * t / PROT)
         + 0.008 * np.sin(4 * np.pi * t / PROT + 0.7))
    return f * (1.0 - DEPTH * in_transit) + rng.normal(0.0, 8e-4, t.size)


def write_spoc(path, sector, t_start, tic=TIC, days=25.0, cadence_s=120.0):
    """A file with the columns and header keywords of a SPOC light curve."""
    t = np.arange(t_start, t_start + days, cadence_s / 86400.0)
    f = 1000.0 * spot_transit_flux(t, seed=sector)
    cols = [fits.Column("TIME", "D", array=t),
            fits.Column("PDCSAP_FLUX", "E", array=f),
            fits.Column("PDCSAP_FLUX_ERR", "E", array=np.full(t.size, 0.8)),
            fits.Column("QUALITY", "J", array=np.zeros(t.size, dtype=int))]
    primary = fits.PrimaryHDU()
    primary.header.update(SECTOR=sector, TICID=tic, CAMERA=1, CCD=2, ORIGIN="test")
    table = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    table.header.update(TIMEDEL=cadence_s / 86400.0, CROWDSAP=0.95, EXPOSURE=20.0)
    fits.HDUList([primary, table]).writeto(path)
    return path


def spoc_name(sector, tic=TIC):
    return f"tess2020000000000-s{sector:04d}-{tic:016d}-0120-s_lc.fits"


@pytest.fixture
def spoc_files(tmp_path):
    return [write_spoc(tmp_path / spoc_name(1), 1, 1000.0),
            write_spoc(tmp_path / spoc_name(2), 2, 1027.0)]


@pytest.fixture
def catalog(tmp_path):
    """A one-star catalogue, so nothing has to ask MAST."""
    path = tmp_path / "stars.csv"
    path.write_text(f"tic_id,ra,dec,tmag,radius,e_radius,mass,e_mass,teff,e_teff,age_myr\n"
                    f"{TIC},10.0,-40.0,9.0,1.0,0.1,1.0,0.1,5000,100,40\n")
    return path

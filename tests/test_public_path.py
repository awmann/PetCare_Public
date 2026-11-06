"""The route someone without our extraction takes: MAST files, no stored Prot."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import PROT, TIC, spoc_name, write_spoc

from petcare import driver, io as pcio
from petcare.scripts import petcare as pc_main


# -- finding files -----------------------------------------------------------

def test_find_lightcurves_matches_mast_names(tmp_path):
    a = tmp_path / spoc_name(1)
    a.touch()
    (tmp_path / spoc_name(1, tic=999)).touch()
    assert pcio.find_lightcurves(TIC, [tmp_path]) == [str(a)]


def test_find_lightcurves_looks_inside_mast_download_tree(tmp_path):
    product = tmp_path / "mastDownload" / "TESS" / spoc_name(1).replace(".fits", "")
    product.mkdir(parents=True)
    f = product / spoc_name(1)
    f.touch()
    assert pcio.find_lightcurves(TIC, [tmp_path]) == [str(f)]


def test_find_lightcurves_matches_qlp_names(tmp_path):
    f = tmp_path / f"hlsp_qlp_tess_ffi_s0011-{TIC:016d}_tess_v01_llc.fits"
    f.touch()
    assert pcio.find_lightcurves(TIC, [tmp_path]) == [str(f)]


def test_find_lightcurves_does_not_match_a_longer_tic(tmp_path):
    (tmp_path / spoc_name(1, tic=TIC * 10)).touch()
    assert pcio.find_lightcurves(TIC, [tmp_path]) == []


def test_find_lightcurves_keeps_the_tessquicklook_layout(tmp_path):
    d = tmp_path / "batch1" / f"tic{TIC:010d}"
    d.mkdir(parents=True)
    f = d / f"tic{TIC:010d}_s0011.npz"
    f.touch()
    assert pcio.find_lightcurves(TIC, [tmp_path]) == [str(f)]


# -- reading -----------------------------------------------------------------

def test_load_spoc_normalises_and_keeps_sectors(spoc_files):
    lc = pcio.load_lightcurve([str(p) for p in spoc_files])
    assert [s.sector for s in lc.sectors] == [1, 2]
    assert np.median(lc.f) == pytest.approx(1.0, abs=0.01)
    assert lc.median_cadence_s == pytest.approx(120.0)


def test_npz_quality_mask_is_honoured(tmp_path):
    """The mask is the difference between the npz route and the CSV route."""
    t = np.arange(1000.0, 1025.0, 120.0 / 86400.0)
    keep = np.ones(t.size, bool)
    keep[100:200] = False
    f = 1.0 + 1e-3 * np.sin(t)
    f[100:200] += 0.05                  # high, so the low-run restore leaves it out
    path = tmp_path / f"tic{TIC:010d}_s0001.npz"
    np.savez(path, t=t, fcor=f, err_photon=np.full(t.size, 1e-3), fit_good_mask=keep,
             cadence_s=np.full(t.size, 120.0), tic_id=TIC, sector=1, dilution=0.9)
    assert len(pcio.load_lightcurve([str(path)])) == t.size - 100


def test_extract_writes_what_the_reader_needs(tmp_path):
    pytest.importorskip("tessquicklook")
    from petcare.scripts.pc_extract import sector_payload
    t = np.arange(1000.0, 1025.0, 120.0 / 86400.0)
    sec = dict(t=t, fcor=1.0 + 1e-3 * np.sin(t), err_photon=np.full(t.size, 1e-3),
               fit_good_mask=np.ones(t.size, bool), crowdsap=0.9, exptime=120.0,
               camera=1, ccd=2, sector=7)
    path = tmp_path / f"tic{TIC:010d}_s0007.npz"
    np.savez(path, **sector_payload(sec, TIC))
    lc = pcio.load_lightcurve([str(path)])
    assert len(lc) == t.size and lc.sectors[0].sector == 7
    assert lc.dilution()[0] == pytest.approx(0.9)


# -- rotation ----------------------------------------------------------------

def test_user_prot_outranks_everything(spoc_files):
    lc = pcio.load_lightcurve([str(p) for p in spoc_files])
    lc.prot_user = 1.23
    rot = lc.rotation_period()
    assert rot["prot_days"] == 1.23 and rot["source"] == "user"
    assert lc.select(lc.t < 1010).rotation_period()["prot_days"] == 1.23


def test_prot_is_measured_when_the_product_has_none(spoc_files):
    rot = pcio.load_lightcurve([str(p) for p in spoc_files]).rotation_period()
    assert rot["source"] == "ls_measured"
    assert rot["prot_days"] == pytest.approx(PROT, rel=0.03)
    assert rot["n_sectors"] == 2


def test_recorded_prot_is_preferred_over_measuring(spoc_files):
    lc = pcio.load_lightcurve([str(p) for p in spoc_files])
    for s in lc.sectors:
        s.prot_acf_days = 5.0
    assert lc.rotation_period()["source"] == "acf"


def test_flat_star_gives_no_prot_rather_than_a_railed_one():
    rng = np.random.default_rng(3)
    t = np.arange(1000.0, 1025.0, 120.0 / 86400.0)
    trend = 1.0 + 0.05 * (t - 1000.0) / 25.0
    rot = pcio.measure_prot_ls(t, trend + rng.normal(0, 1e-3, t.size),
                               np.ones(t.size, int))
    assert rot["prot_days"] is None


def test_tess_orbit_is_not_reported_as_rotation():
    t = np.arange(1000.0, 1050.0, 120.0 / 86400.0)
    f = 1.0 + 0.01 * np.sin(2 * np.pi * t / pcio.TESS_ORBIT_DAYS)
    assert pcio.measure_prot_ls(t, f, np.ones(t.size, int))["prot_days"] is None


def test_too_short_a_sector_gives_no_prot():
    t = np.arange(1000.0, 1000.05, 120.0 / 86400.0)
    assert pcio.measure_prot_ls(t, np.ones(t.size), np.ones(t.size, int))["prot_days"] is None


# -- the driver --------------------------------------------------------------

def test_driver_passes_lc_and_prot_to_each_test():
    argv = driver._argv({"tic": 1, "prot": 2.5, "age_myr": 40}, {"period": 3.0, "t0": 1.0},
                        driver.DriverOptions(lc=("a.fits", "b.fits")), False)
    assert argv[argv.index("--prot") + 1] == "2.5"
    assert argv[argv.index("--age") + 1] == "40"
    assert argv[argv.index("--lc") + 1:argv.index("--lc") + 3] == ["a.fits", "b.fits"]


def test_every_driver_test_accepts_the_driver_arguments():
    """A flag added to one parser and not another takes down a whole candidate."""
    import importlib
    from petcare import cli
    for name, module, _, _ in driver.TESTS:
        mod = importlib.import_module(f"petcare.scripts.{module}")
        parser = cli.base_parser(name, "x")
        if hasattr(mod, "add_arguments"):
            parser = mod.add_arguments(parser)
        parser.parse_args(["--tic", "1", "--prot", "2.0", "--lc", "a.fits", "--period", "3",
                           "--t0", "1", "--duration", "2", "--depth", "100"])


def test_lc_with_a_candidate_table_is_refused(tmp_path):
    table = tmp_path / "c.csv"
    table.write_text("tic,period,t0,duration_h\n1,3.0,1.0,2.0\n")
    with pytest.raises(SystemExit, match="one star"):
        pc_main.main(["--candidates", str(table), "--lc", "a.fits", "--out", str(tmp_path)])


def test_console_scripts_exist_for_every_driver_test():
    text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    for _, module, _, _ in driver.TESTS:
        assert f"petcare.scripts.{module}:main" in text, module


def test_doctor_runs_offline_and_prints_no_secret(monkeypatch, capsys):
    from petcare.scripts import pc_doctor
    monkeypatch.setenv("ATLAS_API_TOKEN", "s3cret-value")
    assert pc_doctor.main(["--offline"]) in (0, 1)
    out = capsys.readouterr().out
    assert "packages" in out and "s3cret-value" not in out


def test_doctor_rejects_upstream_deathstar(monkeypatch, tmp_path, capsys):
    from petcare.scripts import pc_doctor
    (tmp_path / "DEATHSTAR.py").write_text("def setup(): pass\n")
    monkeypatch.setenv("PETCARE_DEATHSTAR", str(tmp_path))
    pc_doctor.check_archival(False)
    assert "awmann/DEATHSTAR" in capsys.readouterr().out


def test_end_to_end_on_spoc_files(tmp_path, spoc_files, catalog):
    """One candidate through the driver, on a MAST-named file, with no network."""
    out = tmp_path / "out"
    code = pc_main.main([
        "--tic", str(TIC), "--ra", "10", "--dec", "-40", "--age", "40",
        "--period", "4.2", "--t0", "1001.0", "--duration", "3.0", "--depth", "10000",
        "--lc-root", str(tmp_path), "--catalog", str(catalog), "--out", str(out),
        "--only", "fit", "depths", "odd_even", "duration", "--no-plot", "--quiet",
    ])
    assert code in (0, 1)
    rec = json.loads((out / f"TIC{TIC}" / "petcare.json").read_text())
    statuses = {k: v["status"] for k, v in rec["tests"].items()}
    assert "ERROR" not in statuses.values(), statuses
    fit = rec["tests"]["fit"]["metrics"]
    assert fit["depth_ppm"] == pytest.approx(10000, rel=0.3)
    assert fit["period"] == pytest.approx(4.2, abs=0.01)
    # the metrics reach both the record and the summary table; no HTML is written
    assert rec["vetting_metrics"]["metrics"]["depth_ppm"] == pytest.approx(fit["depth_ppm"])
    import pandas as pd
    row = pd.read_csv(out / "summary.csv").iloc[0]
    assert row["m_depth_ppm"] == pytest.approx(fit["depth_ppm"])
    assert not list(out.rglob("*.html"))


def test_leak_check_catches_what_it_should(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from check_leaks import scan_text
    allow = {"TIC 166527623"}
    kinds = lambda text, deny=frozenset(): {h[1] for h in scan_text(text, allow, set(deny))}
    n = "9" * 6   # built at run time, so the check does not flag this file
    assert kinds(f"see TIC {n}") == {"TIC"}
    assert kinds(f"run --tic {n}") == {"tic flag"}
    assert kinds("TIC 166527623 is HIP 67522") == set()
    assert kinds("/" + "Users/someone/data") == {"machine path"}
    assert kinds("a bare 999999 here", {"999999"}) == {"denylisted number"}
    assert kinds("a period of 999999.5") == set()

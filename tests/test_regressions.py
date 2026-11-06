"""Regression tests for bugs found in the 2026-11 audit and since fixed.

When a new bug is found and not yet fixed, pin it here as a strict xfail: the fix
then makes the suite fail until the marker comes off."""

import importlib
import json
from pathlib import Path

import pytest
from conftest import TIC

from petcare import cli, driver
from petcare.scripts import petcare as pc_main


def test_negative_depth_does_not_kill_the_test():
    parser = cli.base_parser("x", "x")
    argv = driver._argv({"tic": 1}, {"period": 3.0, "t0": 1.0, "depth": -5e-08},
                        driver.DriverOptions(), False)
    parser.parse_args(argv)


def test_gaia_id_survives_the_stellar_reader():
    from petcare import stellar
    assert stellar._int("6387058411482257536") == 6387058411482257536


def test_all_numeric_candidate_table_runs(tmp_path):
    table = tmp_path / "c.csv"
    table.write_text("tic,period,t0,duration_h\n123456,3.0,1001.0,2.0\n")
    code = pc_main.main(["--candidates", str(table), "--only", "stellar", "--no-plot",
                         "--quiet", "--out", str(tmp_path / "o")])
    assert code in (0, 1)


def test_dilution_note_does_not_ask_for_a_second_correction(spoc_files, tmp_path, catalog):
    out = tmp_path / "o"
    pc_main.main(["--tic", str(TIC), "--ra", "10", "--dec", "-40", "--period", "4.2",
                  "--t0", "1001.0", "--duration", "3.0", "--lc", *[str(p) for p in spoc_files],
                  "--catalog", str(catalog), "--only", "odd_even", "--no-plot",
                  "--quiet", "--out", str(out)])
    rec = json.loads((out / f"TIC{TIC}" / "odd_even.json").read_text())
    assert rec["dilution"]["applied"] is True


def test_one_bad_row_does_not_end_a_batch(tmp_path, monkeypatch):
    # a test whose parser calls sys.exit must become an ERROR record, not end the batch
    def exits(*a, **k):
        raise SystemExit(2)
    monkeypatch.setattr(driver, "_run_one", exits)
    opts = driver.DriverOptions(out=str(tmp_path), only=("stellar",), no_plot=True,
                                quiet=True)
    rec = driver.run_candidate({"tic": 1, "ra": 1.0, "dec": 1.0},
                               {"period": 3.0, "t0": 1.0, "depth": 500.0}, opts)
    assert rec["tests"]["stellar"]["status"] == "ERROR"


SCRIPTS = sorted(p.stem for p in (Path(__file__).parents[1] / "petcare" / "scripts").glob("*.py")
                 if p.stem != "__init__")


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_script_prints_help(name, capsys):
    # argparse formats help with %, so a bare "10-25%" crashed pc_fit --help
    mod = importlib.import_module(f"petcare.scripts.{name}")
    if not hasattr(mod, "main"):
        pytest.skip("no main()")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out

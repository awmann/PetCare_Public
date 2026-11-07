"""Unit tests for the parts that are easy to get quietly wrong.

These are not tests of whether a candidate is a planet -- that is what the
calibration run is for.  They cover the bookkeeping underneath, where a
mistake produces a plausible number rather than an error: epoch numbering,
transit-overlap matching, the density inversion, the noise estimators, and
the schema's promise not to overwrite a human's disposition.

Every test here corresponds to something that actually went wrong, either in
this package during development or in the pipelines that preceded it.

    pytest tests/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from petcare import ephem, noise
from petcare.result import Candidate, Human, KnownSignal, Result, Status, Target
from petcare.stellar import (Star, a_over_rs, duration_range, duration_t14,
                             implied_b, quadratic_ld, reconcile_density, rho_circ)


# --------------------------------------------------------------------------
# epoch numbering


def test_epoch_number_rounds_not_floors():
    """A transit centred on a cycle boundary must be one epoch, not two.

    With floor((t - t0)/P) the cadences just before mid-transit land in one
    epoch and those just after land in the next, so a per-epoch depth test
    sees two half-covered events and reports an inconsistency that is not
    there.
    """
    period, t0 = 5.0, 100.0
    t = np.array([104.9, 104.95, 105.0, 105.05, 105.1])
    assert np.all(ephem.epoch_number(t, period, t0) == 1)
    # the failure mode, for contrast
    assert len(np.unique(np.floor((t - t0) / period).astype(int))) == 2


def test_phase_is_signed_and_centred():
    period, t0 = 3.0, 10.0
    assert ephem.phase(10.0, period, t0) == pytest.approx(0.0)
    assert ephem.phase(11.4, period, t0) == pytest.approx(1.4)
    assert ephem.phase(8.6, period, t0) == pytest.approx(-1.4)
    # wraps rather than growing without bound
    assert abs(ephem.phase(1000.3, period, t0)) <= period / 2


# --------------------------------------------------------------------------
# matching known signals: overlap, never period proximity


def test_match_known_recognises_a_half_period_harmonic():
    """A candidate at P/2 of a known planet is the same object.

    This is the failure that lost 23% of the known planets in the 2026
    search: the search proposed P/2, a period-proximity test did not
    recognise it as the known planet, the known planet was masked at its own
    period anyway, and the candidate was subtracted from its own light curve.
    """
    known = [KnownSignal(label="b", period=8.0, t0_btjd=100.0, duration_h=3.0)]
    m = ephem.match_known(4.0, 100.0, 3.0, known, 90.0, 200.0)
    assert m is not None
    assert m["label"] == "b"
    assert "1/2" in m["relation"]


def test_match_known_recognises_double_period():
    known = [KnownSignal(label="b", period=4.0, t0_btjd=100.0, duration_h=3.0)]
    m = ephem.match_known(8.0, 100.0, 3.0, known, 90.0, 200.0)
    assert m is not None and m["label"] == "b"


def test_match_known_rejects_an_unrelated_period():
    known = [KnownSignal(label="b", period=8.0, t0_btjd=100.0, duration_h=3.0)]
    assert ephem.match_known(7.3, 103.7, 3.0, known, 90.0, 200.0) is None


def test_match_known_rejects_the_same_period_at_the_wrong_epoch():
    """Same period, antiphase: not the same object."""
    known = [KnownSignal(label="b", period=8.0, t0_btjd=100.0, duration_h=3.0)]
    assert ephem.match_known(8.0, 104.0, 3.0, known, 90.0, 200.0) is None


def test_mask_known_is_wider_than_the_transit():
    """The default mask is 2x T14, not 1x.

    Masking exactly one duration leaves the detrending filter's ringing just
    outside the mask, and that ringing was itself detected as three separate
    5-6 sigma candidates.
    """
    t = np.linspace(99.0, 101.0, 2001)
    known = [KnownSignal(label="b", period=100.0, t0_btjd=100.0, duration_h=2.4)]
    m = ephem.mask_known_transits(t, known)
    covered = t[m].max() - t[m].min()
    assert covered == pytest.approx(2 * 2.4 / 24.0, rel=0.05)


# --------------------------------------------------------------------------
# coverage


class _FakeLC:
    """The minimum interface epoch_coverage needs."""

    def __init__(self, t, cadence_s, sectors):
        self.t = t
        self.cadence_s = np.full(t.size, cadence_s, float)
        self.sector_id = np.full(t.size, sectors[0][0], int)
        self.sectors = [type("S", (), {"sector": s, "t_start": a, "t_stop": b})()
                        for s, a, b in sectors]
        self.exp_time_days = cadence_s / 86400.0


def test_coverage_uses_per_point_cadence():
    """A 120 s sector must not be scored against a 20 s global median.

    HIP 67522 has both; using the global median discarded two of its five
    sectors as 17%-covered.
    """
    cad = 120.0
    t = np.arange(0.0, 10.0, cad / 86400.0)
    lc = _FakeLC(t, cad, [(11, t.min(), t.max())])
    cov = ephem.epoch_coverage(lc, period=5.0, t0=2.5, duration_days=0.2)
    usable = [c for c in cov if c.usable]
    assert usable, "a fully sampled transit must be usable"
    assert usable[0].coverage == pytest.approx(1.0, abs=0.05)


def test_coverage_marks_a_partial_transit():
    cad = 120.0
    t = np.arange(0.0, 10.0, cad / 86400.0)
    # remove the second half of the transit at t = 2.5
    t = t[(t < 2.5) | (t > 2.62)]
    lc = _FakeLC(t, cad, [(11, t.min(), t.max())])
    cov = ephem.epoch_coverage(lc, period=5.0, t0=2.5, duration_days=0.2)
    c = next(c for c in cov if c.epoch == 0)
    assert c.coverage < 0.75
    assert not c.egress_covered


# --------------------------------------------------------------------------
# stellar geometry


def test_duration_inverts_consistently():
    """duration_t14 and implied_b must be inverses of one another."""
    period, rho, k = 8.0, 1.5, 0.05
    a = a_over_rs(period, rho)
    for b_true in (0.0, 0.3, 0.6, 0.85):
        t14 = duration_t14(period, a, k, b=b_true)
        b_back = implied_b(period, rho, t14, k)
        assert b_back == pytest.approx(b_true, abs=1e-3)


def test_implied_b_is_none_when_the_transit_is_too_long():
    """A transit longer than a central crossing has no b that explains it."""
    period, rho, k = 8.0, 1.5, 0.05
    _, t_max = duration_range(period, rho, k)
    assert implied_b(period, rho, t_max * 1.5, k) is None


def test_reconcile_density_is_symmetric_in_the_right_way():
    """Required eccentricity depends only on |log ratio|, not its sign."""
    up = reconcile_density(1.0, 2.0)
    down = reconcile_density(2.0, 1.0)
    assert up["ecc_required"] == pytest.approx(down["ecc_required"], rel=1e-6)
    # a larger star lengthens the transit, so a transit that is too long
    # asks for a larger radius
    assert reconcile_density(1.0, 0.5)["radius_error_required"] > 0


def test_rho_circ_round_trips_a_over_rs():
    for period in (1.0, 8.0, 40.0):
        for rho in (0.3, 1.4, 5.0):
            assert rho_circ(period, a_over_rs(period, rho)) == pytest.approx(rho)


def test_density_prefers_mass_and_radius_over_a_catalogue_column():
    """Two derivations of one quantity must not be mixed.

    The TIC's rho is itself derived from M and R; carrying both invites the
    kind of disagreement that had two depth estimators differing by a median
    factor of 0.68 on this project.
    """
    s = Star(mass=1.0, radius=1.0, rho=99.0)
    rho, _ = s.density()
    assert rho == pytest.approx(1.408, rel=1e-3)


def test_density_falls_back_to_the_catalogue_when_it_has_to():
    s = Star(rho=2.5)
    assert s.density()[0] == pytest.approx(2.5)


def test_limb_darkening_has_a_labelled_default():
    u1, u2, src = quadratic_ld(None)
    assert 0 < u1 < 1 and 0 < u2 < 1
    assert "default" in src
    _, _, src2 = quadratic_ld(5500.0)
    assert "Claret" in src2


# --------------------------------------------------------------------------
# noise


def test_point_to_point_ignores_smooth_variability():
    """Differencing must remove a trend, not measure it.

    This is why the estimator is MAD(diff)/sqrt(2) and not the standard
    deviation: on a star with 3% spot modulation the standard deviation is
    the star, not the noise.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 5000)
    white = 300e-6
    f = 1.0 + 0.03 * np.sin(2 * np.pi * t / 1.4) + rng.normal(0, white, t.size)
    measured = noise.point_to_point_ppm(f)
    assert measured == pytest.approx(white * 1e6, rel=0.15)


def test_red_noise_factor_is_one_for_white_noise():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, 20000)
    f = 1.0 + rng.normal(0, 500e-6, t.size)
    rn = noise.red_noise_factor(t, f, bin_days=0.2)
    assert rn["red_noise_factor"] == pytest.approx(1.0, abs=0.25)


def test_red_noise_factor_exceeds_one_for_correlated_noise():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 20, 20000)
    slow = np.cumsum(rng.normal(0, 3e-5, t.size))
    f = 1.0 + slow + rng.normal(0, 200e-6, t.size)
    rn = noise.red_noise_factor(t, f, bin_days=0.2)
    assert rn["red_noise_factor"] > 1.5


def test_depth_uncertainty_includes_the_baseline_term():
    """The out-of-transit term is routinely dropped and matters."""
    a = noise.depth_uncertainty_ppm(100, 100, 1000.0)
    b = noise.depth_uncertainty_ppm(100, 10_000_000, 1000.0)
    assert a > b * 1.3


# --------------------------------------------------------------------------
# the result schema


def _result(**kw):
    r = Result(test="demo", version="0.1.0",
               target=Target(tic=12345), candidate=Candidate(period=3.0, index=0))
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_result_json_has_no_nan(tmp_path):
    """NaN is not valid JSON and breaks every non-Python reader."""
    r = _result()
    r.metrics = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": np.nan}}
    r.passed("fine")
    path = r.write(tmp_path)
    payload = json.loads(path.read_text())    # would raise on a bare NaN
    assert payload["metrics"]["a"] is None
    assert payload["metrics"]["b"][1] is None
    assert payload["metrics"]["c"]["d"] is None


def test_rerun_preserves_a_human_disposition(tmp_path):
    """Most of the value in the last vetting round was the written notes."""
    first = _result()
    first.human = Human(disposition="probable planet", note="looks clean",
                        by="reviewer", date="2026-09-21")
    first.passed("ok")
    first.write(tmp_path)

    second = _result()
    second.flag("something changed")
    path = second.write(tmp_path)

    payload = json.loads(path.read_text())
    assert payload["human"]["disposition"] == "probable planet"
    assert payload["human"]["by"] == "reviewer"
    assert payload["status"] == "FLAG"
    assert payload["previous"]["status"] == "PASS"


def test_error_records_the_traceback():
    r = _result()
    try:
        raise ValueError("deliberate")
    except ValueError as exc:
        r.error(exc)
    assert r.status == Status.ERROR
    assert "deliberate" in r.reason
    assert "ValueError" in (r.traceback or "")


def test_dilution_is_recorded_not_applied():
    r = _result()
    r.set_dilution(factor=0.8, source="tessquicklook", applied=False)
    d = r.to_dict()["dilution"]
    assert d["factor"] == 0.8
    assert d["applied"] is False
    assert d["source"] == "tessquicklook"


def test_filename_separates_candidates_on_one_star():
    a = Result(test="fit", version="0", target=Target(tic=1),
               candidate=Candidate(index=0))
    b = Result(test="fit", version="0", target=Target(tic=1),
               candidate=Candidate(index=1))
    assert a.filename() != b.filename()


def test_target_slug_is_filesystem_safe():
    assert Target(tic=166527623).slug == "TIC166527623"
    assert Target(gaia_dr3=123).slug == "GDR3123"
    assert "/" not in Target(name="HD 1/2").slug


# --------------------------------------------------------------------------
# follow-up archives


def test_required_depth_on_companion_inverts_the_flux_ratio():
    """An equal-brightness companion must eclipse twice as deep as observed.

    Both stars contribute half the blended light, so a 1% observed dip needs a
    2% eclipse on the companion.  Getting the ``1 + 1/ratio`` the wrong way up
    gives 0.5% and turns every companion into a viable blend.
    """
    from petcare.followup import required_depth_on_companion

    assert required_depth_on_companion(10_000.0, 0.0) == pytest.approx(0.02)

    # 5 magnitudes fainter contributes 1% of the light, so a 1% dip needs a
    # 101% eclipse: impossible, and the companion is excluded.
    deep = required_depth_on_companion(10_000.0, 5.0)
    assert deep > 1.0
    assert deep == pytest.approx(0.01 * (1.0 + 10.0 ** 2.0), rel=1e-6)

    assert required_depth_on_companion(1000.0, None) is None


def test_required_depth_rises_with_companion_faintness():
    """Monotonic in dmag: a fainter companion always needs a deeper eclipse."""
    from petcare.followup import required_depth_on_companion

    depths = [required_depth_on_companion(2000.0, d) for d in (0.0, 1.0, 3.0, 6.0)]
    assert depths == sorted(depths)


def test_iso_to_btjd_matches_a_known_epoch():
    """BTJD 0 is BJD 2457000, which is 2014-12-08 12:00 UTC.

    An hour of error here would put an archival frame in the wrong transit
    window, so the zero point is pinned to a date rather than trusted.
    """
    from petcare.followup import iso_to_btjd

    assert iso_to_btjd("2014-12-08T12:00:00Z") == pytest.approx(0.0, abs=1e-6)
    # ...and a day later is BTJD 1.
    assert iso_to_btjd("2014-12-09T12:00:00Z") == pytest.approx(1.0, abs=1e-6)
    assert iso_to_btjd(None) is None
    assert iso_to_btjd("not a date") is None


def test_iso_to_btjd_accepts_the_formats_both_archives_emit():
    """LCO sends ISO with a Z, ExoFOP sends a space-separated stamp."""
    from petcare.followup import iso_to_btjd

    a = iso_to_btjd("2023-02-05T09:30:00Z")
    b = iso_to_btjd("2023-02-05 09:30:00")
    assert a is not None and b is not None
    assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------
# the centroid's coherent-offset refutation


def test_looks_targeted_rejects_serendipitous_field_coverage():
    """A cone search matches footprints, not pointings.

    One star came back with 100 public frames over 10 nights "inside a
    predicted transit".  Every one was pointed at a pulsar or a transient
    elsewhere in the field, six frames a night, with our star sitting 5.4
    arcmin off the field centre as an ordinary field object.  Reporting that as
    archival follow-up sent somebody looking for data that does not exist.

    Cadence is the discriminator: a transit series is tens to hundreds of
    frames in a night, serendipitous coverage is one to six.
    """
    from petcare.scripts.pc_followup import _looks_targeted

    def night(n, prop="PROP2021A-001"):
        return {"2021-04-25": [{"obstype": "EXPOSE", "proposal": prop}
                               for _ in range(n)]}

    rec = {"date": "2021-04-25T23:11:01Z", "obstype": "EXPOSE",
           "proposal": "PROP2021A-001", "offset_arcmin": 5.38}
    assert _looks_targeted(rec, night(6)) is False      # the real case
    assert _looks_targeted(rec, night(100)) is True     # same field, real series


def test_looks_targeted_keeps_real_tfop_observations_off_centre():
    """The pointing tolerance must not be tight.

    LCO's own TFOP transit observations routinely place the target 3-6 arcmin
    off the field centre.  An earlier 2 arcmin cut rejected TOI-837 (4.96') and
    TOI-22 (4.79'), both of which ExoFOP records as observed -- while a
    wide-field mosaic of an extended object leaves our star 11-37 arcmin out
    and must still be rejected.
    """
    from petcare.scripts.pc_followup import _looks_targeted

    dense = {"2023-02-05": [{"obstype": "EXPOSE", "proposal": "KEY2020B-005"}
                            for _ in range(100)]}
    base = {"date": "2023-02-05T09:00:00Z", "obstype": "EXPOSE",
            "proposal": "KEY2020B-005"}
    assert _looks_targeted(dict(base, offset_arcmin=4.96), dense) is True   # TOI-837
    assert _looks_targeted(dict(base, offset_arcmin=4.79), dense) is True   # TOI-22
    assert _looks_targeted(dict(base, offset_arcmin=15.03), dense) is False  # mosaic
    assert _looks_targeted(dict(base, offset_arcmin=36.27), dense) is False  # mosaic


def test_pointing_offset_unwraps_ra_near_zero():
    """The archive wraps footprint longitudes to [-180, 180].

    A field at RA 244 deg comes back as -116, so the polygon must be un-wrapped
    against the target's own RA or the offset lands half a sky away.
    """
    from petcare.scripts.pc_followup import _pointing_offset

    wrapped = {"coordinates": [[[-115.6, -50.39], [-115.6, -50.39],
                                [-115.6, -50.39], [-115.6, -50.39]]]}
    off = _pointing_offset(wrapped, 244.40, -50.39)
    assert off is not None
    assert off < 1.0          # 244.4 and -115.6 are the same place
    assert _pointing_offset(None, 244.40, -50.39) is None


# --------------------------------------------------------------------------
# pc_phasecurve: a transit, or the star's own modulation at that period?


def _synthetic(period=1.4, prot=0.35, amp=0.01, depth=0.0, dur=0.05,
               days=60.0, cad=600.0, noise=3e-4, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, cad / 86400.0)
    f = 1 + amp * np.sin(2 * np.pi * t / prot) + rng.normal(0, noise, t.size)
    ph = ((t / period) + 0.5) % 1.0 - 0.5
    f -= depth * (np.abs(ph * period) < dur / 2)
    return t, f


def test_phasecurve_explains_a_rotation_trough_at_a_harmonic():
    """At P = 4 Prot the rotation folds coherently, and its trough at phase 0
    looks like a deep dip against the median.  A smooth Fourier model fitted
    outside the window follows the trough down, leaving nothing.  In the case
    that prompted this, every other test passed the candidate."""
    from petcare.scripts.pc_phasecurve import _measure

    prot, period, dur = 0.35, 1.4, 0.05
    t, f = _synthetic(period=period, prot=prot, amp=0.01, depth=0.0, dur=dur)
    # put a rotation trough at phase 0
    t0 = prot * 0.75
    m = _measure(t, f, period, t0, dur, order=int(period / (3 * dur)))
    assert m["raw_dip"] > 5 * m["sigma"]                 # it looks like a dip
    assert m["explained"] is not None and m["explained"] > 0.8
    assert abs(m["resid_dip"]) < 3 * m["sigma"]          # ...and it is not one


def test_phasecurve_keeps_a_sharp_transit_on_top_of_rotation():
    """The same star with a real transit at the same phase: the model still
    follows the trough, but cannot represent a feature as short as the
    transit, so the transit survives at close to its injected depth."""
    from petcare.scripts.pc_phasecurve import _measure

    prot, period, dur, depth = 0.35, 1.4, 0.05, 0.004
    t, f = _synthetic(period=period, prot=prot, amp=0.01, depth=depth, dur=dur)
    t0 = 0.0
    m = _measure(t, f, period, t0, dur, order=int(period / (3 * dur)))
    assert m["resid_dip"] > 5 * m["sigma"]
    assert 0.6 * depth < m["resid_dip"] < 1.4 * depth


def test_phasecurve_flat_star_transit_is_entirely_sharp():
    from petcare.scripts.pc_phasecurve import _measure

    period, dur, depth = 3.7, 0.1, 0.002
    t, f = _synthetic(period=period, prot=1.0, amp=0.0, depth=depth, dur=dur)
    m = _measure(t, f, period, 0.0, dur, order=int(period / (3 * dur)))
    assert m["explained"] is not None and abs(m["explained"]) < 0.3
    assert 0.7 * depth < m["resid_dip"] < 1.3 * depth


def test_siblings_catch_a_dip_that_repeats_at_a_rotation_harmonic():
    """A candidate at P = 6.0004 x Prot.  The raw light curve repeats every
    P/6, so the "transit" at phase 0 is one of six identical rotation minima.
    A transit happens once per orbit; this happens six times."""
    from petcare.scripts.pc_phasecurve import _siblings

    prot, n, dur = 0.70, 6, 0.12
    period = n * prot
    t = np.arange(0, 120, 300 / 86400.0)
    rng = np.random.default_rng(4)
    f = 1 + 0.03 * np.cos(2 * np.pi * t / prot) + rng.normal(0, 1e-3, t.size)
    s = _siblings(t, f, period, t0=prot / 2, dur=dur)
    assert s is not None and s["n"] in (2, 3, 6)
    assert s["significance"] > 5
    assert abs(s["sibling_dip"] / s["dip0"] - 1) < 0.2      # it repeats


def test_siblings_leave_a_transit_off_any_harmonic_alone():
    """A transit at a period unrelated to the rotation: siblings land on random
    rotation phases and average to nothing, whatever n is tried."""
    from petcare.scripts.pc_phasecurve import _siblings

    prot, period, dur, depth = 0.70, 3.913, 0.12, 0.004
    t = np.arange(0, 120, 300 / 86400.0)
    rng = np.random.default_rng(5)
    f = 1 + 0.01 * np.cos(2 * np.pi * t / prot) + rng.normal(0, 1e-3, t.size)
    ph = ((t / period) + 0.5) % 1.0 - 0.5
    f -= depth * (np.abs(ph * period) < dur / 2)
    s = _siblings(t, f, period, t0=0.0, dur=dur)
    assert s is None or s["significance"] < 5 or s["sibling_dip"] < 0.5 * s["dip0"]


def test_shapetime_extra_scatter_is_zero_when_values_agree():
    """The headline statistic: the fractional scatter that has to be added,
    beyond errors and floor, for the groups to agree."""
    from petcare.scripts.pc_shapetime import _extra

    assert _extra([1.0, 1.02, 0.98], [0.05, 0.05, 0.05], 0.10) == 0.0
    big = _extra([0.5, 1.0, 1.5], [0.02, 0.02, 0.02], 0.10)
    assert 0.3 < big < 0.6


def test_gaia_id_survives_catalogue_lookup():
    from petcare.stellar import _int, _pick_id
    assert _int(6387058411482257536) == 6387058411482257536
    assert _int("6387058411482257536") == 6387058411482257536
    assert _pick_id({"gaia_dr3_source_id": 6387058411482257536}, "gaia_dr3") == 6387058411482257536


def test_negated_off_target_note_is_not_adverse():
    from petcare.scripts.pc_followup import _off_target_phrase as f
    assert f("no neb on any star") is None
    assert f("not off target; cleared") is None
    assert f("neb on tic 123") == "neb on"
    assert f("event is not on target") == "not on target"
    assert f("no ttv. neb on the 5 arcsec neighbour") == "neb on"


def test_candidate_table_edge_rows(tmp_path):
    """Float TIC columns, blank cells, period_d, --start: each used to crash or mis-mask."""
    from types import SimpleNamespace
    from petcare.scripts import petcare as pc
    tab = tmp_path / "c.csv"
    tab.write_text("tic,period_d,t0,duration_h,depth_ppm,name\n"
                   "100.0,2.0,1500.0,2.0,1000,\n"
                   "100.0,3.0,1501.0,2.0,1000,b\n"
                   "200.5,4.0,1502.0,2.0,1000,c\n")
    args = SimpleNamespace(candidates=str(tab), start=1, limit=None)
    jobs = pc._jobs(args)
    assert jobs[0][0]["tic"] == 100 and jobs[0][1]["index"] == 1
    assert [k.split(":")[1] for k in jobs[0][1]["known"]] == ["2.0"]
    assert jobs[1][1]["known"] == []


def test_comma_csv_columns_by_name(tmp_path):
    from petcare.io import _read_plain_csv
    p = tmp_path / "lc.csv"
    p.write_text("time,quality,flux,flux_err\n" + "".join(f"{1500 + i * .1},0,1.0,0.001\n" for i in range(5)))
    out = _read_plain_csv(str(p), "flux")
    assert np.allclose(out[0]["f"], 1.0)


def test_followup_note_is_only_a_flag():
    src = Path(__file__).parent.parent / "petcare" / "scripts" / "pc_followup.py"
    assert "r.fail(" not in src.read_text()


def test_epoch_metrics_single_event_and_kfold():
    from petcare import metrics
    # four equal transits: no event dominates, dropping one costs sqrt(4/3) in S/N
    even = metrics.epoch_metrics([{"depth_ppm": 1000, "depth_err_ppm": 250}] * 4)
    assert abs(even["max_event_share"] - 0.25) < 1e-9
    assert abs(even["snr"] / even["kfold_snr_min"] - (4 / 3) ** 0.5) < 1e-9
    # one deep event among noise: it carries the signal, and removing it kills it
    one = metrics.epoch_metrics([{"depth_ppm": 3000, "depth_err_ppm": 250}]
                                + [{"depth_ppm": 0, "depth_err_ppm": 250}] * 3)
    assert one["max_event_share"] == 1.0
    assert one["kfold_snr_min"] == 0.0 and one["drop1_snr"] == 0.0
    assert metrics.epoch_metrics([]) == {"n_epochs": 0}


def test_vetting_metrics_from_an_empty_record():
    from petcare import metrics
    out = metrics.collect({"tests": {}})
    assert out["version"] == metrics.VERSION
    assert out["metrics"]["depth_ppm"] is None and out["metrics"]["MS4"] is None


def test_jointgp_restrained_fit_recovers_a_transit_on_a_quiet_star():
    from petcare.fitting import GPSpec
    from petcare.scripts import pc_jointgp as jg
    rng = np.random.default_rng(3)
    t14 = 0.12
    tw = np.arange(-0.48, 0.48, 0.004)
    sw = np.clip((0.5 * t14 - np.abs(tw)) / (0.2 * t14), 0, 1)
    ew = np.full(tw.size, 5e-4)
    # a weak, slow Matern: the restrained fit should return the injected depth
    spec = GPSpec(kind="matern", names=("log_sigma", "log_rho"),
                  values=np.log([2e-4, 2.0]), success=True)
    yw = -2e-3 * sw + rng.normal(0, 5e-4, tw.size)
    d, sd, _ = jg._restrained(spec, tw, yw, ew, sw, tw, t14)
    assert abs(d - 2e-3) < 3 * sd and d / sd > 10
    # and no dip gives a depth consistent with zero
    d0, sd0, _ = jg._restrained(spec, tw, yw + 2e-3 * sw, ew, sw, tw, t14)
    assert abs(d0) < 3 * sd0


def test_crossmatch_per_transit_depth_ignores_rotation():
    from petcare.scripts import pc_crossmatch as xm
    rng = np.random.default_rng(5)
    P, t0, dur = 7.3, 1000.2, 0.15
    t = np.arange(1000.0, 1060.0, 0.01)
    # 2% rotation at 2.9 d dwarfs a 1000 ppm transit; a pooled fold would not survive it
    f = 1 + 0.02 * np.sin(2 * np.pi * t / 2.9) + rng.normal(0, 3e-4, t.size)
    x = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    f[np.abs(x) < dur / 2] -= 1e-3
    m = xm._measure(t, f, P, t0, dur)
    assert abs(m["depth_ppm"] - 1000) < 3 * m["depth_err_ppm"] and m["depth_err_ppm"] < 300

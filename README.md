# PetCare

**Vetting transiting planet candidates around young stars.**

PetCare is a set of small, single-purpose tools. Each one answers exactly one
question about one candidate, writes a JSON record and a figure, and exits.
A driver runs the ones you ask for and merges their answers into a single
record per candidate.

It is built for the regime where the usual vetting tools quietly break: stars
of 0–50 Myr, rotating in under a day, varying by several percent, flaring
constantly, and frequently in binaries. Most of the design decisions in here
exist because a conventional choice failed on exactly such a star.

```
pc-fit --tic 410214986 --period 8.138268 --t0 1332.30991 --duration 3.4
```

---

## What it answers, and what it does not

PetCare answers **"is this signal an astrophysical transit of this star?"**

It does not answer **"is this signal significant?"** Detection thresholds,
null light curves and false-alarm rates belong to the search pipeline.
PetCare assumes it is handed something that already cleared a significance
bar, and asks only whether it looks like a planet.

It also does not decide for you. **There is no overall score.** The driver
reports the per-test statuses side by side, because *which* test objected is
the information, and a scalar hides it.

---

## Install

```bash
git clone https://github.com/awmann/PetCare_Public.git
cd PetCare_Public
pip install -e ".[all]"
pc-doctor
```

Python 3.10+. [`docs/setup.md`](docs/setup.md) covers getting light curves
(including [tessquicklook](https://github.com/awmann/tessquicklook)), rotation
periods, environment variables and credentials. **PetCare assumes your light
curve is already corrected for dilution.**

---

## Quick start

**One test, one candidate.** Every test takes the same target and candidate
vocabulary, and each runs standalone:

```bash
pc-fit      --tic 410214986 --period 8.138268 --t0 1332.30991 --duration 3.4 --out out/
pc-oddeven  --tic 410214986 --use-fit --out out/
pc-centroid --tic 410214986 --use-fit --out out/
```

`--use-fit` reads the ephemeris that `pc-fit` refined out of its JSON, so the
later tests work at the measured period rather than the input one. No test
imports another.

**Everything, for one candidate:**

```bash
petcare --tic 410214986 --period 8.138268 --t0 1332.30991 --duration 3.4 \
        --tiers 0 1 --out out/
```

**A table of candidates:**

```bash
petcare --candidates candidates.csv --tiers 0 1 2 --out out/
```

which writes `out/<target>/petcare.json`, a one-row-per-candidate
`summary.csv`, and a self-contained HTML report per candidate with every
figure inline.

See [`Notebooks/`](Notebooks/) for worked examples with plots.

---

## The tests

Tiered by cost. Tiers 0 and 1 are cheap enough to run on everything.

### Tier 0 — catalogue only, no light curve needed

| tool | question |
|---|---|
| `pc-stellar` | what are this star's parameters, and how well known are they? |
| `pc-neighbours` | which nearby star could reproduce this depth if it were eclipsing? |
| `pc-astrometry` | does Gaia astrometry indicate an unresolved companion? |
| `pc-exofop` | does high-resolution imaging or spectroscopy already exist? |

### Tier 1 — light curve only

| tool | question |
|---|---|
| `pc-fit` | simultaneous variability + transit fit: refined P, t0, depth, T14, b, a/R★ |
| `pc-depths` | are the per-epoch depths mutually consistent, and does one epoch carry the detection? |
| `pc-oddeven` | do odd and even transits have the same depth? |
| `pc-secondary` | is there a secondary eclipse at *any* phase? ellipsoidal or beaming signal? |
| `pc-duration` | is the duration physical for this star's density? |
| `pc-shape` | is the ingress fraction planet-like, or V-shaped and grazing? |
| `pc-companion` | at this age, is the radius a planet, a brown dwarf, or a star? |
| `pc-rotation` | is the period a harmonic of the rotation period? |
| `pc-phasecurve` | is the dip a sharp transit, or the star's own modulation at that period — does it repeat where the star repeats? |
| `pc-shapetime` | does the transit keep its duration and depth over the years, as a planet's does? |
| `pc-flares` | is the fold carried by flare-adjacent cadences? |
| `pc-instrument` | does the signal track the spacecraft rather than the star? |

### Tier 2 — pixels

| tool | question |
|---|---|
| `pc-centroid` | does the flux deficit come from the target's pixels? |

### Tier 3 — external archives

| tool | question |
|---|---|
| `pc-archival` | is the transit present in ground-based archival photometry (via DEATHSTAR)? |
| `pc-crossmatch` | does an independent light curve of the same star measure the same depth? |
| `pc-followup` | what has already been observed, on ExoFOP and in the LCO archive? |

`pc-archival` is a thin driver: it calls
[DEATHSTAR](https://github.com/awmann/DEATHSTAR) (Ross et al. 2024) as a
subprocess rather than vendoring it, so DEATHSTAR stays at its own checkout,
in its own environment, and keeps its own citation. Point `--deathstar` (or
`$PETCARE_DEATHSTAR`) at that checkout and `--deathstar-python` at the
interpreter it was installed into.

### Cross-target — run once over a whole candidate list

| tool | question |
|---|---|
| `pc-dedup` | is the same signal appearing on several nearby stars? |
| `pc-known` | is this already a catalogued TOI, TCE, eclipsing binary or variable? |

`pc-dedup` matters more than its size suggests: one bright eclipsing binary
contaminates every star within a few arcminutes and produces an apparently
independent candidate on each. It is close to free and catches them wholesale,
before any pixel work.

---

## The result schema

Every test writes `out/<target>/<test>.json`:

```json
{
  "test": "odd_even",
  "version": "0.1.0",
  "target": {"tic": 410214986, "gaia_dr3": "6387058411482257536", "name": "DS Tuc A"},
  "candidate": {"period": 8.138268, "t0_btjd": 1332.30991, "duration_h": 3.4,
                "depth_ppm": 2936, "index": 0},
  "status": "PASS",
  "reason": "one sentence a human can read",
  "metrics": {"odd_depth_ppm": 3059, "even_depth_ppm": 2841,
              "difference_sigma": 0.6},
  "noise": {"white_ppm": 1496, "estimator": "1.4826*MAD(diff)/sqrt(2)"},
  "self_check": {"passed": true, "detail": "5 odd and 4 even transits covered"},
  "plots": ["out/DS_Tuc_A/odd_even.png"],
  "provenance": {"lightcurve": "...", "n_points": 18342,
                 "run_utc": "2026-09-21T18:03:11Z", "git_sha": "abc1234"}
}
```

`status`:

- **`PASS`** — ran, self-check passed, candidate looks fine.
- **`FLAG`** — ran cleanly; something here needs a human.
- **`FAIL`** — ran cleanly and the candidate is refuted *by this test alone*.
- **`INCONCLUSIVE`** — could not be applied: too few epochs, no pixels, or the
  self-check failed. Not a pass and not a fail.
- **`ERROR`** — crashed. The traceback is recorded, never swallowed.

Four fields are load-bearing and easy to omit:

- **`self_check`** — every test proves it measured what it claims before it is
  allowed to report. A centroid test that cannot recover the light-curve depth
  from its own difference image returns `INCONCLUSIVE`, not a 3.6σ offset.
- **`noise`** — a null result means nothing without the level it rules out at.
- **`provenance`** — which light curve, how many points, which commit.
- **`human`** — a block for your own disposition and notes. Re-running a test
  preserves it and files the machine's previous answer under `previous`.

---

## Design rules

Each of these is here because the obvious alternative failed on a real star in
this sample.

1. **Fit the transit and the stellar variability simultaneously.** Never
   flatten first. On a star rotating in a day with a 3% amplitude, a filter
   wide enough to leave the transit alone leaves the variability too, and a
   filter narrow enough to remove the variability removes the transit.
2. **Mask the transit when fitting variability hyperparameters, and hold them
   fixed in the joint fit.** The hyperparameter likelihood on these stars is
   nearly flat along an (amplitude, quality-factor) degeneracy, and a joint fit
   free to move along it buys flexibility for free and pays for it with the
   transit.
3. **A global GP is not always the right variability model.** On the most
   active stars the marginalised likelihood prefers *no transit* to a real one.
   `pc-fit` therefore also fits a local polynomial baseline per transit window
   simultaneously with the transit (the "notch" form) and, by default, keeps
   whichever model the two agree on. Both fits and the reason for the choice go
   into the record.
4. **Match known signals by transit overlap, never by period proximity.** A
   known planet at half or twice the candidate's period must still be
   recognised.
5. **Measure uncertainties, do not propagate them.** Per-epoch depth errors
   come from repeating the measurement at off-transit phases. Propagated white
   errors were 60× too small here, which made every candidate look internally
   inconsistent.
6. **Flag, do not veto, anything that rests on young-star parameters.**
   Rotation proximity as a veto previously deleted 25 signals including a
   published one. Odd/even at the textbook 3σ would delete 18% of the known
   planets in this sample — the two worst offenders are both real planets with
   spot crossings.
7. **Record dilution; do not silently apply it.** Say which convention the
   depth is in.
8. **Thresholds are calibrated, not chosen.** Every threshold in the source is
   annotated with the cost it was measured to carry, or marked `PROVISIONAL`
   if it has not been measured yet. See [`calibration/`](calibration/).

---

## Tests

```bash
pytest -q
```

Unit tests cover the schema, ephemeris arithmetic, the fitting primitives and
the light-curve loaders, using synthetic data only — no network, no files from
disk.

The scientific validation is separate and runs against known young planets
(HIP 67522, V1298 Tau, DS Tuc A, AU Mic and others), checking that each test
does not refute a real planet and that the recovered depth, duration and
radius agree with the literature. That sample is
`calibration/samples/known_young_planets.csv`, and `calibration/calibrate.py` measures each test's catch rate and cost
and scans its thresholds.

---

## Credits

PetCare is by Madyson Barber and Andrew Mann, with the paper in preparation
(Barber et al.). The code was written with substantial help from Claude
(Anthropic).

Many of these tests are existing transit-vetting methods adapted to young stars,
not new inventions, and those methods deserve the citation. They include ModShift
(Thompson et al. 2018), SHP from LEO-vetter (Kunimoto et al. 2025), and LATTE's
pixel-level plots (Eisner et al. 2020). PetCare also relies on DEATHSTAR (Ross et al.
2024), the Sonora Diamondback models (Morley et al. 2024), and the TIC, Gaia, and
MAST archives.

---

## Status

Working and validated against known planets; thresholds still being
calibrated. Known limitations are listed in `calibration/README.md` and in the
module docstrings. `pc-archival` depends on an external DEATHSTAR checkout and
is exercised least.

Contributions and bug reports are welcome. If a test refutes a planet you know
to be real, that is the most useful bug report there is — please include the
JSON record.

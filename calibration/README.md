# Calibration

A test with an uncalibrated threshold is not finished, and a test that has never been run
on known planets and known false positives has no defensible threshold at all.
`calibrate.py` measures, for every test, what it costs on real planets and what it catches
on real false positives.

## The samples

**Known planets:** `samples/known_young_planets.csv` lists 45 published young transiting
planets on 31 hosts, all confirmed on the NASA Exoplanet Archive. The table is in the
format `petcare --candidates` reads, with the ephemerides we ran (not refreshed from the
archive; see below), plus radius, age and `truth = planet`.

**False positives:** bring your own. Any candidate table with a `truth` column will do.
Rows whose truth is `false_positive` or `likely_false_positive` are scored (change this
with `--fp-truth`); a `candidate` is carried but never scored, because it is not evidence
either way. Match rows to the run by `tic` and `label`, so two signals on one star can
carry different labels.

## Running it

```bash
petcare --candidates calibration/samples/known_young_planets.csv \
    --tiers 0 1 --out runs/knowns
petcare --candidates my_false_positives.csv --tiers 0 1 --out runs/fps
python calibration/calibrate.py \
    --knowns runs/knowns/summary.csv \
    --false-positives runs/fps/summary.csv \
    --fp-sample my_false_positives.csv \
    --out calibration/out
```

This writes three files to `--out`:

- `confusion_matrix.csv`: per test, the cost on known planets, the catch on false
  positives, and how many of each it could not decide;
- `threshold_scan.json`: for each continuous statistic, both distributions, the threshold
  the code uses now (read from the test module, so it cannot go stale), and the one that
  best separates the samples;
- `CALIBRATION.md`: the same, written out, with the sample size beside every number.

Light curves for the known planets are not shipped. `pc-extract` makes them (see
[`docs/setup.md`](../docs/setup.md)).

## Reading it honestly

**`cost` counts FLAG as an objection.** Most tests flag rather than fail on purpose; a
flag asks for a human and deletes nothing. A test with a high cost *and* a high catch may
be doing exactly its job.

**Small numbers are one object.** A cost of 0.05 on 20 planets is one planet.

**False-positive samples are thin and often eye-dispositioned.** A high catch is
encouraging. A low one is not proof of anything.

**A test that cannot separate the samples is not ready to ship.** The scan marks
`separable: false` where the best achievable separation is under 0.3; treat that test's
threshold as a placeholder however it is set.

**The sample is selected.** Every planet here was found by somebody's pipeline, so it is
easier to see than the planets nobody has found. A planet a search missed can fail a test
that every planet here passes. `interp` and `uniqueness` therefore check, before failing a
candidate, whether they could have seen a transit of its depth on that star (see the
HIP 67522 example).

**Archive ephemerides drift.** The archive's composite table can combine a period and an
epoch from different papers. PetCare tolerates small errors, but refine the ephemeris
(`pc-fit`, then `--use-fit`) before reading a timing-sensitive test.

## What we measured before PetCare

From the search that preceded this package, on 38 to 71 known young planets. Several
thresholds here are set directly from these numbers.

| statistic | cost on real planets |
|---|---|
| per-epoch jackknife at `surv > 0.4` | **0 of 38**, the only cut trusted outright |
| odd/even at 2 σ | 32% |
| odd/even at 3 σ | 18% |
| odd/even at 5 σ | 5% |
| global anti-transit null | 25% |
| local-period null | 31% |
| off-phase null (as a veto) | 12% |
| GP+transit Δln𝓛 > 2 | only 42% of real planets clear it |
| rotation proximity as a veto | removed 25 real-looking signals, one published |

The last two are why `pc_fit` reports the ratio of GP amplitude to depth rather than
cutting on Δln𝓛, and why `pc_rotation` flags and never fails.

# Running many candidates

Nothing in PetCare knows where your files are. Every test takes its light curves, stellar
catalogue and candidate list as arguments, or reads them from environment variables (the
full list is in [setup.md](setup.md#environment-variables)). For a large run, keep those
settings in a small script of your own, outside the repository, so the checkout stays
shareable and nobody inherits one machine's directory layout:

```bash
#!/bin/sh
export PETCARE_LC_ROOTS=/data/lightcurves          # colon-separated
export PETCARE_STELLAR_CATALOG=/data/stellar.parquet
export PETCARE_CACHE=/scratch/petcare_cache
petcare --candidates "$1" --tiers 0 1 --out /data/petcare_out
```

The repository's `.gitignore` ignores `local/`, `*_local.py`, `local_*.py` and `env.sh`, so
a copy kept in the checkout under one of those names will not be committed by accident.

## Getting it right at scale

**Run cheap tiers first.** Tiers 0 and 1 use only the light curve and catalogues. Run
tier 2 (pixels, cutouts, archives) on what survives; it downloads per star.

**Skip the plots.** Every test writes a PNG, and on a large run those dominate the disk.
Pass `--no-plot` for a bulk run, then re-run the candidates you want to read. The
`m_*` columns of `summary.csv` carry the vetting metrics for sorting and cutting.

**Keep outputs and caches off synced folders.** A large run rewrites its outputs and cache
constantly; cloud sync clients struggle with that, and some evict
files to free space mid-run.

**Resuming.** `--start N` skips the first N rows of the table; results are keyed on
(TIC, candidate index), so rows before it keep their numbering and are not overwritten.
`--only` and `--skip` re-run or leave out named tests, and a pass with `--only` merges into
the existing `petcare.json`.

**Look for repeated signals.** `pc-dedup` finds the same signal on several nearby stars,
which is usually one bright eclipsing binary contaminating its neighbours.

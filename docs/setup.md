# Setup

```bash
git clone https://github.com/awmann/PetCare_Public.git
cd PetCare_Public
pip install -e ".[all]"
pc-doctor
```

`pc-doctor` checks packages, paths, network services and credentials, and tells
you what each missing piece costs. Run it before anything else, and again if a
test starts returning `INCONCLUSIVE` everywhere.

Python 3.10 or newer. On a many-core machine set `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, and run candidates in parallel rather than threads
within a fit.

## Light curves

PetCare does not extract light curves itself. **Use tessquicklook, through
`pc-extract`:**

```bash
pc-extract --tic 410214986 --out lightcurves
petcare --tic 410214986 --period 8.138211 --t0 1332.30989 --duration 3.2 --lc-root lightcurves
```

[tessquicklook](https://github.com/awmann/tessquicklook) fits the systematics and
the stellar variability simultaneously, which is what preserves shallow transits
on rapidly rotating stars. `pc-extract` runs it with fixed settings and writes one
`.npz` per sector. Downloads can reach several GB per star; set
`$TESSQUICKLOOK_CACHE` to move them from `~/.tessquicklook`.

Other inputs work but give weaker results:

| input | what you lose |
|---|---|
| tessquicklook's own CSV | The quality mask from the systematics fit. Rejected cadences come back, which can change verdicts. |
| SPOC or QLP `.fits` from MAST | PDCSAP removes systematics before the variability is modelled, which costs precision on fast rotators. Use 120 s products: when a sector has several cadences, the one with the most points is kept. |
| Plain `time,flux[,flux_err]` text | Time in BTJD (BJD - 2457000). No mask, sector information or dilution. |

On DS Tuc A c these choices change the outcome; see
[`examples/ds_tuc`](../examples/ds_tuc/README.md).

Point PetCare at light curves with `--lc file1 file2 ...` for one star, or with
`--lc-root` or `$PETCARE_LC_ROOTS` to search a directory by TIC. Directories of
MAST downloads work with the files under their original names.

### Dilution

**PetCare assumes the light curve is already corrected for contaminating flux.**
tessquicklook's flux columns and SPOC's `PDCSAP_FLUX` both are, so do not divide
depths by the dilution factor again. Using `SAP_FLUX` or your own uncorrected
photometry, correct it before PetCare sees it. The dilution value in each record
(the fraction of aperture flux from the target, higher is better) is a measure of
how much to trust the correction: below about 0.3, small errors in it become large
errors in depth.

### Rotation periods

Several tests need a rotation period. In order of preference PetCare uses:

1. `--prot`, or a `prot` column in a candidate table;
2. a value recorded in the light curve;
3. a per-sector Lomb-Scargle on the light curve (`source: ls_measured`), the usual
   route for `pc-extract` output.

Option 3 is a quick estimate. It rejects periods within 6% of the 13.7 d TESS orbit
and requires three cycles in a sector, so it gives no period rather than a bad one
for slow rotators, and it can land on half the true period of a double-dip star.
If you know the period, give it.

## Environment variables

All are optional; each has a command-line equivalent or a default.

| variable | what it points at | default |
|---|---|---|
| `PETCARE_LC_ROOTS` | colon-separated light-curve directories | none |
| `PETCARE_STELLAR_CATALOG` | parquet or CSV of stellar parameters | TIC, via MAST |
| `PETCARE_CACHE` | cache of queries and cutouts | `$XDG_CACHE_HOME/petcare`, or `~/.cache/petcare` |
| `PETCARE_SONORA` | Sonora Diamondback evolution tables | `<cache>/models/diamondback` |
| `PETCARE_TARS` | parquet of `TICID`, `adopted_period` | not used |
| `PETCARE_DEATHSTAR` | DEATHSTAR checkout, for `pc-archival` | none |
| `PETCARE_DEATHSTAR_PYTHON` | interpreter in DEATHSTAR's environment | none |
| `PETCARE_ONESIDED_MASK` | `0` disables restoring transit bottoms clipped by tessquicklook's quality mask | on |

The cache can grow large: expect roughly 100-400 MB of cutouts per star for tier 2.
Keep it off file-syncing services, which do not cope with this much write churn.

## Credentials

**Tiers 0-2 need no accounts.** MAST, TESScut, Gaia, ExoFOP and the public LCO
archive are queried anonymously. Only `pc-archival` (tier 3) logs in to anything.

### ATLAS (`pc-archival`, southern targets)

The forced-photometry service has its own account, separate from any other ATLAS
login. Register at <https://fallingstar-data.com/forcedphot/>, then either set a
token (<https://fallingstar-data.com/forcedphot/apiguide/>):

```bash
export ATLAS_API_TOKEN=...
```

or your username and password:

```bash
export ATLAS_USERNAME=...
export ATLAS_PASSWORD=...
```

### ZTF (`pc-archival`, northern targets)

ZTF frames are fetched with `ztfquery`, using an
[IRSA account](https://irsa.ipac.caltech.edu/account/signon/login.do). Store the
login once, in DEATHSTAR's environment:

```bash
/path/to/deathstar/env/bin/python -c "import ztfquery; ztfquery.io.set_account('irsa')"
```

### Keeping credentials safe

Put exports in a file only you can read and never in this repository; `.env` and
`secrets*` are ignored by git as a backstop. zsh reads `~/.zshrc` only for
interactive shells, so a batch job will not see exports placed there: use
`~/.zshenv`, or source your file in the job. `pc-doctor` confirms they are visible
without printing them, and `pc-doctor --check-credentials` tries them.

## Ground-based archives

`pc-archival` calls [DEATHSTAR](https://github.com/awmann/DEATHSTAR) as a separate
process, in its own environment (Python 3.9, numpy < 2). It needs this fork: the
upstream repository has no ephemeris scan and no ATLAS downloader.

```bash
git clone https://github.com/awmann/DEATHSTAR.git
git -C DEATHSTAR checkout petcare-2026-11      # the version this PetCare was tested with
export PETCARE_DEATHSTAR=/path/to/DEATHSTAR
export PETCARE_DEATHSTAR_PYTHON=/path/to/deathstar/env/bin/python
```

Runs take about 15 minutes per candidate and raw frames can use a lot of disk, so
reserve this tier for a shortlist.

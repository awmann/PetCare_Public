# Worked example: DS Tuc A

DS Tuc A is a 45 Myr G star that rotates in 2.9 days with spots large enough to
vary by a few percent. It hosts DS Tuc A b (8.14 d) and DS Tuc A c (21.62 d, Barber
et al., in preparation). Planet c is the useful case: at about 800 ppm it is much
shallower than the star's variability.

## Run it

```bash
cd examples/ds_tuc
./run.sh
```

`data/` holds the light curves, trimmed to what PetCare reads (16 MB). Tiers 0 and 1
use them directly; tier 2 downloads about 400 MB of TESS cutouts on the first run.
Allow about 15 minutes the first time and 2 minutes after that.

Results go to `out/TIC410214986/`: one plot and one JSON per test, `fit.png` for planet c
and `fit_c1.png` for planet b, and so on. `out/summary.csv` has one row per candidate, with
each test's status and reason and the vetting metrics as `m_*` columns.

To make the light curves yourself (about 5 GB of downloads and 4 minutes):

```bash
pip install -e "../..[extract]"
pc-extract --tic 410214986 --out data
```

## What it finds

Neither planet is refuted by any test.

| | planet c | planet b |
|---|---|---|
| fit | PASS, 812 ppm, local baseline | PASS, 3984 ppm, GP |
| PASS / FLAG / INCONCLUSIVE / FAIL | 14 / 5 / 2 / 0 | 16 / 5 / 0 / 0 |

Each planet's row in `candidates.csv` masks the other, because the table lists both.

**A FLAG asks you to look; it isn't a verdict.** Planet b, a confirmed planet, has five
of them. That is what vetting looks like on a star this active. Worth reading for c:

- **`neighbours` FLAG:** one star could reproduce the depth if it were eclipsing. It is
  DS Tuc B, a bound companion 5.4" away. PetCare can't rule it out from the catalogue
  alone, which is why the pixel tests exist.
- **`jointgp` FLAG:** the transit is significant when fitted with the variability, but
  depth varies between epochs more than the errors allow.
- **`flares` FLAG:** the transit is 25% deeper with flare cadences masked, so flares
  were filling it in.
- **`secondary` FLAG:** a 4.4 sigma feature at phase 0.175.
- **`centroid` and `pixels` INCONCLUSIVE:** at about 800 ppm, c is not detected well
  enough in the pixels to locate it. They can't say whether it is on target. For planet
  b, at about 4000 ppm, both PASS: on target to 0.3 +/- 0.6". Pixel tests have a depth
  limit, and INCONCLUSIVE is how they report reaching it. It is not a pass.

There is deliberately no overall score. Which test objected, and why, is the information.

## The light curve matters

Planet c shows what the input costs. These are the same candidate and ephemeris, on the
same TESS sectors:

| input | epochs | fit depth | joint GP significance | result |
|---|---:|---:|---:|---|
| tessquicklook, via `pc-extract` | 12 | 812 ppm | 12.3 sigma | no FAIL |
| tessquicklook CSV (no quality mask) | 12 | 768 ppm | 4.8 sigma | `jointgp` FAIL |
| SPOC 120 s PDCSAP | 10 | 860 ppm | 1.4 sigma | `jointgp` FAIL |

The public SPOC light curve is too noisy to recover this planet with confidence. A
local-baseline fit still finds a dip of about the right depth, but it is not significant
once the variability is fitted alongside it. PDCSAP removes systematics before the
variability is modelled; tessquicklook fits both at once, which keeps precision on fast
rotators. Planet b, at five times the depth, is recovered from SPOC too.

The SPOC result isn't always stable either. On that light curve, tiny changes in the
assumed rotation period can send the ephemeris refinement off the transit, so if you
see a depth near zero there, check the fitted `t0` against your input.

## Your own targets

`candidates.csv` shows the columns. `ra` and `dec` here are Gaia positions; add them, or a
stellar catalogue, for high proper motion stars. Give `age_myr` where you know it, and a
`prot` column if you know the rotation period better than a Lomb-Scargle will. Here it
measures 2.93 d.

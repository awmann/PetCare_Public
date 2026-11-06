# Adding a test

A test is one file in `petcare/scripts/`, and it has to satisfy the eight
design rules in the README.  The shape:

```python
"""pc_thing -- the one question this test answers.

Say why it exists and what it costs.  Anything non-obvious in the method
belongs here, next to the code, not in a commit message.
"""
from .. import cli
from ..result import Result

TEST = "thing"
VERSION = "0.1.0"

#: Every threshold is a module constant with a comment saying where the
#: number came from.  If it has not been calibrated yet, say PROVISIONAL.
THING_FLAG = 3.0


def add_arguments(ap):
    g = ap.add_argument_group("thing")
    g.add_argument("--thing-option", type=float, default=1.0)
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)

    # 1. refuse politely when the inputs are not there (rule 8)
    if not cli.require(r, ctx.candidate.period, "no candidate period"):
        return r

    # 2. do the work
    lc = ctx.lc
    ctx.attach_lc_provenance(r, lc)      # which light curve, and its dilution

    # 3. record the noise beside any statistic built on it (rule 5)
    r.set_noise(white_ppm, "how it was estimated")

    # 4. a self-check that can say "I cannot tell" (rule 3)
    r.set_self_check(ok, "what was checked, and what it found")

    # 5. one of PASS / FLAG / FAIL / INCONCLUSIVE, with a readable reason
    r.passed("...") / r.flag("...") / r.fail("...") / r.inconclusive("...")

    # 6. a figure (rule 6)
    if ctx.make_plots:
        r.add_plot(save(fig, ctx.plot_path(TEST)))
    return r


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(TEST, "one-line description"))
    ctx = cli.Context.from_args(ap.parse_args(argv))
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:          # recorded, never swallowed
        r.error(exc)
    return cli.finish(r, ctx)
```

Then:

1. add it to `TESTS` in `petcare/driver.py` with its tier, and whether it
   consumes `pc_fit`'s refined ephemeris;
2. add a console script to `pyproject.toml`;
3. add its continuous statistic to `STATISTICS` in
   `calibration/calibrate.py`, so the calibration run measures what it costs
   on known planets.

## What the shared machinery already does

Do not reimplement these.  Several of them exist because one pipeline
measured the same quantity two ways and the two disagreed.

| need | use |
|---|---|
| load and stitch light curves | `petcare.io.load_lightcurve` |
| fold, count epochs, check coverage | `petcare.ephem` |
| match a known planet | `petcare.ephem.match_known` — **overlap, never period** |
| detrend and get per-epoch depths | `petcare.prep.prepare`, `epoch_depths` |
| choose a detrending window | `petcare.prep.choose_baseline` — by measurement |
| fit a transit | `petcare.fitting.joint_fit` / `local_joint_fit` |
| noise | `petcare.noise` |
| stellar parameters and geometry | `petcare.stellar` |
| catalogue queries | `petcare.catalogs` — all cached, none at CDS |
| figures | `petcare.plotting` |

## Rules that are easy to break

- **`FLAG` is not `FAIL`.**  Very few tests earn the right to refute a
  candidate alone.  At the time of writing only `pc_depths` (the jackknife,
  measured at zero cost across 38 known planets) and `pc_centroid` do.
- **Never match a known signal on period.**  It fails at harmonics, and that
  failure cost 23% of the known planets in the search that preceded this.
- **Print the noise level beside any statistic built on one.**
- **A missing input is `INCONCLUSIVE`, not `PASS`.**  At 30k targets, missing
  sectors, missing catalogue rows and absent pixel data are all normal.

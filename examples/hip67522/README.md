# Worked example: HIP 67522

HIP 67522 is a 17 Myr G star rotating in 1.42 days, twice as fast as DS Tuc A, with two
published transiting planets. Start with [`../ds_tuc`](../ds_tuc/README.md); this example
adds what a faster rotator does, and what a test does when it cannot see a planet.

```bash
cd examples/hip67522
./run.sh
```

`data/` holds the light curves (6 MB), made with `pc-extract --tic 166527623 --out data`.
Tier 2 downloads about 170 MB of cutouts. Allow about 10 minutes the first time.
Plots for planet b are `out/TIC166527623/<test>.png`, and for planet c `<test>_c1.png`.

The rotation period comes from a per-sector Lomb-Scargle: 1.4185 d. One sector lands on
the harmonic at 0.708 d, and taking the median across sectors stops it deciding the answer.

## What it finds

| | planet b | planet c |
|---|---|---|
| fit | PASS, 4655 ppm | PASS, 3228 ppm |
| PASS / FLAG / INCONCLUSIVE / FAIL | 13 / 6 / 1 / 0 | 12 / 6 / 2 / 0 |
| joint GP, above the null phases | FLAG, 40 sigma | PASS, 8.9 sigma |
| `interp` | PASS | FLAG |
| `uniqueness` | PASS | FLAG |

The INCONCLUSIVE on both is `companion`, until the Sonora grid is downloaded (`pc-doctor`
says how). Planet c's second is `pixels`: the difference image peaks off the target in a
single sector, and one sector is not enough to say so.

**Planet c shows what a test does when it cannot see a planet.** On their own statistics,
`interp` and `uniqueness` both object to c:

- `interp`: only 6 epochs, and on a star this active the same detrending at random times
  scatters by 90 times the white noise, so the mean depth stands at 2.7 sigma.
- `uniqueness`: residual variability makes stronger dips and brightenings elsewhere in the
  fold (primary 13 sigma, secondary 28 sigma).

Before either test fails a candidate, it injects copies of the candidate's own depth at 12
other phases and asks how often those pass. For c, 33% and 25% do: on this star the tests
would also object to a real transit this deep, so they cannot refute it, and the result is
a FLAG that records the power. A deeper transit, or one the tests could have seen and did
not, still fails.

b's `jointgp` FLAG is not about significance: b stands 40 sigma above the same fit at 16
phases with no transit, but its 18 per-transit depths scatter more than their errors
allow (chi2 169 for 17 dof). Spots crossed by the planet, or unocculted spots changing
between transits, do that to a real planet, so depth consistency only ever flags.

Read a FLAG as "look at this, for this reason". c's other flags are worth reading too:
`flares` finds the transit 48% deeper with flares masked, and `secondary` a 4.6 sigma bump
at phase 0.115 on a star whose variability puts such bumps everywhere.

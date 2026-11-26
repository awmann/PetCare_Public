"""pc_gap -- is the transit ever seen with data on both sides?

A dip that sits at the edge of a data gap, a momentum dump or a sector
boundary is usually the ramp or step the instrument leaves there.  At long
periods a search can assemble a "candidate" entirely out of such edges,
because only two or three events are predicted and each lands on one.

**The method.**  For every predicted transit:

- it is *bracketed* when in-transit coverage is at least ``MIN_IN``, each
  side (0.5-2.5 durations from mid-transit) is at least ``MIN_SIDE`` covered,
  and there is no hole longer than ``MAX_HOLE`` durations within +/-1.5;
- a box (duration x 0.6-1.25, centre within +/-0.25 durations) is fitted on
  raw flux with a quadratic baseline over +/-3 durations, and its chi-square
  improvement is that epoch's share of the dip.

FAIL when no epoch is bracketed, or when more than ``MAX_UNBRACKETED_SHARE``
of the dip comes from epochs that are not.

**What it costs.**  In our benchmark it fails 1 of 50 known young
planets, 1 of 45 signals judged planet-like by eye, 0 of 100 injected planets
and 0 of 3 control planets, and 22 of 35 survivors of a 40-60 d search list.

A step-against-box comparison is computed per epoch and kept in the metrics
(``n_dip_steplike``), but it is NOT part of the verdict: on raw flux it fails
real planets on active stars and misses steps a human sees.

This test runs on request only (``--only gap``): it was built for long-period
lists and has not been adopted for short periods.

    python -m petcare.scripts.pc_gap --tic 123 --period 52.1 --t0 1500.1 --duration 5
"""

from __future__ import annotations

import numpy as np

from .. import cli
from ..plotting import C_BAD, C_OK, annotate_status, figure, save
from ..result import Result

TEST = "gap"
VERSION = "0.1.0"

#: An epoch counts as bracketed at or above these coverages (fractions).
MIN_IN, MIN_SIDE = 0.9, 0.5
#: ...and with no hole longer than this many durations within +/-1.5.
MAX_HOLE = 0.25
#: FAIL when this share of the summed dip comes from unbracketed epochs.
MAX_UNBRACKETED_SHARE = 0.8

SC = (0.6, 0.8, 1.0, 1.25)


def _lsq(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c, float(r @ r)


def epochs(lc, P, t0, t14d):
    """Per-epoch coverage and box fit.  One dict per epoch with at least 8 points within 3 T14."""
    t = np.asarray(lc.t, float); f = np.asarray(lc.f, float)
    ok = np.isfinite(t) & np.isfinite(f); t, f = t[ok], f[ok]
    cad = np.asarray(lc.cadence_s, float)[ok] / 86400.0
    out = []
    e = np.round((t - t0) / P).astype(int)
    for ep in np.unique(e):
        tc = t0 + ep * P
        x = (t - tc) / t14d
        w = np.abs(x) <= 3.0
        if w.sum() < 8: continue
        xx, ff = x[w], f[w]; cd = np.nanmedian(cad[w]) / t14d
        if not np.isfinite(cd) or cd <= 0: cd = np.median(np.diff(np.sort(xx)))
        cov = lambda a, b: min(((xx >= a) & (xx < b)).sum() * cd / (b - a), 1.0)
        c_in, c_pre, c_post = cov(-0.5, 0.5), cov(-2.5, -0.5), cov(0.5, 2.5)
        core = np.sort(xx[np.abs(xx) <= 1.5])
        edges = np.concatenate([[-1.5], core, [1.5]])
        hole = float(np.max(np.diff(edges)))
        o = dict(epoch=int(ep), tc=float(tc), c_in=c_in, c_pre=c_pre, c_post=c_post, hole=hole, n=int(w.sum()))
        o["bracketed"] = bool(c_in >= MIN_IN and c_pre >= MIN_SIDE and c_post >= MIN_SIDE
                              and hole <= max(MAX_HOLE, 4 * cd))
        if c_in >= 0.3 and (c_pre + c_post) >= 0.5 and w.sum() >= 12:
            med = np.median(ff); y = ff - med
            B = np.vstack([np.ones_like(xx), xx, xx ** 2]).T
            _, chi0 = _lsq(B, y)
            best = (np.inf, 0.0, None)
            for s in SC:
                for c0 in (-0.25, -0.125, 0, 0.125, 0.25):
                    box = (np.abs(xx - c0) < 0.5 * s).astype(float)
                    if box.sum() < 3 or box.sum() > len(xx) - 5: continue
                    c, chi = _lsq(np.column_stack([B, -box]), y)
                    if chi < best[0]: best = (chi, float(c[3]), (s, c0, int(box.sum())))
            bs = (np.inf, 0.0, None)
            for ts in np.linspace(-1, 1, 21):
                H = (xx > ts).astype(float)
                if H.sum() < 3 or H.sum() > len(xx) - 3: continue
                c, chi = _lsq(np.column_stack([B, H]), y)
                if chi < bs[0]: bs = (chi, float(c[3]), float(ts))
            if best[2] is not None:
                nin = best[2][2]; dof = max(len(xx) - 4, 1)
                # red-noise-aware depth error: scatter of the box-fit residuals binned to the box length
                A = np.column_stack([B, -(np.abs(xx - best[2][1]) < 0.5 * best[2][0]).astype(float)])
                c, _ = _lsq(A, y); r = y - A @ c
                order = np.argsort(xx); rb = r[order]; k = max(nin, 1)
                bins = [rb[i:i + k].mean() for i in range(0, len(rb) - k + 1, k)]
                sd_bin = float(np.std(bins)) if len(bins) >= 4 else float(np.sqrt(best[0] / dof) / np.sqrt(k))
                sd_white = float(np.sqrt(best[0] / dof)) * np.sqrt(1.0 / nin + 1.0 / max(len(xx) - nin, 1))
                sig = max(sd_bin, sd_white, 1e-12)
                o.update(depth=best[1], snr=best[1] / sig, d_box=chi0 - best[0],
                         d_step=(chi0 - bs[0]) if np.isfinite(bs[0]) else 0.0, step_amp=bs[1], step_t=bs[2],
                         box_scale=best[2][0], box_c=best[2][1])
                o["boxlike"] = bool(o["d_box"] > o["d_step"] and best[1] > 0)
        o["clean"] = bool(o["bracketed"] and o.get("boxlike") and o.get("snr", 0) >= 3.0)
        out.append(o)
    return out



def judge(eps: list) -> tuple[str, dict]:
    """(status, metrics) from the per-epoch list."""
    fit = [e for e in eps if "depth" in e and e["depth"] > 0]
    tot = sum(max(e["d_box"], 0) for e in fit)
    nbr = sum(1 for e in eps if e["bracketed"])
    share = (sum(max(e["d_box"], 0) for e in fit if not e["bracketed"]) / tot) if tot > 0 else float("nan")
    fail = nbr == 0 or share > MAX_UNBRACKETED_SHARE
    dip = [e for e in fit if e.get("snr", 0) >= 3]
    return ("FAIL" if fail else "PASS"), {
        "n_epochs": len(eps), "n_bracketed": nbr,
        "unbracketed_dip_share": None if share != share else round(float(share), 3),
        "n_dip": len(dip),
        "n_dip_steplike": sum(1 for e in dip if e["bracketed"] and not e.get("boxlike"))}


def add_arguments(ap):
    return ap


def run(ctx: cli.Context) -> Result:
    r = ctx.result(TEST, VERSION)
    if not cli.require(r, ctx.candidate.period and ctx.candidate.t0_btjd is not None,
                       "no candidate ephemeris: --period and --t0 (or --use-fit)"):
        return r
    if not cli.require(r, ctx.candidate.duration_h,
                       "no transit duration: pass --duration or run pc_fit first"):
        return r
    try:
        lc = ctx.lc
    except Exception as exc:  # noqa: BLE001 - missing data is normal
        return r.inconclusive(f"could not load the light curve: {exc}")
    ctx.attach_lc_provenance(r, lc)
    period, t0 = float(ctx.candidate.period), float(ctx.candidate.t0_btjd)
    dur = float(ctx.candidate.duration_h) / 24.0
    eps = epochs(lc, period, t0, dur)
    if not eps:
        r.set_self_check(False, "no epoch has data")
        return r.inconclusive("no predicted transit has at least 8 points within 3 durations of it")
    status, m = judge(eps)
    r.metrics.update(m)
    r.metrics["epochs"] = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                            for k, v in e.items()} for e in eps]
    r.set_self_check(True, f"{m['n_epochs']} epoch(s) with data")
    share = m["unbracketed_dip_share"]
    desc = (f"{m['n_bracketed']} of {m['n_epochs']} epoch(s) have data on both sides"
            + (f"; {share:.0%} of the dip comes from the others" if share is not None else ""))
    if status == "FAIL":
        r.fail(("no transit is seen with data on both sides: " if m["n_bracketed"] == 0 else
                "the dip comes from epochs at the edge of the data: ") + desc
               + ". A ramp or step at a gap makes this; a transit does not need one.")
    else:
        r.passed(f"the transit is seen away from gaps. {desc}")
    if ctx.make_plots:
        try:
            r.add_plot(_plot(ctx, r, eps))
        except Exception as exc:  # noqa: BLE001 - a failed figure is not a verdict
            ctx.notes.append(f"gap plot failed: {type(exc).__name__}: {exc}")
    return r


def _plot(ctx, r, eps):
    fig, ax = figure(nrows=1, ncols=2, figsize=(11, 0.28 * len(eps) + 2.2))
    yy = np.arange(len(eps))
    col = [C_OK if e["bracketed"] else C_BAD for e in eps]
    for k, lab in (("c_pre", "before"), ("c_in", "in transit"), ("c_post", "after")):
        off = {"c_pre": -0.27, "c_in": 0.0, "c_post": 0.27}[k]
        ax[0].barh(yy + off, [e[k] for e in eps], height=0.25, color=col,
                   alpha={"c_pre": 0.45, "c_in": 1.0, "c_post": 0.45}[k])
    ax[0].axvline(MIN_SIDE, color="0.5", lw=0.6, ls=":")
    ax[0].axvline(MIN_IN, color="0.5", lw=0.6, ls="--")
    ax[0].set_xlabel("coverage before / in / after the transit")
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels([f"E{e['epoch']}" for e in eps], fontsize=6)
    ax[1].barh(yy, [max(e.get("d_box", 0.0), 0.0) for e in eps], color=col)
    ax[1].set_xlabel("dip strength (chi-square improvement of a box)")
    ax[1].set_yticks([])
    ax[0].set_title("green: data on both sides; red: at a gap or edge", fontsize=9)
    annotate_status(fig, r)
    return save(fig, ctx.plot_path(TEST))


def main(argv=None) -> int:
    ap = add_arguments(cli.base_parser(TEST, "is the transit seen with data on both sides?"))
    ctx = cli.Context.from_args(ap.parse_args(argv))
    r = ctx.result(TEST, VERSION)
    try:
        r = run(ctx)
    except Exception as exc:  # recorded, never swallowed
        r.error(exc)
    return cli.finish(r, ctx)


if __name__ == "__main__":
    raise SystemExit(main())

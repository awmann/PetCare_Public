"""Injection power: can a test that just refuted a candidate refute anything else?

pc_interp and pc_uniqueness measure the dip against what this star's noise
makes on its own.  On a fast, spotted rotator that noise is so large that the
test fails real planets too: copies of HIP 67522 c injected at other phases
FAIL pc_interp 75% of the time and pc_uniqueness 83% of the time.  A FAIL from
a test that fails most planets of this depth on this star says nothing.

So before either test refutes, it is re-run on copies of the candidate's own
signal -- same depth, period and duration -- injected at other phases of the
real light curve.  Each copy is tested at its own ephemeris with the real
signal masked as a known sibling, exactly as a candidate would be.  The FAIL
stands only if more than POWER_FAIL_MIN of the copies pass (status != FAIL);
otherwise the result is a FLAG that records the power.

Adopted after a replay of production runs, in which 59% of sole pc_interp
FAILs and 43% of sole pc_uniqueness FAILs had power <= 0.5: those candidates
are let through, at the cost of more eye vetting.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses

import numpy as np

from .result import KnownSignal

#: Copies per candidate, and the phases they are drawn from: evenly spaced and
#: clear of the real transit, skipping any that land on a sibling's transits.
N_COPIES = 12
PHASES = np.linspace(0.12, 0.88, N_COPIES + 4)
#: A copy is skipped if more than this fraction of its window overlaps a sibling.
MAX_SIBLING_OVERLAP = 0.2
#: The FAIL stands only when more than this fraction of the copies pass.
POWER_FAIL_MIN = 0.5
#: Ingress/egress of the injected trapezoid, hours (or T14/4 if shorter).
INGRESS_H = 0.3


def injection_power(ctx, run) -> dict | None:
    """Re-run ``run`` (a test's ``run(ctx)``) on injected copies of the candidate.

    Returns ``{"power", "n_copies", "depth_ppm", "statuses"}``, or None when
    power cannot be measured (no positive depth, no light curve, no copy
    survived), in which case the caller's FAIL stands.  Never recurses: the
    copies' own FAILs are not gated.
    """
    if getattr(ctx, "_power_trial", False):
        return None
    cand = ctx.candidate
    depth = cand.depth_ppm
    if not (cand.period and cand.t0_btjd is not None and cand.duration_h and depth and depth > 0):
        return None
    try:
        lc = ctx.lc
    except Exception:  # noqa: BLE001 -- the caller already reported the light curve
        return None
    period, t0, dur_h = float(cand.period), float(cand.t0_btjd), float(cand.duration_h)
    half, tau = dur_h / 48.0, min(INGRESS_H, dur_h / 4.0) / 24.0
    real = KnownSignal(label="candidate under test", period=period, t0_btjd=t0, duration_h=dur_h)
    args = argparse.Namespace(**{**vars(ctx.args), "no_plot": True})

    statuses = []
    for ph in PHASES:
        if len(statuses) >= N_COPIES:
            break
        ti = t0 + ph * period
        dt = (lc.t - ti + 0.5 * period) % period - 0.5 * period
        inwin = np.abs(dt) < half
        if inwin.any() and any(
                (inwin & (np.abs((lc.t - k.t0_btjd + 0.5 * k.period) % k.period - 0.5 * k.period)
                          < k.duration_h / 48.0)).sum() > MAX_SIBLING_OVERLAP * inwin.sum()
                for k in ctx.known if k.period):
            continue
        shape = np.clip((half - np.abs(dt)) / tau, 0.0, 1.0)
        c = copy.copy(ctx)
        c._power_trial = True
        c.args = args
        c.notes = []
        c.candidate = dataclasses.replace(cand, t0_btjd=float(ti))
        c.known = list(ctx.known) + [real]
        c._lc = dataclasses.replace(lc, f=lc.f * (1.0 - depth * 1e-6 * shape))
        try:
            statuses.append(str(run(c).status))
        except Exception:  # noqa: BLE001 -- a copy that crashes is not counted
            continue
    if not statuses:
        return None
    return {"power": sum(s != "FAIL" for s in statuses) / len(statuses),
            "n_copies": len(statuses), "depth_ppm": float(depth), "statuses": statuses,
            "fail_min": POWER_FAIL_MIN}


def gate(r, ctx, run, why: str) -> None:
    """Record ``why`` on ``r`` as a FAIL if this test has the power to refute, else a FLAG."""
    p = injection_power(ctx, run)
    if p is not None:
        r.metrics["injection_power"] = p
    if p is None or p["power"] > POWER_FAIL_MIN:
        extra = (f" Copies of this signal injected at {p['n_copies']} other phases pass "
                 f"{p['power']:.0%} of the time, so the test can see a transit this deep here."
                 if p else "")
        r.fail(why + extra)
    else:
        r.flag(why + f" But copies of this signal injected at {p['n_copies']} other phases "
                     f"pass only {p['power']:.0%} of the time: on this star the test fails "
                     f"real transits this deep too, so it cannot refute.")

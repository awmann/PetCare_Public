"""Checks that need the rest of the candidate list, not just one candidate.

A per-candidate test cannot know what else was found on the star.  These are plain
functions for the local driver, which does know, to call with that information.
"""

from __future__ import annotations

#: Relative tolerance for calling one period an integer multiple of another.
MULTIPLE_TOL = 2e-4


def period_multiple(period: float, refuted_periods, tol: float = MULTIPLE_TOL):
    """Is ``period`` an integer multiple (n >= 2) of a period already refuted on the same star?

    A search that reaches long periods finds short-period false positives again at 2x, 3x,
    ... their period, where fewer events are predicted and fewer tests have power.  If the
    short period was refuted, the multiple is the same signal.

    Returns ``(n, base_period)`` for the first match, else ``None``.  Pass only periods that
    were REFUTED: a multiple of a surviving candidate may be a real alias question, which is
    a different check.
    """
    for p in refuted_periods:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if not p > 0:
            continue
        n = period / p
        if round(n) >= 2 and abs(n / round(n) - 1) < tol:
            return int(round(n)), p
    return None

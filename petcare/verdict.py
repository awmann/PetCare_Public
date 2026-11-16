"""Which test's answer stands when two tests measure the same thing.

PetCare computes no overall score, and nothing here adds one.  This module
holds the one place where a test's status is superseded by another's: the
per-epoch audit (``pc_epochaudit``) fits each transit together with the star's
variability, which is the better measurement of what ``pc_depths``,
``pc_shapetime`` and ``pc_oddeven`` measure on a masked-GP detrend.  So:

* audit PASS: a FAIL from any of those three counts as a FLAG;
* audit FAIL: it is a FAIL in its own right (nothing here is needed);
* audit odd/even consistent (whatever its overall verdict): a FAIL from
  ``pc_oddeven`` counts as a FLAG.

The per-test records are never edited.  ``effective_statuses`` returns the
statuses to count, and the list of what was overruled and why, and the driver
writes both into the merged summary.
"""

from __future__ import annotations

from .result import Status

#: Tests whose FAIL a passing per-epoch audit turns into a FLAG.
EPOCH_AUDIT_OVERRULES = frozenset({"depths", "shapetime", "odd_even"})


def effective_statuses(records: dict) -> tuple:
    """``({test: status}, [overruled])`` for a ``{test: record}`` mapping."""
    status = {name: rec.get("status", Status.ERROR) for name, rec in records.items()}
    overruled = []
    audit = records.get("epoch_audit") or {}
    metrics = audit.get("metrics") or {}

    def demote(name, why):
        if status.get(name) == Status.FAIL:
            status[name] = Status.FLAG
            overruled.append({"test": name, "from": Status.FAIL, "to": Status.FLAG,
                              "by": "epoch_audit", "why": why})

    if audit.get("status") == Status.PASS:
        for name in sorted(EPOCH_AUDIT_OVERRULES):
            demote(name, "the transits agree when each is fitted together with the star")
    if audit.get("status") in (Status.PASS, Status.FAIL) and metrics.get("oddeven_consistent"):
        demote("odd_even", "odd and even transits agree when fitted together with the star")
    return status, overruled

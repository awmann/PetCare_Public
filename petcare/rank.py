"""An ordering for a list of surviving candidates.  Not a verdict.

The driver computes no overall score, on purpose: which test objected is the information,
and a scalar hides it.  This module does not change that.  It takes numbers from tests that
CANNOT refute a candidate -- fit quality, shape, rotation, instrument, uniqueness, per-epoch
votes and epoch statistics -- and turns them into one number whose only use is to decide what
a human looks at first.  Nothing here may be used to drop a candidate.

**The score.**  For each number, the candidate population is cut into eight quantile bins;
each bin holds the log of (fraction of injected planets in it) / (fraction of all candidates
in it), capped at +/- ``cap`` so that no single number dominates.  The score is the sum over
the numbers a candidate has; a missing number adds zero.

**What it buys.**  On the candidates judged by eye in the search it was built on, it ranks
known and planet-like signals above rejected ones with AUC 0.92, against 0.82 for ordering by
the count of passed quality checks and then search margin.

The table in ``data/rank_model.json`` belongs to that search (young stars, TESS).
A different survey should refit it: ``fit()`` needs the metrics of injected planets and of
the candidate population.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "data" / "rank_model.json"
_model = {}


def load_model(path=None) -> dict:
    path = Path(path) if path else MODEL_PATH
    key = str(path)
    if key not in _model:
        _model[key] = json.loads(path.read_text())
    return _model[key]


def parts(metrics: dict, model: dict | None = None) -> dict:
    """Each number's contribution to the score."""
    model = model or load_model()
    out = {}
    for name, t in model["features"].items():
        x = metrics.get(name)
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        if x != x or x in (float("inf"), float("-inf")):
            continue
        out[name] = t["logodds"][bisect.bisect_left(t["edges"], x)]
    return out


def score(metrics: dict, model: dict | None = None) -> float:
    """The ranking score from ``petcare.metrics.collect(record)["metrics"]``."""
    return round(float(sum(parts(metrics, model).values())), 3)


def score_record(rec: dict, model: dict | None = None) -> float:
    from . import metrics as pm
    return score(pm.collect(rec)["metrics"], model)


def fit(injected: dict, population: dict, cap: float = 1.5, bins: int = 8,
        min_injected: int = 40, min_population: int = 500) -> dict:
    """Build a model from {name: array} of injected-planet and population values."""
    feats = {}
    for name in injected:
        if name not in population:
            continue
        a = np.asarray(injected[name], float)
        b = np.asarray(population[name], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < min_injected or b.size < min_population:
            continue
        e = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1))[1:-1])
        ca = np.bincount(np.searchsorted(e, a), minlength=len(e) + 1) + 1.0
        cb = np.bincount(np.searchsorted(e, b), minlength=len(e) + 1) + 1.0
        lo = np.clip(np.log((ca / ca.sum()) / (cb / cb.sum())), -cap, cap)
        feats[name] = {"edges": [float(x) for x in e], "logodds": [round(float(x), 4) for x in lo]}
    return {"cap": cap, "features": feats}

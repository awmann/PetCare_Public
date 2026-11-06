"""Phase folding, epoch bookkeeping, and matching against known signals.

Two things here are less obvious than they look and both caused real damage in
earlier versions of this pipeline:

**Epoch numbering.**  The epoch of a cadence is ``round((t - t0) / P)``, not
``floor(...)``.  With ``floor`` a transit centred near a cycle boundary is
split across two epoch numbers, so a per-epoch depth test sees two
half-covered events instead of one good one and reports an inconsistency that
is not there.

**Matching a candidate to a known planet.**  Always by *transit overlap*,
never by period proximity.  A period-proximity match fails at harmonics: when
the search proposes P/2 for a known planet, a proximity test does not
recognise it, the known planet gets masked out at its own period anyway, and
the candidate is subtracted from its own light curve.  That mechanism lost 23%
of the known planets in the 2026 search before any vetting statistic was
computed.  :func:`match_known` compares the *set of transit times*.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# folding


def phase(t, period: float, t0: float) -> np.ndarray:
    """Signed phase in days, in ``[-P/2, +P/2)``, zero at mid-transit."""
    t = np.asarray(t, dtype=float)
    return ((t - t0 + 0.5 * period) % period) - 0.5 * period


def phase_frac(t, period: float, t0: float) -> np.ndarray:
    """Phase as a fraction of the period, in ``[0, 1)``, zero at mid-transit."""
    t = np.asarray(t, dtype=float)
    return ((t - t0) % period) / period


def epoch_number(t, period: float, t0: float) -> np.ndarray:
    """Integer transit epoch of each cadence.  Round, never floor -- see module docstring."""
    t = np.asarray(t, dtype=float)
    return np.round((t - t0) / period).astype(int)


def transit_time(epoch, period: float, t0: float) -> np.ndarray:
    return t0 + np.asarray(epoch, dtype=float) * period


def in_transit(t, period: float, t0: float, duration_days: float,
               width: float = 1.0) -> np.ndarray:
    """Boolean mask of cadences within ``width * T14 / 2`` of a mid-transit."""
    return np.abs(phase(t, period, t0)) < 0.5 * width * duration_days


def out_of_transit(t, period: float, t0: float, duration_days: float,
                   inner: float = 0.75, outer: float = 3.0) -> np.ndarray:
    """Local out-of-transit baseline: a ring around each transit.

    ``inner`` is in units of the full duration measured from mid-transit, so
    the default leaves a 0.25*T14 guard band outside fourth contact -- enough
    that an underestimated duration does not drag in-transit cadences into the
    baseline, which biases every depth low.
    """
    p = np.abs(phase(t, period, t0))
    return (p >= inner * duration_days) & (p < outer * duration_days)


# --------------------------------------------------------------------------
# coverage


@dataclass
class EpochCoverage:
    """What the data actually contain for one transit epoch.

    ``coverage`` is the fraction of the in-transit window that has cadences in
    it, computed from the cadence count rather than from the time span, so a
    window with a hole in the middle is scored below 1 as it should be.
    """

    epoch: int
    t_center: float
    n_in: int
    n_oot: int
    coverage: float
    ingress_covered: bool
    egress_covered: bool
    in_gap: bool
    near_sector_edge: bool
    sector: int | None
    t_first: float | None = None
    t_last: float | None = None

    @property
    def usable(self) -> bool:
        """Enough in- and out-of-transit data to measure a depth at all."""
        return self.n_in >= 3 and self.n_oot >= 5 and self.coverage >= 0.3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usable"] = self.usable
        return d


def epoch_coverage(lc, period: float, t0: float, duration_days: float,
                   oot_inner: float = 0.75, oot_outer: float = 3.0,
                   edge_days: float = 0.5) -> list[EpochCoverage]:
    """Per-epoch coverage bookkeeping over every epoch the data could contain.

    Epochs with no data at all are omitted; epochs with partial data are kept
    and marked, because "this transit is half in a downlink gap" is exactly
    the information a depth-consistency test needs in order not to report a
    shallow epoch as a physical inconsistency.

    ``edge_days`` is how close to the start or end of a sector's data a
    transit centre has to be to count as sector-edge-affected.
    """
    t = np.asarray(lc.t, dtype=float)
    if t.size == 0 or not np.isfinite(period) or period <= 0:
        return []

    # Per-point cadence, not a global median.  A star with both 120 s and
    # 20 s sectors has a global median of 20 s, against which every 120 s
    # epoch scores a coverage of 0.17 and is silently discarded as unusable.
    # That dropped two of HIP 67522's five sectors on the first run.
    cad_all = np.asarray(lc.cadence_s, float)
    if not np.isfinite(cad_all).any():
        cad_all = np.full(t.size, lc.exp_time_days * 86400.0)
    cad_all = np.where(np.isfinite(cad_all) & (cad_all > 0), cad_all, np.nan)
    default_cad_days = lc.exp_time_days
    if not np.isfinite(default_cad_days) or default_cad_days <= 0:
        default_cad_days = 30.0 / 1440.0

    # sector time spans, for the edge test
    spans = [(s.sector, s.t_start, s.t_stop) for s in lc.sectors
             if np.isfinite(s.t_start) and np.isfinite(s.t_stop)]

    e_lo = int(np.floor((t.min() - t0) / period)) - 1
    e_hi = int(np.ceil((t.max() - t0) / period)) + 1

    ph = phase(t, period, t0)
    ep = epoch_number(t, period, t0)
    half = 0.5 * duration_days

    out: list[EpochCoverage] = []
    for e in range(e_lo, e_hi + 1):
        tc = t0 + e * period
        sel = ep == e
        if not sel.any():
            continue
        in_mask = sel & (np.abs(ph) < half)
        oot_mask = sel & (np.abs(ph) >= oot_inner * duration_days) & \
                   (np.abs(ph) < oot_outer * duration_days)
        n_in = int(in_mask.sum())
        n_oot = int(oot_mask.sum())

        cad_here = cad_all[sel]
        cad_here = cad_here[np.isfinite(cad_here)]
        cad_days = (float(np.median(cad_here)) / 86400.0 if cad_here.size
                    else default_cad_days)

        expected = max(duration_days / cad_days, 1.0)
        coverage = min(n_in / expected, 1.0)

        # ingress/egress present?  Ask whether there is a cadence within one
        # cadence-length of first and fourth contact.
        tol = max(cad_days, 0.02 * duration_days)
        t_in = t[in_mask]
        ingress = bool(t_in.size and np.min(np.abs(t_in - (tc - half))) <= tol + cad_days)
        egress = bool(t_in.size and np.min(np.abs(t_in - (tc + half))) <= tol + cad_days)

        # a gap *inside* the transit, distinct from missing wings
        # "gap" means a hole big enough to lose real signal, so it is scaled
        # to the transit, not to the cadence: at 20 s sampling a 3-cadence
        # threshold fires on ordinary data-quality dropouts and flagged 9 of
        # 12 clean epochs on the first run.
        gap_tol = max(10 * cad_days, 0.08 * duration_days)
        in_gap = False
        if t_in.size >= 2:
            in_gap = bool(np.max(np.diff(np.sort(t_in))) > gap_tol)
        elif n_in > 0:
            in_gap = coverage < 0.5

        sector = None
        near_edge = False
        for sec, ts, te in spans:
            if ts - period <= tc <= te + period:
                if ts <= tc <= te:
                    sector = sec
                if min(abs(tc - ts), abs(tc - te)) < edge_days + half:
                    near_edge = True
        if sector is None and n_in:
            sids = lc.sector_id[in_mask]
            vals, counts = np.unique(sids, return_counts=True)
            sector = int(vals[np.argmax(counts)])

        out.append(EpochCoverage(
            epoch=int(e), t_center=float(tc), n_in=n_in, n_oot=n_oot,
            coverage=float(coverage), ingress_covered=ingress,
            egress_covered=egress, in_gap=in_gap, near_sector_edge=near_edge,
            sector=sector,
            t_first=float(t_in.min()) if t_in.size else None,
            t_last=float(t_in.max()) if t_in.size else None,
        ))
    return out


def coverage_summary(cov: Sequence[EpochCoverage]) -> dict:
    usable = [c for c in cov if c.usable]
    return {
        "n_epochs_with_data": len(cov),
        "n_epochs_usable": len(usable),
        "n_epochs_full": sum(1 for c in cov if c.coverage >= 0.9),
        "n_epochs_ingress_or_egress_missing": sum(
            1 for c in usable if not (c.ingress_covered and c.egress_covered)),
        "n_epochs_with_internal_gap": sum(1 for c in usable if c.in_gap),
        "n_epochs_near_sector_edge": sum(1 for c in usable if c.near_sector_edge),
        "median_coverage": float(np.median([c.coverage for c in usable])) if usable else None,
        "epochs": [c.epoch for c in usable],
        "sectors": sorted({c.sector for c in usable if c.sector is not None}),
    }


# --------------------------------------------------------------------------
# known signals: overlap-based matching


def transit_times(period: float, t0: float, t_min: float, t_max: float) -> np.ndarray:
    """Every mid-transit time of an ephemeris inside ``[t_min, t_max]``."""
    if not np.isfinite(period) or period <= 0:
        return np.array([])
    e_lo = int(np.floor((t_min - t0) / period))
    e_hi = int(np.ceil((t_max - t0) / period))
    tt = t0 + np.arange(e_lo, e_hi + 1) * period
    return tt[(tt >= t_min) & (tt <= t_max)]


#: Fractional period agreement for the drifting-ephemeris fallback in
#: match_known.  0.3% at a ratio of 1, 2 or 3 and coincidence at the known
#: epoch is a random match well under 1 in 1000.
DRIFT_RATIO_TOL = 0.003


def match_known(period: float, t0: float, duration_h: float,
                known, t_min: float, t_max: float,
                tol_frac: float = 0.5) -> dict | None:
    """Does this candidate's transit train coincide with a known signal's?

    Matching is on transit *times*, not periods, so a candidate at P/2, 2P or
    3P of a known planet is recognised as the same object -- which is the
    whole point (see module docstring).

    Returns ``None`` if nothing matches, else a dict with the matched signal,
    the fraction of the candidate's transits that land on a known one, and the
    period ratio, which tells you *how* it matched.

    ``tol_frac`` is the coincidence tolerance as a fraction of the shorter of
    the two durations.
    """
    cand_tt = transit_times(period, t0, t_min, t_max)
    if cand_tt.size == 0:
        return None

    best = None
    for k in (known or []):
        k_per = float(getattr(k, "period", None) or k["period"])
        k_t0 = float(getattr(k, "t0_btjd", None) or k["t0_btjd"])
        k_dur = float(getattr(k, "duration_h", None) or k["duration_h"])
        k_label = getattr(k, "label", None) or k.get("label", "known")

        known_tt = transit_times(k_per, k_t0, t_min, t_max)
        if known_tt.size == 0:
            continue
        tol = tol_frac * min(duration_h, k_dur) / 24.0
        if tol <= 0:
            continue

        # fraction of candidate transits that coincide with a known transit
        d_cand = np.min(np.abs(cand_tt[:, None] - known_tt[None, :]), axis=1)
        frac_cand = float(np.mean(d_cand <= tol))
        # and the reverse, which distinguishes P/2 (all known hit, half the
        # candidate's hit) from 2P (all candidate hit, half the known's)
        d_known = np.min(np.abs(known_tt[:, None] - cand_tt[None, :]), axis=1)
        frac_known = float(np.mean(d_known <= tol))

        score = max(frac_cand, frac_known)
        how = "transit trains coincide"
        if score < 0.8:
            # Fallback for ephemerides that drift.  A catalogue ephemeris fitted to
            # one early sector can be hours out five years later, so the trains
            # stop coinciding even for the same object (one case: 3.78995 vs
            # 3.78833 d, same depth, called "separate").  Same object if the
            # periods agree to DRIFT_RATIO_TOL at a simple ratio AND the
            # transits coincide at the known signal's own epoch, where its
            # ephemeris is good.
            r_ = period / k_per
            for n in (1, 2, 3):
                for val in (n, 1.0 / n):
                    if abs(r_ / val - 1) < DRIFT_RATIO_TOL:
                        # Extrapolate the CANDIDATE's ephemeris (fitted to the
                        # whole baseline) to a few of the known signal's
                        # transits near its own epoch, where the known
                        # ephemeris is good.  Using the known period instead
                        # throws one known planet off by 0.8 d (its catalogue
                        # period is 0.1% out).  Several known transits, because at 2P the
                        # candidate skips every other one.
                        kt = k_t0 + k_per * np.arange(-3, 4)
                        dts = ((kt - t0 + 0.5 * period) % period) - 0.5 * period
                        dt0 = float(np.min(np.abs(dts)))
                        if dt0 <= max(duration_h, k_dur) / 24.0:
                            score, how = 0.8, ("periods agree and transits coincide at "
                                               "the known epoch; the ephemerides drift "
                                               "apart over the baseline")
                        break
        if score >= 0.8 and (best is None or score > best["score"]):
            ratio = period / k_per
            best = {
                "label": k_label,
                "known_period": k_per,
                "known_t0_btjd": k_t0,
                "period_ratio": float(ratio),
                "frac_candidate_transits_matched": frac_cand,
                "frac_known_transits_matched": frac_known,
                "score": score,
                "relation": _describe_ratio(ratio),
                "tolerance_days": float(tol),
                "how": how,
            }
    return best


def _describe_ratio(ratio: float) -> str:
    for num in (1, 2, 3, 4, 5):
        for den in (1, 2, 3, 4, 5):
            if abs(ratio - num / den) < 0.02 * max(num / den, 1.0):
                if num == den:
                    return "same period"
                return f"{num}/{den} of the known period"
    return f"period ratio {ratio:.3f}"


def mask_known_transits(t, known, width: float = 2.0) -> np.ndarray:
    """Mask cadences inside any known signal's transit.

    ``width`` multiplies T14.  The default of 2 is wider than the transit on
    purpose: masking exactly 1x T14 leaves the detrending filter's ringing
    across the masked gap just outside the mask, and that ringing was itself
    detected as three separate 5-6 sigma "candidates" in the 2026 search.
    """
    t = np.asarray(t, dtype=float)
    mask = np.zeros(t.size, dtype=bool)
    for k in (known or []):
        k_per = float(getattr(k, "period", None) or k["period"])
        k_t0 = float(getattr(k, "t0_btjd", None) or k["t0_btjd"])
        k_dur = float(getattr(k, "duration_h", None) or k["duration_h"]) / 24.0
        if not (np.isfinite(k_per) and k_per > 0 and np.isfinite(k_dur)):
            continue
        mask |= np.abs(phase(t, k_per, k_t0)) < 0.5 * width * k_dur
    return mask


# --------------------------------------------------------------------------
# ephemeris drift


def ephemeris_drift(period_old: float, t0_old: float,
                    period_new: float, t0_new: float,
                    t_ref: float, duration_h: float) -> dict:
    """How far the refined ephemeris has moved at ``t_ref``, in durations.

    A stale catalogue ephemeris looks exactly like a shallow or inconsistent
    transit, and it hit 2 of 27 candidates in one run.  Any test that folds on a
    catalogue value needs this number reported beside its result.
    """
    n_old = np.round((t_ref - t0_old) / period_old)
    n_new = np.round((t_ref - t0_new) / period_new)
    tc_old = t0_old + n_old * period_old
    tc_new = t0_new + n_new * period_new
    shift_days = float(tc_new - tc_old)
    dur_days = duration_h / 24.0
    return {
        "t0_shift_days": float(t0_new - t0_old),
        "period_shift_days": float(period_new - period_old),
        "period_shift_frac": float((period_new - period_old) / period_old),
        "shift_at_tref_days": shift_days,
        "shift_at_tref_durations": float(shift_days / dur_days) if dur_days > 0 else None,
        "t_ref": float(t_ref),
    }

"""Shared command-line scaffolding.

Rule 1 of this package is *one script, one question, one CLI*: every test must
run standalone from the command line and none may require the driver.  That is
only sustainable if they all share an argument vocabulary, which is what this
module is.

A test script is therefore:

.. code-block:: python

    def main(argv=None):
        ap = cli.base_parser("odd_even", "do odd and even transits match?")
        args = ap.parse_args(argv)
        ctx = cli.Context.from_args(args)
        result = run(ctx)
        return cli.finish(result, ctx)

Tests consume each other's output through the JSON files, never by importing
each other's internals: ``--use-fit`` reads the refined ephemeris out of
``<out>/<target>/fit.json`` if it is there, and says so in the provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io as pcio
from .result import Candidate, KnownSignal, Result, Status, Target
from .stellar import Star, assume_radius_error, get_star

DEFAULT_OUT = "out"


# --------------------------------------------------------------------------


def base_parser(test_name: str, description: str, *, needs_lightcurve: bool = True,
                needs_candidate: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"pc_{test_name}",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = ap.add_argument_group("target")
    g.add_argument("--tic", type=int, help="TIC ID")
    g.add_argument("--gaia", type=int, help="Gaia DR3 source_id")
    g.add_argument("--name", help="display name (defaults to 'TIC <id>')")
    g.add_argument("--ra", type=float, help="RA in degrees (else looked up)")
    g.add_argument("--dec", type=float, help="Dec in degrees (else looked up)")
    g.add_argument("--age", type=float, default=None,
                   help="stellar age in Myr, when the catalogue lacks one. "
                        "pc_duration's planet-radius ceiling is age-dependent "
                        "-- young giants are genuinely puffy -- so without an "
                        "age it falls back to the most generous bound.")
    g.add_argument("--catalog", help="stellar parameter catalogue "
                                     "(parquet/CSV); else $PETCARE_STELLAR_CATALOG")

    if needs_lightcurve:
        g = ap.add_argument_group("light curve")
        g.add_argument("--lc", nargs="+", default=None,
                       help="light-curve file(s) or glob(s); format is sniffed")
        g.add_argument("--lc-root", nargs="+", default=None,
                       help="directories to search by TIC; else $PETCARE_LC_ROOTS")
        g.add_argument("--prot", type=float, default=None,
                       help="rotation period [d]. Without it the extraction's "
                            "value is used, else a Lomb-Scargle on this light "
                            "curve, which can land on P/2 of a double-dip star")
        g.add_argument("--flux-column", default="fcor",
                       help="fcor (corrected, variability retained) | fflat | f")
        g.add_argument("--sectors", type=int, nargs="+", default=None)
        g.add_argument("--drop-sectors", type=int, nargs="+", default=None)

    if needs_candidate:
        g = ap.add_argument_group("candidate")
        g.add_argument("--period", type=float, help="orbital period [d]")
        g.add_argument("--t0", type=float, help="mid-transit [BTJD]; BJD is converted")
        g.add_argument("--duration", type=float, help="T14 [hours]")
        g.add_argument("--depth", type=float, help="depth [ppm]")
        g.add_argument("--index", type=int, default=0,
                       help="candidate index on this star, for the output filename")
        g.add_argument("--candidates", help="candidate table (CSV/parquet)")
        g.add_argument("--row", type=int, default=None,
                       help="row of --candidates to use (else matched on --tic)")
        g.add_argument("--use-fit", action="store_true",
                       help="take the refined ephemeris from <out>/<target>/fit.json")

    g = ap.add_argument_group("known signals")
    g.add_argument("--known", nargs="+", default=None,
                   help="a JSON file, or inline 'label:P:t0:duration_h' entries. "
                        "Matched by transit overlap, never by period.")
    g.add_argument("--no-auto-known", action="store_true",
                   help="do not pull known planets from the candidate table")

    g = ap.add_argument_group("output")
    g.add_argument("--out", default=DEFAULT_OUT, help="output root directory")
    g.add_argument("--no-plot", action="store_true")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--json", action="store_true",
                   help="print the result JSON to stdout")

    return ap


# --------------------------------------------------------------------------


@dataclass
class Context:
    """Everything a test was given, resolved once and shared."""

    args: argparse.Namespace
    target: Target
    candidate: Candidate
    known: list = field(default_factory=list)
    out: Path = Path(DEFAULT_OUT)
    _lc: object = None
    _star: Star | None = None
    notes: list = field(default_factory=list)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "Context":
        target = _resolve_target(args)
        candidate = _resolve_candidate(args, target)
        out = Path(getattr(args, "out", DEFAULT_OUT))
        ctx = cls(args=args, target=target, candidate=candidate, out=out)
        ctx.input_depth_ppm = candidate.depth_ppm
        if getattr(args, "use_fit", False):
            ctx._apply_fit_refinement()
        age = getattr(args, "age", None)
        if age:
            ctx._age_override = float(age)
        ctx._fill_duration()
        ctx.known = _resolve_known(args, target, ctx)
        return ctx

    # -- lazily resolved resources -------------------------------------------

    _age_override = None
    #: pc_fit ran and did not converge, so candidate.depth_ppm is still the input
    #: table's, never confirmed by a fit of the light curve.
    depth_unverified = False

    @property
    def lc(self):
        """The stitched light curve.  Raises LightCurveError if unavailable."""
        if self._lc is None:
            paths = getattr(self.args, "lc", None)
            if not paths:
                if self.target.tic is None:
                    raise pcio.LightCurveError(
                        "no --lc given and no --tic to search by")
                paths = pcio.find_lightcurves(
                    self.target.tic, getattr(self.args, "lc_root", None))
                if not paths:
                    raise pcio.LightCurveError(
                        f"no light curves found for TIC {self.target.tic}; "
                        f"pass --lc or set $PETCARE_LC_ROOTS")
            self._lc = pcio.load_lightcurve(
                paths,
                flux_column=getattr(self.args, "flux_column", "fcor"),
                sectors=getattr(self.args, "sectors", None),
                drop_sectors=getattr(self.args, "drop_sectors", None),
            )
            self._lc.prot_user = getattr(self.args, "prot", None)
            if self.target.tic is None and self._lc.target.get("tic"):
                self.target.tic = int(self._lc.target["tic"])
        return self._lc

    @property
    def star(self) -> Star:
        if self._star is None:
            self._star = get_star(tic=self.target.tic, gaia_dr3=self.target.gaia_dr3,
                                  catalog=getattr(self.args, "catalog", None))
            assume_radius_error(self._star)
            if self.target.ra is None and self._star.ra is not None:
                self.target.ra, self.target.dec = self._star.ra, self._star.dec
            if self.target.gaia_dr3 is None and self._star.gaia_dr3:
                self.target.gaia_dr3 = self._star.gaia_dr3
            # A cluster age from the candidate table fills in where the
            # stellar catalogue has none.  Never overrides a catalogue value.
            if self._age_override and not self._star.age_myr:
                self._star.age_myr = float(self._age_override)
                self._star.source_detail.setdefault("age_from", "candidate table")
        return self._star

    # -- results -------------------------------------------------------------

    def result(self, test: str, version: str) -> Result:
        r = Result(test=test, version=version, target=self.target,
                   candidate=self.candidate)
        r.set_known_signals(self.known)
        return r

    def attach_lc_provenance(self, r: Result, lc=None) -> Result:
        """Record which light curve produced the numbers, and its dilution.

        Both are mandatory context: a candidate can pass on QLP and fail on
        SPOC, and a low factor means the depth rests on a large correction.
        """
        lc = lc if lc is not None else self.lc
        factor, source = lc.dilution()
        r.set_provenance(
            lightcurve=";".join(Path(p).name for p in lc.paths),
            n_points=len(lc),
            lightcurve_paths=lc.paths,
            flux_column=lc.flux_column,
            sectors=[s.sector for s in lc.sectors],
            cadence_s=lc.median_cadence_s,
            baseline_days=lc.baseline_days,
        )
        sap = "sap" in str(lc.flux_column).lower()
        r.set_dilution(factor=factor, source=source or "none", applied=not sap,
                       note="SAP flux: not corrected for dilution" if sap else
                            "the light curve is already de-diluted; "
                            "do not divide depths by this factor")
        if self.notes:
            r.provenance["notes"] = list(self.notes)
        return r

    def plot_path(self, name: str) -> Path:
        d = self.out / self.target.slug
        d.mkdir(parents=True, exist_ok=True)
        suffix = f"_c{self.candidate.index}" if self.candidate.index else ""
        return d / f"{name}{suffix}.png"

    @property
    def make_plots(self) -> bool:
        return not getattr(self.args, "no_plot", False)

    # -- internals -----------------------------------------------------------

    def _fill_duration(self) -> None:
        """Derive a transit duration when the candidate has none.

        A candidate table from a box search often has a period and an epoch
        and nothing else.  Without a duration every light-curve test returns
        INCONCLUSIVE -- five of twenty-two candidates in one run lost their
        entire tier-1 chain that way, including the three a human had already
        annotated with "secondary eclipse", which pc_secondary never got to
        look for.

        The estimate is the same one pc_fit makes: the duration a central-ish
        crossing implies for this star's density.  It is recorded in
        ``notes`` and in ``candidate.source`` so nothing downstream mistakes
        it for a measurement.
        """
        if self.candidate.duration_h is not None or not self.candidate.period:
            return
        from .stellar import a_over_rs, duration_t14
        try:
            rho, _ = self.star.density()
        except Exception:  # noqa: BLE001 - no catalogue is not an error here
            rho = None
        p = float(self.candidate.period)
        if rho and rho > 0:
            dur = duration_t14(p, a_over_rs(p, rho), 0.05, b=0.5) * 24.0
            how = f"stellar density {rho:.3f} g/cm3 at b = 0.5"
        else:
            dur = 0.05 * p ** (1 / 3) * 24.0
            how = "crude P^(1/3) scaling (no stellar density available)"
        self.candidate.duration_h = float(dur)
        self.notes.append(
            f"no transit duration was supplied; estimated {dur:.2f} h from "
            f"{how}. This is a starting value, not a measurement -- run "
            f"pc_fit and pass --use-fit for a fitted one.")

    def _apply_fit_refinement(self) -> None:
        path = self.out / self.target.slug / (
            f"fit_c{self.candidate.index}.json" if self.candidate.index else "fit.json")
        if not path.exists():
            self.notes.append(f"--use-fit: no {path.name}, using the input ephemeris")
            return
        try:
            fit = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.notes.append(f"--use-fit: could not read {path.name}: {exc}")
            return
        if fit.get("status") in (Status.ERROR, Status.INCONCLUSIVE):
            self.depth_unverified = True
            # An inconclusive fit is not a refinement, but it may still be the
            # only duration anyone has: search output routinely carries a
            # period and an epoch and no duration at all.  Take the duration
            # if we have none, take nothing else, and say so.
            m = (fit.get("metrics") or {})
            dur = m.get("duration_h")
            if self.candidate.duration_h is None and dur and np.isfinite(float(dur)):
                self.candidate.duration_h = float(dur)
                self.notes.append(
                    f"--use-fit: fit.json is {fit['status']}, so the ephemeris "
                    f"is the input one, but its T14 = {dur:.2f} h was the only "
                    f"duration available and has been adopted")
            else:
                self.notes.append(
                    f"--use-fit: fit.json is {fit['status']}, using the input "
                    f"ephemeris")
            return
        m = fit.get("metrics") or {}
        before = (self.candidate.period, self.candidate.t0_btjd, self.candidate.duration_h)
        for key, attr in (("period", "period"), ("t0_btjd", "t0_btjd"),
                          ("duration_h", "duration_h"), ("depth_ppm", "depth_ppm")):
            if m.get(key) is not None and np.isfinite(float(m[key])):
                setattr(self.candidate, attr, float(m[key]))
        self.candidate.source = "pc_fit"
        self.notes.append(
            f"--use-fit: ephemeris from {path.name} "
            f"(P {before[0]} -> {self.candidate.period}, "
            f"t0 {before[1]} -> {self.candidate.t0_btjd})")


# --------------------------------------------------------------------------
# resolution helpers


def _resolve_target(args) -> Target:
    tic = getattr(args, "tic", None)
    gaia = getattr(args, "gaia", None)
    if tic is None and gaia is None and not getattr(args, "lc", None):
        raise SystemExit("need at least one of --tic, --gaia or --lc")
    return Target(tic=tic, gaia_dr3=gaia, name=getattr(args, "name", None),
                  ra=getattr(args, "ra", None), dec=getattr(args, "dec", None))


def _to_btjd(t0: float | None) -> float | None:
    """Accept BJD or BTJD and return BTJD, so no caller has to remember which."""
    if t0 is None:
        return None
    t0 = float(t0)
    return t0 - 2_457_000.0 if t0 > 2_400_000 else t0


#: Column aliases in a candidate table, best first.
#: Column names accepted for each field, best first.  Real search output does
#: not agree on any of them -- one table calls the period ``period_d`` and the
#: depth ``depth_gp_ppm``, another ``pl_orbper`` and ``pl_trandep`` -- and a
#: table that silently fails to match is worse than one that errors, because
#: every test then reports "no ephemeris" and the run looks like a data
#: problem.
_CAND_COLS = {
    "period": ["period", "per", "pl_orbper", "P", "period_days", "period_d",
               "period_refined"],
    "t0_btjd": ["t0_btjd", "t0", "T0", "pl_tranmid", "epoch", "t0_refined"],
    "duration_h": ["duration_h", "duration", "T14", "dur_hours", "pl_trandur",
                   "duration_hours"],
    "depth_ppm": ["depth_ppm", "depth", "pl_trandep", "Depth", "depth_gp_ppm",
                  "depth_fit_ppm"],
    "tic": ["tic", "tic_id", "TIC", "ticid"],
    "label": ["label", "pl_name", "name", "candidate"],
}


def _col(row, key):
    for c in _CAND_COLS[key]:
        if c in row and row[c] is not None:
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                if key == "label":
                    return str(row[c])
                continue
            if np.isfinite(v):
                return v
    return None


def _read_table(path):
    import pandas as pd
    p = str(path)
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    if p.endswith((".json", ".jsonl")):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    return pd.read_csv(p)


def _resolve_candidate(args, target: Target) -> Candidate:
    idx = getattr(args, "index", 0) or 0
    cand = Candidate(
        period=getattr(args, "period", None),
        t0_btjd=_to_btjd(getattr(args, "t0", None)),
        duration_h=getattr(args, "duration", None),
        depth_ppm=getattr(args, "depth", None),
        index=idx, source="command line",
    )

    path = getattr(args, "candidates", None)
    if path:
        df = _read_table(path)
        row_no = getattr(args, "row", None)
        if row_no is not None:
            row = df.iloc[int(row_no)]
        else:
            if target.tic is None:
                raise SystemExit("--candidates needs --row or --tic")
            m = None
            for c in _CAND_COLS["tic"]:
                if c in df.columns:
                    key = df[c].astype(str).str.replace("TIC", "", regex=False).str.strip()
                    m = key == str(int(target.tic))
                    if m.any():
                        break
            if m is None or not m.any():
                raise SystemExit(f"TIC {target.tic} not in {path}")
            sub = df[m]
            if idx >= len(sub):
                raise SystemExit(
                    f"--index {idx} but only {len(sub)} row(s) for TIC {target.tic}")
            row = sub.iloc[idx]

        d = row.to_dict()
        cand = Candidate(
            period=_col(d, "period") or cand.period,
            t0_btjd=_to_btjd(_col(d, "t0_btjd")) or cand.t0_btjd,
            duration_h=_col(d, "duration_h") or cand.duration_h,
            depth_ppm=_col(d, "depth_ppm") or cand.depth_ppm,
            index=idx, source=f"{Path(path).name}",
        )
        if target.tic is None:
            tv = _col(d, "tic")
            if tv:
                target.tic = int(tv)
        label = _col(d, "label")
        if label and target.name in (None, f"TIC {target.tic}"):
            target.name = str(label)

    # Depths are quoted as a fraction in some tables (pl_trandep is a percent
    # in the NASA archive).  Normalise to ppm, and say so rather than guessing
    # silently downstream.
    if cand.depth_ppm is not None and 0 < cand.depth_ppm < 100:
        cand.depth_ppm = cand.depth_ppm * 1e4   # percent -> ppm
        cand.source = (cand.source or "") + " (depth read as percent)"

    return cand


def _resolve_known(args, target: Target, ctx: Context) -> list:
    """Known signals on this star: from --known, or from the candidate table."""
    known: list[KnownSignal] = []
    spec = getattr(args, "known", None) or []

    for item in spec:
        p = Path(item)
        if p.exists():
            payload = json.loads(p.read_text())
            entries = payload if isinstance(payload, list) else payload.get("known", [])
            for e in entries:
                if target.tic is not None and e.get("tic") not in (None, target.tic):
                    continue
                known.append(KnownSignal(
                    label=str(e.get("label", "known")),
                    period=float(e["period"]), t0_btjd=_to_btjd(e["t0_btjd"]),
                    duration_h=float(e["duration_h"]),
                    depth_ppm=e.get("depth_ppm"), kind=e.get("kind", "planet")))
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise SystemExit(
                f"--known entry {item!r} is neither a file nor 'label:P:t0:duration_h'")
        known.append(KnownSignal(label=parts[0], period=float(parts[1]),
                                 t0_btjd=_to_btjd(float(parts[2])),
                                 duration_h=float(parts[3])))

    # Other rows of the same candidate table on the same star are, for
    # modelling purposes, known signals too.
    path = getattr(args, "candidates", None)
    if path and target.tic is not None and not getattr(args, "no_auto_known", False):
        try:
            df = _read_table(path)
        except Exception:  # noqa: BLE001 - already reported by _resolve_candidate
            df = None
        if df is not None:
            for c in _CAND_COLS["tic"]:
                if c in df.columns:
                    key = df[c].astype(str).str.replace("TIC", "", regex=False).str.strip()
                    sub = df[key == str(int(target.tic))]
                    for j, (_, row) in enumerate(sub.iterrows()):
                        if j == (getattr(args, "index", 0) or 0):
                            continue
                        d = row.to_dict()
                        per, t0 = _col(d, "period"), _to_btjd(_col(d, "t0_btjd"))
                        dur = _col(d, "duration_h")
                        if per and t0 is not None and dur:
                            known.append(KnownSignal(
                                label=str(_col(d, "label") or f"row{j}"),
                                period=per, t0_btjd=t0, duration_h=dur,
                                depth_ppm=_col(d, "depth_ppm"), kind="candidate"))
                    break
    return known


# --------------------------------------------------------------------------
# finishing


def require(result: Result, condition: bool, reason: str) -> bool:
    """Mark ``result`` INCONCLUSIVE and return False when ``condition`` fails.

    The idiom for rule 8: missing data is normal and must produce
    INCONCLUSIVE, never a crash and never a silent pass.
    """
    if condition:
        return True
    result.inconclusive(reason)
    return False


def finish(result: Result, ctx: Context) -> int:
    """Write the JSON, print a one-line summary, and return an exit code.

    The exit code is 0 for every status a test can legitimately reach,
    including FAIL -- a refuted candidate is a successful test run.  Only
    ERROR returns non-zero, so a batch driver can tell a crash from a verdict.
    """
    path = result.write(ctx.out)
    if not getattr(ctx.args, "quiet", False):
        print(result.summary_line())
        print(f"  -> {path}")
        for p in result.plots:
            print(f"  -> {p}")
        for note in result.provenance.get("notes", []):
            print(f"  note: {note}")
    if getattr(ctx.args, "json", False):
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    if result.status == Status.ERROR and result.traceback and not getattr(
            ctx.args, "quiet", False):
        print(result.traceback, file=sys.stderr)
    return 1 if result.status == Status.ERROR else 0

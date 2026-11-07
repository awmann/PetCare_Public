"""The driver: run the tests and merge their JSONs and vetting metrics.

It runs each test by building that test's own argument parser and calling its
``run(ctx)``.  Nothing is reimplemented here, and the driver has no vetting
logic of its own -- it merges JSONs, it does not re-derive anything.  A test
must behave identically whether it is invoked from here or from the command
line, and reusing its parser is what guarantees that.

**No overall score is computed.**  The per-test statuses are reported side by
side, because *which* test objected is the information and a scalar hides it.
The standing preference on this project is to let a few bad candidates through
and disprove them with follow-up rather than to build an aggressive automatic
cut; a single number invites the opposite.

This module is deliberately free of hardcoded paths.  The light-curve roots,
the stellar catalogue and the candidate list are inputs.  The local script
that knows where those live on a particular machine lives outside the
repository.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import cli
from . import metrics as vetting_metrics
from .result import Status, utc_now

#: Every test, with its tier and whether it consumes ``pc_fit``'s output.
#: Tier 0 is catalogue-only, 1 is light-curve-only, 2 needs pixels, 3 hits
#: external archives.  Order within a tier is the order they run in.
TESTS = [
    # (name, module, tier, uses_fit)
    ("stellar",    "pc_stellar",    0, False),
    ("followup",   "pc_followup",   3, True),
    ("astrometry", "pc_astrometry", 0, False),
    ("neighbours", "pc_neighbours", 0, True),
    ("fit",        "pc_fit",        1, False),
    ("depths",     "pc_depths",     1, True),
    ("odd_even",   "pc_oddeven",    1, True),
    ("shape",      "pc_shape",      1, True),
    ("duration",   "pc_duration",   1, True),
    ("companion",  "pc_companion",  1, True),
    ("secondary",  "pc_secondary",  1, True),
    ("instrument", "pc_instrument", 1, True),
    ("rotation",   "pc_rotation",   1, True),
    ("phasecurve", "pc_phasecurve", 1, True),
    ("shapetime",  "pc_shapetime",  1, True),
    ("interp",     "pc_interp",     1, True),
    ("jointgp",    "pc_jointgp",    1, True),
    ("uniqueness", "pc_uniqueness", 1, True),
    ("flares",     "pc_flares",     1, True),
    ("centroid",   "pc_centroid",   2, True),
    ("pixels",     "pc_pixels",     2, True),
    ("archival",   "pc_archival",   3, True),
    ("crossmatch", "pc_crossmatch", 3, True),
]

TEST_BY_NAME = {name: (mod, tier, uses_fit) for name, mod, tier, uses_fit in TESTS}


@dataclass
class DriverOptions:
    """Everything the driver needs, all of it passed in."""

    out: str = "out"
    tiers: tuple = (0, 1)
    only: tuple = ()
    skip: tuple = ()
    catalog: str | None = None
    lc_root: tuple = ()
    lc: tuple = ()
    known: tuple = ()
    no_plot: bool = False
    quiet: bool = False
    extra: dict = field(default_factory=dict)   # {test_name: [cli tokens]}

    def selected(self) -> list:
        names = [n for n, _, tier, _ in TESTS if tier in self.tiers]
        if self.only:
            names = [n for n in names if n in self.only]
            # --only overrides the tier filter: asking for a test by name is
            # asking for it, not asking whether its tier was enabled
            for n in self.only:
                if n not in names and n in TEST_BY_NAME:
                    names.append(n)
            names = [n for n, *_ in TESTS if n in names]
        names = [n for n in names if n not in self.skip]

        # A test that reads the fit has to run after it, whatever tier it is
        # in.  pc_neighbours is tier 0 and needs the depth: run in catalogue
        # order it has no depth to attribute to a neighbour and returns
        # INCONCLUSIVE on every candidate whose input table lacks one.
        if "fit" in names:
            before = [n for n in names
                      if n != "fit" and not TEST_BY_NAME[n][2]]
            after = [n for n in names if n != "fit" and TEST_BY_NAME[n][2]]
            names = before + ["fit"] + after
        return names


# --------------------------------------------------------------------------


def run_candidate(target: dict, candidate: dict, opts: DriverOptions) -> dict:
    """Run the selected tests on one candidate and merge the records.

    ``target`` carries ``tic``/``gaia``/``name``/``ra``/``dec``;
    ``candidate`` carries ``period``/``t0``/``duration``/``depth``/``index``.
    Both are plain dicts so the caller need not import anything from here.
    """
    records, order = {}, []
    t_start = time.time()

    for name in opts.selected():
        module, _, uses_fit = TEST_BY_NAME[name]
        started = time.time()
        try:
            record = _run_one(name, module, target, candidate, opts, uses_fit)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - a crashing test must not
            # take the candidate down with it; record it and carry on
            record = {"test": name, "status": Status.ERROR,
                      "reason": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        record["seconds"] = round(time.time() - started, 2)
        records[name] = record
        order.append(name)
        if not opts.quiet:
            _print_line(name, record)

    merged = {
        "schema": "petcare-merged/1.0",
        "run_utc": utc_now(),
        "seconds": round(time.time() - t_start, 2),
        "target": target,
        "candidate": candidate,
        "tests_run": order,
        "tests": records,
        "summary": summarise(records),
        "note": ("no overall score is computed on purpose: which test "
                 "objected is the information, and a scalar hides it"),
    }

    slug = _slug(target)
    outdir = Path(opts.out) / slug
    outdir.mkdir(parents=True, exist_ok=True)
    idx = candidate.get("index") or 0
    path = outdir / (f"petcare_c{idx}.json" if idx else "petcare.json")
    # Merge into the previous record rather than replacing it.  The intended
    # workflow is tiers 0-1 over everything and then tier 2 on a shortlist, and
    # a --only run that replaced the record threw away the twelve tests that
    # had already run -- which is how a tier-2 pass on one candidate list left
    # each merged record holding one test.  Tests not re-run are carried
    # forward, with the newer answer winning wherever both exist.
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            old = {}
        if old.get("human"):
            merged["human"] = old["human"]
        kept = {k: v for k, v in (old.get("tests") or {}).items()
                if k not in records}
        if kept:
            merged["tests"] = {**kept, **records}
            merged["tests_run"] = ([n for n, *_ in TESTS
                                    if n in merged["tests"]]
                                   + [n for n in merged["tests"]
                                      if n not in TEST_BY_NAME])
            merged["summary"] = summarise(merged["tests"])
            merged["carried_forward"] = sorted(kept)
            merged["ran_this_pass"] = list(order)
    # Reported, never graded: recomputed from every test on record, so a --only
    # pass refreshes it too (see petcare.metrics).
    try:
        merged["vetting_metrics"] = vetting_metrics.collect(merged)
    except Exception as exc:  # noqa: BLE001 -- the metrics must never cost a verdict
        merged["vetting_metrics"] = {"version": vetting_metrics.VERSION,
                                     "error": f"{type(exc).__name__}: {exc}"}
    path.write_text(json.dumps(merged, indent=2, allow_nan=False, default=str))
    merged["path"] = str(path)
    return merged


def _run_one(name, module_name, target, candidate, opts, uses_fit) -> dict:
    import importlib

    mod = importlib.import_module(f".scripts.{module_name}", package="petcare")
    parser = cli.base_parser(name, mod.__doc__ or name)
    if hasattr(mod, "add_arguments"):
        parser = mod.add_arguments(parser)

    argv = _argv(target, candidate, opts, uses_fit)
    argv += list(opts.extra.get(name, []))
    args = parser.parse_args(argv)
    ctx = cli.Context.from_args(args)
    result = mod.run(ctx)
    result.write(ctx.out)
    return result.to_dict()


def _argv(target, candidate, opts, uses_fit) -> list:
    argv = []
    for key, flag in (("tic", "--tic"), ("gaia", "--gaia"), ("name", "--name"),
                      ("ra", "--ra"), ("dec", "--dec"), ("age_myr", "--age"),
                      ("prot", "--prot")):
        if target.get(key) is not None:
            argv += [flag, str(target[key])]
    for key, flag in (("period", "--period"), ("t0", "--t0"),
                      ("duration", "--duration"), ("depth", "--depth"),
                      ("index", "--index")):
        if candidate.get(key) is None:
            continue
        # A non-positive search depth is no depth at all (and "--depth -12"
        # is not even parseable): the tests then treat the depth as unknown.
        if key == "depth" and float(candidate[key]) <= 0:
            continue
        argv += [f"{flag}={candidate[key]}"]

    argv += ["--out", str(opts.out)]
    if opts.catalog:
        argv += ["--catalog", str(opts.catalog)]
    if opts.lc_root:
        argv += ["--lc-root", *[str(p) for p in opts.lc_root]]
    if opts.lc:
        argv += ["--lc", *[str(p) for p in opts.lc]]
    # Known signals: the driver's global list plus anything specific to this
    # candidate (the other planets on the same star, which every modelling
    # test has to mask).
    known = list(opts.known) + list(candidate.get("known") or [])
    if known:
        argv += ["--known", *[str(k) for k in known]]
    if opts.no_plot:
        argv += ["--no-plot"]
    argv += ["--quiet"]
    if uses_fit:
        argv += ["--use-fit"]
    return argv


# --------------------------------------------------------------------------


def summarise(records: dict) -> dict:
    """Counts by status, and the tests that objected, worst first."""
    counts = {s: 0 for s in Status.ALL}
    for rec in records.values():
        counts[rec.get("status", Status.ERROR)] = \
            counts.get(rec.get("status", Status.ERROR), 0) + 1

    def sev(item):
        return Status.SEVERITY.get(item[1].get("status"), 9)

    ranked = sorted(records.items(), key=sev)
    return {
        "counts": counts,
        "n_tests": len(records),
        "objections": [
            {"test": n, "status": rec.get("status"), "reason": rec.get("reason")}
            for n, rec in ranked
            if rec.get("status") in (Status.FAIL, Status.FLAG, Status.ERROR)],
        "inconclusive": [n for n, rec in records.items()
                         if rec.get("status") == Status.INCONCLUSIVE],
        "passed": [n for n, rec in records.items()
                   if rec.get("status") == Status.PASS],
    }


def summary_row(merged: dict) -> dict:
    """One flat row per candidate, for the cross-target summary table."""
    t, c = merged.get("target", {}), merged.get("candidate", {})
    row = {"tic": t.get("tic"), "name": t.get("name"),
           "candidate_index": c.get("index") or 0,
           "period": c.get("period"), "t0_btjd": c.get("t0"),
           "duration_h": c.get("duration"), "depth_ppm": c.get("depth"),
           "n_known_signals": len(c.get("known") or [])}

    fit = (merged.get("tests", {}).get("fit") or {}).get("metrics") or {}
    for key in ("period", "t0_btjd", "depth_ppm", "duration_h", "b", "rp_rs",
                "rp_earth", "gp_sigma_over_depth"):
        if fit.get(key) is not None:
            row[f"fit_{key}"] = fit[key]

    for name, rec in (merged.get("tests") or {}).items():
        row[f"{name}_status"] = rec.get("status")
        row[f"{name}_reason"] = rec.get("reason")

    s = merged.get("summary", {})
    row["n_fail"] = s.get("counts", {}).get(Status.FAIL, 0)
    row["n_flag"] = s.get("counts", {}).get(Status.FLAG, 0)
    row["n_inconclusive"] = s.get("counts", {}).get(Status.INCONCLUSIVE, 0)
    row["n_error"] = s.get("counts", {}).get(Status.ERROR, 0)
    row["objections"] = "; ".join(
        f"{o['test']}:{o['status']}" for o in s.get("objections", []))
    # the vetting metrics, one column each, so a run is one table to sort and cut
    for key, val in ((merged.get("vetting_metrics") or {}).get("metrics") or {}).items():
        if not isinstance(val, (dict, list)):
            row[f"m_{key}"] = val
    return row


def _print_line(name, record) -> None:
    status = record.get("status", "?")
    reason = (record.get("reason") or "")[:96]
    print(f"    {name:<12s} {status:<13s} {reason}")


def _slug(target: dict) -> str:
    if target.get("tic"):
        return f"TIC{int(target['tic'])}"
    if target.get("gaia"):
        return f"GDR3{int(target['gaia'])}"
    name = target.get("name") or "target"
    return "".join(ch if ch.isalnum() else "_" for ch in name)

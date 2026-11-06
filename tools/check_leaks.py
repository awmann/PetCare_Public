#!/usr/bin/env python
"""Fail if the tree contains things that do not belong in a public repository.

Looks for target IDs outside ``tools/leak_allowlist.txt``, home-directory and
machine paths, and terms from private projects.  With ``--denylist FILE`` it
also fails on any bare number listed in that file (one per line), which catches
an ID with no "TIC" in front of it.  Keep that file outside the repository: a
list of what must not be published is itself something that must not be
published.

    python tools/check_leaks.py
    python tools/check_leaks.py --denylist /path/outside/the/repo/denylist.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = Path(__file__).with_name("leak_allowlist.txt")

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist",
             "out", "cache"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".fits", ".npz", ".gz", ".zip"}
#: This tool and its list have to spell out what they look for.
SKIP_FILES = {"tools/check_leaks.py", "tools/leak_allowlist.txt"}

TARGET_PATTERNS = {
    "TIC": re.compile(r"\bTIC[ _-]?(\d{5,10})\b"),
    "tic flag": re.compile(r"--tic[ =](\d{5,10})\b"),
    "TOI": re.compile(r"\b(?:C?TOI)[ _-]?(\d{3,5})\b"),
    "PATHOS": re.compile(r"\bPATHOS-(\d+)\b"),
}
TERMS = {
    "machine path": re.compile(r"/Users/|/Volumes/|/home/[a-z]"),
    "synced folder": re.compile(r"\bDropbox\b|CloudStorage"),
    "private directory": re.compile(r"petcare_local|tidye_lc_run|tidye_vet|run_petcare"),
    "machine name": re.compile(r"WanShiTong|Hawky|ThunderBay|SPHERExData"),
    "private project": re.compile(r"\bTIDYE\b|\bVela\b|\bFRANK\b"),
    "initials": re.compile(r"\bAWM\b|\bMGB\b"),
}
BARE_NUMBER = re.compile(r"(?<![\d.])\d{6,10}(?![\d.])")


def tracked_files(root: Path) -> list[Path]:
    """Files under ``root``, skipping build products and data."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & SKIP_DIRS:
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES or rel.as_posix() in SKIP_FILES:
            continue
        out.append(p)
    return out


def load_allowlist(path: Path = ALLOWLIST) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def scan_text(text: str, allow: set[str], deny: set[str]) -> list[tuple[int, str, str]]:
    """(line number, kind, match) for everything objectionable in ``text``."""
    hits = []
    allowed_numbers = {a.split()[-1] for a in allow}
    for n, line in enumerate(text.splitlines(), start=1):
        for kind, pat in TARGET_PATTERNS.items():
            for m in pat.finditer(line):
                ident = {"tic flag": "TIC", "TIC": "TIC"}.get(kind, kind)
                if f"{ident} {m.group(1)}" not in allow:
                    hits.append((n, kind, m.group(0)))
        for kind, pat in TERMS.items():
            for m in pat.finditer(line):
                hits.append((n, kind, m.group(0)))
        if deny:
            for m in BARE_NUMBER.finditer(line):
                if m.group(0) in deny and m.group(0) not in allowed_numbers:
                    hits.append((n, "denylisted number", m.group(0)))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--denylist", type=Path,
                    help="file of numbers that must not appear; keep it outside the repo")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--summary", action="store_true", help="counts per file only")
    args = ap.parse_args(argv)

    allow = load_allowlist()
    deny = (set(args.denylist.read_text().split()) if args.denylist else set())
    if args.denylist:
        inside = args.denylist.resolve().is_relative_to(args.root.resolve())
        tracked = subprocess.run(["git", "-C", str(args.root), "rev-parse"],
                                 capture_output=True).returncode == 0
        if inside and tracked:
            print(f"refusing: {args.denylist} is inside the repository")
            return 2

    total = 0
    for path in tracked_files(args.root):
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        hits = scan_text(text, allow, deny)
        total += len(hits)
        rel = path.relative_to(args.root)
        if hits and args.summary:
            print(f"{len(hits):4d}  {rel}")
        elif hits:
            for n, kind, match in hits:
                print(f"{rel}:{n}: {kind}: {match}")
    print(f"{total} problem(s)" if total else "clean")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

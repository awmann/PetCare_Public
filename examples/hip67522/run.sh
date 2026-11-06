#!/bin/sh
# Vet both planets of HIP 67522 from the light curves in data/. Tier 2 downloads
# about 170 MB of TESS cutouts on the first run.
cd "$(dirname "$0")" || exit 1
petcare --candidates candidates.csv --lc-root data --tiers 0 1 2 --out out

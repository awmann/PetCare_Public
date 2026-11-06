#!/bin/sh
# Vet both planets of DS Tuc A from the light curves in data/. Tiers 0 and 1 take
# about a minute; tier 2 adds a download of about 400 MB of TESS cutouts.
cd "$(dirname "$0")" || exit 1
petcare --candidates candidates.csv --lc-root data --tiers 0 1 2 --out out

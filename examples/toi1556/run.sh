#!/bin/sh
# Vet TOI-1556.01 from the light curves in data/. Tier 2 downloads TESS cutouts on the
# first run; tier 3 needs DEATHSTAR and a ZTF (IRSA) login, and reports INCONCLUSIVE
# without them.
cd "$(dirname "$0")" || exit 1
petcare --candidates candidates.csv --lc-root data --tiers 0 1 2 3 --out out

#!/usr/bin/env bash
# Evaluate the v4 secret-key model against v3 (Part 1 of the final phase).
# Runs every test one after another (parallel torch processes thrash the CPU)
# and keeps each test's console output in output/v4_eval/<test>.log.
#
#   bash scripts/evaluate_v4.sh [checkpoints/hidden_v4_best.pt]
#
# Expected total on an M1 Pro CPU: about 3-4 hours (most of it test_v3_vs_v2 and
# test_overwrite_cost, which build 500-decoy informed checks per photo).
# PYTHON can point to another interpreter, e.g. PYTHON=~/Desktop/.../.venv/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."
V4=${1:-checkpoints/hidden_v4_best.pt}
PY=${PYTHON:-python}
LOG=output/v4_eval
mkdir -p "$LOG"
[ -f "$V4" ] || { echo "missing $V4"; exit 1; }

run() {
  local name=$1; shift
  echo "== $name  ($(date +%H:%M))"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$LOG/$name.log" | tail -3
  echo "   done in $(( (SECONDS - t0) / 60 )) min"
}

run v4_keys "$PY" tests/test_v4_keys.py "$V4" 20 20                              # ~10 min
run hidden_v2_eval "$PY" tests/test_hidden_v2_eval.py "$V4" data/products_png           # ~10 min
run v3_vs_v4 env ONLY=v3,v4 "$PY" tests/test_v3_vs_v2.py 30 v4="$V4"        # ~60-90 min
run overwrite_cost "$PY" tests/test_overwrite_cost.py 20 20 v4="$V4"                    # ~60-90 min
run rewatermark_v4 "$PY" tests/test_rewatermark.py 20 20 "$V4"                          # ~30 min
run rewatermark_v3 "$PY" tests/test_rewatermark.py 20 20 checkpoints/hidden_v3_best.pt  # ~30 min
run in_the_wild_v4 "$PY" tests/test_in_the_wild.py data/products_png 12 1.0 none "$V4"  # ~3 min
run in_the_wild_v3 "$PY" tests/test_in_the_wild.py data/products_png 12 1.0 none checkpoints/hidden_v3_best.pt
echo "All logs in $LOG/"

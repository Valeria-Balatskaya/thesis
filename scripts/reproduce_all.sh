#!/usr/bin/env bash
# Reproduce every thesis table and figure, one test after another (parallel torch
# processes thrash the CPU). Console output of each step: output/thesis/logs/<step>.log.
# At the end scripts/thesis_tables.py writes the tables (Markdown, with 95% Wilson
# intervals) to output/thesis/ and copies the figures to thesis/figures/.
#
#   bash scripts/reproduce_all.sh               # everything (~13-15 h on an M1 Pro CPU)
#   bash scripts/reproduce_all.sh v3_vs_v4 ablation tables    # only these steps
#   PYTHON=/path/to/python bash scripts/reproduce_all.sh      # another interpreter
#
# Needs: checkpoints/hidden_v3_best.pt (+ hidden_v4_best.pt, hidden_v2_best.pt,
# hidden_final.pt for the history rows), data/products_png, data/abo_sample,
# data/coco_sample, data/abo_negatives (scripts/fetch_reference_images.py), and for the
# baselines: bash scripts/get_baselines.sh. Steps whose inputs are missing are skipped.
# Expected runtimes are for an Apple M1 Pro, CPU only, nothing else running.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=output/thesis/logs
mkdir -p "$LOG" output/thesis thesis/figures
V4=checkpoints/hidden_v4_best.pt
V4ARG=$([ -f "$V4" ] && echo "v4=$V4" || echo "")

step() {   # step <name> <expected> <command...>
  local name=$1 expected=$2; shift 2
  if [ ${#SELECTED[@]} -gt 0 ] && [[ ! " ${SELECTED[*]} " =~ " $name " ]]; then return; fi
  echo "== $name (expected ~$expected)  started $(date +%H:%M)"
  local t0=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "   ok in $(( (SECONDS - t0) / 60 )) min"
  else
    echo "   FAILED (see $LOG/$name.log)"
  fi
}
SELECTED=("$@")

# Ch. 3 — e-commerce images vs regular photos (1.2.1)
step ecommerce_analysis "2 min"   "$PY" scripts/analyse_ecommerce_images.py
step ecommerce_vs_regular "40 min" "$PY" tests/test_ecommerce_vs_regular.py 30
# Ch. 6.2/6.3 — visibility and robustness, v2 vs v3 (vs v4), 23 attacks (1.2.2, 1.2.3)
step v3_vs_v4 "90-150 min"        "$PY" tests/test_v3_vs_v2.py 30 $V4ARG
step hidden_eval_own "10 min"     "$PY" tests/test_hidden_v2_eval.py checkpoints/hidden_v3_best.pt data/products_png
step in_the_wild "5 min"          "$PY" tests/test_in_the_wild.py data/products_png 12 1.0 none checkpoints/hidden_v3_best.pt
# Ch. 6.4 — alignment (legacy v2 benchmark, as in CLAUDE.md)
step defence_benchmark "10 min"   "$PY" tests/test_defence_benchmark.py data/products_png 35 0.5
step aspect_ratios "10 min"       "$PY" tests/test_aspect_ratios.py
# Ch. 6.5 — overwriting and re-watermarking (1.2.5)
step rewatermark "40 min"         "$PY" tests/test_rewatermark.py 20 20 checkpoints/hidden_v3_best.pt
step overwrite_cost "90-150 min"  "$PY" tests/test_overwrite_cost.py 20 20 $V4ARG
step overwritten_verdict "40 min" "$PY" tests/test_overwritten_verdict.py 20 20
if [ -f "$V4" ]; then
  step v4_keys "15 min"           "$PY" tests/test_v4_keys.py "$V4" 20 20
fi
# Ch. 6.6 — copy attack / content binding
step copy_attack "20 min"         "$PY" tests/test_copy_attack.py 20 20
# Ch. 6.7 — real captures (only once data/real_world/ has captures)
if [ -f data/real_world/captures.csv ]; then
  step real_world "5 min"         "$PY" tests/test_real_world.py
fi
# Ch. 6.8 — overhead (1.2.4)
step overhead "15 min"            "$PY" tests/test_overhead.py
# Ch. 6.9 — pretrained baselines (TrustMark, Watermark Anything)
step baselines "3-4 h"            "$PY" tests/test_baselines.py 30
# Ch. 6.10 — ablation; appendix: parameter study and texture masks (v2 era)
step ablation "60 min"            "$PY" tests/test_ablation.py 30 30 500
step parameter_study "40 min"     "$PY" tests/test_parameter_study.py 30
step texture_masks "60 min"       "$PY" tests/test_texture_masks.py 30 30
# tables + figures
step tables "10 s"                "$PY" scripts/thesis_tables.py
echo "Tables: output/thesis/*.md   Figures: thesis/figures/   Logs: $LOG/"

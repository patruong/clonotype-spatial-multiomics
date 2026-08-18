#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-mofaflex/bin/python}"

cd "$ROOT"

OUT="results/integration/mofaflex/f12/sign_aligned_factors/patient5_bc2059_V13Y10-038_B1_Free_03__vs__bc2004_V13Y10-060_B1_Free_07"

rm -rf "$OUT"

"$PY" \
  scripts/integration/mofaflex/04_plot_sign_aligned_factors.py \
  --patients 5 \
  --left-factors Free_03 \
  --top-loadings 10 \
  --right-rotation cw \
  --crop-padding 0.03 \
  --spot-size-scale 0.045

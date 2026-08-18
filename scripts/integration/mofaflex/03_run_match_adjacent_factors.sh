#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-mofaflex/bin/python}"

cd "$ROOT"

rm -rf \
  results/integration/mofaflex/f12/adjacent_factor_matching

"$PY" \
  scripts/integration/mofaflex/03_match_adjacent_factors.py

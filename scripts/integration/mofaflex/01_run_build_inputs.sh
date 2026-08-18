#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-mofaflex/bin/python}"

cd "$ROOT"

rm -rf data/intermediate/integration/mofaflex/inputs

"$PY" \
  scripts/integration/mofaflex/01_build_inputs.py

echo
echo "Built Figure 2 MOFA-FLEX inputs:"
cat data/intermediate/integration/mofaflex/inputs/qc_summary.tsv

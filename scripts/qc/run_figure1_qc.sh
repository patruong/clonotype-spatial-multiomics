#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-python/bin/python}"

cd "$ROOT"

echo "=== Figure 1: modality QC ==="
"$PY" scripts/qc/01_figure1_modality_qc.py

echo
echo "=== Figure 1: clonotype QC ==="
"$PY" scripts/qc/02_figure1_clonotype_qc.py

echo
echo "=== Done ==="

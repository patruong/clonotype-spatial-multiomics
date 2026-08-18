#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-python/bin/python}"

cd "$ROOT"

echo "=== MSI NMF: fit models ==="
"$PY" scripts/msi/nmf/01_run_msi_nmf.py   --random-state 42

echo
echo "=== MSI NMF: select non-technical factors ==="
"$PY" scripts/msi/nmf/02_select_nontechnical_factors.py

echo
echo "=== MSI NMF: plot selected loadings ==="
"$PY" scripts/msi/nmf/03_plot_selected_loadings.py

echo
echo "=== Done ==="

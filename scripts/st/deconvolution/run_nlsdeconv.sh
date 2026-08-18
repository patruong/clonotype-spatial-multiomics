#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-python/bin/python}"

cd "$ROOT"

echo "=== NLSDeconv ==="
"$PY" scripts/st/deconvolution/01_run_nlsdeconv.py

echo
echo "=== Plot NLSDeconv spatial results ==="
"$PY" scripts/st/deconvolution/02_plot_nlsdeconv_spatial.py

echo
echo "=== Done ==="

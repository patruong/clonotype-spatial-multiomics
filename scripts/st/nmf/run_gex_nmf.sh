#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-python/bin/python}"

cd "$ROOT"

echo "=== GEX NMF ==="
"$PY" scripts/st/nmf/01_run_gex_nmf.py   --random-state 42

echo "=== Done ==="

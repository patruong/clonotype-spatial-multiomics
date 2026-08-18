#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-python/bin/python}"

cd "$ROOT"

echo "=== Build sample manifest ==="
"$PY" scripts/metadata/build_sample_manifest.py

echo "=== Done ==="

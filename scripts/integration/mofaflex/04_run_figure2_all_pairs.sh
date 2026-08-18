#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-mofaflex/bin/python}"

SCRIPT="$ROOT/scripts/integration/mofaflex/04_plot_sign_aligned_factors.py"
OUT="$ROOT/results/integration/mofaflex/f12/sign_aligned_factors"

cd "$ROOT"

echo "=== Figure 2: all adjacent-section MOFA-FLEX pairs ==="
echo "Top loadings per pole: 10"
echo "Crop padding: 0.03"
echo "Spot-size scale: 0.045"
echo

# Remove previous output so no stale orientations remain.
rm -rf "$OUT"

run_pair() {
    local patient="$1"
    local factor="$2"
    local rotation="$3"
    local flip_horizontal="$4"

    echo
    echo "------------------------------------------------------------"
    echo "Patient: $patient"
    echo "Left factor: $factor"
    echo "Right rotation: $rotation"
    echo "Right horizontal flip: $flip_horizontal"
    echo "------------------------------------------------------------"

    args=(
        --patients "$patient"
        --left-factors "$factor"
        --top-loadings 10
        --right-rotation "$rotation"
        --crop-padding 0.03
        --spot-size-scale 0.045
    )

    if [[ "$flip_horizontal" == "yes" ]]; then
        args+=(--right-flip-horizontal)
    fi

    "$PY" "$SCRIPT" "${args[@]}"
}


# Patient 3
# bc2067 9AA Free_11
# bc2012 DHBA Free_12
# bc2012: 180 degrees
run_pair 3 Free_11 180 no


# Patient 4
# bc2051 9AA Free_03
# bc2091 DHBA Free_07
# bc2091: horizontal flip + 180 degrees
run_pair 4 Free_03 180 yes


# Patient 5
# bc2059 9AA Free_03
# bc2004 DHBA Free_07
# bc2004: 90 degrees clockwise
run_pair 5 Free_03 cw no


# Patient 7
# bc2075 9AA Free_02
# bc2020 DHBA Free_11
# bc2020: no rotation
run_pair 7 Free_02 none no


echo
echo "============================================================"
echo "Figure 2 plotting complete"
echo "Output:"
echo "$OUT"
echo "============================================================"

find "$OUT" -maxdepth 1 -mindepth 1 -type d | sort

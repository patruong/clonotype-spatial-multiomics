#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-/data/patrick.truong/conda/envs/sma-vdj-mofaflex/bin/python}"

INPUT="$ROOT/data/intermediate/integration/mofaflex/inputs"
OUT="$ROOT/results/integration/mofaflex/f12"
RUNNER="$ROOT/scripts/integration/mofaflex/02_run_mofaflex.py"

# Space-separated physical GPU IDs.
# Examples:
#   GPUS="0" ./02_run_mofaflex_f12.sh
#   GPUS="0 2 3 5" ./02_run_mofaflex_f12.sh
read -ra GPU_ARRAY <<< "${GPUS:-0}"

cd "$ROOT"

rm -rf "$OUT"
mkdir -p "$OUT/logs"

mapfile -t SAMPLES < <(
    tail -n +2 "$INPUT/qc_summary.tsv" | cut -f1
)

n=${#SAMPLES[@]}
ngpu=${#GPU_ARRAY[@]}

echo "=== MOFA-FLEX f12 ==="
echo "Samples: $n"
echo "GPUs: ${GPU_ARRAY[*]}"
echo

for ((batch_start=0; batch_start<n; batch_start+=ngpu)); do

    pids=()
    names=()
    gpus=()

    for ((i=0; i<ngpu; i++)); do

        idx=$((batch_start + i))
        (( idx >= n )) && break

        sample="${SAMPLES[$idx]}"
        gpu="${GPU_ARRAY[$i]}"

        echo "$(date '+%F %T') START GPU=$gpu $sample"

        CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY" "$RUNNER" \
          --samples "$sample" \
          --n-factors 12 \
          --seed 1 \
          --max-epochs 4000 \
          > "$OUT/logs/${sample}.log" 2>&1 &

        pids+=("$!")
        names+=("$sample")
        gpus+=("$gpu")
    done

    for i in "${!pids[@]}"; do

        if wait "${pids[$i]}"; then
            echo "$(date '+%F %T') DONE  GPU=${gpus[$i]} ${names[$i]}"
        else
            rc=$?
            echo "$(date '+%F %T') FAILED GPU=${gpus[$i]} ${names[$i]} rc=$rc"
            exit "$rc"
        fi

    done
done

echo
echo "Completed models:"
find "$OUT" -name mofaflex_results.npz -type f | wc -l

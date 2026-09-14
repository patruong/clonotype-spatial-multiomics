#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/cnv/infercnv"

NLS_ROOT="$ROOT/data/processed/st/nlsdeconv"
PATHOLOGY_DIR="$ROOT/data/processed/st/morphology_annotations/all_annotations"
GTF="$ROOT/data/references/genome/refdata-gex-GRCh38-2020-A/genes/genes.gtf"

OUT_ROOT="$ROOT/data/processed/cnv/infercnv/figure3"
PATHOLOGY_ROOT="$OUT_ROOT/pathology_filtered_nlsdeconv"
QC_ROOT="$OUT_ROOT/qc_nlsdeconv"
PREP_ROOT="$OUT_ROOT/prepared"

# For validating against the historical analysis, run with:
#   CNV_ENV=infercnv_r_nonecrosis bash ...
ENV_NAME="${CNV_ENV:-sma-vdj-st-cnv}"

MODE="noHMM"
RESOLUTION="0.70"
SEED="0"
THREADS="${THREADS:-8}"

SAMPLES=(
    V13Y10-061_D1
    V13Y10-060_D1
    V13Y10-038_D1
)

NAMES=(
    bc2043_V13Y10-061_D1_synthref
    bc2020_V13Y10-060_D1_synthref
    bc2075_V13Y10-038_D1_synthref
)

echo "=========================================="
echo "Figure 3 inferCNV workflow"
echo "=========================================="
echo "ROOT=$ROOT"
echo "ENV=$ENV_NAME"
echo "MODE=$MODE"
echo "RESOLUTION=$RESOLUTION"
echo "SEED=$SEED"
echo "THREADS=$THREADS"
echo

# -------------------------
# Check inputs
# -------------------------

test -f "$GTF"

for SAMPLE in "${SAMPLES[@]}"; do
    test -f "$NLS_ROOT/${SAMPLE}.nlsdeconv.h5ad"
done

for F in \
    00_attach_histopathology.py \
    01_prepare_spot_qc.py \
    02_prepare_synthetic_reference.py \
    03_run_infercnv.R \
    04_export_for_leiden.R \
    05_leiden_posthoc.py
do
    test -f "$SCRIPT_DIR/$F"
done

# Prevent accidental destruction of an existing completed run.
if [[ -d "$OUT_ROOT" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $OUT_ROOT"
    echo
    echo "Use FORCE=1 to replace it."
    exit 1
fi

if [[ "${FORCE:-0}" == "1" ]]; then
    rm -rf "$OUT_ROOT"
fi

mkdir -p "$PATHOLOGY_ROOT" "$QC_ROOT" "$PREP_ROOT"

echo "===== ENVIRONMENT ====="
conda run -n "$ENV_NAME" python -c '
import sys, scanpy, igraph, leidenalg
print("Python:", sys.version.split()[0])
print("scanpy:", scanpy.__version__)
print("igraph:", igraph.__version__)
print("leidenalg:", getattr(leidenalg, "__version__", "installed"))
'

conda run -n "$ENV_NAME" Rscript -e '
cat("R:", R.version.string, "\n")
cat("infercnv:", as.character(packageVersion("infercnv")), "\n")
'

# -------------------------
# 1. Attach pathology / remove necrosis
# -------------------------

echo
echo "===== 1/6 HISTOPATHOLOGY / NECROSIS FILTER ====="

conda run -n "$ENV_NAME" \
    python "$SCRIPT_DIR/00_attach_histopathology.py" \
    --input-root "$NLS_ROOT" \
    --pathology-annot-dir "$PATHOLOGY_DIR" \
    --out-root "$PATHOLOGY_ROOT" \
    --samples "${SAMPLES[@]}" \
    --exclude-labels "Necrosis"

# -------------------------
# 2. Spot QC
# -------------------------

echo
echo "===== 2/6 SPOT QC ====="

conda run -n "$ENV_NAME" \
    python "$SCRIPT_DIR/01_prepare_spot_qc.py" \
    --nls-root "$PATHOLOGY_ROOT" \
    --out-root "$QC_ROOT" \
    --samples "${SAMPLES[@]}" \
    --min-features 500 \
    --min-counts 1000 \
    --exclude-labels "Necrosis"

# -------------------------
# 2. Synthetic reference
# -------------------------

echo
echo "===== 3/6 SYNTHETIC REFERENCE ====="

conda run -n "$ENV_NAME" \
    python "$SCRIPT_DIR/02_prepare_synthetic_reference.py" \
    --nls-root "$QC_ROOT" \
    --gtf "$GTF" \
    --out-root "$PREP_ROOT" \
    --samples "${SAMPLES[@]}" \
    --names "${NAMES[@]}" \
    --cancer-col "Cancer Epithelial" \
    --train-cancer-max 0.20 \
    --train-nonmal-min 0.60 \
    --max-train 600 \
    --ridge-alpha 1.0 \
    --n-ref-cells 25 \
    --target-sum 10000 \
    --seed "$SEED"

# -------------------------
# 3. inferCNV
# -------------------------

echo
echo "===== 4/6 INFERCNV ====="

for NAME in "${NAMES[@]}"; do
    echo "[inferCNV] $NAME"

    conda run -n "$ENV_NAME" \
        Rscript "$SCRIPT_DIR/03_run_infercnv.R" \
        "$PREP_ROOT/$NAME" \
        "$THREADS" \
        "$MODE"
done

# -------------------------
# 4. Export inferCNV matrix
# -------------------------

echo
echo "===== 5/6 EXPORT FOR LEIDEN ====="

for NAME in "${NAMES[@]}"; do
    echo "[EXPORT] $NAME"

    conda run -n "$ENV_NAME" \
        Rscript "$SCRIPT_DIR/04_export_for_leiden.R" \
        "$PREP_ROOT/$NAME" \
        "$MODE"
done

# -------------------------
# 5. Leiden res 0.70
# -------------------------

echo
echo "===== 6/6 LEIDEN RESOLUTION 0.70 ====="

for NAME in "${NAMES[@]}"; do
    echo "[LEIDEN] $NAME"

    # bc2043_V13Y10-061_D1_synthref -> V13Y10-061_D1
    SID="${NAME#*_}"
    SID="${SID%_synthref}"
    SPATIAL_H5AD="$NLS_ROOT/${SID}.nlsdeconv.h5ad"

    test -f "$SPATIAL_H5AD"


    conda run -n "$ENV_NAME" \
        python "$SCRIPT_DIR/05_leiden_posthoc.py" \
        --prepared-dir "$PREP_ROOT/$NAME" \
        --mode "$MODE" \
        --resolution "$RESOLUTION" \
        --n-pcs 30 \
        --n-neighbors 15 \
        --seed "$SEED" \
        --spatial-h5ad "$SPATIAL_H5AD" \
        --write-h5ad
done

echo
echo "=========================================="
echo "DONE"
echo "=========================================="

for NAME in "${NAMES[@]}"; do
    echo
    echo "$NAME"
    ls -lh \
        "$PREP_ROOT/$NAME/infercnv_R_out_${MODE}/leiden_posthoc_res_${RESOLUTION}/infercnv_R_leiden_cluster_mean_profiles.tsv" \
        "$PREP_ROOT/$NAME/infercnv_R_out_${MODE}/leiden_posthoc_res_${RESOLUTION}/infercnv_R_leiden_spatial_table.tsv"
done

#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Self-bootstrap the pinned IgDiscover environment
# =============================================================================

ENV_NAME="igdiscover-repro"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd -P
)"

REPO_ROOT="$(
    cd "$SCRIPT_DIR/../../.."
    pwd -P
)"

ENV_FILE="$REPO_ROOT/envs/conda/igdiscover.yml"

environment_exists() {
    conda env list |
        awk 'NF && $1 !~ /^#/ {print $1}' |
        grep -qx "$ENV_NAME"
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    command -v conda >/dev/null 2>&1 || {
        echo "ERROR: Conda is not available in PATH." >&2
        exit 1
    }

    [[ -f "$ENV_FILE" ]] || {
        echo "ERROR: Environment file not found: $ENV_FILE" >&2
        exit 1
    }

    if ! environment_exists; then
        echo "Creating Conda environment: $ENV_NAME"

        if command -v mamba >/dev/null 2>&1; then
            mamba env create \
                --name "$ENV_NAME" \
                --file "$ENV_FILE"
        else
            conda env create \
                --name "$ENV_NAME" \
                --file "$ENV_FILE"
        fi
    fi

    echo "Running inside Conda environment: $ENV_NAME"

    exec conda run \
        --no-capture-output \
        --name "$ENV_NAME" \
        bash "$0" "$@"
fi

# =============================================================================
# Paths and parameters
# =============================================================================

RAW_READS="$REPO_ROOT/data/raw/svdj/sma_vdj/reads"

# Temporary pipeline workspace. Removed after successful publication by default.
WORK_DIR="$REPO_ROOT/data/intermediate/svdj/igdiscover/sma_vdj"
WORK_READS="$WORK_DIR/reads"
NATIVE_FINAL="$WORK_DIR/final"

# Validated downstream-ready outputs.
PROCESSED_DIR="$REPO_ROOT/data/processed/svdj/igdiscover/sma_vdj"

REFERENCE_DIR="$REPO_ROOT/data/references/igdiscover_ref_db"
BCR_DB="$REFERENCE_DIR/BCR_db"
TCR_DB="$REFERENCE_DIR/TCR_db"
CONSTANT_FASTA="$REFERENCE_DIR/ref_C.fasta"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/results/logs/svdj/igdiscover/$RUN_ID"

CORES="${IGDISCOVER_CORES:-16}"

# Set to 1 to retain tmp/, .snakemake/, staged reads, and native final files.
KEEP_WORKSPACE="${KEEP_IGDISCOVER_WORKSPACE:-0}"

SAMPLES=(
    bc2004 bc2012 bc2019 bc2020 bc2027 bc2028 bc2035 bc2036
    bc2043 bc2044 bc2051 bc2052 bc2059 bc2067 bc2075 bc2083
    bc2091
)

BARCODES="$(
    IFS=,
    echo "${SAMPLES[*]}"
)"

# =============================================================================
# Input validation
# =============================================================================

command -v igdiscover >/dev/null 2>&1 || {
    echo "ERROR: igdiscover is unavailable in $ENV_NAME." >&2
    exit 1
}

[[ -d "$BCR_DB" ]] || {
    echo "ERROR: BCR database not found: $BCR_DB" >&2
    exit 1
}

[[ -d "$TCR_DB" ]] || {
    echo "ERROR: TCR database not found: $TCR_DB" >&2
    exit 1
}

[[ -s "$CONSTANT_FASTA" ]] || {
    echo "ERROR: Constant-region FASTA not found: $CONSTANT_FASTA" >&2
    exit 1
}

mkdir -p "$WORK_READS" "$LOG_DIR"

for sample in "${SAMPLES[@]}"; do
    source_fasta="$RAW_READS/${sample}.fasta"
    staged_fasta="$WORK_READS/${sample}.fasta"

    [[ -s "$source_fasta" ]] || {
        echo "ERROR: Missing raw FASTA: $source_fasta" >&2
        exit 1
    }

    if [[ ! -f "$staged_fasta" ]] ||
       ! cmp -s "$source_fasta" "$staged_fasta"; then
        cp -p "$source_fasta" "$staged_fasta"
        echo "Staged: $sample"
    fi
done

echo
echo "Active environment:  ${CONDA_DEFAULT_ENV:-unknown}"
echo "Repository:          $REPO_ROOT"
echo "Raw inputs:          $RAW_READS"
echo "Temporary workspace: $WORK_DIR"
echo "Processed outputs:   $PROCESSED_DIR"
echo "Log directory:       $LOG_DIR"
echo "Cores:               $CORES"
echo

# =============================================================================
# Run IgDiscover
# =============================================================================

run_receptor() {
    local receptor="$1"
    local database="$2"
    local logfile="$LOG_DIR/igdiscover_${receptor}.log"

    echo "Running IgDiscover receptor type: $receptor"

    igdiscover run_spatial \
        --smrt_barcodes "$BARCODES" \
        --folder "$WORK_DIR" \
        --receptor_type "$receptor" \
        --database "$database" \
        --C_fasta "$CONSTANT_FASTA" \
        --cores "$CORES" \
        2>&1 | tee "$logfile"
}

run_receptor "Ig" "$BCR_DB"
run_receptor "TCR" "$TCR_DB"

# =============================================================================
# Validate all expected native outputs
# =============================================================================

IG_CHAINS=(IGH IGK IGL)
TCR_CHAINS=(TRA TRB TRG TRD)
ALL_CHAINS=(IGH IGK IGL TRA TRB TRG TRD)

for sample in "${SAMPLES[@]}"; do
    for chain in "${ALL_CHAINS[@]}"; do
        matrix="$NATIVE_FINAL/${sample}_${chain}_count_matrix.tsv"

        [[ -s "$matrix" ]] || {
            echo "ERROR: Missing or empty count matrix: $matrix" >&2
            exit 1
        }
    done
done

for chain in "${ALL_CHAINS[@]}"; do
    [[ -s "$NATIVE_FINAL/${chain}_clonotypes.tsv" ]] || {
        echo "ERROR: Missing clonotype table for $chain" >&2
        exit 1
    }

    [[ -s "$NATIVE_FINAL/${chain}_clonotypes_members.tsv" ]] || {
        echo "ERROR: Missing clonotype-member table for $chain" >&2
        exit 1
    }
done

MATRIX_COUNT="$(
    find "$NATIVE_FINAL" -maxdepth 1 -type f \
        -name 'bc*_*_count_matrix.tsv' |
    wc -l |
    tr -d ' '
)"

CLONOTYPE_COUNT="$(
    find "$NATIVE_FINAL" -maxdepth 1 -type f \
        -name '*_clonotypes.tsv' |
    wc -l |
    tr -d ' '
)"

MEMBER_COUNT="$(
    find "$NATIVE_FINAL" -maxdepth 1 -type f \
        -name '*_clonotypes_members.tsv' |
    wc -l |
    tr -d ' '
)"

[[ "$MATRIX_COUNT" == "119" ]] || {
    echo "ERROR: Expected 119 matrices; found $MATRIX_COUNT." >&2
    exit 1
}

[[ "$CLONOTYPE_COUNT" == "7" ]] || {
    echo "ERROR: Expected 7 clonotype tables; found $CLONOTYPE_COUNT." >&2
    exit 1
}

[[ "$MEMBER_COUNT" == "7" ]] || {
    echo "ERROR: Expected 7 member tables; found $MEMBER_COUNT." >&2
    exit 1
}

# =============================================================================
# Publish validated outputs atomically
# =============================================================================

PUBLISH_TMP="${PROCESSED_DIR}.tmp.$$"
PREVIOUS="${PROCESSED_DIR}.previous"

rm -rf "$PUBLISH_TMP" "$PREVIOUS"
mkdir -p "$PUBLISH_TMP"

cp -a "$NATIVE_FINAL"/. "$PUBLISH_TMP"/

{
    echo "created=$(date --iso-8601=seconds)"
    echo "igdiscover_version=$(python -c \
        'import importlib.metadata; print(importlib.metadata.version("igdiscover"))')"
    echo "sample_count=${#SAMPLES[@]}"
    echo "matrix_count=$MATRIX_COUNT"
    echo "clonotype_count=$CLONOTYPE_COUNT"
    echo "member_count=$MEMBER_COUNT"

    if git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        echo "repository_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    fi
} > "$PUBLISH_TMP/PROVENANCE.txt"

(
    cd "$PUBLISH_TMP"
    find . -maxdepth 1 -type f \
        ! -name SHA256SUMS.txt \
        -print0 |
        sort -z |
        xargs -0 sha256sum > SHA256SUMS.txt
)

if [[ -d "$PROCESSED_DIR" ]]; then
    mv "$PROCESSED_DIR" "$PREVIOUS"
fi

mv "$PUBLISH_TMP" "$PROCESSED_DIR"
rm -rf "$PREVIOUS"

echo
echo "IgDiscover preprocessing and validation completed."
echo
echo "Processed outputs:"
echo "  $PROCESSED_DIR"
echo
echo "Validated:"
echo "  $MATRIX_COUNT count matrices"
echo "  $CLONOTYPE_COUNT clonotype tables"
echo "  $MEMBER_COUNT clonotype-member tables"
echo
echo "Logs:"
echo "  $LOG_DIR"

# =============================================================================
# Remove temporary workspace only after successful publication
# =============================================================================

if [[ "$KEEP_WORKSPACE" == "1" ]]; then
    echo
    echo "Intermediate workspace retained:"
    echo "  $WORK_DIR"
else
    rm -rf "$WORK_DIR"

    echo
    echo "Intermediate workspace removed after successful publication."
fi

#!/usr/bin/env python3
"""
Build per-section MOFA-FLEX inputs for the Figure 2 ST–MSI–VDJ analysis.

Canonical preprocessing reproduced from the historical workflow:

ST
  - Visium in-tissue spots
  - remove pathologist-annotated necrosis before all downstream alignment
  - spot QC: 100 <= counts <= 60000, >=100 genes, <=20% mitochondrial
  - genes present in >=50 spots
  - remove MT-, RPL*, RPS*, and historical housekeeping set
  - HVGs: seurat_v3 on raw counts, n=2000
  - retain HVGs detected in >=5% of spots
  - CP10K + log1p model matrix

MSI
  - RMS-normalized fake-Space-Ranger matrix
  - require >=5 detected peaks per MSI spot
  - matrix-specific technical blacklist, +/-2 ppm
  - peaks present in >=3 MSI spots and >=10% of MSI spots
  - HVGs: seurat_v3, n=500
  - no further normalization
  - kNN imputation to retained ST spots: k=3, inverse-distance^2

VDJ
  - IGH/IGK/IGL -> BCR
  - TRA/TRB/TRD/TRG -> TCR
  - clones present in >=5 ST spots
  - retain top 50% by spatial presence
  - log1p(10000 * clone_count / ST RNA library)
  - BCR row-wise L2 normalization
  - TCR row-wise L2 normalization only if >=20 retained features
  - merge BCR and TCR into one VDJ view with BCR__/TCR__ prefixes

Only 9AA/DHBA metabolite/lipid sections are built. CHCA/peptide sections are excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse as sp
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------
# Canonical parameters
# ---------------------------------------------------------------------

BCR_TYPES = ("IGH", "IGK", "IGL")
TCR_TYPES = ("TRA", "TRB", "TRD", "TRG")

ST_MIN_TOTAL_COUNTS = 100
ST_MAX_TOTAL_COUNTS = 60_000
ST_MIN_N_GENES = 100
ST_MAX_PCT_MT = 20.0
ST_MIN_GENE_SPOTS = 50
ST_MIN_DETECTION_FRACTION = 0.05
ST_N_HVG = 2000

MSI_DETECT_EPS = 1e-8
MSI_MIN_PEAKS_PER_SPOT = 5
MSI_MIN_FEATURE_SPOTS = 3
MSI_MIN_DETECTION_FRACTION = 0.10
MSI_N_HVG = 500
MSI_BLACKLIST_PPM = 2.0

VDJ_MIN_SPOTS = 5
VDJ_TOP_FRACTION = 0.50
TCR_MIN_FEATURES_FOR_L2 = 20

TARGET_SUM = 1e4

MSI_KNN_K = 3
MSI_KNN_POWER = 2.0
MSI_KNN_EPS = 1e-6

HOUSEKEEPING_GENES = {
    "ACTB", "B2M", "GAPDH", "GUSB", "HPRT1", "PGK1", "PPIA",
    "RPL13A", "RPL19", "RPLP0", "RPLP1", "RPLP2", "RPS18",
    "RPS27A", "TBP", "TFRC", "TUBA1B", "TUBB", "UBC", "YWHAZ",
    "SDHA", "PUM1", "POLR2A", "HMBS", "HNRNPL", "NONO",
    "RPL11", "RPL23A", "RPL31", "RPS3", "RPS6", "RPS9",
    "RPS10", "RPS28", "EEF1A1", "EEF2",
}

PATHOLOGY_LABEL_MAP = {
    "Benign glands": "Benign glands",
    "Normal gland": "Normal gland",
    "Blood vessel": "Blood vessel",
    "In situ": "In situ",
    "Invasive ca": "Invasive carcinoma",
    "Invasiv ca": "Invasive carcinoma",
    "Lymphocytes": "Lymphocytes",
    "Lymfocytes": "Lymphocytes",
    "Necrosis": "Necrosis",
    "Stroma": "Stroma",
    "Cluster 1": "Cluster 1",
    "Cluster 2": "Cluster 2",
    "Cluster 3": "Cluster 3",
    "Cluster 4": "Cluster 4",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def cp10k_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lib = np.maximum(x.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(TARGET_SUM * x / lib).astype(np.float32)


def row_l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    zero = np.sum(np.abs(x), axis=1) == 0
    norms = np.maximum(norms, eps)
    out = x / norms
    out[zero, :] = 0.0
    return out.astype(np.float32)


def canonical_gene_symbols(names) -> pd.Index:
    idx = pd.Index(np.asarray(names, dtype=object))
    return (
        idx.str.upper()
        .str.split(".", n=1).str[0]
        .str.replace(r"-\d+$", "", regex=True)
    )


def first_float(x) -> float:
    m = re.search(r"[-+]?\d+\.\d+|[-+]?\d+", str(x))
    return float(m.group(0)) if m else np.nan


def read_blacklist(path: Path) -> np.ndarray:
    values = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in re.split(r"[,\s]+", line):
                try:
                    values.append(float(token))
                except ValueError:
                    pass

    values = sorted(set(round(float(x), 9) for x in values))
    if not values:
        raise RuntimeError(f"Blacklist is empty: {path}")

    return np.asarray(values, dtype=float)


def filter_msi_blacklist(
    x: np.ndarray,
    features: np.ndarray,
    blacklist: np.ndarray,
    ppm: float,
):
    feature_mz = np.asarray([first_float(v) for v in features], dtype=float)
    remove = np.zeros(len(features), dtype=bool)

    for i, mz in enumerate(feature_mz):
        if not np.isfinite(mz):
            continue

        delta = np.abs(mz - blacklist) / np.maximum(np.abs(blacklist), 1e-12) * 1e6
        if float(np.min(delta)) <= ppm:
            remove[i] = True

    return x[:, ~remove], features[~remove], remove


def filter_by_presence(
    x: np.ndarray,
    features: np.ndarray,
    min_spots: int,
    eps: float = 0.0,
):
    present = (x > eps).sum(axis=0)
    keep = present >= min_spots
    return x[:, keep], features[keep]


def select_vdj_clones(
    x: np.ndarray,
    features: np.ndarray,
):
    if x.shape[1] == 0:
        return x, features

    presence = (x > 0).sum(axis=0)
    keep = presence >= VDJ_MIN_SPOTS

    x = x[:, keep]
    features = features[keep]
    presence = presence[keep]

    if x.shape[1] == 0:
        return x, features

    n_keep = max(1, int(np.ceil(VDJ_TOP_FRACTION * x.shape[1])))

    # Deterministic ranking:
    #   1. spatial prevalence, descending
    #   2. total abundance, descending
    #   3. canonical clonotype feature ID, ascending
    #
    # IgDiscover's native sequential clone IDs are not stable between
    # equivalent runs, so upstream canonicalization makes the final
    # tie-break reproducible.
    abundance = x.sum(axis=0)
    feature_key = features.astype(str)

    order = np.lexsort(
        (
            feature_key,
            -abundance,
            -presence,
        )
    )
    order = order[:n_keep]

    return x[:, order], features[order]


def knn_impute(
    x_src: np.ndarray,
    coords_src: np.ndarray,
    coords_target: np.ndarray,
) -> np.ndarray:
    n_src = x_src.shape[0]
    k = min(MSI_KNN_K, n_src)

    tree = cKDTree(coords_src)
    distances, indices = tree.query(coords_target, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)

    weights = 1.0 / np.power(distances + MSI_KNN_EPS, MSI_KNN_POWER)
    weights[~np.isfinite(weights)] = 0.0

    row_sum = weights.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0

    if bad.any():
        weights[bad, :] = 1.0 / k
        row_sum = weights.sum(axis=1, keepdims=True)

    weights /= row_sum

    rows = np.repeat(np.arange(coords_target.shape[0]), k)
    cols = indices.reshape(-1)

    w = sp.csr_matrix(
        (weights.reshape(-1), (rows, cols)),
        shape=(coords_target.shape[0], n_src),
        dtype=np.float32,
    )

    return np.asarray(w @ x_src, dtype=np.float32)


# ---------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------

def read_st(path: Path) -> ad.AnnData:
    obj = sc.read_visium(str(path))
    obj.var_names_make_unique()

    positions_path = path / "spatial" / "tissue_positions.csv"
    if positions_path.exists():
        positions = pd.read_csv(positions_path).set_index("barcode")

        for column in (
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ):
            if column not in obj.obs.columns and column in positions.columns:
                obj.obs[column] = positions.reindex(obj.obs_names)[column]

    if "in_tissue" in obj.obs:
        keep = pd.to_numeric(obj.obs["in_tissue"], errors="coerce") == 1
        obj = obj[keep.to_numpy()].copy()

    return obj


def read_msi(path: Path) -> ad.AnnData:
    obj = sc.read_10x_mtx(
        path / "filtered_feature_bc_matrix",
        var_names="gene_symbols",
        make_unique=True,
    )

    positions = pd.read_csv(
        path / "spatial" / "tissue_positions.csv",
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    ).set_index("barcode")

    obj.obs = obj.obs.join(positions, how="left")

    keep = pd.to_numeric(obj.obs["in_tissue"], errors="coerce") == 1
    obj = obj[keep.to_numpy()].copy()
    obj.var_names_make_unique()

    return obj


def find_pathology_file(pathology_root: Path, capture: str) -> Path:
    suffix = capture[-6:]
    path = pathology_root / f"Morphology_{suffix}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Required morphology annotation missing for {capture}: {path}"
        )

    return path


def pathology_label_column(df: pd.DataFrame) -> str:
    if "Morphology" in df.columns:
        return "Morphology"
    if "Graph-based" in df.columns:
        return "Graph-based"

    for col in df.columns:
        if any(x in col.lower() for x in ("morph", "histo", "path", "graph")):
            return col

    raise RuntimeError(
        f"Could not identify pathology label column. Columns: {list(df.columns)}"
    )


def add_pathology_and_remove_necrosis(
    obj: ad.AnnData,
    pathology_file: Path,
):
    df = pd.read_csv(pathology_file)

    if "Barcode" not in df.columns:
        raise RuntimeError(f"'Barcode' column missing from {pathology_file}")

    label_col = pathology_label_column(df)

    lookup = {}
    for barcode, value in zip(df["Barcode"].astype(str), df[label_col]):
        lookup[barcode] = value

        if barcode.endswith("-1"):
            lookup[barcode[:-2]] = value
        else:
            lookup[barcode + "-1"] = value

    raw = pd.Series(obj.obs_names.astype(str), index=obj.obs_names).map(lookup)

    labels = raw.map(
        lambda x: (
            "Unannotated"
            if pd.isna(x) or str(x).strip() == ""
            else PATHOLOGY_LABEL_MAP.get(str(x).strip(), str(x).strip())
        )
    )

    obj.obs["histopathology_raw"] = raw
    obj.obs["histopathology"] = labels.astype(str)

    keep = obj.obs["histopathology"].str.lower() != "necrosis"
    n_removed = int((~keep).sum())

    return obj[keep.to_numpy()].copy(), n_removed, label_col


def read_vdj_family(vdj_root: Path, barcode: str, families) -> pd.DataFrame | None:
    frames = []

    for family in families:
        path = vdj_root / f"{barcode}_{family}_count_matrix.tsv"

        if not path.exists():
            continue

        df = pd.read_csv(path, sep="\t", index_col=0)
        df.columns = family + "_" + df.columns.astype(str)
        frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def matrix_label(row: pd.Series) -> str:
    matrix = str(row.get("matrix", "")).upper().replace("-", "").replace("_", "")
    target = str(row.get("MSI target", "")).lower()

    if "9AA" in matrix or "9AMINO" in matrix or "metabolite" in target:
        return "9AA"

    if "DHBA" in matrix or "DHB" in matrix or "lipid" in target:
        return "DHBA"

    return "UNKNOWN"


def select_figure2_sections(metadata: pd.DataFrame) -> pd.DataFrame:
    keep = metadata.apply(lambda row: matrix_label(row) in {"9AA", "DHBA"}, axis=1)
    out = metadata.loc[keep].copy()

    if out.empty:
        raise RuntimeError("No 9AA/DHBA sections found in metadata.")

    return out


# ---------------------------------------------------------------------
# Per-section processing
# ---------------------------------------------------------------------

def build_section(
    row: pd.Series,
    args,
    blacklists: dict[str, np.ndarray],
):
    capture = str(row["capture_area_id"])
    barcode = str(row["barcode"])
    matrix = matrix_label(row)

    sample = f"{barcode}_{capture}"
    out_dir = args.output_root / sample

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} already contains files. Use --overwrite to rebuild."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sample} [{matrix}] ===")

    # --------------------------------------------------------------
    # ST
    # --------------------------------------------------------------

    st_path = args.st_root / capture
    if not st_path.exists():
        raise FileNotFoundError(f"Space Ranger output missing: {st_path}")

    st = read_st(st_path)
    n_st_raw = st.n_obs

    pathology_file = find_pathology_file(args.pathology_root, capture)
    st, n_necrosis_removed, pathology_col = add_pathology_and_remove_necrosis(
        st, pathology_file
    )

    if st.n_obs == 0:
        raise RuntimeError(f"{sample}: no ST spots remain after necrosis removal.")

    st.var["mt"] = st.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(st, qc_vars=["mt"], inplace=True)

    spot_keep = (
        (st.obs["total_counts"] >= ST_MIN_TOTAL_COUNTS)
        & (st.obs["total_counts"] <= ST_MAX_TOTAL_COUNTS)
        & (st.obs["n_genes_by_counts"] >= ST_MIN_N_GENES)
        & (st.obs["pct_counts_mt"] <= ST_MAX_PCT_MT)
    )

    st = st[spot_keep.to_numpy()].copy()

    if st.n_obs == 0:
        raise RuntimeError(f"{sample}: all ST spots failed QC.")

    sc.pp.filter_genes(st, min_cells=ST_MIN_GENE_SPOTS)

    genes = np.asarray(st.var_names, dtype=object)
    canonical = canonical_gene_symbols(genes)

    remove = (
        canonical.str.startswith("MT-")
        | canonical.str.startswith("RPL")
        | canonical.str.startswith("RPS")
        | canonical.isin(HOUSEKEEPING_GENES)
    )

    st = st[:, ~np.asarray(remove, dtype=bool)].copy()

    st_raw = dense_float32(st.X)
    genes = np.asarray(st.var_names, dtype=object)

    st_model_full = cp10k_log1p(st_raw)

    detection = (st_model_full > 0).mean(axis=0)
    detection_keep = detection >= ST_MIN_DETECTION_FRACTION

    tmp = ad.AnnData(X=st_raw.copy())
    tmp.var_names = pd.Index(genes.astype(str))
    tmp.var_names_make_unique()

    sc.pp.highly_variable_genes(
        tmp,
        flavor="seurat_v3",
        n_top_genes=min(ST_N_HVG, tmp.n_vars),
    )

    hvg = tmp.var["highly_variable"].to_numpy(dtype=bool)
    keep = hvg & detection_keep

    if not keep.any():
        raise RuntimeError(f"{sample}: ST HVG filtering retained zero genes.")

    st_model = st_model_full[:, keep]
    st_features = genes[keep]

    st_barcodes = np.asarray(st.obs_names, dtype=str)

    x_st = pd.to_numeric(
        st.obs["pxl_col_in_fullres"], errors="coerce"
    ).to_numpy(dtype=float)

    y_st = pd.to_numeric(
        st.obs["pxl_row_in_fullres"], errors="coerce"
    ).to_numpy(dtype=float)

    coords_st = np.column_stack([x_st, y_st]).astype(np.float32)

    if not np.isfinite(coords_st).all():
        raise RuntimeError(f"{sample}: non-finite ST pixel coordinates.")

    # Historical VDJ scaling used the filtered raw ST matrix's library size.
    rna_lib = np.maximum(st_raw.sum(axis=1, keepdims=True), 1.0)

    print(
        f"ST: {n_st_raw} raw -> {st.n_obs} retained spots; "
        f"{st_model.shape[1]} genes"
    )
    print(f"  necrosis spots removed: {n_necrosis_removed}")

    # --------------------------------------------------------------
    # MSI
    # --------------------------------------------------------------

    msi_path = args.msi_root / capture
    if not msi_path.exists():
        raise FileNotFoundError(
            f"Required MSI input missing for Figure 2 section {sample}: {msi_path}"
        )

    msi = read_msi(msi_path)
    x_msi = dense_float32(msi.X)

    peak_count = (x_msi > MSI_DETECT_EPS).sum(axis=1)
    msi_spot_keep = peak_count >= MSI_MIN_PEAKS_PER_SPOT

    msi = msi[msi_spot_keep].copy()
    x_msi = x_msi[msi_spot_keep, :]

    if msi.n_obs == 0:
        raise RuntimeError(f"{sample}: all MSI spots failed MSI QC.")

    msi_features = np.asarray(msi.var_names, dtype=object)

    x_msi, msi_features, blacklist_removed = filter_msi_blacklist(
        x_msi,
        msi_features,
        blacklists[matrix],
        MSI_BLACKLIST_PPM,
    )

    x_msi, msi_features = filter_by_presence(
        x_msi,
        msi_features,
        MSI_MIN_FEATURE_SPOTS,
        MSI_DETECT_EPS,
    )

    detection = (x_msi > MSI_DETECT_EPS).mean(axis=0)
    detection_keep = detection >= MSI_MIN_DETECTION_FRACTION

    x_msi = x_msi[:, detection_keep]
    msi_features = msi_features[detection_keep]

    if x_msi.shape[1] == 0:
        raise RuntimeError(f"{sample}: MSI filtering retained zero features.")

    tmp = ad.AnnData(X=x_msi.copy())
    tmp.var_names = pd.Index(msi_features.astype(str))
    tmp.var_names_make_unique()

    sc.pp.highly_variable_genes(
        tmp,
        flavor="seurat_v3",
        n_top_genes=min(MSI_N_HVG, tmp.n_vars),
    )

    hvg = tmp.var["highly_variable"].to_numpy(dtype=bool)

    if hvg.any():
        x_msi = x_msi[:, hvg]
        msi_features = msi_features[hvg]

    x_msi_coord = pd.to_numeric(
        msi.obs["pxl_col_in_fullres"], errors="coerce"
    ).to_numpy(dtype=float)

    y_msi_coord = pd.to_numeric(
        msi.obs["pxl_row_in_fullres"], errors="coerce"
    ).to_numpy(dtype=float)

    coords_msi = np.column_stack([x_msi_coord, y_msi_coord]).astype(np.float32)

    if not np.isfinite(coords_msi).all():
        raise RuntimeError(f"{sample}: non-finite MSI pixel coordinates.")

    msi_on_st = knn_impute(x_msi, coords_msi, coords_st)

    print(
        f"MSI: {msi.n_obs} retained MSI spots; "
        f"{x_msi.shape[1]} features; "
        f"{int(blacklist_removed.sum())} blacklist features removed"
    )

    # --------------------------------------------------------------
    # VDJ
    # --------------------------------------------------------------

    vdj_parts = []
    vdj_features = []
    vdj_stats = {}

    for label, families in (("BCR", BCR_TYPES), ("TCR", TCR_TYPES)):
        df = read_vdj_family(args.vdj_root, barcode, families)

        if df is None:
            vdj_stats[f"n_{label.lower()}"] = 0
            continue

        df = df.reindex(st_barcodes).fillna(0.0)

        x = df.to_numpy(dtype=np.float32)
        features = np.asarray(df.columns, dtype=object)

        x, features = select_vdj_clones(x, features)

        if x.shape[1] == 0:
            vdj_stats[f"n_{label.lower()}"] = 0
            continue

        x = np.log1p(TARGET_SUM * x / rna_lib).astype(np.float32)

        if label == "BCR":
            x = row_l2_normalize(x)
        elif x.shape[1] >= TCR_MIN_FEATURES_FOR_L2:
            x = row_l2_normalize(x)

        features = np.asarray(
            [f"{label}__{feature}" for feature in features],
            dtype=object,
        )

        vdj_parts.append(x)
        vdj_features.extend(features.tolist())
        vdj_stats[f"n_{label.lower()}"] = int(x.shape[1])

    if not vdj_parts:
        raise RuntimeError(
            f"{sample}: no BCR or TCR features remained after filtering."
        )

    vdj_model = np.hstack(vdj_parts).astype(np.float32)
    vdj_features = np.asarray(vdj_features, dtype=object)

    print(
        f"VDJ: {vdj_model.shape[1]} features "
        f"(BCR={vdj_stats.get('n_bcr', 0)}, "
        f"TCR={vdj_stats.get('n_tcr', 0)})"
    )

    # --------------------------------------------------------------
    # Shared observation table
    # --------------------------------------------------------------

    obs = pd.DataFrame(
        index=pd.Index(
            np.asarray([f"ST_{x}" for x in st_barcodes], dtype=object),
            dtype=object,
        )
    )

    # Keep string metadata as ordinary object dtype for broad H5AD compatibility.
    obs["capture_id"] = np.full(len(obs), capture, dtype=object)
    obs["barcode_prefix"] = np.full(len(obs), barcode, dtype=object)
    obs["x_pixel"] = coords_st[:, 0]
    obs["y_pixel"] = coords_st[:, 1]
    obs["histopathology"] = np.asarray(
        st.obs["histopathology"].astype(str),
        dtype=object,
    )

    # --------------------------------------------------------------
    # Write views
    # --------------------------------------------------------------

    st_out = ad.AnnData(
        X=st_model.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(np.asarray(st_features, dtype=object), dtype=object)),
    )
    st_out.var_names_make_unique()
    st_out.obsm["spatial"] = coords_st
    st_out.write_h5ad(out_dir / "ST.h5ad", convert_strings_to_categoricals=False)

    msi_out = ad.AnnData(
        X=msi_on_st.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(np.asarray(msi_features, dtype=object), dtype=object)),
    )
    msi_out.var_names_make_unique()
    msi_out.obsm["spatial"] = coords_st
    msi_out.write_h5ad(out_dir / "MSI.h5ad", convert_strings_to_categoricals=False)

    vdj_out = ad.AnnData(
        X=vdj_model.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(np.asarray(vdj_features, dtype=object), dtype=object)),
    )
    vdj_out.var_names_make_unique()
    vdj_out.obsm["spatial"] = coords_st
    vdj_out.write_h5ad(out_dir / "VDJ.h5ad", convert_strings_to_categoricals=False)

    # --------------------------------------------------------------
    # Provenance
    # --------------------------------------------------------------

    meta = {
        "sample": sample,
        "barcode": barcode,
        "capture_area_id": capture,
        "matrix": matrix,
        "msi_target": str(row.get("MSI target", "")),
        "paths": {
            "st": str(st_path),
            "msi": str(msi_path),
            "vdj": str(args.vdj_root),
            "pathology": str(pathology_file),
            "blacklist": str(args.blacklist_9aa if matrix == "9AA" else args.blacklist_dhba),
        },
        "dimensions": {
            "n_st_spots": int(st_model.shape[0]),
            "n_st_genes": int(st_model.shape[1]),
            "n_msi_features": int(msi_on_st.shape[1]),
            "n_vdj_features": int(vdj_model.shape[1]),
            "n_bcr_features": int(vdj_stats.get("n_bcr", 0)),
            "n_tcr_features": int(vdj_stats.get("n_tcr", 0)),
        },
        "pathology": {
            "label_column": pathology_col,
            "n_necrosis_spots_removed": n_necrosis_removed,
        },
        "parameters": {
            "st_n_hvg": ST_N_HVG,
            "st_min_gene_spots": ST_MIN_GENE_SPOTS,
            "st_min_detection_fraction": ST_MIN_DETECTION_FRACTION,
            "msi_n_hvg": MSI_N_HVG,
            "msi_min_feature_spots": MSI_MIN_FEATURE_SPOTS,
            "msi_min_detection_fraction": MSI_MIN_DETECTION_FRACTION,
            "msi_blacklist_ppm": MSI_BLACKLIST_PPM,
            "msi_knn_k": MSI_KNN_K,
            "msi_knn_power": MSI_KNN_POWER,
            "vdj_min_spots": VDJ_MIN_SPOTS,
            "vdj_top_fraction": VDJ_TOP_FRACTION,
            "tcr_min_features_for_l2": TCR_MIN_FEATURES_FOR_L2,
            "target_sum": TARGET_SUM,
        },
    }

    with (out_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)

    return {
        "sample": sample,
        "barcode": barcode,
        "capture_area_id": capture,
        "matrix": matrix,
        "n_st_spots": st_model.shape[0],
        "n_st_genes": st_model.shape[1],
        "n_msi_features": msi_on_st.shape[1],
        "n_vdj_features": vdj_model.shape[1],
        "n_bcr_features": vdj_stats.get("n_bcr", 0),
        "n_tcr_features": vdj_stats.get("n_tcr", 0),
        "n_necrosis_removed": n_necrosis_removed,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    repo = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo / "metadata" / "metadata_short.tsv",
    )
    parser.add_argument(
        "--st-root",
        type=Path,
        default=repo / "data" / "processed" / "st" / "space_ranger_outs",
    )
    parser.add_argument(
        "--msi-root",
        type=Path,
        default=repo / "data" / "processed" / "msi" / "fake_spaceranger" / "msi_in_he",
    )
    parser.add_argument(
        "--vdj-root",
        type=Path,
        default=repo / "data" / "processed" / "svdj" / "igdiscover" / "sma_vdj",
    )
    parser.add_argument(
        "--pathology-root",
        type=Path,
        default=repo / "data" / "processed" / "st" / "morphology_annotations" / "all_annotations",
    )
    parser.add_argument(
        "--blacklist-9aa",
        type=Path,
        default=repo / "config" / "msi" / "blacklists" / "9AA_matrix_exclude_mz_nmf_informed_v2.txt",
    )
    parser.add_argument(
        "--blacklist-dhba",
        type=Path,
        default=repo / "config" / "msi" / "blacklists" / "DHBA_matrix_exclude_mz_nmf_informed_v3_recurrent_family.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "data" / "intermediate" / "integration" / "mofaflex" / "inputs",
    )

    parser.add_argument(
        "--samples",
        nargs="*",
        help="Optional barcode IDs to build, e.g. bc2004 bc2059.",
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    for required in (
        args.metadata,
        args.st_root,
        args.msi_root,
        args.vdj_root,
        args.pathology_root,
        args.blacklist_9aa,
        args.blacklist_dhba,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    metadata = pd.read_csv(args.metadata, sep="\t")
    metadata = select_figure2_sections(metadata)

    if args.samples:
        wanted = set(args.samples)
        metadata = metadata[metadata["barcode"].astype(str).isin(wanted)].copy()

        missing = wanted - set(metadata["barcode"].astype(str))
        if missing:
            raise RuntimeError(
                f"Requested samples are not eligible 9AA/DHBA sections: {sorted(missing)}"
            )

    blacklists = {
        "9AA": read_blacklist(args.blacklist_9aa),
        "DHBA": read_blacklist(args.blacklist_dhba),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Building {len(metadata)} Figure 2 sections")
    print(f"Output: {args.output_root}")

    summary = []

    for _, row in metadata.iterrows():
        summary.append(build_section(row, args, blacklists))

    summary_df = pd.DataFrame(summary)
    summary_path = args.output_root / "qc_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print("\n=== complete ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()

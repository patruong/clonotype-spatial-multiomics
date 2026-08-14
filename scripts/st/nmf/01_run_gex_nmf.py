#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import NMF


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

METADATA = ROOT / "metadata" / "metadata_short.tsv"
ST_ROOT = ROOT / "data" / "processed" / "st" / "space_ranger_outs"

DATA_OUT = ROOT / "data" / "intermediate" / "st" / "nmf" / "gex"
FIG_OUT = ROOT / "results" / "st" / "nmf" / "gex"


# ---------------------------------------------------------------------
# Canonical settings
# ---------------------------------------------------------------------

K = 3
RANDOM_STATE = 42

MIN_SPOT_PRESENCE = 3
MIN_TOTAL_COUNTS = 1.0
TARGET_SUM = 1e4
MAX_ITER = 3000

# Remove common nuisance/high-abundance gene families.
# Do NOT broadly remove housekeeping genes.
EXCLUDE_GENE_REGEX = r"^(MT-|RPL|RPS|HB[ABDEGQMZ][0-9]*$)"

IG_GROUPS = [
    ("IGK",  r"^IGK"),
    ("IGL",  r"^IGL"),
    ("IGHA", r"^IGHA"),
    ("IGHG", r"^IGHG"),
    ("IGHD", r"^IGHD"),
    ("IGHM", r"^IGHM"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_metadata():
    md = pd.read_csv(METADATA, sep="\t", dtype=str)

    required = {"capture_area_id", "barcode"}
    missing = required - set(md.columns)

    if missing:
        raise ValueError(
            f"metadata_short.tsv missing columns: {sorted(missing)}"
        )

    return md


def aggregate_ig_families(X, gene_names):
    """
    Optional sensitivity analysis.

    Collapse:
      IGK*  -> IGK
      IGL*  -> IGL
      IGHA* -> IGHA
      IGHG* -> IGHG
      IGHD* -> IGHD
      IGHM* -> IGHM

    Other genes remain unchanged.
    """

    if sparse.issparse(X):
        X = X.tocsr().astype(np.float64, copy=True)
    else:
        X = sparse.csr_matrix(
            np.asarray(X, dtype=np.float64)
        )

    gene_names = np.asarray(gene_names).astype(str)

    assigned = np.zeros(
        len(gene_names),
        dtype=bool,
    )

    grouped_columns = []
    grouped_names = []
    counts = {}

    for group_name, pattern in IG_GROUPS:
        rgx = re.compile(pattern)

        mask = np.array(
            [bool(rgx.search(g)) for g in gene_names],
            dtype=bool,
        )

        mask &= ~assigned
        n = int(mask.sum())
        counts[group_name] = n

        if n:
            col = X[:, np.where(mask)[0]].sum(axis=1)

            grouped_columns.append(
                sparse.csr_matrix(
                    np.asarray(col)
                )
            )

            grouped_names.append(group_name)
            assigned |= mask

    keep = ~assigned

    X_keep = X[:, keep].tocsr()
    genes_keep = gene_names[keep]

    if grouped_columns:
        X_new = sparse.hstack(
            [X_keep] + grouped_columns,
            format="csr",
        )

        genes_new = np.concatenate(
            [
                genes_keep,
                np.asarray(grouped_names),
            ]
        )

    else:
        X_new = X_keep
        genes_new = genes_keep

    X_new.eliminate_zeros()

    stats = {
        "ig_grouping": True,
        "n_original_ig_genes_grouped": int(
            assigned.sum()
        ),
        "groups": ";".join(
            f"{name}:{n}"
            for name, n in counts.items()
            if n
        ),
    }

    return X_new, genes_new, stats


def filter_genes(
    X,
    gene_names,
    min_spot_presence,
    min_total_counts,
    exclude_regex,
):
    """
    Gene-level filtering only.

    No per-entry count thresholding.
    """

    if sparse.issparse(X):
        X = X.tocsr().astype(
            np.float64,
            copy=True,
        )
    else:
        X = sparse.csr_matrix(
            np.asarray(X, dtype=np.float64)
        )

    gene_names = np.asarray(
        gene_names
    ).astype(str)

    present = np.asarray(
        X.getnnz(axis=0)
    ).ravel()

    total = np.asarray(
        X.sum(axis=0)
    ).ravel()

    keep = (
        (present >= min_spot_presence)
        & (total >= min_total_counts)
    )

    if exclude_regex:
        rgx = re.compile(exclude_regex)

        nuisance = np.array(
            [
                bool(rgx.search(g))
                for g in gene_names
            ]
        )

        keep &= ~nuisance

    X = X[:, keep].tocsr()
    X.eliminate_zeros()

    return (
        X,
        gene_names[keep],
        {
            "n_genes_before_filter":
                int(len(gene_names)),
            "n_genes_after_filter":
                int(X.shape[1]),
            "min_spot_presence":
                int(min_spot_presence),
            "min_total_counts":
                float(min_total_counts),
            "exclude_gene_regex":
                exclude_regex,
        },
    )


def cp10k_log1p(X, target_sum):
    X = X.tocsr().astype(
        np.float64,
        copy=True,
    )

    library_size = np.asarray(
        X.sum(axis=1)
    ).ravel()

    scale = (
        float(target_sum)
        / np.maximum(library_size, 1.0)
    )

    X = X.multiply(
        scale[:, None]
    ).tocsr()

    X.data = np.log1p(X.data)
    X.eliminate_zeros()

    return X


def read_plot_geometry(
    adata,
    sample_dir,
):
    spatial_dir = sample_dir / "spatial"

    with open(
        spatial_dir / "scalefactors_json.json"
    ) as f:
        scalefactors = json.load(f)

    hires_scale = float(
        scalefactors["tissue_hires_scalef"]
    )

    spot_diameter = float(
        scalefactors.get(
            "spot_diameter_fullres",
            100.0,
        )
    )

    spot_diameter *= hires_scale

    positions_path = (
        spatial_dir / "tissue_positions.csv"
    )

    try:
        pos = pd.read_csv(
            positions_path
        )

        if "barcode" not in pos.columns:
            raise ValueError

    except Exception:
        pos = pd.read_csv(
            positions_path,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )

    pos["barcode"] = pos[
        "barcode"
    ].astype(str)

    pos = pos.set_index(
        "barcode"
    ).reindex(
        adata.obs_names.astype(str)
    )

    x = pd.to_numeric(
        pos["pxl_col_in_fullres"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        pos["pxl_row_in_fullres"],
        errors="coerce",
    ).to_numpy(dtype=float)

    x *= hires_scale
    y *= hires_scale

    image = plt.imread(
        spatial_dir / "tissue_hires_image.png"
    )

    return x, y, image, spot_diameter


def robust01(
    values,
    p_low=1,
    p_high=99,
    gamma=0.7,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    lo, hi = np.nanpercentile(
        values,
        [p_low, p_high],
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):
        return np.zeros_like(values)

    values = np.clip(
        (values - lo) / (hi - lo),
        0,
        1,
    )

    return values ** gamma


def save_figure(fig, stem):
    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_factor(
    image,
    x,
    y,
    spot_diameter,
    values,
    title,
    stem,
):
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    lo, hi = np.nanpercentile(
        values[valid],
        [1, 99],
    )

    size = max(
        (spot_diameter * 0.75) ** 2
        * 0.02,
        4.0,
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 5.8)
    )

    ax.imshow(image)

    scatter = ax.scatter(
        x[valid],
        y[valid],
        c=values[valid],
        s=size,
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        edgecolors="none",
        linewidths=0,
        alpha=0.95,
    )

    ax.set_axis_off()
    ax.set_title(
        title,
        fontsize=16,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.035,
        pad=0.02,
    )

    colorbar.ax.tick_params(
        labelsize=11
    )

    save_figure(
        fig,
        stem,
    )


def plot_rgb(
    image,
    x,
    y,
    spot_diameter,
    W,
    stem,
):
    if W.shape[1] < 3:
        return

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    rgb = np.column_stack(
        [
            robust01(W[:, 0]),
            robust01(W[:, 1]),
            robust01(W[:, 2]),
        ]
    )

    size = max(
        (spot_diameter * 0.75) ** 2
        * 0.02,
        4.0,
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 5.8)
    )

    ax.imshow(image)

    ax.scatter(
        x[valid],
        y[valid],
        c=rgb[valid],
        s=size,
        edgecolors="none",
        linewidths=0,
        alpha=0.95,
    )

    ax.set_axis_off()
    ax.set_title(
        "GEX NMF RGB",
        fontsize=16,
    )

    fig.text(
        0.82,
        0.80,
        "R: NMF1",
        fontsize=12,
    )

    fig.text(
        0.82,
        0.76,
        "G: NMF2",
        fontsize=12,
    )

    fig.text(
        0.82,
        0.72,
        "B: NMF3",
        fontsize=12,
    )

    save_figure(
        fig,
        stem,
    )


def plot_loadings(
    H,
    genes,
    stem,
    top_n,
):
    genes = np.asarray(
        genes
    ).astype(str)

    n_factors = H.shape[0]

    ncols = 2
    nrows = int(
        np.ceil(
            n_factors / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            8.0,
            max(
                5.0,
                3.4 * nrows,
            ),
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for factor in range(
        n_factors
    ):
        ax = axes[factor]

        values = np.asarray(
            H[factor]
        )

        idx = np.argsort(
            values
        )[-top_n:]

        idx = idx[
            np.argsort(
                values[idx]
            )
        ]

        ax.barh(
            np.arange(len(idx)),
            values[idx],
        )

        ax.set_yticks(
            np.arange(len(idx))
        )

        ax.set_yticklabels(
            genes[idx],
            fontsize=10,
        )

        ax.tick_params(
            axis="x",
            labelsize=10,
        )

        ax.set_title(
            f"NMF{factor + 1}",
            fontsize=14,
            fontweight="bold",
            loc="left",
        )

        ax.set_xlabel(
            "Loading",
            fontsize=11,
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

    for ax in axes[n_factors:]:
        ax.axis("off")

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# ---------------------------------------------------------------------
# Run one sample
# ---------------------------------------------------------------------

def run_sample(
    sample,
    barcode,
    args,
):
    sample_dir = ST_ROOT / sample

    if not sample_dir.is_dir():
        raise FileNotFoundError(
            sample_dir
        )

    adata = sc.read_visium(
        str(sample_dir)
    )

    adata.var_names_make_unique()

    X = adata.X

    genes = np.asarray(
        adata.var_names
    ).astype(str)

    if args.group_ig:
        X, genes, ig_stats = (
            aggregate_ig_families(
                X,
                genes,
            )
        )

        variant = "ig_grouped"

    else:
        if sparse.issparse(X):
            X = X.tocsr().astype(
                np.float64,
                copy=True,
            )
        else:
            X = sparse.csr_matrix(
                np.asarray(
                    X,
                    dtype=np.float64,
                )
            )

        ig_stats = {
            "ig_grouping": False,
            "n_original_ig_genes_grouped": 0,
            "groups": "",
        }

        variant = "ig_ungrouped"

    X, genes, filter_stats = (
        filter_genes(
            X=X,
            gene_names=genes,
            min_spot_presence=
                args.min_spot_presence,
            min_total_counts=
                args.min_total_counts,
            exclude_regex=
                args.exclude_gene_regex,
        )
    )

    X_nmf = cp10k_log1p(
        X,
        target_sum=args.target_sum,
    )

    model = NMF(
        n_components=args.k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        random_state=
            args.random_state,
        max_iter=args.max_iter,
        tol=1e-4,
    )

    start = time.perf_counter()

    W = model.fit_transform(
        X_nmf
    )

    H = model.components_

    runtime = (
        time.perf_counter()
        - start
    )

    data_dir = (
        DATA_OUT
        / variant
        / sample
    )

    fig_dir = (
        FIG_OUT
        / variant
        / sample
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    factor_names = [
        f"NMF{i + 1}"
        for i in range(
            args.k
        )
    ]

    W_df = pd.DataFrame(
        W,
        index=adata.obs_names,
        columns=factor_names,
    )

    H_df = pd.DataFrame(
        H,
        index=factor_names,
        columns=genes,
    )

    W_df.to_csv(
        data_dir / "W.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    H_df.to_csv(
        data_dir / "H.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    with open(
        data_dir
        / "kept_genes.txt",
        "w",
    ) as handle:
        for gene in genes:
            handle.write(
                f"{gene}\n"
            )

    metrics = {
        "sample": sample,
        "barcode": barcode,
        "variant": variant,
        "method":
            "cp10k_log1p_euclidean",
        "k": int(args.k),
        "random_state":
            int(args.random_state),
        "n_spots":
            int(adata.n_obs),
        "n_genes":
            int(len(genes)),
        "runtime_sec":
            float(runtime),
        "reconstruction_err":
            float(
                model.reconstruction_err_
            ),
        "n_iter":
            int(model.n_iter_),
        **ig_stats,
        **filter_stats,
    }

    with open(
        data_dir
        / "metrics.json",
        "w",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    if not args.skip_plots:
        (
            x,
            y,
            image,
            spot_diameter,
        ) = read_plot_geometry(
            adata,
            sample_dir,
        )

        for factor in range(
            args.k
        ):
            plot_factor(
                image=image,
                x=x,
                y=y,
                spot_diameter=
                    spot_diameter,
                values=
                    W[:, factor],
                title=
                    f"NMF{factor + 1}",
                stem=
                    fig_dir
                    / f"NMF{factor + 1}",
            )

        plot_rgb(
            image=image,
            x=x,
            y=y,
            spot_diameter=
                spot_diameter,
            W=W,
            stem=
                fig_dir / "rgb",
        )

        plot_loadings(
            H=H,
            genes=genes,
            stem=
                fig_dir
                / "top_loadings",
            top_n=args.top_n,
        )

    print(
        f"[OK] {sample} | "
        f"{variant} | "
        f"{adata.n_obs} spots | "
        f"{len(genes)} genes | "
        f"{runtime:.1f}s",
        flush=True,
    )

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help=(
            "Optional capture_area_id "
            "subset"
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=K,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
    )

    parser.add_argument(
        "--min-spot-presence",
        type=int,
        default=
            MIN_SPOT_PRESENCE,
    )

    parser.add_argument(
        "--min-total-counts",
        type=float,
        default=
            MIN_TOTAL_COUNTS,
    )

    parser.add_argument(
        "--target-sum",
        type=float,
        default=TARGET_SUM,
    )

    parser.add_argument(
        "--exclude-gene-regex",
        default=
            EXCLUDE_GENE_REGEX,
    )

    parser.add_argument(
        "--group-ig",
        action="store_true",
        help=(
            "Sensitivity analysis: "
            "collapse IG genes into "
            "IGK/IGL/IGHA/IGHG/"
            "IGHD/IGHM families."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
    )

    args = parser.parse_args()

    md = load_metadata()

    if args.samples:
        requested = set(
            args.samples
        )

        available = set(
            md[
                "capture_area_id"
            ]
        )

        unknown = (
            requested
            - available
        )

        if unknown:
            raise ValueError(
                "Samples not in "
                "manuscript metadata: "
                + ", ".join(
                    sorted(unknown)
                )
            )

        md = md[
            md[
                "capture_area_id"
            ].isin(requested)
        ].copy()

    variant = (
        "ig_grouped"
        if args.group_ig
        else "ig_ungrouped"
    )

    rows = []

    for _, row in md.iterrows():
        sample = str(
            row[
                "capture_area_id"
            ]
        )

        barcode = str(
            row["barcode"]
        )

        try:
            metrics = run_sample(
                sample,
                barcode,
                args,
            )

            metrics["status"] = (
                "completed"
            )

            rows.append(
                metrics
            )

        except Exception as exc:
            print(
                f"[FAILED] {sample}: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            rows.append(
                {
                    "sample": sample,
                    "barcode": barcode,
                    "variant": variant,
                    "status": "failed",
                    "error":
                        f"{type(exc).__name__}: "
                        f"{exc}",
                }
            )

    summary_dir = (
        FIG_OUT / variant
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        summary_dir
        / "gex_nmf_summary.tsv",
        sep="\t",
        index=False,
    )

    print(
        "\nDone.\n"
        f"Data: {DATA_OUT / variant}\n"
        f"Figures: {FIG_OUT / variant}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""
Plot spatial NLSDeconv cell-type proportions for the manuscript cohort.

Inputs
------
metadata/metadata_short.tsv
data/processed/st/nlsdeconv/<sample>.nlsdeconv.h5ad

Outputs
-------
results/st/nlsdeconv/spatial/<sample>/
    <celltype>.{png,pdf,svg}
    all_celltypes_panel.{png,pdf,svg}

results/st/nlsdeconv/nlsdeconv_plot_summary.tsv
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[3]

METADATA = ROOT / "metadata" / "metadata_short.tsv"
NLS_ROOT = ROOT / "data" / "processed" / "st" / "nlsdeconv"
OUT_ROOT = ROOT / "results" / "st" / "nlsdeconv"


def load_metadata() -> pd.DataFrame:
    md = pd.read_csv(METADATA, sep="\t", dtype=str)

    if "capture_area_id" not in md.columns:
        raise ValueError("metadata_short.tsv lacks capture_area_id")

    md["capture_area_id"] = md["capture_area_id"].str.strip()
    return md


def safe_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    return re.sub(r"[^A-Za-z0-9_.-]", "", value)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)

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


def load_nls(sample: str):
    path = NLS_ROOT / f"{sample}.nlsdeconv.h5ad"

    if not path.is_file():
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)

    if "nlsdeconv_props" not in adata.obsm:
        raise KeyError(
            f"{sample}: missing adata.obsm['nlsdeconv_props']"
        )

    if "nlsdeconv_celltypes" not in adata.uns:
        raise KeyError(
            f"{sample}: missing adata.uns['nlsdeconv_celltypes']"
        )

    celltypes = [
        str(x)
        for x in adata.uns["nlsdeconv_celltypes"]
    ]

    props = np.asarray(
        adata.obsm["nlsdeconv_props"],
        dtype=float,
    )

    if props.shape != (adata.n_obs, len(celltypes)):
        raise ValueError(
            f"{sample}: deconvolution matrix shape {props.shape} "
            f"does not match {adata.n_obs} spots x "
            f"{len(celltypes)} cell types"
        )

    return adata, props, celltypes


def select_celltypes(
    available: list[str],
    requested: list[str] | None,
) -> list[str]:

    if not requested:
        return available

    lookup = {
        name.lower(): name
        for name in available
    }

    selected = []
    missing = []

    for name in requested:
        match = lookup.get(name.lower())

        if match is None:
            missing.append(name)
        else:
            selected.append(match)

    if missing:
        raise ValueError(
            "Requested cell types not found: "
            + ", ".join(missing)
            + "\nAvailable: "
            + ", ".join(available)
        )

    return selected


def plot_celltype(
    adata,
    values: np.ndarray,
    celltype: str,
    stem: Path,
    qmin: float,
    qmax: float,
    cmap: str,
) -> None:

    key = "__nls_plot_value"
    adata.obs[key] = values

    finite = values[np.isfinite(values)]

    if len(finite):
        vmin = float(np.quantile(finite, qmin))
        vmax = float(np.quantile(finite, qmax))

        if vmax <= vmin:
            vmax = float(np.max(finite))

        if vmax <= vmin:
            vmax = None
    else:
        vmin = None
        vmax = None

    fig, ax = plt.subplots(figsize=(6.2, 5.5))

    sc.pl.spatial(
        adata,
        color=key,
        img_key="hires",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        show=False,
        title=celltype,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    save_figure(fig, stem)
    plt.close(fig)

    del adata.obs[key]


def plot_panel(
    adata,
    props: np.ndarray,
    celltypes: list[str],
    indices: list[int],
    stem: Path,
    qmin: float,
    qmax: float,
    cmap: str,
) -> None:

    n = len(celltypes)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 5.2 * nrows),
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, celltype, index in zip(
        axes,
        celltypes,
        indices,
    ):
        values = props[:, index]

        finite = values[np.isfinite(values)]

        if len(finite):
            vmin = float(np.quantile(finite, qmin))
            vmax = float(np.quantile(finite, qmax))

            if vmax <= vmin:
                vmax = float(np.max(finite))

            if vmax <= vmin:
                vmax = None
        else:
            vmin = None
            vmax = None

        key = "__nls_plot_value"
        adata.obs[key] = values

        sc.pl.spatial(
            adata,
            color=key,
            img_key="hires",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            show=False,
            title=celltype,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n:]:
        ax.axis("off")

    if "__nls_plot_value" in adata.obs:
        del adata.obs["__nls_plot_value"]

    fig.tight_layout()

    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional manuscript sample subset",
    )

    parser.add_argument(
        "--celltypes",
        nargs="*",
        default=None,
        help="Optional cell types; default plots all",
    )

    parser.add_argument(
        "--qmin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--qmax",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--cmap",
        default="viridis",
    )

    args = parser.parse_args()

    if not 0 <= args.qmin < args.qmax <= 1:
        raise ValueError(
            "Require 0 <= qmin < qmax <= 1"
        )

    md = load_metadata()
    cohort = md["capture_area_id"].tolist()

    if args.samples:
        requested = set(args.samples)

        unknown = requested - set(cohort)

        if unknown:
            raise ValueError(
                "Samples not in manuscript cohort: "
                + ", ".join(sorted(unknown))
            )

        samples = [
            sample
            for sample in cohort
            if sample in requested
        ]
    else:
        samples = cohort

    print(
        f"Plotting NLSDeconv for {len(samples)} sections",
        flush=True,
    )

    rows = []

    for sample in samples:
        print(f"\n===== {sample} =====", flush=True)

        adata, props, available = load_nls(sample)

        selected = select_celltypes(
            available,
            args.celltypes,
        )

        indices = [
            available.index(celltype)
            for celltype in selected
        ]

        sample_out = OUT_ROOT / "spatial" / sample
        sample_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        for celltype, index in zip(
            selected,
            indices,
        ):
            plot_celltype(
                adata=adata,
                values=props[:, index],
                celltype=celltype,
                stem=sample_out / safe_name(celltype),
                qmin=args.qmin,
                qmax=args.qmax,
                cmap=args.cmap,
            )

        plot_panel(
            adata=adata,
            props=props,
            celltypes=selected,
            indices=indices,
            stem=sample_out / "all_celltypes_panel",
            qmin=args.qmin,
            qmax=args.qmax,
            cmap=args.cmap,
        )

        rows.append(
            {
                "sample": sample,
                "spots": adata.n_obs,
                "n_celltypes": len(selected),
                "celltypes": ";".join(selected),
            }
        )

        print(
            f"[OK] {sample}: "
            f"{len(selected)} cell types",
            flush=True,
        )

    summary = pd.DataFrame(rows)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT_ROOT / "nlsdeconv_plot_summary.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"\nWrote plots under: {OUT_ROOT / 'spatial'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

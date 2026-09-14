#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
import scanpy as sc
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None exists: " + " | ".join(map(str, paths)))

def natural_key(x):
    s = str(x)
    try:
        return (0, int(s))
    except Exception:
        return (1, s)

def load_hires_background(h5ad_path):
    adata = sc.read_h5ad(h5ad_path, backed="r")

    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        raise RuntimeError(f"No Visium spatial metadata in {h5ad_path}")

    library_id = next(iter(adata.uns["spatial"]))
    spatial = adata.uns["spatial"][library_id]

    if "hires" not in spatial.get("images", {}):
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        raise RuntimeError(f"No hires H&E image in {h5ad_path}")

    image = np.asarray(spatial["images"]["hires"])
    scale = float(spatial["scalefactors"]["tissue_hires_scalef"])

    if getattr(adata, "file", None) is not None:
        adata.file.close()

    return image, scale


def plot_spatial(
    df,
    key,
    out_png,
    title,
    categorical=False,
    background=None,
    hires_scale=1.0,
    rotation="none",
):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if background is not None:
        x = df["pxl_col_in_fullres"].astype(float).to_numpy() * hires_scale
        y = df["pxl_row_in_fullres"].astype(float).to_numpy() * hires_scale

        # Figure 3 section-specific display orientation.
        # Rotate image and coordinates together so registration is preserved.
        if rotation == "cw":
            h0, w0 = background.shape[:2]
            background = np.rot90(background, k=3)
            x, y = (h0 - 1) - y, x
        elif rotation == "ccw":
            h0, w0 = background.shape[:2]
            background = np.rot90(background, k=1)
            x, y = y, (w0 - 1) - x
        elif rotation == "180":
            h0, w0 = background.shape[:2]
            background = np.rot90(background, k=2)
            x, y = (w0 - 1) - x, (h0 - 1) - y
        elif rotation != "none":
            raise ValueError(f"Unknown spatial rotation: {rotation}")

        ax.imshow(background, origin="upper")
    else:
        x = df["pxl_col_in_fullres"].astype(float).to_numpy()
        y = df["pxl_row_in_fullres"].astype(float).to_numpy()

    if categorical:
        cats = pd.Categorical(df[key].astype(str))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in cats.codes]

        ax.scatter(
            x, y,
            c=colors,
            s=18,
            linewidths=0,
            alpha=0.88,
        )

        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                label=str(cat),
                markerfacecolor=cmap(i % 20),
                markeredgecolor="none",
                markersize=6,
            )
            for i, cat in enumerate(cats.categories)
        ]

        ax.legend(
            handles=handles,
            title=key,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

    else:
        sca = ax.scatter(
            x, y,
            c=df[key].astype(float),
            s=18,
            linewidths=0,
            alpha=0.88,
        )

        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(key)

    if background is not None:
        # Crop around retained Visium spots while keeping H&E context.
        h, w = background.shape[:2]
        pad = 0.035 * max(w, h)

        ax.set_xlim(
            max(0, x.min() - pad),
            min(w, x.max() + pad),
        )
        ax.set_ylim(
            min(h, y.max() + pad),
            max(0, y.min() - pad),
        )
    else:
        ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["noHMM", "HMM_i6"])
    ap.add_argument("--resolution", type=float, required=True)
    ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-h5ad", action="store_true")
    ap.add_argument(
        "--spatial-h5ad",
        default=None,
        help="Visium/NLSDeconv H5AD containing uns['spatial'] H&E image and scalefactors",
    )
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    infer_dir = prepared / f"infercnv_R_out_{args.mode}"
    export_dir = infer_dir / "leiden_export"
    res_label = f"{args.resolution:.2f}"
    out_dir = infer_dir / f"leiden_posthoc_res_{res_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = mmread(export_dir / "infercnv_expr_genes_by_spots.mtx").tocsr()
    genes_path = read_first_existing([export_dir / "infercnv_genes.tsv", export_dir / "genes.tsv"])
    barcodes_path = read_first_existing([export_dir / "infercnv_barcodes.tsv", export_dir / "barcodes.tsv", export_dir / "spots.tsv"])

    genes = pd.read_csv(genes_path, sep="\t", header=None)[0].astype(str).tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str).tolist()

    X = X.T.tocsr()

    coords = pd.read_csv(prepared / "spatial_coords.tsv", sep="\t")
    coords["barcode"] = coords["barcode"].astype(str)
    coords = coords.set_index("barcode")

    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    obs = obs.join(coords, how="left")

    keep = obs["pxl_col_in_fullres"].notna() & obs["pxl_row_in_fullres"].notna()
    X = X[keep.values, :]
    obs = obs.loc[keep].copy()

    X_dense = X.toarray().astype(np.float32)
    cnv_dev = X_dense - 1.0

    adata = ad.AnnData(
        X=cnv_dev,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obs["cnv_abs_dev_mean"] = np.mean(np.abs(cnv_dev), axis=1).astype(np.float32)

    sc.pp.scale(adata, max_value=5)
    sc.tl.pca(adata, n_comps=min(args.n_pcs, adata.n_obs - 1, adata.n_vars - 1), random_state=args.seed)
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=min(args.n_pcs, adata.obsm["X_pca"].shape[1]), random_state=args.seed)
    sc.tl.leiden(adata, resolution=args.resolution, key_added="cnv_leiden", random_state=args.seed)

    tab = adata.obs.copy()
    tab.to_csv(out_dir / "infercnv_R_leiden_spatial_table.tsv", sep="\t")

    background = None
    hires_scale = 1.0

    # Historical Figure 3 display orientation:
    # bc2043 is rotated 90 degrees clockwise; other sections are unchanged.
    spatial_rotation = (
        "cw" if prepared.name.startswith("bc2043_") else "none"
    )

    if args.spatial_h5ad is not None:
        background, hires_scale = load_hires_background(
            Path(args.spatial_h5ad)
        )
        print(
            f"[H&E] {args.spatial_h5ad}: "
            f"shape={background.shape}, hires_scale={hires_scale}"
        )

    plot_spatial(
        tab,
        "cnv_abs_dev_mean",
        out_dir / "spatial_cnv_abs_dev_mean.png",
        f"{prepared.name} R inferCNV CNV magnitude {args.mode}, res={res_label}",
        categorical=False,
        background=background,
        hires_scale=hires_scale,
        rotation=spatial_rotation,
    )
    plot_spatial(
        tab,
        "cnv_leiden",
        out_dir / "spatial_cnv_leiden.png",
        f"{prepared.name} R inferCNV Leiden {args.mode}, res={res_label}",
        categorical=True,
        background=background,
        hires_scale=hires_scale,
        rotation=spatial_rotation,
    )

    labels = adata.obs["cnv_leiden"].astype(str).values
    uniq = sorted(pd.unique(labels), key=natural_key)
    mean_rows = []
    means = []
    for u in uniq:
        mask = labels == u
        vec = cnv_dev[mask, :].mean(axis=0)
        means.append(vec)
        row = {"cluster_label": str(u), "n_spots": int(mask.sum()), "cnv_abs_dev_mean": float(np.mean(np.abs(vec)))}
        row.update({g: float(vec[i]) for i, g in enumerate(genes)})
        mean_rows.append(row)

    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(out_dir / "infercnv_R_leiden_cluster_mean_profiles.tsv", sep="\t", index=False)

    M = np.vstack(means)
    var = M.var(axis=0)
    top = np.argsort(var)[::-1][:min(3000, M.shape[1])]
    top = np.sort(top)
    Mplot = M[:, top]

    vmax = max(0.1, float(np.nanpercentile(np.abs(Mplot), 99)))
    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(uniq))))
    im = ax.imshow(Mplot, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(uniq)))
    ax.set_yticklabels([f"Leiden {u} (n={(labels == u).sum()})" for u in uniq])
    ax.set_xticks([])
    ax.set_xlabel("Genomic genes, top-variable subset")
    ax.set_title(f"{prepared.name} R inferCNV cluster means {args.mode}, res={res_label}")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("inferCNV deviation from neutral")
    fig.tight_layout()
    fig.savefig(out_dir / "infercnv_R_leiden_cluster_mean_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "infercnv_R_leiden_cluster_mean_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    if args.write_h5ad:
        adata.write_h5ad(out_dir / "infercnv_R_leiden_posthoc.h5ad", compression="gzip")

    counts = tab["cnv_leiden"].astype(str).value_counts().sort_index()
    counts.to_csv(out_dir / "cnv_leiden_cluster_sizes.tsv", sep="\t", header=["n_spots"])
    print(f"[OK] {prepared.name} {args.mode} res={res_label} -> {out_dir}")
    print(counts.to_string())

if __name__ == "__main__":
    main()

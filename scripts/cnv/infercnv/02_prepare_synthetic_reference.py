#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
import scanpy as sc


def open_maybe_gzip(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_gtf_attrs(attr):
    out = {}
    for part in str(attr).split(";"):
        part = part.strip()
        if not part or " " not in part:
            continue
        k, v = part.split(" ", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def build_gene_pos(gtf, cache):
    cache = Path(cache)
    if cache.exists():
        return pd.read_csv(cache, sep="\t")

    rows = []
    with open_maybe_gzip(gtf) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom, start, end, attrs = f[0], f[3], f[4], f[8]
            a = parse_gtf_attrs(attrs)
            gene = a.get("gene_name") or a.get("gene_id")
            if gene is None:
                continue
            rows.append((gene, chrom, int(start), int(end)))

    df = pd.DataFrame(rows, columns=["gene", "chr", "start", "end"])
    df = df.drop_duplicates("gene")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, sep="\t", index=False)
    return df


def close_composition(P, eps=1e-6):
    P = np.asarray(P, dtype=float)
    P = np.where(np.isfinite(P), P, 0)
    P = np.clip(P, eps, None)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def ilr_pivot(P, eps=1e-6):
    P = close_composition(P, eps=eps)
    n, K = P.shape
    logP = np.log(P)
    Z = np.zeros((n, K - 1), dtype=np.float64)
    for j in range(1, K):
        gm_prev = logP[:, :j].mean(axis=1)
        Z[:, j - 1] = np.sqrt(j / (j + 1.0)) * (gm_prev - logP[:, j])
    return Z


def fit_ridge(Y, Z, alpha):
    X = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    R = np.eye(X.shape[1])
    R[0, 0] = 0.0
    B = np.linalg.solve(X.T @ X + alpha * R, X.T @ Y)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nls-root", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="Output folder names, same length as samples")
    ap.add_argument("--cancer-col", default="Cancer Epithelial")
    ap.add_argument("--train-cancer-max", type=float, default=0.20)
    ap.add_argument("--train-nonmal-min", type=float, default=0.60)
    ap.add_argument("--max-train", type=int, default=600)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--n-ref-cells", type=int, default=25)
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert len(args.samples) == len(args.names)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    gene_pos = build_gene_pos(args.gtf, out_root / "_gene_positions_from_gtf.tsv")
    gene_pos = gene_pos.drop_duplicates("gene").set_index("gene")

    summaries = []

    for sid, outname in zip(args.samples, args.names):
        print("\n" + "=" * 80)
        print("[SAMPLE]", sid, "->", outname)
        print("=" * 80)

        h5ad = Path(args.nls_root) / sid / f"{sid}.nlsdeconv.h5ad"
        ad = sc.read_h5ad(h5ad)

        # Gene intersection and genomic order.
        if "gene_name" in ad.var.columns:
            genes0 = ad.var["gene_name"].astype(str).values
        else:
            genes0 = ad.var_names.astype(str).values

        keep = pd.Index(genes0).isin(gene_pos.index)
        ad = ad[:, keep].copy()
        genes0 = genes0[keep]

        gp = gene_pos.loc[genes0].copy()
        gp["gene"] = genes0
        gp = gp.reset_index(drop=True)
        gp["_chrom_sort"] = gp["chr"].astype(str).str.replace("chr", "", regex=False)
        gp["_chrom_num"] = pd.to_numeric(gp["_chrom_sort"].replace({"X": 23, "Y": 24, "M": 25, "MT": 25}), errors="coerce").fillna(999).astype(int)
        gp = gp.sort_values(["_chrom_num", "start", "end"], kind="mergesort")

        order = gp.index.values
        ad = ad[:, order].copy()
        gp = gp.iloc[np.arange(len(gp))].copy()
        genes = gp["gene"].astype(str).tolist()

        # Raw/query expression.
        X_raw = ad.layers["counts"] if "counts" in ad.layers else ad.X
        if sp.issparse(X_raw):
            X_raw = X_raw.tocsr()
        else:
            X_raw = sp.csr_matrix(X_raw)

        # Build normalized log matrix for regression.
        X = X_raw.astype(np.float64).copy()
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib <= 0] = 1.0
        X_norm = X.multiply(args.target_sum / lib[:, None])
        X_log = X_norm.copy()
        X_log.data = np.log1p(X_log.data)
        Y = X_log.toarray()

        # Deconvolution table.
        P = np.asarray(ad.obsm["nlsdeconv_props"], dtype=float)
        celltypes = [str(x) for x in list(ad.uns["nlsdeconv_celltypes"])]
        if args.cancer_col not in celltypes:
            raise ValueError(f"Missing cancer col {args.cancer_col}. Available: {celltypes}")
        cancer = P[:, celltypes.index(args.cancer_col)]
        cancer = np.clip(cancer, 0, 1)
        nonmal = 1.0 - cancer

        train = np.isfinite(cancer) & (cancer <= args.train_cancer_max) & (nonmal >= args.train_nonmal_min)
        if train.sum() < 25:
            idx = np.argsort(cancer)[:min(args.max_train, ad.n_obs)]
        else:
            idx = np.where(train)[0]
            if len(idx) > args.max_train:
                rng = np.random.default_rng(args.seed)
                idx = np.sort(rng.choice(idx, size=args.max_train, replace=False))

        P_closed = close_composition(P)
        Z = ilr_pivot(P_closed[idx, :])
        B = fit_ridge(Y[idx, :], Z, args.ridge_alpha)

        p_ref = close_composition(P_closed[idx, :]).mean(axis=0)
        p_ref = p_ref / p_ref.sum()
        z_ref = ilr_pivot(p_ref[None, :])
        ref_log = np.concatenate([np.ones((1, 1)), z_ref], axis=1) @ B
        ref_log = np.clip(ref_log.ravel(), 0, None)

        # Convert predicted log-normalized expression back to pseudo-count scale.
        ref_counts = np.expm1(ref_log)
        ref_counts = np.clip(ref_counts, 0, None)
        ref_counts = ref_counts / max(ref_counts.sum(), 1e-9) * args.target_sum
        ref_mat = np.tile(ref_counts[None, :], (args.n_ref_cells, 1))
        ref_mat = sp.csr_matrix(ref_mat)

        # Use normalized count-like query matrix too, so query/ref scales are compatible.
        query_mat = X_norm.tocsr()
        full_mat = sp.vstack([query_mat, ref_mat], format="csr")  # spots x genes

        barcodes = list(ad.obs_names.astype(str)) + [f"SYNTHREF_{i:03d}" for i in range(args.n_ref_cells)]
        annotations = pd.DataFrame({
            "barcode": barcodes,
            "group": ["query"] * ad.n_obs + ["synthref"] * args.n_ref_cells,
        })

        outdir = out_root / outname
        outdir.mkdir(parents=True, exist_ok=True)

        # R inferCNV expects genes x cells.
        mmwrite(outdir / "counts.mtx", full_mat.T.tocoo())

        pd.Series(genes).to_csv(outdir / "genes.tsv", sep="\t", index=False, header=False)
        pd.Series(barcodes).to_csv(outdir / "barcodes.tsv", sep="\t", index=False, header=False)

        annotations.to_csv(outdir / "annotations.tsv", sep="\t", index=False, header=False)
        pd.Series(["synthref"]).to_csv(outdir / "ref_group_names.txt", index=False, header=False)

        gene_order = gp[["gene", "chr", "start", "end"]].copy()
        gene_order.to_csv(outdir / "gene_order.tsv", sep="\t", index=False, header=False)

        if "spatial" in ad.obsm:
            coords = pd.DataFrame({
                "barcode": ad.obs_names.astype(str),
                "pxl_col_in_fullres": ad.obsm["spatial"][:, 0],
                "pxl_row_in_fullres": ad.obsm["spatial"][:, 1],
            })
            coords.to_csv(outdir / "spatial_coords.tsv", sep="\t", index=False)

        refcomp = pd.DataFrame({
            "celltype": celltypes,
            "p_ref_mean_train": p_ref,
            "mean_train": P_closed[idx, :].mean(axis=0),
        })
        refcomp.to_csv(outdir / "synthref_reference_composition.tsv", sep="\t", index=False)

        qc = pd.DataFrame({
            "barcode": ad.obs_names.astype(str),
            "cancer_frac": cancer,
            "nonmal_frac": nonmal,
            "used_train": np.isin(np.arange(ad.n_obs), idx),
            "histopathology": ad.obs["histopathology"].astype(str).values if "histopathology" in ad.obs else "NA",
        })
        qc.to_csv(outdir / "synthref_qc.tsv", sep="\t", index=False)

        summaries.append({
            "sample": sid,
            "outname": outname,
            "n_query_spots": ad.n_obs,
            "n_ref_cells": args.n_ref_cells,
            "n_genes": len(genes),
            "n_train": len(idx),
            "outdir": str(outdir),
        })

        print("[OK] wrote", outdir)

    pd.DataFrame(summaries).to_csv(out_root / "prepare_synthref_ILR_R_infercnv_summary.tsv", sep="\t", index=False)
    print("[DONE]", out_root / "prepare_synthref_ILR_R_infercnv_summary.tsv")


if __name__ == "__main__":
    main()

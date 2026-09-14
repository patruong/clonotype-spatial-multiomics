#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

def qc_from_X(X):
    if sparse.issparse(X):
        X = X.tocsr()
        n_counts = np.asarray(X.sum(axis=1)).ravel()
        n_features = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        A = np.asarray(X)
        n_counts = A.sum(axis=1)
        n_features = (A > 0).sum(axis=1)
    return n_features, n_counts

def find_pathology_col(obs, requested):
    cols = list(obs.columns)
    if requested and requested in cols:
        return requested
    preferred = [
        "histopathology",
        "histopathology_raw",
        "pathology",
        "pathology_annotation",
        "annotation",
        "morphology",
        "Morphology",
        "Graph-based",
    ]
    for c in preferred:
        if c in cols:
            return c
    hits = [c for c in cols if "path" in c.lower() or "morph" in c.lower() or "histo" in c.lower()]
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nls-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--min-features", type=float, default=500)
    ap.add_argument("--min-counts", type=float, default=1000)
    ap.add_argument("--exclude-labels", default="Necrosis")
    ap.add_argument("--pathology-col", default="")
    args = ap.parse_args()

    nls_root = Path(args.nls_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    exclude = {x.strip().lower() for x in args.exclude_labels.split(",") if x.strip()}
    rows = []

    for sid in args.samples:
        # Support both the historical nested layout:
        #   <root>/<sample>/<sample>.nlsdeconv.h5ad
        # and the clean-repo flat layout:
        #   <root>/<sample>.nlsdeconv.h5ad
        candidates = [
            nls_root / sid / f"{sid}.nlsdeconv.h5ad",
            nls_root / f"{sid}.nlsdeconv.h5ad",
        ]
        in_h5 = next((x for x in candidates if x.exists()), None)
        if in_h5 is None:
            raise FileNotFoundError(
                "Could not find NLSDeconv input; tried: "
                + " | ".join(map(str, candidates))
            )

        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5 = out_dir / f"{sid}.nlsdeconv.h5ad"

        print(f"[READ] {in_h5}")
        ad = sc.read_h5ad(in_h5)

        X_qc = ad.layers["counts"] if "counts" in ad.layers else ad.X
        n_features, n_counts = qc_from_X(X_qc)

        keep_qc = (n_features > args.min_features) & (n_counts > args.min_counts)

        keep_tissue = np.ones(ad.n_obs, dtype=bool)
        if "in_tissue" in ad.obs.columns:
            keep_tissue = ad.obs["in_tissue"].astype(int).to_numpy() == 1

        path_col = find_pathology_col(ad.obs, args.pathology_col)
        keep_path = np.ones(ad.n_obs, dtype=bool)
        if path_col is not None and exclude:
            keep_path = ~ad.obs[path_col].astype(str).str.lower().isin(exclude).to_numpy()

        keep = keep_qc & keep_tissue & keep_path

        ad.obs["nFeature_spot_qc"] = n_features
        ad.obs["nCount_spot_qc"] = n_counts
        ad.obs["pass_spot_qc_infercnv"] = keep

        before = ad.n_obs
        ad2 = ad[keep, :].copy()
        ad2.write_h5ad(out_h5, compression="gzip")

        rows.append({
            "sample": sid,
            "input_h5ad": str(in_h5),
            "output_h5ad": str(out_h5),
            "spots_before": int(before),
            "spots_after": int(ad2.n_obs),
            "spots_removed": int(before - ad2.n_obs),
            "pct_removed": float((before - ad2.n_obs) / max(before, 1) * 100),
            "min_features": args.min_features,
            "min_counts": args.min_counts,
            "pathology_col": path_col if path_col is not None else "",
            "exclude_labels": ",".join(sorted(exclude)),
            "median_nFeature_before": float(np.median(n_features)),
            "median_nCount_before": float(np.median(n_counts)),
        })

        qc = pd.DataFrame({
            "barcode": ad.obs_names.astype(str),
            "nFeature_spot_qc": n_features,
            "nCount_spot_qc": n_counts,
            "keep_qc": keep_qc,
            "keep_tissue": keep_tissue,
            "keep_pathology": keep_path,
            "keep_final": keep,
        })
        if path_col is not None:
            qc[path_col] = ad.obs[path_col].astype(str).values
        qc.to_csv(out_dir / f"{sid}.spotQC500_1000_nonecrosis_audit.tsv", sep="\t", index=False)

        print(f"[WRITE] {out_h5}")
        print(f"[QC] {sid}: kept {ad2.n_obs}/{before}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "spotQC500_1000_nonecrosis_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print("[DONE]", out_root / "spotQC500_1000_nonecrosis_summary.tsv")

if __name__ == "__main__":
    main()

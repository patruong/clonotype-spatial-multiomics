#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc


LABEL_MAP = {
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


def clean_label(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unannotated"
    x = str(x).strip()
    return LABEL_MAP.get(x, x)


def find_morphology(sample, annot_dir):
    suffix = sample[-6:]
    p = annot_dir / f"Morphology_{suffix}.csv"

    if p.exists():
        return p

    hits = sorted(annot_dir.glob(f"*{suffix}*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No morphology annotation found for {sample}"
        )

    return hits[0]


def pick_label_col(df):
    for c in ["Morphology", "Graph-based"]:
        if c in df.columns:
            return c

    for c in df.columns:
        if any(x in c.lower() for x in ["morph", "path", "graph"]):
            return c

    raise RuntimeError(
        f"No pathology label column found. Columns={df.columns.tolist()}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--pathology-annot-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--exclude-labels", default="Necrosis")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    annot_dir = Path(args.pathology_annot_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    exclude = {
        x.strip().lower()
        for x in args.exclude_labels.split(",")
        if x.strip()
    }

    summary = []

    for sample in args.samples:
        print()
        print("=" * 80)
        print("[SAMPLE]", sample)
        print("=" * 80)

        in_h5ad = input_root / f"{sample}.nlsdeconv.h5ad"
        if not in_h5ad.exists():
            raise FileNotFoundError(in_h5ad)

        pathology_file = find_morphology(sample, annot_dir)

        print("[READ]", in_h5ad)
        print("[PATHOLOGY]", pathology_file)

        adata = sc.read_h5ad(in_h5ad)

        morph = pd.read_csv(pathology_file)
        label_col = pick_label_col(morph)

        if "Barcode" not in morph.columns:
            raise KeyError(f"{pathology_file} missing Barcode column")

        morph["_raw"] = morph["Barcode"].astype(str)
        morph["_nosuf"] = morph["_raw"].str.replace(
            r"-1$", "", regex=True
        )

        obs = pd.DataFrame(index=adata.obs_names)
        obs["_raw"] = obs.index.astype(str)
        obs["_nosuf"] = obs["_raw"].str.replace(
            r"-1$", "", regex=True
        )

        raw_map = (
            morph.drop_duplicates("_raw")
            .set_index("_raw")[label_col]
        )

        nosuf_map = (
            morph.drop_duplicates("_nosuf")
            .set_index("_nosuf")[label_col]
        )

        labels = (
            obs["_raw"]
            .map(raw_map)
            .fillna(obs["_nosuf"].map(nosuf_map))
        )

        adata.obs["histopathology_raw"] = labels.values
        adata.obs["histopathology"] = (
            labels.map(clean_label)
            .fillna("Unannotated")
            .values
        )
        adata.obs["pathology_file"] = str(pathology_file)

        print("[LABELS BEFORE FILTER]")
        print(
            adata.obs["histopathology"]
            .value_counts(dropna=False)
            .to_string()
        )

        before = adata.n_obs

        keep = ~(
            adata.obs["histopathology"]
            .astype(str)
            .str.lower()
            .isin(exclude)
        )

        adata = adata[keep].copy()
        after = adata.n_obs

        out_dir = out_root / sample
        out_dir.mkdir(parents=True, exist_ok=True)

        out_h5ad = out_dir / f"{sample}.nlsdeconv.h5ad"
        adata.write_h5ad(out_h5ad, compression="gzip")

        print(
            f"[FILTER] {sample}: "
            f"{before} -> {after} "
            f"(removed {before-after})"
        )
        print("[WRITE]", out_h5ad)

        summary.append({
            "sample": sample,
            "input_h5ad": str(in_h5ad),
            "pathology_file": str(pathology_file),
            "label_column": label_col,
            "spots_before": before,
            "spots_after": after,
            "spots_removed": before - after,
            "exclude_labels": ",".join(sorted(exclude)),
        })

    pd.DataFrame(summary).to_csv(
        out_root / "histopathology_filter_summary.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()

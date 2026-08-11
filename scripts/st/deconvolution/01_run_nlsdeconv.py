#!/usr/bin/env python3
from __future__ import annotations

"""
Run NLSDeconv on the manuscript ST cohort.

Inputs
------
metadata/metadata_short.tsv
data/processed/st/space_ranger_outs/
data/references/single_cell/sc_miniatlas/

Outputs
-------
data/processed/st/nlsdeconv/
    <capture_area_id>.nlsdeconv_props.csv.gz
    <capture_area_id>.nlsdeconv.h5ad

NLSDeconv provenance
--------------------
https://github.com/tinachentc/NLSDeconv
commit: 3bbe0a53a8b0f6798eacf7eab9f7298742fd27fd

Parameters reproduce the original SMA-VDJ analysis:
lr=1e-2, reg=0.1, n_fold=5, warm_start=True,
num_epochs=800, device="cpu".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[3]

METADATA = ROOT / "metadata" / "metadata_short.tsv"

SC_ROOT = ROOT / "data" / "references" / "single_cell" / "sc_miniatlas"
SC_MTX_DIR = SC_ROOT / "expression" / "5f1936d7771a5b0dbb904757"
SC_META = SC_ROOT / "metadata" / "Whole_miniatlas_meta.csv"

ST_ROOT = ROOT / "data" / "processed" / "st" / "space_ranger_outs"
OUTDIR = ROOT / "data" / "processed" / "st" / "nlsdeconv"

NLS_PATH = ROOT / "envs" / "conda" / "src" / "NLSDeconv"
NLS_COMMIT = "3bbe0a53a8b0f6798eacf7eab9f7298742fd27fd"

if not NLS_PATH.is_dir():
    raise SystemExit(
        f"NLSDeconv source not found: {NLS_PATH}\n"
        "Clone the pinned NLSDeconv repository first."
    )

sys.path.insert(0, str(NLS_PATH))

from preprocess import Preprocess  # noqa: E402
from deconv import Deconv  # noqa: E402


def load_metadata() -> pd.DataFrame:
    md = pd.read_csv(METADATA, sep="\t", dtype=str)

    if "capture_area_id" not in md.columns:
        raise ValueError("metadata_short.tsv lacks capture_area_id")

    md["capture_area_id"] = md["capture_area_id"].str.strip()

    if md["capture_area_id"].duplicated().any():
        raise ValueError("capture_area_id must be unique in metadata")

    return md


def load_scrna_reference() -> sc.AnnData:
    mtx = SC_MTX_DIR / "matrix.mtx.gz"
    features = SC_MTX_DIR / "features.tsv.gz"
    barcodes = SC_MTX_DIR / "barcodes.tsv.gz"

    for path in (mtx, features, barcodes, SC_META):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading single-cell reference...", flush=True)

    ad = sc.read_mtx(mtx).T

    var = pd.read_csv(features, sep="\t", header=None)

    if var.shape[1] >= 2:
        ad.var["gene_id"] = var.iloc[:, 0].astype(str).values
        ad.var["gene_symbol"] = var.iloc[:, 1].astype(str).values
        ad.var_names = ad.var["gene_symbol"].values
    else:
        ad.var_names = var.iloc[:, 0].astype(str).values

    ad.var_names_make_unique()

    obs = pd.read_csv(barcodes, sep="\t", header=None)
    ad.obs_names = obs.iloc[:, 0].astype(str).values

    meta = pd.read_csv(SC_META, low_memory=False)

    required = {"NAME", "celltype_major"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"Single-cell metadata missing columns: {sorted(missing)}"
        )

    labels = (
        meta[["NAME", "celltype_major"]]
        .dropna()
        .drop_duplicates(subset=["NAME"])
        .set_index("NAME")["celltype_major"]
        .astype(str)
        .reindex(ad.obs_names)
    )

    matched = int(labels.notna().sum())

    print(
        f"Single-cell metadata matched: "
        f"{matched:,}/{ad.n_obs:,} cells",
        flush=True,
    )

    if matched == 0:
        raise RuntimeError(
            "No cells matched between single-cell barcodes and metadata."
        )

    keep = labels.notna().to_numpy()

    if not keep.all():
        ad = ad[keep].copy()
        labels = labels.loc[ad.obs_names]

    ad.obs["celltype"] = labels.to_numpy()
    ad.obs["cell_type"] = ad.obs["celltype"].astype(str)

    sc.pp.filter_genes(ad, min_cells=5)

    print(
        f"Reference: {ad.n_obs:,} cells x {ad.n_vars:,} genes",
        flush=True,
    )

    return ad


def load_visium(sample: str) -> sc.AnnData:
    path = ST_ROOT / sample

    if not path.is_dir():
        raise FileNotFoundError(path)

    ad = sc.read_visium(str(path))
    ad.var_names_make_unique()

    # Restrict to GEX if feature_types are present.
    if "feature_types" in ad.var.columns:
        mask = (
            ad.var["feature_types"]
            .astype(str)
            .eq("Gene Expression")
        )
        if mask.any():
            ad = ad[:, mask].copy()

    return ad


def run_sample(
    sample: str,
    ad_sc: sc.AnnData,
    lr: float,
    reg: float,
    epochs: int,
    device: str,
    force: bool,
) -> dict[str, object]:

    out_csv = OUTDIR / f"{sample}.nlsdeconv_props.csv.gz"
    out_h5ad = OUTDIR / f"{sample}.nlsdeconv.h5ad"

    if out_csv.exists() and out_h5ad.exists() and not force:
        print(f"[SKIP] {sample}: outputs already exist", flush=True)
        return {
            "sample": sample,
            "status": "existing",
        }

    print(f"\n===== {sample} =====", flush=True)

    ad_st = load_visium(sample)

    common = len(
        set(ad_sc.var_names).intersection(ad_st.var_names)
    )

    print(
        f"ST: {ad_st.n_obs:,} spots x {ad_st.n_vars:,} genes",
        flush=True,
    )
    print(
        f"Common genes before preprocessing: {common:,}",
        flush=True,
    )

    pp = Preprocess(
        ad_st,
        ad_sc,
        celltype_key="celltype",
    )

    ad_st_pp, ad_sc_pp = pp.preprocess()

    print(
        f"Selected genes: {ad_st_pp.n_vars:,}",
        flush=True,
    )

    model = Deconv(
        ad_sc_pp,
        ad_st_pp,
        normalization=True,
    )

    res, runtime, celltypes = model.NLS(
        lr=lr,
        reg=reg,
        n_fold=5,
        warm_start=True,
        num_epochs=epochs,
        device=device,
    )

    if hasattr(res, "detach"):
        res = res.detach().cpu().numpy()
    else:
        res = np.asarray(res)

    df = pd.DataFrame(
        res,
        index=ad_st_pp.obs_names,
        columns=[str(x) for x in celltypes],
    ).reindex(ad_st.obs_names)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out_csv,
        compression="gzip",
    )

    ad_st.obsm["nlsdeconv_props"] = df.to_numpy()
    ad_st.uns["nlsdeconv_celltypes"] = list(df.columns)
    ad_st.uns["nlsdeconv_runtime_sec"] = float(runtime)
    ad_st.uns["nlsdeconv_commit"] = NLS_COMMIT
    ad_st.uns["nlsdeconv_parameters"] = {
        "lr": float(lr),
        "reg": float(reg),
        "n_fold": 5,
        "warm_start": True,
        "num_epochs": int(epochs),
        "device": str(device),
    }

    ad_st.write_h5ad(out_h5ad)

    print(f"[OK] {out_csv}", flush=True)
    print(f"[OK] {out_h5ad}", flush=True)

    return {
        "sample": sample,
        "status": "completed",
        "spots": ad_st.n_obs,
        "genes_preprocessed": ad_st_pp.n_vars,
        "celltypes": len(celltypes),
        "runtime_sec": float(runtime),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional capture_area_id subset",
    )

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    md = load_metadata()

    cohort = md["capture_area_id"].tolist()

    if args.samples:
        requested = set(args.samples)
        samples = [x for x in cohort if x in requested]

        unknown = requested - set(cohort)
        if unknown:
            raise ValueError(
                f"Requested samples not in manuscript metadata: "
                f"{sorted(unknown)}"
            )
    else:
        samples = cohort

    print(f"NLSDeconv commit: {NLS_COMMIT}")
    print(f"Cohort sections: {len(samples)}")
    print(
        f"Parameters: lr={args.lr}, reg={args.reg}, "
        f"epochs={args.epochs}, device={args.device}",
        flush=True,
    )

    ad_sc = load_scrna_reference()

    summaries = []

    for sample in samples:
        summaries.append(
            run_sample(
                sample=sample,
                ad_sc=ad_sc,
                lr=args.lr,
                reg=args.reg,
                epochs=args.epochs,
                device=args.device,
                force=args.force,
            )
        )

    summary = pd.DataFrame(summaries)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTDIR / "nlsdeconv_run_summary.tsv",
        sep="\t",
        index=False,
    )

    print("\nDone.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

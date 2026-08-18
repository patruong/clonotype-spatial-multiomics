#!/usr/bin/env python3

"""
Match MOFA-FLEX factors across adjacent 9AA/metabolite and DHBA/lipid sections.

Matching anchor:
    Pearson correlation of ST factor loadings over genes shared between
    the two independently fitted adjacent-section models.

Outputs:
    - complete 12 x 12 pairwise correlation matrices
    - best right factor for every left factor
    - best left factor for every right factor
    - mutual-best matches (primary)
    - Hungarian one-to-one matches (sensitivity)

For historical reproducibility, both are reported:
    pearson_r:
        conventional Pearson correlation

    historical_r:
        exact calculation used by the historical SMA-VDJ script:
        population-SD z-score followed by division by (n - 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def orient_w(W, n_features, n_factors):
    W = np.asarray(W, dtype=float)

    if W.shape == (n_features, n_factors):
        return W

    if W.shape == (n_factors, n_features):
        return W.T

    if W.shape[0] == n_factors:
        return W.T

    if W.shape[1] == n_factors:
        return W

    raise RuntimeError(
        f"Cannot orient W shape={W.shape}; "
        f"n_features={n_features}, n_factors={n_factors}"
    )


def load_result(path):
    d = np.load(path, allow_pickle=True)

    Z = np.asarray(d["Z"], dtype=float)
    n_factors = Z.shape[1]

    factor_names = np.asarray(
        d["factor_names"]
        if "factor_names" in d.files
        else [f"F{i + 1}" for i in range(n_factors)],
        dtype=str,
    )

    features = np.asarray(d["features_ST"], dtype=str)
    W = orient_w(
        d["W_ST"],
        n_features=len(features),
        n_factors=n_factors,
    )

    return {
        "factor_names": factor_names,
        "features_ST": features,
        "W_ST": W,
        "path": str(path),
    }


def align_st_loadings(left, right):
    lf = pd.Index(left["features_ST"])
    rf = pd.Index(right["features_ST"])

    common = lf.intersection(rf)

    if len(common) < 3:
        raise RuntimeError(
            f"Only {len(common)} shared ST genes."
        )

    li = lf.get_indexer(common)
    ri = rf.get_indexer(common)

    A = left["W_ST"][li, :]
    B = right["W_ST"][ri, :]

    return A, B, common.to_numpy(dtype=str)


def zscore_population(X):
    X = np.asarray(X, dtype=float)

    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True, ddof=0)

    sd[(sd == 0) | ~np.isfinite(sd)] = np.nan

    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0

    return Z


def correlation_matrices(A, B):
    """
    Historical implementation:
        zscore(ddof=0), then dot / (n - 1)

    Conventional Pearson:
        same zscores, then dot / n
    """

    ZA = zscore_population(A)
    ZB = zscore_population(B)

    n = ZA.shape[0]

    historical = (ZA.T @ ZB) / max(n - 1, 1)
    pearson = (ZA.T @ ZB) / n

    return historical, pearson


def build_pairs(metadata):
    md = pd.read_csv(metadata, sep="\t")

    required = {
        "patient",
        "barcode",
        "capture_area_id",
        "MSI target",
    }

    missing = required - set(md.columns)
    if missing:
        raise RuntimeError(
            f"Metadata missing columns: {sorted(missing)}"
        )

    md["sample"] = (
        md["barcode"].astype(str)
        + "_"
        + md["capture_area_id"].astype(str)
    )

    target = md["MSI target"].astype(str).str.lower()

    left = md[target == "metabolites"].copy()
    right = md[target == "lipids"].copy()

    rows = []

    patients = sorted(
        set(left["patient"]).intersection(right["patient"])
    )

    for patient in patients:
        L = left[left["patient"] == patient]
        R = right[right["patient"] == patient]

        if len(L) != 1 or len(R) != 1:
            raise RuntimeError(
                f"Patient {patient}: expected exactly one metabolite "
                f"and one lipid section; got {len(L)} and {len(R)}."
            )

        l = L.iloc[0]
        r = R.iloc[0]

        rows.append(
            {
                "patient": patient,
                "left_sample": l["sample"],
                "right_sample": r["sample"],
                "left_barcode": l["barcode"],
                "right_barcode": r["barcode"],
                "left_capture_area_id": l["capture_area_id"],
                "right_capture_area_id": r["capture_area_id"],
            }
        )

    return pd.DataFrame(rows)


def main():
    repo = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        default=repo / "metadata" / "metadata_short.tsv",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo / "results" / "integration" / "mofaflex" / "f12",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repo
            / "results"
            / "integration"
            / "mofaflex"
            / "f12"
            / "adjacent_factor_matching"
        ),
    )

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.metadata)

    print("=== Adjacent 9AA -> DHBA pairs ===")
    print(pairs.to_string(index=False))
    print()

    pairs.to_csv(
        args.output_root / "adjacent_pairs.tsv",
        sep="\t",
        index=False,
    )

    all_pairwise = []
    left_best_rows = []
    right_best_rows = []
    mutual_rows = []
    hungarian_rows = []

    for _, pair in pairs.iterrows():

        patient = pair["patient"]
        left_sample = pair["left_sample"]
        right_sample = pair["right_sample"]

        left_path = (
            args.results_root
            / left_sample
            / "mofaflex_results.npz"
        )

        right_path = (
            args.results_root
            / right_sample
            / "mofaflex_results.npz"
        )

        if not left_path.exists():
            raise FileNotFoundError(left_path)

        if not right_path.exists():
            raise FileNotFoundError(right_path)

        left = load_result(left_path)
        right = load_result(right_path)

        A, B, common_genes = align_st_loadings(
            left,
            right,
        )

        historical_r, pearson_r = correlation_matrices(A, B)

        abs_r = np.abs(pearson_r)

        left_names = left["factor_names"]
        right_names = right["factor_names"]

        pair_name = (
            f"patient{patient}_"
            f"{left_sample}__vs__{right_sample}"
        )

        pair_dir = args.output_root / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            historical_r,
            index=left_names,
            columns=right_names,
        ).to_csv(
            pair_dir / "st_loading_historical_correlation.csv"
        )

        pd.DataFrame(
            pearson_r,
            index=left_names,
            columns=right_names,
        ).to_csv(
            pair_dir / "st_loading_pearson_correlation.csv"
        )

        pd.Series(common_genes, name="gene").to_csv(
            pair_dir / "shared_st_genes.tsv",
            sep="\t",
            index=False,
        )

        # ----------------------------------------------------------
        # Complete pairwise table
        # ----------------------------------------------------------

        for i, left_factor in enumerate(left_names):
            for j, right_factor in enumerate(right_names):
                all_pairwise.append(
                    {
                        "patient": patient,
                        "left_sample": left_sample,
                        "right_sample": right_sample,
                        "left_factor": left_factor,
                        "right_factor": right_factor,
                        "pearson_r": float(pearson_r[i, j]),
                        "abs_pearson_r": float(abs_r[i, j]),
                        "historical_r": float(historical_r[i, j]),
                        "sign": (
                            1 if pearson_r[i, j] >= 0 else -1
                        ),
                        "n_shared_st_genes": len(common_genes),
                    }
                )

        # ----------------------------------------------------------
        # Best factor in each direction
        # ----------------------------------------------------------

        left_best = np.nanargmax(abs_r, axis=1)
        right_best = np.nanargmax(abs_r, axis=0)

        for i, j in enumerate(left_best):
            left_best_rows.append(
                {
                    "patient": patient,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "left_factor": left_names[i],
                    "best_right_factor": right_names[j],
                    "pearson_r": float(pearson_r[i, j]),
                    "abs_pearson_r": float(abs_r[i, j]),
                    "historical_r": float(historical_r[i, j]),
                    "right_best_left_factor": left_names[right_best[j]],
                    "mutual_best": bool(right_best[j] == i),
                    "n_shared_st_genes": len(common_genes),
                }
            )

        for j, i in enumerate(right_best):
            right_best_rows.append(
                {
                    "patient": patient,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "right_factor": right_names[j],
                    "best_left_factor": left_names[i],
                    "pearson_r": float(pearson_r[i, j]),
                    "abs_pearson_r": float(abs_r[i, j]),
                    "historical_r": float(historical_r[i, j]),
                    "left_best_right_factor": right_names[left_best[i]],
                    "mutual_best": bool(left_best[i] == j),
                    "n_shared_st_genes": len(common_genes),
                }
            )

        # ----------------------------------------------------------
        # Mutual-best matches: primary
        # ----------------------------------------------------------

        for i, j in enumerate(left_best):
            if right_best[j] != i:
                continue

            mutual_rows.append(
                {
                    "patient": patient,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "left_factor": left_names[i],
                    "right_factor": right_names[j],
                    "pearson_r": float(pearson_r[i, j]),
                    "abs_pearson_r": float(abs_r[i, j]),
                    "historical_r": float(historical_r[i, j]),
                    "sign": (
                        1 if pearson_r[i, j] >= 0 else -1
                    ),
                    "n_shared_st_genes": len(common_genes),
                }
            )

        # ----------------------------------------------------------
        # Hungarian one-to-one matching: sensitivity
        # ----------------------------------------------------------

        rows, cols = linear_sum_assignment(-abs_r)

        for i, j in zip(rows, cols):
            hungarian_rows.append(
                {
                    "patient": patient,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "left_factor": left_names[i],
                    "right_factor": right_names[j],
                    "pearson_r": float(pearson_r[i, j]),
                    "abs_pearson_r": float(abs_r[i, j]),
                    "historical_r": float(historical_r[i, j]),
                    "sign": (
                        1 if pearson_r[i, j] >= 0 else -1
                    ),
                    "n_shared_st_genes": len(common_genes),
                }
            )

        strongest = np.unravel_index(
            np.nanargmax(abs_r),
            abs_r.shape,
        )

        i, j = strongest

        print(
            f"Patient {patient}: "
            f"{left_sample} vs {right_sample}"
        )
        print(
            f"  shared ST genes: {len(common_genes)}"
        )
        print(
            f"  strongest pair: "
            f"{left_names[i]} <-> {right_names[j]}  "
            f"Pearson r={pearson_r[i, j]:+.4f}  "
            f"historical r={historical_r[i, j]:+.4f}"
        )

    # --------------------------------------------------------------
    # Global output tables
    # --------------------------------------------------------------

    pairwise_df = pd.DataFrame(all_pairwise)
    left_df = pd.DataFrame(left_best_rows)
    right_df = pd.DataFrame(right_best_rows)
    mutual_df = pd.DataFrame(mutual_rows)
    hungarian_df = pd.DataFrame(hungarian_rows)

    pairwise_df.to_csv(
        args.output_root / "all_pairwise_correlations.tsv",
        sep="\t",
        index=False,
    )

    left_df.to_csv(
        args.output_root / "best_right_factor_per_left_factor.tsv",
        sep="\t",
        index=False,
    )

    right_df.to_csv(
        args.output_root / "best_left_factor_per_right_factor.tsv",
        sep="\t",
        index=False,
    )

    mutual_df = mutual_df.sort_values(
        ["patient", "abs_pearson_r"],
        ascending=[True, False],
    )

    mutual_df.to_csv(
        args.output_root / "mutual_best_matches.tsv",
        sep="\t",
        index=False,
    )

    hungarian_df.to_csv(
        args.output_root / "hungarian_matches.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("=== Mutual-best matches ===")
    print(
        mutual_df[
            [
                "patient",
                "left_factor",
                "right_factor",
                "pearson_r",
                "historical_r",
                "n_shared_st_genes",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Outputs: {args.output_root}")


if __name__ == "__main__":
    main()

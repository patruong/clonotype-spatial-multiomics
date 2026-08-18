#!/usr/bin/env python3

"""
Plot sign-aligned MOFA-FLEX factors and loadings for adjacent 9AA/DHBA sections.

Primary use:
    Figure 2 patient-5 matched axis
        bc2059 Free_03 (9AA reference)
        bc2004 Free_07 (DHBA aligned)

Logic:
    - left/9AA factor defines the orientation
    - if Pearson r < 0, multiply right/DHBA scores and loadings by -1
    - original MOFA-FLEX result files are never modified

Outputs for each selected match:
    - sign-aligned spatial factor-score pair on H&E
    - spatial score tables
    - top sign-aligned loadings for ST, MSI and VDJ
    - loading bar plots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIEWS = ("ST", "MSI", "VDJ")


def sanitize(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def capture_from_sample(sample):
    if "_" not in sample:
        raise RuntimeError(f"Cannot extract capture area from {sample}")
    return sample.split("_", 1)[1]


def orient_w(W, n_features, n_factors):
    W = np.asarray(W, dtype=float)

    if W.shape == (n_features, n_factors):
        return W

    if W.shape == (n_factors, n_features):
        return W.T

    raise RuntimeError(
        f"Cannot orient W shape={W.shape}; "
        f"n_features={n_features}, n_factors={n_factors}"
    )


def load_result(path):
    d = np.load(path, allow_pickle=True)

    Z = np.asarray(d["Z"], dtype=float)
    samples = np.asarray(d["samples"], dtype=str)

    n_factors = Z.shape[1]

    factor_names = np.asarray(
        d["factor_names"]
        if "factor_names" in d.files
        else [f"F{i + 1}" for i in range(n_factors)],
        dtype=str,
    )

    views = np.asarray(
        d["view_names"]
        if "view_names" in d.files
        else [],
        dtype=str,
    )

    W_map = {}
    feature_map = {}

    for view in views:
        wk = f"W_{view}"
        fk = f"features_{view}"

        if wk not in d.files or fk not in d.files:
            continue

        features = np.asarray(d[fk], dtype=str)

        W_map[view] = orient_w(
            d[wk],
            n_features=len(features),
            n_factors=n_factors,
        )

        feature_map[view] = features

    return {
        "Z": Z,
        "samples": samples,
        "factor_names": factor_names,
        "views": views,
        "W": W_map,
        "features": feature_map,
    }


def factor_index(result, factor):
    names = list(result["factor_names"])

    if factor not in names:
        raise RuntimeError(
            f"{factor} not found. Available factors: {names}"
        )

    return names.index(factor)


def model_barcodes(samples):
    return np.asarray(
        [
            x[3:] if x.startswith("ST_") else x
            for x in samples
        ],
        dtype=str,
    )


def read_tissue_positions(path):
    df = pd.read_csv(path)

    if "barcode" not in df.columns:
        df = pd.read_csv(
            path,
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

    return df.set_index("barcode")


def spatial_context(spaceranger_root, sample, model_ids):
    capture = capture_from_sample(sample)

    spatial = spaceranger_root / capture / "spatial"

    positions_path = spatial / "tissue_positions.csv"
    scale_path = spatial / "scalefactors_json.json"
    image_path = spatial / "tissue_hires_image.png"

    for path in (positions_path, scale_path, image_path):
        if not path.exists():
            raise FileNotFoundError(path)

    positions = read_tissue_positions(positions_path)

    missing = [
        x for x in model_ids
        if x not in positions.index
    ]

    if missing:
        raise RuntimeError(
            f"{sample}: {len(missing)} model barcodes absent from "
            f"tissue_positions.csv; first examples: {missing[:5]}"
        )

    positions = positions.reindex(model_ids)

    with scale_path.open() as handle:
        scale = json.load(handle)

    hires_scale = float(scale["tissue_hires_scalef"])
    spot_fullres = float(
        scale.get("spot_diameter_fullres", 100.0)
    )

    x = pd.to_numeric(
        positions["pxl_col_in_fullres"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        positions["pxl_row_in_fullres"],
        errors="coerce",
    ).to_numpy(dtype=float)

    coords = np.column_stack(
        [x * hires_scale, y * hires_scale]
    )

    if not np.isfinite(coords).all():
        raise RuntimeError(
            f"{sample}: non-finite spatial coordinates"
        )

    return {
        "image": plt.imread(image_path),
        "coords": coords,
        "spot_diameter": spot_fullres * hires_scale,
    }


def symmetric_limit(values, percentile=99.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 1.0

    vmax = float(
        np.percentile(np.abs(values), percentile)
    )

    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0

    return vmax


def plot_spatial(ax, context, values, title, vmax):
    image = context["image"]
    coords = context["coords"]

    H, W = image.shape[:2]

    # Approximate spot display size from Space Ranger scale factors.
    s = (context["spot_diameter"] ** 2) * 1.6e-2

    ax.imshow(image, origin="upper")

    sca = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=s,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="none",
        alpha=0.98,
    )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)

    return sca


def write_spatial_pair(
    out_base,
    left_context,
    right_context,
    left_values,
    right_values,
    left_title,
    right_title,
    title,
    shared_scale=False,
):
    if shared_scale:
        vmax = symmetric_limit(
            np.concatenate(
                [left_values, right_values]
            )
        )
        vmax_left = vmax
        vmax_right = vmax
    else:
        vmax_left = symmetric_limit(left_values)
        vmax_right = symmetric_limit(right_values)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.5),
    )

    left_scatter = plot_spatial(
        axes[0],
        left_context,
        left_values,
        left_title,
        vmax_left,
    )

    right_scatter = plot_spatial(
        axes[1],
        right_context,
        right_values,
        right_title,
        vmax_right,
    )

    fig.colorbar(
        left_scatter,
        ax=axes[0],
        fraction=0.046,
        pad=0.02,
        label="factor score",
    )

    fig.colorbar(
        right_scatter,
        ax=axes[1],
        fraction=0.046,
        pad=0.02,
        label="sign-aligned factor score",
    )

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def top_loadings(
    result,
    view,
    factor_idx,
    sign,
    top_n,
    side,
    sample,
    factor,
):
    if view not in result["W"]:
        return []

    W = result["W"][view]
    features = result["features"][view]

    raw = np.asarray(
        W[:, factor_idx],
        dtype=float,
    )

    aligned = raw * float(sign)

    order = np.argsort(
        -np.abs(aligned)
    )[:top_n]

    rows = []

    for rank, j in enumerate(order, start=1):
        rows.append(
            {
                "side": side,
                "sample": sample,
                "view": view,
                "factor": factor,
                "factor_idx": factor_idx,
                "sign_applied": int(sign),
                "rank": rank,
                "feature": str(features[j]),
                "raw_loading": float(raw[j]),
                "aligned_loading": float(aligned[j]),
                "abs_aligned_loading": float(
                    abs(aligned[j])
                ),
                "direction_after_alignment": (
                    "positive"
                    if aligned[j] >= 0
                    else "negative"
                ),
            }
        )

    return rows


def plot_loading_rows(rows, out_base, title):
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(
        "aligned_loading",
        ascending=True,
    )

    values = df["aligned_loading"].to_numpy()
    labels = df["feature"].astype(str).tolist()

    fig_h = max(
        3.2,
        0.22 * len(df) + 1.2,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, fig_h)
    )

    y = np.arange(len(df))

    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("sign-aligned loading")
    ax.set_title(title, fontsize=9)

    fig.tight_layout()

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
    )

    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            repo
            / "results"
            / "integration"
            / "mofaflex"
            / "f12"
        ),
    )

    parser.add_argument(
        "--match-table",
        type=Path,
        default=(
            repo
            / "results"
            / "integration"
            / "mofaflex"
            / "f12"
            / "adjacent_factor_matching"
            / "mutual_best_matches.tsv"
        ),
    )

    parser.add_argument(
        "--spaceranger-root",
        type=Path,
        default=(
            repo
            / "data"
            / "processed"
            / "st"
            / "space_ranger_outs"
        ),
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
            / "sign_aligned_factors"
        ),
    )

    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--left-factors",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--top-loadings",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--views",
        nargs="+",
        default=list(VIEWS),
    )

    parser.add_argument(
        "--shared-pair-scale",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = pd.read_csv(
        args.match_table,
        sep="\t",
    )

    required = {
        "patient",
        "left_sample",
        "right_sample",
        "left_factor",
        "right_factor",
        "pearson_r",
        "abs_pearson_r",
    }

    missing = required - set(matches.columns)

    if missing:
        raise RuntimeError(
            f"Match table missing columns: {sorted(missing)}"
        )

    matches = matches[
        matches["abs_pearson_r"]
        >= args.min_abs_corr
    ].copy()

    if args.patients:
        wanted = {
            str(x)
            for x in args.patients
        }

        matches = matches[
            matches["patient"]
            .astype(str)
            .isin(wanted)
        ].copy()

    if args.left_factors:
        wanted = set(args.left_factors)

        matches = matches[
            matches["left_factor"]
            .astype(str)
            .isin(wanted)
        ].copy()

    if matches.empty:
        raise RuntimeError(
            "No matches remain after filtering."
        )

    index_rows = []
    all_loading_rows = []

    cache = {}

    def get_result(sample):
        if sample not in cache:
            path = (
                args.results_root
                / sample
                / "mofaflex_results.npz"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            cache[sample] = load_result(path)

        return cache[sample]

    for _, row in matches.iterrows():

        patient = row["patient"]

        left_sample = str(row["left_sample"])
        right_sample = str(row["right_sample"])

        left_factor = str(row["left_factor"])
        right_factor = str(row["right_factor"])

        corr = float(row["pearson_r"])
        abs_corr = float(row["abs_pearson_r"])

        # 9AA/left is always the reference orientation.
        right_sign = -1 if corr < 0 else 1

        left = get_result(left_sample)
        right = get_result(right_sample)

        left_idx = factor_index(
            left,
            left_factor,
        )

        right_idx = factor_index(
            right,
            right_factor,
        )

        left_ids = model_barcodes(
            left["samples"]
        )

        right_ids = model_barcodes(
            right["samples"]
        )

        left_context = spatial_context(
            args.spaceranger_root,
            left_sample,
            left_ids,
        )

        right_context = spatial_context(
            args.spaceranger_root,
            right_sample,
            right_ids,
        )

        left_scores = np.asarray(
            left["Z"][:, left_idx],
            dtype=float,
        )

        right_scores_raw = np.asarray(
            right["Z"][:, right_idx],
            dtype=float,
        )

        right_scores_aligned = (
            right_scores_raw * right_sign
        )

        pair_name = (
            f"patient{patient}_"
            f"{left_sample}_{left_factor}"
            f"__vs__"
            f"{right_sample}_{right_factor}"
        )

        out = (
            args.output_root
            / sanitize(pair_name)
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        flip_label = (
            "right flipped"
            if right_sign == -1
            else "no flip"
        )

        write_spatial_pair(
            out / "sign_aligned_spatial_pair",
            left_context,
            right_context,
            left_scores,
            right_scores_aligned,
            f"{left_sample}\n{left_factor} / 9AA reference",
            f"{right_sample}\n{right_factor} × {right_sign}",
            (
                f"Patient {patient} | "
                f"r={corr:+.3f} | {flip_label}"
            ),
            shared_scale=args.shared_pair_scale,
        )

        pd.DataFrame(
            {
                "barcode": left_ids,
                "sample": left_sample,
                "factor": left_factor,
                "raw_factor_score": left_scores,
                "aligned_factor_score": left_scores,
                "sign_applied": 1,
            }
        ).to_csv(
            out / "left_reference_spatial_scores.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(
            {
                "barcode": right_ids,
                "sample": right_sample,
                "factor": right_factor,
                "raw_factor_score": right_scores_raw,
                "aligned_factor_score": right_scores_aligned,
                "sign_applied": right_sign,
            }
        ).to_csv(
            out / "right_sign_aligned_spatial_scores.tsv",
            sep="\t",
            index=False,
        )

        local_rows = []

        for view in args.views:

            left_rows = top_loadings(
                left,
                view,
                left_idx,
                sign=1,
                top_n=args.top_loadings,
                side="left_reference",
                sample=left_sample,
                factor=left_factor,
            )

            right_rows = top_loadings(
                right,
                view,
                right_idx,
                sign=right_sign,
                top_n=args.top_loadings,
                side="right_aligned",
                sample=right_sample,
                factor=right_factor,
            )

            local_rows.extend(left_rows)
            local_rows.extend(right_rows)

            if left_rows:
                plot_loading_rows(
                    left_rows,
                    out
                    / f"left_reference_{view}_top_loadings",
                    (
                        f"{left_sample} {left_factor} "
                        f"| {view}"
                    ),
                )

            if right_rows:
                plot_loading_rows(
                    right_rows,
                    out
                    / f"right_aligned_{view}_top_loadings",
                    (
                        f"{right_sample} {right_factor} "
                        f"| {view} | sign × {right_sign}"
                    ),
                )

        if local_rows:
            local_df = pd.DataFrame(
                local_rows
            )

            local_df.insert(
                0,
                "patient",
                patient,
            )

            local_df.insert(
                1,
                "pearson_r",
                corr,
            )

            local_df.to_csv(
                out / "sign_aligned_top_loadings.tsv",
                sep="\t",
                index=False,
            )

            all_loading_rows.extend(
                local_df.to_dict("records")
            )

        index_rows.append(
            {
                "patient": patient,
                "left_sample": left_sample,
                "right_sample": right_sample,
                "left_factor": left_factor,
                "right_factor": right_factor,
                "pearson_r": corr,
                "abs_pearson_r": abs_corr,
                "right_sign_applied": right_sign,
                "output_dir": str(out),
            }
        )

        print(
            f"[OK] patient {patient}: "
            f"{left_sample} {left_factor} <-> "
            f"{right_sample} {right_factor} "
            f"r={corr:+.4f}, "
            f"right sign={right_sign:+d}"
        )

    pd.DataFrame(
        index_rows
    ).to_csv(
        args.output_root / "plot_index.tsv",
        sep="\t",
        index=False,
    )

    if all_loading_rows:
        pd.DataFrame(
            all_loading_rows
        ).to_csv(
            args.output_root
            / "all_sign_aligned_top_loadings.tsv",
            sep="\t",
            index=False,
        )

    print()
    print(
        f"Outputs: {args.output_root}"
    )


if __name__ == "__main__":
    main()

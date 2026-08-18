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



def flip_spatial_context_horizontal(context):
    """
    Flip hires image and spatial coordinates horizontally.
    """
    image = context["image"]
    coords = np.asarray(
        context["coords"],
        dtype=float,
    ).copy()

    _, W = image.shape[:2]

    image_flip = np.fliplr(image)

    coords[:, 0] = (W - 1) - coords[:, 0]

    return {
        "image": image_flip,
        "coords": coords,
        "spot_diameter": context["spot_diameter"],
    }


def rotate_spatial_context(context, direction="cw"):
    """
    Rotate hires image and spatial coordinates together by 90 degrees.

    Coordinates use image convention:
        x = column
        y = row
    """

    if direction == "none":
        return context

    image = context["image"]
    coords = np.asarray(
        context["coords"],
        dtype=float,
    ).copy()

    H, W = image.shape[:2]

    x = coords[:, 0]
    y = coords[:, 1]

    if direction == "cw":
        # np.rot90(..., k=3)
        image_rot = np.rot90(image, k=3)

        x_rot = (H - 1) - y
        y_rot = x

    elif direction == "ccw":
        # np.rot90(..., k=1)
        image_rot = np.rot90(image, k=1)

        x_rot = y
        y_rot = (W - 1) - x

    elif direction == "180":
        # 180-degree rotation; CW and CCW are equivalent.
        image_rot = np.rot90(image, k=2)

        x_rot = (W - 1) - x
        y_rot = (H - 1) - y

    else:
        raise ValueError(
            "direction must be 'cw', 'ccw', '180', or 'none'"
        )

    return {
        "image": image_rot,
        "coords": np.column_stack(
            [x_rot, y_rot]
        ),
        "spot_diameter": context["spot_diameter"],
    }


def plot_spatial(
    ax,
    context,
    values,
    title,
    vmax,
    crop_padding=0.03,
    spot_size_scale=0.035,
):
    image = context["image"]
    coords = np.asarray(
        context["coords"],
        dtype=float,
    )

    # Approximate displayed spot size from Space Ranger scale factors.
    s = (context["spot_diameter"] ** 2) * spot_size_scale

    ax.imshow(
        image,
        origin="upper",
    )

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

    # ------------------------------------------------------------
    # Tight tissue crop
    # ------------------------------------------------------------

    xmin = float(np.nanmin(coords[:, 0]))
    xmax = float(np.nanmax(coords[:, 0]))
    ymin = float(np.nanmin(coords[:, 1]))
    ymax = float(np.nanmax(coords[:, 1]))

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    # Small fractional margin, but never less than roughly one spot.
    xpad = max(
        dx * crop_padding,
        context["spot_diameter"],
    )

    ypad = max(
        dy * crop_padding,
        context["spot_diameter"],
    )

    ax.set_xlim(
        xmin - xpad,
        xmax + xpad,
    )

    # Image coordinates have y increasing downward.
    ax.set_ylim(
        ymax + ypad,
        ymin - ypad,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove visible frame for Illustrator-ready panel.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        title,
        fontsize=9,
    )

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
    right_rotation="cw",
    right_flip_horizontal=False,
    crop_padding=0.03,
    spot_size_scale=0.035,
):
    right_context_plot = right_context

    if right_flip_horizontal:
        right_context_plot = flip_spatial_context_horizontal(
            right_context_plot
        )

    right_context_plot = rotate_spatial_context(
        right_context_plot,
        direction=right_rotation,
    )

    if shared_scale:
        vmax = symmetric_limit(
            np.concatenate(
                [left_values, right_values]
            )
        )
        vmax_left = vmax
        vmax_right = vmax

    else:
        vmax_left = symmetric_limit(
            left_values
        )

        vmax_right = symmetric_limit(
            right_values
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.3),
    )

    left_scatter = plot_spatial(
        axes[0],
        left_context,
        left_values,
        left_title,
        vmax_left,
        crop_padding=crop_padding,
        spot_size_scale=spot_size_scale,
    )

    right_scatter = plot_spatial(
        axes[1],
        right_context_plot,
        right_values,
        right_title,
        vmax_right,
        crop_padding=crop_padding,
        spot_size_scale=spot_size_scale,
    )

    fig.colorbar(
        left_scatter,
        ax=axes[0],
        fraction=0.035,
        pad=0.015,
        label="factor score",
    )

    fig.colorbar(
        right_scatter,
        ax=axes[1],
        fraction=0.035,
        pad=0.015,
        label="sign-aligned factor score",
    )

    fig.suptitle(
        title,
        fontsize=10,
        y=0.985,
    )

    # Keep adjacent sections close together.
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.90,
        wspace=0.20,
    )

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.01,
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

    positive_idx = np.flatnonzero(
        aligned > 0
    )

    negative_idx = np.flatnonzero(
        aligned < 0
    )

    # Strongest positive loadings first.
    positive_idx = positive_idx[
        np.argsort(
            -aligned[positive_idx]
        )
    ][:top_n]

    # Most negative loadings first.
    negative_idx = negative_idx[
        np.argsort(
            aligned[negative_idx]
        )
    ][:top_n]

    rows = []

    for pole, indices in (
        ("positive", positive_idx),
        ("negative", negative_idx),
    ):
        for rank, j in enumerate(
            indices,
            start=1,
        ):
            rows.append(
                {
                    "side": side,
                    "sample": sample,
                    "view": view,
                    "factor": factor,
                    "factor_idx": factor_idx,
                    "sign_applied": int(sign),
                    "pole": pole,
                    "rank": rank,
                    "feature": str(features[j]),
                    "raw_loading": float(raw[j]),
                    "aligned_loading": float(aligned[j]),
                    "abs_aligned_loading": float(
                        abs(aligned[j])
                    ),
                    "direction_after_alignment": pole,
                }
            )

    return rows



def plot_loading_rows(
    rows,
    out_base,
    title,
):
    if not rows:
        return

    df = pd.DataFrame(rows)

    positive = (
        df[df["aligned_loading"] > 0]
        .sort_values(
            "aligned_loading",
            ascending=False,
        )
        .copy()
    )

    negative = (
        df[df["aligned_loading"] < 0]
        .sort_values(
            "aligned_loading",
            ascending=True,
        )
        .copy()
    )

    if positive.empty and negative.empty:
        return

    max_abs = max(
        positive["aligned_loading"].abs().max()
        if not positive.empty
        else 0.0,
        negative["aligned_loading"].abs().max()
        if not negative.empty
        else 0.0,
    )

    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    max_abs *= 1.06

    n_rows = max(
        len(positive),
        len(negative),
        1,
    )

    fig_h = max(
        3.2,
        0.31 * n_rows + 1.45,
    )

    fig, (ax_pos, ax_neg) = plt.subplots(
        1,
        2,
        figsize=(5.0, fig_h),
        gridspec_kw={
            "wspace": 0.18,
        },
    )

    # ------------------------------------------------------------
    # Positive pole: LEFT
    # ------------------------------------------------------------

    if not positive.empty:
        y = np.arange(
            len(positive)
        )

        ax_pos.barh(
            y,
            positive["aligned_loading"],
            color="#2f62ad",
            edgecolor="none",
        )

        ax_pos.set_yticks(y)

        ax_pos.set_yticklabels(
            positive["feature"].astype(str),
            fontsize=8,
        )

        ax_pos.invert_yaxis()

    ax_pos.set_xlim(
        0,
        max_abs,
    )

    ax_pos.set_title(
        "Positive pole",
        fontsize=9,
        fontweight="bold",
        color="#2f62ad",
    )

    ax_pos.set_xlabel(
        "Loading",
        fontsize=9,
    )

    # ------------------------------------------------------------
    # Negative pole: RIGHT
    # ------------------------------------------------------------

    if not negative.empty:
        y = np.arange(
            len(negative)
        )

        ax_neg.barh(
            y,
            negative["aligned_loading"],
            color="#e53935",
            edgecolor="none",
        )

        ax_neg.set_yticks(y)

        ax_neg.set_yticklabels(
            negative["feature"].astype(str),
            fontsize=8,
        )

        ax_neg.invert_yaxis()

    ax_neg.set_xlim(
        -max_abs,
        0,
    )

    ax_neg.yaxis.tick_right()
    ax_neg.yaxis.set_label_position(
        "right"
    )

    ax_neg.set_title(
        "Negative pole",
        fontsize=9,
        fontweight="bold",
        color="#e53935",
    )

    ax_neg.set_xlabel(
        "Loading",
        fontsize=9,
    )

    # Clean manuscript-style axes.
    for ax in (ax_pos, ax_neg):
        ax.spines["top"].set_visible(False)

        ax.tick_params(
            axis="x",
            labelsize=8,
        )

        ax.tick_params(
            axis="y",
            length=0,
        )

    ax_pos.spines["right"].set_visible(
        False
    )

    ax_neg.spines["left"].set_visible(
        False
    )

    fig.suptitle(
        title,
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.84,
        bottom=0.10,
        top=0.88,
        wspace=0.18,
    )

    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
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

    parser.add_argument(
        "--right-rotation",
        choices=["cw", "ccw", "180", "none"],
        default="cw",
        help=(
            "Rotate the right/DHBA tissue image and coordinates "
            "before plotting. Default: 90 degrees clockwise."
        ),
    )

    parser.add_argument(
        "--right-flip-horizontal",
        action="store_true",
        help=(
            "Horizontally flip the right-section image and "
            "coordinates before applying right-section rotation."
        ),
    )

    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.03,
        help=(
            "Fractional padding around retained spatial spots. "
            "Default: 0.03."
        ),
    )

    parser.add_argument(
        "--spot-size-scale",
        type=float,
        default=0.035,
        help=(
            "Matplotlib spot area scale relative to Space Ranger "
            "spot diameter squared. Default: 0.035."
        ),
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
            right_rotation=args.right_rotation,
            right_flip_horizontal=args.right_flip_horizontal,
            crop_padding=args.crop_padding,
            spot_size_scale=args.spot_size_scale,
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

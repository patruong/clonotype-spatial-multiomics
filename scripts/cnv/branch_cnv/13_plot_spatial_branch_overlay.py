#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]

BCR = (
    ROOT / "data/processed/svdj/figure3/"
    "IGH_clone9953_historical14280_unique_spot_classes.tsv"
)

PREP = ROOT / "data/processed/cnv/infercnv/figure3/prepared"

NLS = ROOT / "data/processed/st/nlsdeconv"

OUT = (
    ROOT / "results/cnv/figure3/branch_cnv_seed0/"
    "spatial_branch_overlay"
)
OUT.mkdir(parents=True, exist_ok=True)


SECTIONS = {
    "bc2043": {
        "sid": "V13Y10-061_D1",
        "prepared": "bc2043_V13Y10-061_D1_synthref",
        "rotate_cw": True,
    },
    "bc2020": {
        "sid": "V13Y10-060_D1",
        "prepared": "bc2020_V13Y10-060_D1_synthref",
        "rotate_cw": False,
    },
    "bc2075": {
        "sid": "V13Y10-038_D1",
        "prepared": "bc2075_V13Y10-038_D1_synthref",
        "rotate_cw": False,
    },
}

BLUE = "#377EB8"
RED = "#E41A1C"
MIXED = "#984EA3"
BACKGROUND = "#BDBDBD"


def norm_barcode(values):
    s = pd.Series(values, dtype="string")
    return s.str.extract(r"([ACGT]{16}-1)", expand=False)


def classify_branch(row):
    blue = float(row["raw_A"]) > 0
    red = float(row["raw_B"]) > 0

    if blue and red:
        return "Both"
    if blue:
        return "Blue"
    if red:
        return "Red"
    return "Other"


def load_spatial_h5ad(path):
    a = ad.read_h5ad(path)

    if "spatial" not in a.obsm:
        raise RuntimeError(f"{path}: obsm['spatial'] missing")

    if "spatial" not in a.uns:
        raise RuntimeError(f"{path}: uns['spatial'] missing")

    libs = list(a.uns["spatial"].keys())

    if len(libs) != 1:
        raise RuntimeError(
            f"{path}: expected one spatial library, found {libs}"
        )

    lib = libs[0]
    info = a.uns["spatial"][lib]

    images = info.get("images", {})
    scalefactors = info.get("scalefactors", {})

    if "hires" not in images:
        raise RuntimeError(f"{path}: hires H&E image missing")

    if "tissue_hires_scalef" not in scalefactors:
        raise RuntimeError(
            f"{path}: tissue_hires_scalef missing"
        )

    image = np.asarray(images["hires"])
    scale = float(scalefactors["tissue_hires_scalef"])

    coords = np.asarray(a.obsm["spatial"], dtype=float) * scale

    coord_df = pd.DataFrame(
        {
            "barcode": norm_barcode(a.obs_names).to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )

    return a, image, coord_df


def rotate_clockwise(image, coords):
    """
    Rotate image and coordinates 90 degrees clockwise.

    Input coordinates are in imshow image coordinates:
      x = horizontal
      y = vertical
    """
    h, w = image.shape[:2]

    image2 = np.rot90(image, k=-1)

    c = coords.copy()

    x_old = c["x"].to_numpy()
    y_old = c["y"].to_numpy()

    c["x"] = (h - 1) - y_old
    c["y"] = x_old

    return image2, c


def crop_limits(coords, image, margin_frac=0.04):
    xmin = float(coords["x"].min())
    xmax = float(coords["x"].max())
    ymin = float(coords["y"].min())
    ymax = float(coords["y"].max())

    dx = max(xmax - xmin, 1)
    dy = max(ymax - ymin, 1)

    mx = dx * margin_frac
    my = dy * margin_frac

    h, w = image.shape[:2]

    return (
        max(0, xmin - mx),
        min(w - 1, xmax + mx),
        max(0, ymin - my),
        min(h - 1, ymax + my),
    )


# ------------------------------------------------------------------
# BCR table
# ------------------------------------------------------------------
bcr = pd.read_csv(BCR, sep="\t")

bcr["barcode"] = norm_barcode(bcr["barcode"])

bcr["branch_plot"] = bcr.apply(
    classify_branch,
    axis=1,
)

bcr = bcr[
    bcr["branch_plot"].isin(["Blue", "Red", "Both"])
].copy()


summary_rows = []
panel_data = {}


# ------------------------------------------------------------------
# Build exact final-CNV branch overlays
# ------------------------------------------------------------------
for section, info in SECTIONS.items():

    sid = info["sid"]
    prepared = info["prepared"]

    spatial_table = (
        PREP
        / prepared
        / "infercnv_R_out_noHMM"
        / "leiden_posthoc_res_0.70"
        / "infercnv_R_leiden_spatial_table.tsv"
    )

    h5ad_path = NLS / f"{sid}.nlsdeconv.h5ad"

    if not spatial_table.exists():
        raise FileNotFoundError(spatial_table)

    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    cnv = pd.read_csv(spatial_table, sep="\t")
    cnv["barcode"] = norm_barcode(cnv["barcode"])

    if cnv["barcode"].duplicated().any():
        raise RuntimeError(
            f"{section}: duplicate barcode in final CNV table"
        )

    x = bcr[bcr["section"] == section].copy()

    joined = x.merge(
        cnv[["barcode", "cnv_leiden"]],
        on="barcode",
        how="inner",
        validate="one_to_one",
    )

    a, image, coords = load_spatial_h5ad(h5ad_path)

    # Background = exactly all spots in final CNV analysis.
    bg = cnv[["barcode"]].merge(
        coords,
        on="barcode",
        how="left",
        validate="one_to_one",
    )

    if bg[["x", "y"]].isna().any().any():
        raise RuntimeError(
            f"{section}: final CNV spots missing H&E coordinates"
        )

    joined = joined.merge(
        coords,
        on="barcode",
        how="left",
        validate="one_to_one",
    )

    if joined[["x", "y"]].isna().any().any():
        raise RuntimeError(
            f"{section}: branch spots missing H&E coordinates"
        )

    if info["rotate_cw"]:
        image, bg = rotate_clockwise(image, bg)

        _, joined = rotate_clockwise(
            np.empty((1, 1)),
            joined,
        )

        # Above dummy-image rotation cannot be used because dimensions
        # differ. Recompute using original hires height.
        _, original_image, original_coords = load_spatial_h5ad(h5ad_path)
        h = original_image.shape[0]

        joined = joined.copy()
        x_old = joined["x"].to_numpy()
        y_old = joined["y"].to_numpy()

        joined["x"] = (h - 1) - y_old
        joined["y"] = x_old

        a.file.close() if getattr(a, "file", None) else None

    crop = crop_limits(bg, image)

    counts = (
        joined["branch_plot"]
        .value_counts()
        .reindex(["Blue", "Red", "Both"], fill_value=0)
    )

    print(
        f"[{section}] final-CNV branch spots: "
        f"Blue={counts['Blue']} "
        f"Red={counts['Red']} "
        f"Both={counts['Both']}"
    )

    summary_rows.append(
        {
            "section": section,
            "Blue": int(counts["Blue"]),
            "Red": int(counts["Red"]),
            "Both": int(counts["Both"]),
            "total_branch_spots": int(len(joined)),
            "final_cnv_spots": int(len(cnv)),
        }
    )

    joined[
        [
            "section",
            "barcode",
            "branch_plot",
            "cnv_leiden",
            "x",
            "y",
        ]
    ].to_csv(
        OUT / f"{section}_branch_overlay_spots.tsv",
        sep="\t",
        index=False,
    )

    panel_data[section] = {
        "image": image,
        "background": bg,
        "branches": joined,
        "crop": crop,
        "counts": counts,
    }


# ------------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------------
def draw_panel(ax, section, dat, title=True):

    image = dat["image"]
    bg = dat["background"]
    z = dat["branches"]

    ax.imshow(image)

    # all CNV-assigned spots
    ax.scatter(
        bg["x"],
        bg["y"],
        s=5,
        c=BACKGROUND,
        alpha=0.22,
        linewidths=0,
        zorder=2,
    )

    # focal branch spots
    for branch, color in [
        ("Red", RED),
        ("Blue", BLUE),
        ("Both", MIXED),
    ]:
        q = z[z["branch_plot"] == branch]

        if len(q) == 0:
            continue

        ax.scatter(
            q["x"],
            q["y"],
            s=24,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            zorder=5,
        )

    xmin, xmax, ymin, ymax = dat["crop"]

    ax.set_xlim(xmin, xmax)

    # imshow uses an inverted visual y axis
    ax.set_ylim(ymax, ymin)

    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        c = dat["counts"]

        ax.set_title(
            f"{section}\n"
            f"Blue={int(c['Blue'])}, "
            f"Red={int(c['Red'])}, "
            f"Mixed={int(c['Both'])}",
            fontsize=10,
        )


# ------------------------------------------------------------------
# Individual panels
# ------------------------------------------------------------------
for section, dat in panel_data.items():

    fig, ax = plt.subplots(figsize=(6, 6))

    draw_panel(
        ax,
        section,
        dat,
    )

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor="white",
                markersize=8,
                label="Blue branch",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=RED,
                markeredgecolor="white",
                markersize=8,
                label="Red branch",
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=MIXED,
                markeredgecolor="white",
                markersize=8,
                label="Mixed",
            ),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.05, top=0.82, wspace=0.18)

    base = OUT / f"{section}_bcr_branch_spatial_overlay"

    fig.savefig(
        base.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ------------------------------------------------------------------
# Three-section contact sheet
# ------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5.3),
)

for ax, section in zip(
    axes,
    ["bc2043", "bc2020", "bc2075"],
):
    draw_panel(
        ax,
        section,
        panel_data[section],
    )

fig.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="", label="Blue branch",
               markerfacecolor="#377eb8", markeredgecolor="white", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", label="Red branch",
               markerfacecolor="#e41a1c", markeredgecolor="white", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", label="Mixed Blue+Red",
               markerfacecolor="#984ea3", markeredgecolor="white", markersize=8),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    frameon=False,
    fontsize=12,
    handletextpad=0.4,
    columnspacing=1.0,
)

fig.suptitle(
    "Spatial localization of focal BCR branches",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.91]
)

base = OUT / "bcr_branch_spatial_overlay_all_sections"

fig.savefig(
    base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)


pd.DataFrame(summary_rows).to_csv(
    OUT / "branch_spatial_overlay_summary.tsv",
    sep="\t",
    index=False,
)

print()
print("[DONE]", OUT)
print(
    pd.DataFrame(summary_rows)
    .to_string(index=False)
)

#!/usr/bin/env python3

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

NMF_ROOT = ROOT / "data" / "intermediate" / "msi" / "nmf"
SELECTED_ROOT = ROOT / "results" / "msi" / "nmf_selected"

TOP_N = 15


def read_selection(path):
    text = path.read_text()

    k = int(
        re.search(r"best_K:\s*(\d+)", text).group(1)
    )

    m = re.search(
        r"selected_factors:\s*R=(NMF\d+),\s*G=(NMF\d+),\s*B=(NMF\d+)",
        text,
    )

    if m is None:
        raise ValueError(f"Could not parse selected factors from {path}")

    return k, list(m.groups())


def savefig3(fig, stem):
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
    plt.close(fig)


def plot_one_factor(H, factor, stem, top_n=TOP_N):
    values = H.loc[factor].astype(float)

    idx = values.nlargest(top_n).index
    vals = values.loc[idx]

    # Highest loading at top
    idx = idx[::-1]
    vals = vals[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    ax.barh(
        np.arange(len(idx)),
        vals.values,
    )

    ax.set_yticks(
        np.arange(len(idx))
    )

    ax.set_yticklabels(
        idx,
        fontsize=11,
    )

    ax.tick_params(
        axis="x",
        labelsize=11,
    )

    ax.set_xlabel(
        "NMF loading",
        fontsize=13,
    )

    ax.set_title(
        factor,
        fontsize=16,
        fontweight="bold",
        loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    savefig3(fig, stem)


def plot_selected_panel(H, selected, stem, top_n=TOP_N):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.8),
    )

    channel_names = ["R", "G", "B"]

    for ax, channel, factor in zip(
        axes,
        channel_names,
        selected,
    ):
        values = H.loc[factor].astype(float)

        idx = values.nlargest(top_n).index
        vals = values.loc[idx]

        idx = idx[::-1]
        vals = vals[::-1]

        ax.barh(
            np.arange(len(idx)),
            vals.values,
        )

        ax.set_yticks(
            np.arange(len(idx))
        )

        ax.set_yticklabels(
            idx,
            fontsize=10,
        )

        ax.tick_params(
            axis="x",
            labelsize=10,
        )

        ax.set_xlabel(
            "NMF loading",
            fontsize=11,
        )

        ax.set_title(
            f"{channel}: {factor}",
            fontsize=14,
            fontweight="bold",
            loc="left",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    savefig3(fig, stem)


def main():
    selection_files = sorted(
        SELECTED_ROOT.glob("*/selected_factors.txt")
    )

    if not selection_files:
        raise FileNotFoundError(
            f"No selected_factors.txt files under {SELECTED_ROOT}"
        )

    for selection_file in selection_files:
        sample = selection_file.parent.name

        k, selected = read_selection(
            selection_file
        )

        h_path = (
            NMF_ROOT
            / sample
            / f"k{k}"
            / "H.tsv.gz"
        )

        if not h_path.exists():
            print(
                f"[WARN] {sample}: missing {h_path}"
            )
            continue

        H = pd.read_csv(
            h_path,
            sep="\t",
            index_col=0,
        )

        missing = [
            f for f in selected
            if f not in H.index
        ]

        if missing:
            print(
                f"[WARN] {sample}: missing factors {missing}"
            )
            continue

        out = selection_file.parent

        for factor in selected:
            plot_one_factor(
                H,
                factor,
                out / f"{factor}_top_loadings",
            )

        plot_selected_panel(
            H,
            selected,
            out / "selected_top_loadings",
        )

        print(
            f"[OK] {sample}: "
            f"K={k}; "
            + ", ".join(selected)
        )


if __name__ == "__main__":
    main()

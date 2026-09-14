#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

INDIR = ROOT / "results/cnv/figure3/domain_matching_seed0"
PAIRWISE = INDIR / "all_pairwise_similarity.tsv"
MAPPING = INDIR / "cnv_family_mapping_seed0.tsv"

OUT_PNG = INDIR / "cnv_domain_similarity_heatmap_ordered.png"
OUT_PDF = INDIR / "cnv_domain_similarity_heatmap_ordered.pdf"
OUT_MATRIX = INDIR / "cnv_domain_correlation_matrix_ordered.tsv"
OUT_ORDER = INDIR / "ordered_domain_labels.tsv"


# ---------------------------------------------------------------------
# Load exact correlations produced by 06_match_cnv_domains.py
# ---------------------------------------------------------------------
pairs = pd.read_csv(PAIRWISE, sep="\t")
mapping = pd.read_csv(MAPPING, sep="\t")

print("[INFO] pairwise columns:", list(pairs.columns))
print("[INFO] mapping columns:", list(mapping.columns))


def find_col(df, candidates):
    for x in candidates:
        if x in df.columns:
            return x
    raise RuntimeError(
        f"Could not find any of {candidates} in columns: {list(df.columns)}"
    )


A = find_col(
    pairs,
    ["domain1", "domain_1", "domain_a", "domain_i", "row_domain", "group1"]
)

B = find_col(
    pairs,
    ["domain2", "domain_2", "domain_b", "domain_j", "col_domain", "group2"]
)

R = find_col(
    pairs,
    ["pearson_r", "pearson", "correlation", "corr", "similarity"]
)


# ---------------------------------------------------------------------
# Domain ordering from the curated mapping
# ---------------------------------------------------------------------
mapping["cluster_label"] = (
    pd.to_numeric(mapping["cluster_label"], errors="raise")
    .astype(int)
)

mapping["domain"] = mapping["domain"].astype(str)

family_rank = {
    "CNV-A": 0,
    "CNV-A-like": 1,
    "CNV-B": 2,
    "CNV-C": 3,
    "CNV-C/other": 4,
}

section_rank = {
    "bc2043": 0,
    "bc2020": 1,
    "bc2075": 2,
}

mapping["family_rank"] = mapping["cnv_family"].map(family_rank)
mapping["section_rank"] = mapping["section"].map(section_rank)

if mapping["family_rank"].isna().any():
    raise RuntimeError(
        "Unknown CNV family: "
        + str(mapping.loc[mapping["family_rank"].isna(), "cnv_family"].unique())
    )

mapping = mapping.sort_values(
    ["family_rank", "section_rank", "cluster_label"]
).reset_index(drop=True)

order = mapping["domain"].tolist()

print("\n[ORDER]")
for _, row in mapping.iterrows():
    print(f'{row["cnv_family"]:12s}  {row["domain"]}')


# ---------------------------------------------------------------------
# Reconstruct exact symmetric correlation matrix
# ---------------------------------------------------------------------
domains = sorted(set(pairs[A].astype(str)) | set(pairs[B].astype(str)))

matrix = pd.DataFrame(
    np.nan,
    index=domains,
    columns=domains,
    dtype=float,
)

for d in domains:
    matrix.loc[d, d] = 1.0

for _, row in pairs.iterrows():
    a = str(row[A])
    b = str(row[B])
    r = float(row[R])

    matrix.loc[a, b] = r
    matrix.loc[b, a] = r


missing_domains = [x for x in order if x not in matrix.index]
if missing_domains:
    raise RuntimeError(
        f"Mapped domains missing from exact pairwise table: {missing_domains}"
    )

ordered = matrix.loc[order, order]

if ordered.isna().any().any():
    where = np.argwhere(ordered.isna().to_numpy())
    raise RuntimeError(
        f"Missing pairwise correlations in reconstructed matrix: {len(where)} cells"
    )

ordered.to_csv(OUT_MATRIX, sep="\t")

mapping[
    ["domain", "section", "cluster_label", "cnv_family"]
].to_csv(OUT_ORDER, sep="\t", index=False)


# ---------------------------------------------------------------------
# Plot — display changes only; values are untouched
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 10))

im = ax.imshow(
    ordered.to_numpy(),
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)

ax.set_xticks(np.arange(len(order)))
ax.set_yticks(np.arange(len(order)))

ax.set_xticklabels(order, rotation=90)
ax.set_yticklabels(order)

for i in range(len(order)):
    for j in range(len(order)):
        value = ordered.iloc[i, j]
        ax.text(
            j, i, f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=6,
        )


# separators between mapped families
families = mapping["cnv_family"].tolist()

for i in range(1, len(families)):
    if families[i] != families[i - 1]:
        pos = i - 0.5
        ax.axhline(pos, color="black", linewidth=1.2)
        ax.axvline(pos, color="black", linewidth=1.2)


cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Pearson correlation of smoothed inferCNV profiles")

ax.set_title(
    "Seeded Figure 3 CNV-domain similarity\n"
    "ordered by CNV-family assignment"
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)

print("\n[DONE]", OUT_PNG)
print("[DONE]", OUT_PDF)
print("[DONE]", OUT_MATRIX)
print("[DONE]", OUT_ORDER)

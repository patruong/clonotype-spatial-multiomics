#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

PREP = ROOT / "data/processed/cnv/infercnv/figure3/prepared"
OUT = ROOT / "results/cnv/figure3/domain_matching_seed0"
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = {
    "bc2043": "bc2043_V13Y10-061_D1_synthref",
    "bc2020": "bc2020_V13Y10-060_D1_synthref",
    "bc2075": "bc2075_V13Y10-038_D1_synthref",
}

WINDOW = 51


def load_profile(section, name):
    p = (
        PREP / name /
        "infercnv_R_out_noHMM" /
        "leiden_posthoc_res_0.70" /
        "infercnv_R_leiden_cluster_mean_profiles.tsv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, sep="\t")
    df["cluster_label"] = df["cluster_label"].astype(str)

    meta = {"cluster_label", "n_spots", "cnv_abs_dev_mean"}
    genes = [c for c in df.columns if c not in meta]

    return df, genes


# ------------------------------------------------------------
# Load all three sections
# ------------------------------------------------------------

dfs = {}
gene_sets = []

for sec, name in SECTIONS.items():
    df, genes = load_profile(sec, name)
    dfs[sec] = df
    gene_sets.append(set(genes))

common = set.intersection(*gene_sets)

print("Common inferCNV genes:", len(common))


# ------------------------------------------------------------
# Recover genomic order
# ------------------------------------------------------------

gene_order_path = (
    PREP /
    SECTIONS["bc2043"] /
    "gene_order.tsv"
)

go = pd.read_csv(
    gene_order_path,
    sep="\t",
    header=None,
    comment="#",
)

if go.shape[1] < 4:
    raise RuntimeError(
        f"Unexpected gene_order.tsv shape: {go.shape}"
    )

go = go.iloc[:, :4].copy()
go.columns = ["gene", "chrom", "start", "end"]

# Robust to a possible header row.
if str(go.iloc[0]["gene"]).lower() in {
    "gene", "gene_name", "symbol"
}:
    go = go.iloc[1:].copy()

go["gene"] = go["gene"].astype(str)
go["chrom"] = go["chrom"].astype(str)

# Keep first occurrence only.
go = go.drop_duplicates("gene", keep="first")

ordered_genes = [
    g for g in go["gene"].tolist()
    if g in common
]

if len(ordered_genes) < 1000:
    raise RuntimeError(
        f"Too few common ordered genes: {len(ordered_genes)}"
    )

go = go.set_index("gene").loc[ordered_genes]

print("Ordered common genes:", len(ordered_genes))


# ------------------------------------------------------------
# Build and smooth profiles chromosome-by-chromosome
# ------------------------------------------------------------

profiles = {}
metadata = []

chroms = go["chrom"].to_numpy()

for sec in ["bc2043", "bc2020", "bc2075"]:
    df = dfs[sec]

    for _, row in df.iterrows():
        label = str(row["cluster_label"])
        domain = f"{sec}_L{label}"

        raw = row[ordered_genes].astype(float).to_numpy()

        smoothed = np.empty_like(raw, dtype=float)

        for chrom in pd.unique(chroms):
            idx = np.flatnonzero(chroms == chrom)

            smoothed[idx] = (
                pd.Series(raw[idx])
                .rolling(
                    WINDOW,
                    center=True,
                    min_periods=1,
                )
                .mean()
                .to_numpy()
            )

        profiles[domain] = smoothed

        metadata.append({
            "domain": domain,
            "section": sec,
            "cluster_label": label,
            "n_spots": int(row["n_spots"]),
            "cnv_abs_dev_mean": float(
                row["cnv_abs_dev_mean"]
            ),
        })


meta = pd.DataFrame(metadata)
meta.to_csv(
    OUT / "domain_metadata.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# Pairwise similarity
# ------------------------------------------------------------

domains = meta["domain"].tolist()

corr = pd.DataFrame(
    np.eye(len(domains)),
    index=domains,
    columns=domains,
    dtype=float,
)

rows = []

section_of = dict(zip(meta["domain"], meta["section"]))

for i, a in enumerate(domains):
    for j, b in enumerate(domains):

        x = profiles[a]
        y = profiles[b]

        r = float(np.corrcoef(x, y)[0, 1])
        corr.loc[a, b] = r

        if j <= i:
            continue

        rmse = float(
            np.sqrt(np.mean((x - y) ** 2))
        )

        rows.append({
            "domain_a": a,
            "section_a": section_of[a],
            "domain_b": b,
            "section_b": section_of[b],
            "pearson_r": r,
            "rmse": rmse,
            "cross_section":
                section_of[a] != section_of[b],
        })


pairs = pd.DataFrame(rows)

pairs.to_csv(
    OUT / "all_pairwise_similarity.tsv",
    sep="\t",
    index=False,
)

cross = (
    pairs[pairs["cross_section"]]
    .sort_values(
        ["pearson_r", "rmse"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

cross.to_csv(
    OUT / "cross_section_similarity.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# Best cross-section match for every domain
# ------------------------------------------------------------

best_rows = []

for a in domains:
    candidates = []

    for b in domains:
        if a == b:
            continue
        if section_of[a] == section_of[b]:
            continue

        x = profiles[a]
        y = profiles[b]

        candidates.append({
            "domain": a,
            "section": section_of[a],
            "best_match": b,
            "best_match_section": section_of[b],
            "pearson_r": float(
                np.corrcoef(x, y)[0, 1]
            ),
            "rmse": float(
                np.sqrt(np.mean((x - y) ** 2))
            ),
        })

    candidates = sorted(
        candidates,
        key=lambda z: (-z["pearson_r"], z["rmse"]),
    )

    best_rows.append(candidates[0])


best = pd.DataFrame(best_rows)

best_map = dict(
    zip(best["domain"], best["best_match"])
)

best["reciprocal_best"] = [
    best_map.get(b) == a
    for a, b in zip(
        best["domain"],
        best["best_match"],
    )
]

best.to_csv(
    OUT / "best_cross_section_matches.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

M = corr.loc[domains, domains].to_numpy()

fig, ax = plt.subplots(figsize=(10, 9))

im = ax.imshow(
    M,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    interpolation="nearest",
)

ax.set_xticks(np.arange(len(domains)))
ax.set_yticks(np.arange(len(domains)))

ax.set_xticklabels(
    domains,
    rotation=90,
    fontsize=8,
)
ax.set_yticklabels(
    domains,
    fontsize=8,
)

for i in range(len(domains)):
    for j in range(len(domains)):
        ax.text(
            j,
            i,
            f"{M[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=6,
        )

cb = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)
cb.set_label(
    "Pearson correlation of smoothed inferCNV profiles"
)

ax.set_title(
    "Seeded Figure 3 CNV-domain similarity\n"
    f"{WINDOW}-gene chromosome-aware smoothing"
)

fig.tight_layout()

fig.savefig(
    OUT / "cnv_domain_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT / "cnv_domain_correlation_heatmap.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# Save smoothed profiles for downstream MEDICC2/family inspection
# ------------------------------------------------------------

profile_df = pd.DataFrame(
    {d: profiles[d] for d in domains},
    index=ordered_genes,
)

profile_df.index.name = "gene"

profile_df.to_csv(
    OUT / "smoothed_domain_profiles.tsv.gz",
    sep="\t",
    compression="gzip",
)


# ------------------------------------------------------------
# Terminal summary
# ------------------------------------------------------------

print()
print("=" * 78)
print("DOMAINS")
print("=" * 78)
print(
    meta.to_string(index=False)
)

print()
print("=" * 78)
print("BEST CROSS-SECTION MATCH FOR EACH DOMAIN")
print("=" * 78)
print(
    best.to_string(index=False)
)

print()
print("=" * 78)
print("TOP 20 CROSS-SECTION PAIRS")
print("=" * 78)
print(
    cross[
        [
            "domain_a",
            "domain_b",
            "pearson_r",
            "rmse",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("[DONE]", OUT)

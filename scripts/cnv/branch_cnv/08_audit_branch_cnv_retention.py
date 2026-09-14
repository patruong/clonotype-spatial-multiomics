#!/usr/bin/env python3

from pathlib import Path
from itertools import product
import math

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[3]

BRANCH = (
    ROOT / "data/processed/svdj/figure3/"
    "IGH_clone9953_historical14280_unique_spot_classes.tsv"
)

CNV_ROOT = ROOT / "data/processed/cnv/infercnv/figure3"
RAW_ROOT = ROOT / "data/processed/st/nlsdeconv"

OUT = ROOT / "results/cnv/figure3/branch_cnv_seed0"
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = {
    "bc2043": "V13Y10-061_D1",
    "bc2020": "V13Y10-060_D1",
    "bc2075": "V13Y10-038_D1",
}

NAMES = {
    "bc2043": "bc2043_V13Y10-061_D1_synthref",
    "bc2020": "bc2020_V13Y10-060_D1_synthref",
    "bc2075": "bc2075_V13Y10-038_D1_synthref",
}


def norm_barcode(values):
    return (
        pd.Series(values, dtype=str)
        .str.strip()
        .str.replace(r"-1$", "", regex=True)
    )


def h5ad_barcodes(path):
    a = ad.read_h5ad(path, backed="r")
    vals = set(norm_barcode(a.obs_names).tolist())
    a.file.close()
    return vals


def find_one(root, sid):
    hits = sorted(root.rglob(f"{sid}.nlsdeconv.h5ad"))
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one {sid}.nlsdeconv.h5ad under {root}; "
            f"found {len(hits)}: {hits}"
        )
    return hits[0]


branches = pd.read_csv(BRANCH, sep="\t")
branches["barcode_key"] = norm_barcode(branches["barcode"])

focal = branches[
    branches["spot_branch_class"].isin(["blue", "red"])
].copy()

audit_parts = []


for section, sid in SECTIONS.items():

    raw_file = RAW_ROOT / f"{sid}.nlsdeconv.h5ad"

    pathology_file = find_one(
        CNV_ROOT / "pathology_filtered_nlsdeconv",
        sid,
    )

    qc_file = find_one(
        CNV_ROOT / "qc_nlsdeconv",
        sid,
    )

    final_file = (
        CNV_ROOT /
        "prepared" /
        NAMES[section] /
        "infercnv_R_out_noHMM" /
        "leiden_posthoc_res_0.70" /
        "infercnv_R_leiden_spatial_table.tsv"
    )

    raw = h5ad_barcodes(raw_file)
    pathology = h5ad_barcodes(pathology_file)
    qc = h5ad_barcodes(qc_file)

    final_df = pd.read_csv(final_file, sep="\t")
    final = set(norm_barcode(final_df["barcode"]).tolist())

    # Retention audit starts from Space Ranger in-tissue / filtered-GEX spots.
    # The raw NLSDeconv object was created with sc.read_visium(), which uses
    # Space Ranger filtered_feature_bc_matrix. BCR-positive off-tissue barcodes
    # are therefore outside the integrated ST-CNV analysis universe, not losses.
    x_all = focal[focal["section"] == section].copy()

    off = x_all[~x_all["barcode_key"].isin(raw)]
    if len(off):
        counts = off["spot_branch_class"].value_counts().to_dict()
        print(
            f"[PROVENANCE] {section}: excluded before retention audit "
            f"(Space Ranger off-tissue / outside filtered GEX): {counts}"
        )

    x = x_all[x_all["barcode_key"].isin(raw)].copy()

    x["in_raw"] = x["barcode_key"].isin(raw)
    x["after_pathology"] = x["barcode_key"].isin(pathology)
    x["after_expression_qc"] = x["barcode_key"].isin(qc)
    x["in_final_cnv"] = x["barcode_key"].isin(final)

    x["loss_stage"] = np.select(
        [
            ~x["in_raw"],
            x["in_raw"] & ~x["after_pathology"],
            x["after_pathology"] & ~x["after_expression_qc"],
            x["after_expression_qc"] & ~x["in_final_cnv"],
        ],
        [
            "absent_from_raw_nlsdeconv",
            "removed_pathology_necrosis",
            "removed_expression_qc",
            "missing_after_qc",
        ],
        default="retained_final_cnv",
    )

    audit_parts.append(x)


audit = pd.concat(audit_parts, ignore_index=True)

audit.to_csv(
    OUT / "branch_cnv_retention_audit.tsv",
    sep="\t",
    index=False,
)

summary = (
    audit
    .groupby(
        ["section", "spot_branch_class", "loss_stage"],
        observed=True,
    )
    .size()
    .rename("n")
    .reset_index()
)

summary.to_csv(
    OUT / "branch_cnv_retention_summary.tsv",
    sep="\t",
    index=False,
)


print()
print("=" * 88)
print("RETENTION / LOSS STAGE")
print("=" * 88)

for section in SECTIONS:
    print(f"\n--- {section} ---")
    z = summary[summary["section"] == section]
    print(
        z.pivot(
            index="loss_stage",
            columns="spot_branch_class",
            values="n",
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )


print()
print("=" * 88)
print("FINAL CNV RETENTION: BLUE vs RED")
print("=" * 88)

retention_rows = []

for section in SECTIONS:

    z = audit[audit["section"] == section]

    counts = {}

    for branch in ["blue", "red"]:

        b = z[z["spot_branch_class"] == branch]

        counts[branch] = {
            "retained": int(b["in_final_cnv"].sum()),
            "removed": int((~b["in_final_cnv"]).sum()),
        }

    table = [
        [
            counts["blue"]["retained"],
            counts["blue"]["removed"],
        ],
        [
            counts["red"]["retained"],
            counts["red"]["removed"],
        ],
    ]

    OR, p = fisher_exact(table)

    retention_rows.append({
        "section": section,
        "blue_retained": counts["blue"]["retained"],
        "blue_removed": counts["blue"]["removed"],
        "red_retained": counts["red"]["retained"],
        "red_removed": counts["red"]["removed"],
        "retention_OR_blue_vs_red": OR,
        "p_value": p,
    })


ret = pd.DataFrame(retention_rows)

ret.to_csv(
    OUT / "branch_cnv_retention_fisher.tsv",
    sep="\t",
    index=False,
)

print(ret.to_string(index=False))


# ------------------------------------------------------------
# Fisher-Freeman-Halton exact test for the complete bc2043 table
# ------------------------------------------------------------

joined = pd.read_csv(
    OUT / "branch_spots_joined_to_seeded_cnv.tsv",
    sep="\t",
)


def ffh_2xk(df, column):

    tab = pd.crosstab(
        df["spot_branch_class"],
        df[column],
    ).reindex(["blue", "red"])

    blue = tab.loc["blue"].to_numpy(dtype=int)
    totals = tab.sum(axis=0).to_numpy(dtype=int)

    B = int(blue.sum())
    N = int(totals.sum())

    def logcomb(n, k):
        if k < 0 or k > n:
            return -np.inf
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )

    denom = logcomb(N, B)

    def logprob(x):
        return (
            sum(
                logcomb(int(n), int(k))
                for n, k in zip(totals, x)
            )
            - denom
        )

    obs_lp = logprob(blue)

    p = 0.0

    ranges = [
        range(int(n) + 1)
        for n in totals[:-1]
    ]

    for xs in product(*ranges):

        last = B - sum(xs)

        if last < 0 or last > totals[-1]:
            continue

        full = list(xs) + [last]
        lp = logprob(full)

        if lp <= obs_lp + 1e-12:
            p += math.exp(lp)

    return tab, min(p, 1.0)


print()

print(f"\n[DONE] {OUT}")

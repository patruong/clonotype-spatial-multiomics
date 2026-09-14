#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


ROOT = Path(__file__).resolve().parents[3]

BCR = (
    ROOT / "data/processed/svdj/figure3/"
    "IGH_clone9953_historical14280_unique_spot_classes.tsv"
)

MAPPING = (
    ROOT / "results/cnv/figure3/domain_matching_seed0/"
    "cnv_family_mapping_seed0.tsv"
)

OUT = (
    ROOT / "results/cnv/figure3/branch_cnv_seed0/"
    "manuscript_analysis"
)
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = ["bc2020", "bc2043", "bc2075"]

PREPARED = {
    "bc2020": "bc2020_V13Y10-060_D1_synthref",
    "bc2043": "bc2043_V13Y10-061_D1_synthref",
    "bc2075": "bc2075_V13Y10-038_D1_synthref",
}

BRANCH_ORDER = ["Blue", "Red", "Both"]

PERMUTATIONS = 10000
SEED = 14280

EXPECTED_DIRECTION = {
    "bc2043_L0": "red",
    "bc2043_L1": "blue",
    "bc2043_L2": "red",
    "bc2043_L3": "red",
}


# ---------------------------------------------------------------------
# Exact historical utility functions
# ---------------------------------------------------------------------

def normalize_barcode(series):
    return (
        series.astype(str)
        .str.extract(r"([ACGT]{16}-1)", expand=False)
    )


def classify_branch(row):
    blue = row["raw_A"] > 0
    red = row["raw_B"] > 0

    if blue and red:
        return "Both"
    if blue:
        return "Blue"
    if red:
        return "Red"

    return "Other"


def bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty_like(adjusted)
    out[order] = adjusted

    return out


# ---------------------------------------------------------------------
# Current deterministic CNV assignments
# ---------------------------------------------------------------------

def load_current_cnv():
    frames = []

    for section in SECTIONS:
        name = PREPARED[section]

        path = (
            ROOT
            / "data/processed/cnv/infercnv/figure3/prepared"
            / name
            / "infercnv_R_out_noHMM"
            / "leiden_posthoc_res_0.70"
            / "infercnv_R_leiden_spatial_table.tsv"
        )

        df = pd.read_csv(path, sep="\t", low_memory=False)

        required = {"barcode", "cnv_leiden"}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{path} missing columns: {sorted(missing)}"
            )

        df["barcode"] = normalize_barcode(df["barcode"])
        df["section"] = section
        df["cnv_leiden"] = pd.to_numeric(
            df["cnv_leiden"],
            errors="raise",
        ).astype(int)

        df["cluster_key"] = (
            df["section"]
            + "_L"
            + df["cnv_leiden"].astype(str)
        )

        frames.append(df)

    cnv = pd.concat(frames, ignore_index=True)

    if cnv.duplicated(["section", "barcode"]).any():
        raise RuntimeError(
            "Duplicate CNV assignments by section + barcode."
        )

    return cnv


def load_mapping():
    m = pd.read_csv(
        MAPPING,
        sep="\t",
        dtype={"cluster_label": str},
    )

    m["cluster_key"] = m["domain"]
    m["plot_family"] = m["cnv_family"]

    if m.duplicated("cluster_key").any():
        raise RuntimeError("Duplicate cluster_key in mapping.")

    return m[
        [
            "section",
            "cluster_label",
            "cluster_key",
            "plot_family",
        ]
    ].copy()


# ---------------------------------------------------------------------
# Historical summary tables
# ---------------------------------------------------------------------

def make_summary(df, group_columns):
    counts = (
        df.groupby(group_columns + ["branch_plot"])
        .size()
        .unstack(fill_value=0)
    )

    for branch in BRANCH_ORDER:
        if branch not in counts.columns:
            counts[branch] = 0

    counts = counts[BRANCH_ORDER].reset_index()
    counts["n_focal_spots"] = counts[BRANCH_ORDER].sum(axis=1)

    return counts


# ---------------------------------------------------------------------
# Exact historical stratified Pearson statistic + permutation
# ---------------------------------------------------------------------

def pearson_statistic_stratified(df, group_column):
    statistic = 0.0

    for section, section_df in df.groupby("section"):

        table = pd.crosstab(
            section_df[group_column],
            section_df["branch_binary"],
        ).reindex(
            columns=["Blue", "Red"],
            fill_value=0,
        )

        table = table.loc[table.sum(axis=1) > 0]

        if table.shape[0] < 2:
            continue

        observed = table.to_numpy(dtype=float)
        total = observed.sum()

        expected = (
            observed.sum(axis=1, keepdims=True)
            @ observed.sum(axis=0, keepdims=True)
            / total
        )

        valid = expected > 0

        statistic += np.sum(
            ((observed - expected) ** 2)[valid]
            / expected[valid]
        )

    return float(statistic)


def stratified_permutation_test(
    df,
    group_column,
    permutations,
    seed,
):
    rng = np.random.default_rng(seed)

    observed = pearson_statistic_stratified(
        df,
        group_column,
    )

    permuted_statistics = np.empty(permutations)

    sections = {
        section: section_df.copy()
        for section, section_df in df.groupby("section")
    }

    for iteration in range(permutations):

        shuffled_frames = []

        for section, section_df in sections.items():

            shuffled = section_df.copy()

            shuffled["branch_binary"] = rng.permutation(
                shuffled["branch_binary"].to_numpy()
            )

            shuffled_frames.append(shuffled)

        permuted = pd.concat(
            shuffled_frames,
            ignore_index=True,
        )

        permuted_statistics[iteration] = (
            pearson_statistic_stratified(
                permuted,
                group_column,
            )
        )

    p_value = (
        1
        + np.sum(permuted_statistics >= observed)
    ) / (permutations + 1)

    return {
        "grouping": group_column,
        "observed_statistic": observed,
        "permutations": permutations,
        "seed": seed,
        "permutation_p_value": p_value,
    }


# ---------------------------------------------------------------------
# Historical two-sided group-vs-rest Fisher tests
# ---------------------------------------------------------------------

def group_vs_rest_fisher(binary_df, group_column):
    rows = []

    for section, section_df in binary_df.groupby("section"):

        for group_value in sorted(
            section_df[group_column].dropna().unique()
        ):
            inside = section_df[group_column].eq(group_value)

            blue_in = int(
                (
                    inside
                    & section_df["branch_binary"].eq("Blue")
                ).sum()
            )
            red_in = int(
                (
                    inside
                    & section_df["branch_binary"].eq("Red")
                ).sum()
            )
            blue_out = int(
                (
                    ~inside
                    & section_df["branch_binary"].eq("Blue")
                ).sum()
            )
            red_out = int(
                (
                    ~inside
                    & section_df["branch_binary"].eq("Red")
                ).sum()
            )

            table = [
                [blue_in, red_in],
                [blue_out, red_out],
            ]

            odds_ratio, p_value = fisher_exact(
                table,
                alternative="two-sided",
            )

            corrected_odds = (
                (blue_in + 0.5)
                * (red_out + 0.5)
                / (
                    (red_in + 0.5)
                    * (blue_out + 0.5)
                )
            )

            rows.append(
                {
                    "section": section,
                    "grouping": group_column,
                    "group": group_value,
                    "blue_in_group": blue_in,
                    "red_in_group": red_in,
                    "blue_outside_group": blue_out,
                    "red_outside_group": red_out,
                    "odds_ratio_blue_vs_red": odds_ratio,
                    "haldane_corrected_log2_odds_ratio":
                        np.log2(corrected_odds),
                    "fisher_p_value": p_value,
                }
            )

    result = pd.DataFrame(rows)

    result["fisher_bh_q_value"] = np.nan

    for _, index in result.groupby(
        ["section", "grouping"]
    ).groups.items():

        index = list(index)

        result.loc[
            index,
            "fisher_bh_q_value",
        ] = bh_adjust(
            result.loc[
                index,
                "fisher_p_value",
            ].to_numpy()
        )

    return result


# ---------------------------------------------------------------------
# Exact manuscript bc2043 directional one-sided tests
# ---------------------------------------------------------------------

def bc2043_directional_tests(local_counts):
    df = (
        local_counts[
            local_counts["section"].eq("bc2043")
        ]
        .sort_values("cnv_leiden")
        .reset_index(drop=True)
    )

    rows = []

    for row in df.itertuples(index=False):

        cluster = row.cluster_key
        direction = EXPECTED_DIRECTION[cluster]

        blue_in = int(row.Blue)
        red_in = int(row.Red)

        others = df[
            ~df["cluster_key"].eq(cluster)
        ]

        blue_out = int(others["Blue"].sum())
        red_out = int(others["Red"].sum())

        table = [
            [blue_in, red_in],
            [blue_out, red_out],
        ]

        alternative = (
            "greater"
            if direction == "blue"
            else "less"
        )

        odds_ratio, p_one = fisher_exact(
            table,
            alternative=alternative,
        )

        _, p_two = fisher_exact(
            table,
            alternative="two-sided",
        )

        corrected_or = (
            (blue_in + 0.5)
            * (red_out + 0.5)
            / (
                (red_in + 0.5)
                * (blue_out + 0.5)
            )
        )

        rows.append(
            {
                "section": "bc2043",
                "cluster_key": cluster,
                "cnv_leiden": int(row.cnv_leiden),
                "plot_family": row.plot_family,
                "expected_direction": direction,
                "Blue": blue_in,
                "Red": red_in,
                "Both": int(row.Both),
                "blue_outside": blue_out,
                "red_outside": red_out,
                "odds_ratio_blue_vs_red": odds_ratio,
                "haldane_log2_odds_ratio":
                    np.log2(corrected_or),
                "one_sided_alternative": alternative,
                "one_sided_fisher_p": p_one,
                "two_sided_fisher_p": p_two,
            }
        )

    res = pd.DataFrame(rows)

    res["one_sided_bh_q"] = bh_adjust(
        res["one_sided_fisher_p"].to_numpy()
    )

    return res


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

bcr = pd.read_csv(
    BCR,
    sep="\t",
    low_memory=False,
)

required = {
    "section",
    "barcode",
    "raw_A",
    "raw_B",
    "raw_other",
}

missing = required - set(bcr.columns)

if missing:
    raise RuntimeError(
        f"BCR table missing columns: {sorted(missing)}"
    )

# Exact historical behavior.
bcr["barcode"] = normalize_barcode(bcr["barcode"])
bcr["branch_plot"] = bcr.apply(
    classify_branch,
    axis=1,
)

if bcr.duplicated(["section", "barcode"]).any():
    raise RuntimeError(
        "BCR table is not unique by section + barcode."
    )


cnv = load_current_cnv()
mapping = load_mapping()

cnv = cnv.merge(
    mapping[
        [
            "cluster_key",
            "plot_family",
        ]
    ],
    on="cluster_key",
    how="left",
    validate="many_to_one",
)

if cnv["plot_family"].isna().any():
    missing_keys = sorted(
        cnv.loc[
            cnv["plot_family"].isna(),
            "cluster_key",
        ].unique()
    )

    raise RuntimeError(
        "Missing CNV family mappings: "
        + ", ".join(missing_keys)
    )


joined = bcr.merge(
    cnv[
        [
            "section",
            "barcode",
            "cnv_leiden",
            "cluster_key",
            "plot_family",
            "cnv_abs_dev_mean",
        ]
    ],
    on=["section", "barcode"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

joined.to_csv(
    OUT / "IGH_clone9953_barcode_branch_CNV_join_all_spots.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Join audit
# ---------------------------------------------------------------------

audit_rows = []

for section in SECTIONS:

    section_all = joined["section"].eq(section)

    section_focal = (
        section_all
        & joined["branch_plot"].isin(BRANCH_ORDER)
    )

    section_matched = (
        section_focal
        & joined["plot_family"].notna()
    )

    audit_rows.append(
        {
            "section": section,
            "all_clone_spots": int(section_all.sum()),
            "red_blue_bearing_spots": int(
                section_focal.sum()
            ),
            "red_blue_bearing_spots_with_CNV_assignment":
                int(section_matched.sum()),
            "red_blue_bearing_spots_without_CNV_assignment":
                int(
                    section_focal.sum()
                    - section_matched.sum()
                ),
            "CNV_assignment_fraction":
                (
                    section_matched.sum()
                    / section_focal.sum()
                    if section_focal.sum() > 0
                    else np.nan
                ),
        }
    )

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    OUT / "CNV_join_overlap_audit.tsv",
    sep="\t",
    index=False,
)


focal = joined[
    joined["branch_plot"].isin(BRANCH_ORDER)
    & joined["plot_family"].notna()
].copy()

binary = focal[
    focal["branch_plot"].isin(["Blue", "Red"])
].copy()

binary["branch_binary"] = binary["branch_plot"]


# ---------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------

local_counts = make_summary(
    focal,
    [
        "section",
        "cnv_leiden",
        "cluster_key",
        "plot_family",
    ],
)

family_counts = make_summary(
    focal,
    [
        "section",
        "plot_family",
    ],
)

local_counts.to_csv(
    OUT / "branch_composition_by_local_CNV_cluster_counts.tsv",
    sep="\t",
    index=False,
)

family_counts.to_csv(
    OUT / "branch_composition_by_CNV_family_and_section_counts.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Historical two-sided Fisher analysis
# ---------------------------------------------------------------------

fisher_results = pd.concat(
    [
        group_vs_rest_fisher(
            binary,
            "cluster_key",
        ),
        group_vs_rest_fisher(
            binary,
            "plot_family",
        ),
    ],
    ignore_index=True,
)

fisher_results.to_csv(
    OUT / "group_vs_rest_fisher_two_sided.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Manuscript directional one-sided bc2043 analysis
# ---------------------------------------------------------------------

directional = bc2043_directional_tests(
    local_counts
)

directional.to_csv(
    OUT / "bc2043_directional_one_sided_fisher.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Exact historical section-stratified permutation
# ---------------------------------------------------------------------

permutation_results = pd.DataFrame(
    [
        stratified_permutation_test(
            binary,
            "cluster_key",
            PERMUTATIONS,
            SEED,
        ),
        stratified_permutation_test(
            binary,
            "plot_family",
            PERMUTATIONS,
            SEED + 1,
        ),
    ]
)

permutation_results.to_csv(
    OUT / "section_stratified_permutation_tests.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Historical-vs-seeded bc2043 comparison
# ---------------------------------------------------------------------

historical_counts = pd.DataFrame(
    [
        ["bc2043_L0", "CNV-B",       3,  5],
        ["bc2043_L1", "CNV-C/other", 36, 19],
        ["bc2043_L2", "CNV-A",       7, 10],
        ["bc2043_L3", "CNV-C",       6,  9],
    ],
    columns=[
        "cluster_key",
        "plot_family",
        "Blue",
        "Red",
    ],
)

hist_rows = []

for row in historical_counts.itertuples(index=False):

    others = historical_counts[
        ~historical_counts["cluster_key"].eq(
            row.cluster_key
        )
    ]

    blue_out = int(others["Blue"].sum())
    red_out = int(others["Red"].sum())

    direction = EXPECTED_DIRECTION[
        row.cluster_key
    ]

    alternative = (
        "greater"
        if direction == "blue"
        else "less"
    )

    OR, p = fisher_exact(
        [
            [row.Blue, row.Red],
            [blue_out, red_out],
        ],
        alternative=alternative,
    )

    hist_rows.append(
        {
            "cluster_key": row.cluster_key,
            "historical_Blue": row.Blue,
            "historical_Red": row.Red,
            "historical_OR": OR,
            "historical_one_sided_p": p,
        }
    )

hist = pd.DataFrame(hist_rows)

hist["historical_one_sided_q"] = bh_adjust(
    hist["historical_one_sided_p"]
)

new = directional[
    [
        "cluster_key",
        "Blue",
        "Red",
        "odds_ratio_blue_vs_red",
        "one_sided_fisher_p",
        "one_sided_bh_q",
    ]
].rename(
    columns={
        "Blue": "seeded_Blue",
        "Red": "seeded_Red",
        "odds_ratio_blue_vs_red": "seeded_OR",
        "one_sided_fisher_p": "seeded_one_sided_p",
        "one_sided_bh_q": "seeded_one_sided_q",
    }
)

comparison = hist.merge(
    new,
    on="cluster_key",
    validate="one_to_one",
)

comparison.to_csv(
    OUT / "bc2043_historical_vs_seeded.tsv",
    sep="\t",
    index=False,
)


# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------

print()
print("=" * 92)
print("CURRENT LOCAL CNV COUNTS")
print("=" * 92)
print(local_counts.to_string(index=False))

print()
print("=" * 92)
print("CURRENT FAMILY COUNTS")
print("=" * 92)
print(family_counts.to_string(index=False))

print()
print("=" * 92)
print("bc2043 MANUSCRIPT-STYLE ONE-SIDED FISHER")
print("=" * 92)

print(
    directional[
        [
            "cluster_key",
            "plot_family",
            "expected_direction",
            "Blue",
            "Red",
            "blue_outside",
            "red_outside",
            "odds_ratio_blue_vs_red",
            "one_sided_fisher_p",
            "one_sided_bh_q",
        ]
    ].to_string(index=False)
)

print()
print("=" * 92)
print("SECTION-STRATIFIED PERMUTATION")
print("=" * 92)
print(permutation_results.to_string(index=False))

print()
print("=" * 92)
print("bc2043 HISTORICAL vs SEEDED")
print("=" * 92)
print(comparison.to_string(index=False))

print()
print("=" * 92)
print("HISTORICAL MANUSCRIPT REFERENCE")
print("=" * 92)
print(
    "local-domain permutation: "
    "stat=52.07330620217665, p=0.006799320067993201"
)
print(
    "family permutation:       "
    "stat=8.288387696237423, p=0.30416958304169583"
)

print()
print("[DONE]", OUT)

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


CHAINS = ["IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"]


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


def normalize_values(series: pd.Series) -> list[str]:
    return sorted(
        set(
            series.fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )
    )


def clonotype_fingerprint(group: pd.DataFrame) -> str:
    """
    Stable identity of an already-inferred IgDiscover clonotype.

    Deliberately independent of:
      - IgDiscover clone_id
      - row order
      - abundance
      - spatial barcode order

    Uses the fields defining the inferred clonotype identity.
    """
    fields = []

    for col in ["locus", "v_call", "j_call"]:
        if col not in group.columns:
            raise ValueError(f"Required column missing: {col}")
        vals = normalize_values(group[col])
        fields.append(f"{col}=" + "|".join(vals))

    if "cdr3" not in group.columns:
        raise ValueError("Required column missing: cdr3")

    cdr3s = normalize_values(group["cdr3"])
    fields.append("cdr3=" + "|".join(cdr3s))

    payload = "\n".join(fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_mapping(members: pd.DataFrame, chain: str) -> pd.DataFrame:
    records = []

    for old_id, group in members.groupby("clone_id", sort=False):
        fp = clonotype_fingerprint(group)
        records.append(
            {
                "clone_id_igdiscover": int(old_id),
                "fingerprint": fp,
            }
        )

    mapping = pd.DataFrame(records)

    if mapping["fingerprint"].duplicated().any():
        dup = mapping.loc[
            mapping["fingerprint"].duplicated(False),
            ["clone_id_igdiscover", "fingerprint"],
        ]
        raise RuntimeError(
            f"{chain}: fingerprint collision detected:\n"
            + dup.to_string(index=False)
        )

    # Deterministic integer numbering:
    # same complete clonotype set -> same integer IDs.
    mapping = mapping.sort_values(
        "fingerprint",
        kind="mergesort",
    ).reset_index(drop=True)

    mapping["clone_id"] = range(1, len(mapping) + 1)

    # Full SHA-256 is retained as permanent cross-run identity.
    mapping["clone_uid"] = (
        chain + "_" + mapping["fingerprint"]
    )

    return mapping[
        [
            "clone_id_igdiscover",
            "clone_id",
            "clone_uid",
            "fingerprint",
        ]
    ]


def canonicalize_table(
    table: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    if "clone_id" not in table.columns:
        raise ValueError("Input table has no clone_id column")

    x = table.rename(
        columns={"clone_id": "clone_id_igdiscover"}
    ).copy()

    x = x.merge(
        mapping[
            ["clone_id_igdiscover", "clone_id", "clone_uid"]
        ],
        on="clone_id_igdiscover",
        how="left",
        validate="many_to_one",
    )

    if x["clone_id"].isna().any():
        raise RuntimeError("Some clone IDs could not be canonicalized")

    x["clone_id"] = x["clone_id"].astype(int)

    return x


def parse_sample(sequence_id: str) -> str:
    return str(sequence_id).split("_molecule")[0]


def parse_cb(sequence_id: str) -> str:
    return str(sequence_id).rsplit(";")[-1][3:]


def parse_umi(sequence_id: str) -> str:
    return str(sequence_id).rsplit(";")[-2][3:]


def rebuild_count_matrices(
    members: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    chain: str,
) -> None:

    x = members.copy()

    x["sample"] = x["sequence_id"].map(parse_sample)
    x["CB"] = x["sequence_id"].map(parse_cb)
    x["UMI"] = x["sequence_id"].map(parse_umi)
    x["CB:UMI"] = x["CB"] + ":" + x["UMI"]

    # Preserve the complete sample set represented by the original
    # IgDiscover count-matrix files, including samples with zero clones.
    suffix = f"_{chain}_count_matrix.tsv"
    samples = sorted(
        p.name[:-len(suffix)]
        for p in input_dir.glob(f"*{suffix}")
    )

    for sample in samples:
        sub = x[x["sample"] == sample].copy()

        out = output_dir / f"{sample}_{chain}_count_matrix.tsv"

        if sub.empty:
            pd.DataFrame([""]).to_csv(out, sep="\t")
            continue

        # Match IgDiscover CreateCountMatrix behavior:
        # discard CB:UMI observations assigned to >1 clone.
        ambiguity = (
            sub.groupby("CB:UMI")["clone_id"]
            .nunique()
        )
        ambiguous = ambiguity[ambiguity > 1].index
        sub = sub[~sub["CB:UMI"].isin(ambiguous)]

        counts = (
            sub.groupby(["CB", "clone_id"])["UMI"]
            .nunique()
            .unstack(fill_value=0)
        )

        counts = counts.reindex(
            sorted(counts.columns),
            axis=1,
        )

        counts.index = [
            reverse_complement(cb) + "-1"
            for cb in counts.index
        ]

        counts.to_csv(out, sep="\t")


def process_chain(
    input_dir: Path,
    output_dir: Path,
    chain: str,
) -> None:

    members_path = input_dir / f"{chain}_clonotypes_members.tsv"
    reps_path = input_dir / f"{chain}_clonotypes.tsv"

    if not members_path.exists():
        print(f"{chain}: no members file; skipping")
        return

    members = pd.read_csv(members_path, sep="\t")

    if members.empty or "clone_id" not in members.columns:
        shutil.copy2(members_path, output_dir / members_path.name)
        if reps_path.exists():
            shutil.copy2(reps_path, output_dir / reps_path.name)
        print(f"{chain}: empty; copied unchanged")
        return

    mapping = build_mapping(members, chain)

    canonical_members = canonicalize_table(
        members,
        mapping,
    )

    canonical_members.to_csv(
        output_dir / members_path.name,
        sep="\t",
        index=False,
    )

    if reps_path.exists():
        reps = pd.read_csv(reps_path, sep="\t")

        if not reps.empty and "clone_id" in reps.columns:
            canonical_reps = canonicalize_table(
                reps,
                mapping,
            )
            canonical_reps.to_csv(
                output_dir / reps_path.name,
                sep="\t",
                index=False,
            )
        else:
            shutil.copy2(
                reps_path,
                output_dir / reps_path.name,
            )

    mapping.to_csv(
        output_dir / f"{chain}_clone_id_mapping.tsv",
        sep="\t",
        index=False,
    )

    rebuild_count_matrices(
        canonical_members,
        input_dir,
        output_dir,
        chain,
    )

    print(
        f"{chain}: {len(mapping):,} clonotypes canonicalized"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for chain in CHAINS:
        process_chain(
            args.input_dir,
            args.output_dir,
            chain,
        )


if __name__ == "__main__":
    main()

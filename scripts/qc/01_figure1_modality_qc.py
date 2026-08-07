#!/usr/bin/env python3
from __future__ import annotations

import gzip
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix, issparse


ROOT = Path(__file__).resolve().parents[2]

META = ROOT / "metadata" / "metadata_short.tsv"
ST_ROOT = ROOT / "data" / "processed" / "st" / "space_ranger_outs"
MSI_ROOT = ROOT / "data" / "processed" / "msi" / "fake_spaceranger"
VDJ_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "svdj"
    / "igdiscover"
    / "sma_vdj"
)

OUT_TSV = ROOT / "results" / "qc" / "figure1_modality_qc.tsv"
OUT_FIG = ROOT / "results" / "figures" / "figure1"

BCR_CHAINS = ("IGH", "IGK", "IGL")
TCR_CHAINS = ("TRA", "TRB", "TRD", "TRG")
ALL_CHAINS = BCR_CHAINS + TCR_CHAINS

CONDITION_ORDER = (
    "no_molecule",
    "metabolite",
    "lipid",
    "peptide",
)

# Manuscript-panel typography
TITLE_FONTSIZE = 19
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
LEGEND_TITLE_FONTSIZE = 14


def normalize_condition(value: str) -> str:
    value = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "no_molecule": "no_molecule",
        "no_molecules": "no_molecule",
        "metabolite": "metabolite",
        "metabolites": "metabolite",
        "lipid": "lipid",
        "lipids": "lipid",
        "peptide": "peptide",
        "peptides": "peptide",
    }

    if value not in aliases:
        raise ValueError(f"Unknown MSI target: {value!r}")

    return aliases[value]


def count_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as handle:
        return sum(1 for _ in handle)


def first_existing(
    directory: Path,
    names: tuple[str, ...],
) -> Path:
    for name in names:
        path = directory / name

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"None of {names} exists in {directory}"
    )


def read_st_qc(section_id: str) -> dict[str, float | int]:
    path = (
        ST_ROOT
        / section_id
        / "filtered_feature_bc_matrix.h5"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]

        shape = tuple(
            int(value)
            for value in group["shape"][:]
        )

        matrix = csc_matrix(
            (
                group["data"][:],
                group["indices"][:],
                group["indptr"][:],
            ),
            shape=shape,
        )

    umi_per_spot = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    genes_per_spot = np.diff(matrix.indptr)

    return {
        "st_n_spots": int(matrix.shape[1]),
        "st_detected_genes": int(
            np.unique(matrix.indices).size
        ),
        "st_total_umis": float(umi_per_spot.sum()),
        "st_mean_umis_per_spot": float(
            umi_per_spot.mean()
        ),
        "st_median_umis_per_spot": float(
            np.median(umi_per_spot)
        ),
        "st_mean_genes_per_spot": float(
            genes_per_spot.mean()
        ),
        "st_median_genes_per_spot": float(
            np.median(genes_per_spot)
        ),
    }


def read_msi_qc(section_id: str) -> dict[str, object]:
    candidate_sources = (
        (
            "he",
            MSI_ROOT
            / "msi_in_he"
            / section_id
            / "filtered_feature_bc_matrix",
        ),
        (
            "bw",
            MSI_ROOT
            / "msi_in_bw"
            / section_id
            / "filtered_feature_bc_matrix",
        ),
    )

    source = next(
        (
            (source_name, directory)
            for source_name, directory in candidate_sources
            if directory.is_dir()
        ),
        None,
    )

    if source is None:
        return {
            "msi_status": "not_measured",
            "msi_coordinate_source": "",
            "msi_n_pixels": np.nan,
            "msi_detected_features": np.nan,
            "msi_mean_features_per_pixel": np.nan,
            "msi_median_features_per_pixel": np.nan,
        }

    source_name, directory = source

    matrix_path = first_existing(
        directory,
        ("matrix.mtx.gz", "matrix.mtx"),
    )

    features_path = first_existing(
        directory,
        ("features.tsv.gz", "features.tsv"),
    )

    barcodes_path = first_existing(
        directory,
        ("barcodes.tsv.gz", "barcodes.tsv"),
    )

    n_features = count_lines(features_path)
    n_barcodes = count_lines(barcodes_path)

    opener = (
        gzip.open
        if matrix_path.suffix == ".gz"
        else open
    )

    with opener(matrix_path, "rb") as handle:
        matrix = mmread(handle)

    matrix = (
        matrix.tocsc()
        if issparse(matrix)
        else csc_matrix(matrix)
    )

    if matrix.shape == (n_barcodes, n_features):
        matrix = matrix.T.tocsc()

    expected_shape = (n_features, n_barcodes)

    if matrix.shape != expected_shape:
        raise ValueError(
            f"{section_id}: MSI matrix shape "
            f"{matrix.shape} does not match "
            f"{expected_shape}"
        )

    features_per_pixel = np.diff(matrix.indptr)

    return {
        "msi_status": "available",
        "msi_coordinate_source": source_name,
        "msi_n_pixels": int(matrix.shape[1]),
        "msi_detected_features": int(
            np.unique(matrix.indices).size
        ),
        "msi_mean_features_per_pixel": float(
            features_per_pixel.mean()
        ),
        "msi_median_features_per_pixel": float(
            np.median(features_per_pixel)
        ),
    }


def collect_svdj_qc(
    sample_barcodes: list[str],
) -> pd.DataFrame:
    feature_sets = {
        barcode: {
            "BCR": set(),
            "TCR": set(),
        }
        for barcode in sample_barcodes
    }

    total_counts = {
        barcode: {
            "BCR": 0.0,
            "TCR": 0.0,
        }
        for barcode in sample_barcodes
    }

    for chain in ALL_CHAINS:
        path = VDJ_ROOT / f"{chain}_clonotypes_members.tsv"

        if not path.is_file():
            raise FileNotFoundError(path)

        table = pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "sequence_id",
                "clone_id",
                "count",
            ],
            low_memory=False,
        )

        table["barcode"] = (
            table["sequence_id"]
            .astype(str)
            .str.extract(
                r"^(bc\d{4})",
                expand=False,
            )
        )

        table = table.dropna(
            subset=["barcode", "clone_id"]
        )

        table = table[
            table["barcode"].isin(feature_sets)
        ].copy()

        receptor = (
            "BCR"
            if chain in BCR_CHAINS
            else "TCR"
        )

        # Prefix clone IDs with chain because clone_id=1
        # in IGH is not the same feature as clone_id=1
        # in IGK, IGL, TRA, etc.
        table["feature_id"] = (
            chain
            + ":"
            + table["clone_id"].astype(str)
        )

        table["count"] = pd.to_numeric(
            table["count"],
            errors="coerce",
        ).fillna(0)

        for barcode, group in table.groupby(
            "barcode",
            sort=False,
        ):
            feature_sets[barcode][receptor].update(
                group["feature_id"]
            )

            total_counts[barcode][receptor] += float(
                group["count"].sum()
            )

    rows = []

    for barcode in sample_barcodes:
        rows.append(
            {
                "barcode": barcode,
                "bcr_unique_clonotypes": len(
                    feature_sets[barcode]["BCR"]
                ),
                "tcr_unique_clonotypes": len(
                    feature_sets[barcode]["TCR"]
                ),
                "bcr_total_count": total_counts[
                    barcode
                ]["BCR"],
                "tcr_total_count": total_counts[
                    barcode
                ]["TCR"],
            }
        )

    return pd.DataFrame(rows)


def collect_qc_table() -> pd.DataFrame:
    metadata = pd.read_csv(
        META,
        sep="\t",
        dtype=str,
    )

    required_columns = {
        "capture_area_id",
        "matrix",
        "MSI target",
        "barcode",
        "patient",
        "subtype",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    metadata["condition"] = (
        metadata["MSI target"]
        .map(normalize_condition)
    )

    if metadata["capture_area_id"].duplicated().any():
        raise ValueError(
            "capture_area_id must be unique"
        )

    if metadata["barcode"].duplicated().any():
        raise ValueError(
            "barcode must be unique"
        )

    svdj_table = collect_svdj_qc(
        metadata["barcode"].tolist()
    )

    records = []

    for _, sample in metadata.iterrows():
        section_id = sample["capture_area_id"]

        record = {
            "capture_area_id": section_id,
            "barcode": sample["barcode"],
            "patient": sample["patient"],
            "subtype": sample["subtype"],
            "matrix": sample["matrix"],
            "msi_target_original": sample["MSI target"],
            "condition": sample["condition"],
        }

        record.update(
            read_st_qc(section_id)
        )

        record.update(
            read_msi_qc(section_id)
        )

        records.append(record)

    result = pd.DataFrame(records)

    result = result.merge(
        svdj_table,
        on="barcode",
        how="left",
        validate="one_to_one",
    )

    OUT_TSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_TSV,
        sep="\t",
        index=False,
        na_rep="NA",
    )

    print(f"Wrote QC table: {OUT_TSV}")

    return result


def patient_order(values: pd.Series) -> list[str]:
    def sort_key(value: str) -> tuple[int, object]:
        value = str(value)

        if value.isdigit():
            return 0, int(value)

        return 1, value

    return sorted(
        values.astype(str).unique(),
        key=sort_key,
    )


def draw_grouped_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    patient_colors: dict[str, object],
) -> None:
    patients = patient_order(table["patient"])

    group_centres = np.arange(
        len(CONDITION_ORDER),
        dtype=float,
    )

    bar_width = 0.8 / len(patients)

    for patient_index, patient in enumerate(patients):
        patient_table = (
            table.loc[
                table["patient"].astype(str) == patient
            ]
            .set_index("condition")
            .reindex(CONDITION_ORDER)
        )

        values = pd.to_numeric(
            patient_table[metric],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(values)

        offset = (
            patient_index
            - (len(patients) - 1) / 2
        ) * bar_width

        axis.bar(
            group_centres[valid] + offset,
            values[valid],
            width=bar_width * 0.92,
            label=patient,
            color=patient_colors[patient],
        )

    axis.set_title(
        title,
        fontsize=TITLE_FONTSIZE,
    )
    axis.set_ylabel(
        ylabel,
        fontsize=LABEL_FONTSIZE,
    )

    axis.set_xticks(group_centres)
    axis.set_xticklabels(
        CONDITION_ORDER,
        rotation=20,
        ha="right",
        fontsize=TICK_FONTSIZE,
    )

    axis.tick_params(
        axis="y",
        labelsize=TICK_FONTSIZE,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def save_individual_panel(
    table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_stem: str,
    patient_colors: dict[str, object],
) -> None:
    figure, axis = plt.subplots(
        figsize=(6.4, 4.6)
    )

    draw_grouped_panel(
        axis,
        table,
        metric,
        title,
        ylabel,
        patient_colors,
    )

    axis.legend(
        title="Patient",
        frameon=False,
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )

    figure.tight_layout()

    for extension in ("png", "pdf", "svg"):
        kwargs = (
            {"dpi": 300}
            if extension == "png"
            else {}
        )

        figure.savefig(
            OUT_FIG
            / f"{output_stem}_grouped.{extension}",
            **kwargs,
        )

    plt.close(figure)


def plot_qc(table: pd.DataFrame) -> None:
    OUT_FIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    patients = patient_order(table["patient"])

    default_palette = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    patient_colors = {
        patient: default_palette[
            index % len(default_palette)
        ]
        for index, patient in enumerate(patients)
    }

    panels = (
        (
            "st_median_genes_per_spot",
            "ST QC",
            "Median detected genes per spot",
            "st_qc",
        ),
        (
            "msi_median_features_per_pixel",
            "MSI QC",
            "Median detected features per pixel",
            "msi_qc",
        ),
        (
            "bcr_unique_clonotypes",
            "BCR QC",
            "Detected BCR clonotypes",
            "bcr_qc",
        ),
        (
            "tcr_unique_clonotypes",
            "TCR QC",
            "Detected TCR clonotypes",
            "tcr_qc",
        ),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9.0),
    )

    for axis, panel in zip(axes.flat, panels):
        metric, title, ylabel, output_stem = panel

        draw_grouped_panel(
            axis,
            table,
            metric,
            title,
            ylabel,
            patient_colors,
        )

        save_individual_panel(
            table,
            metric,
            title,
            ylabel,
            output_stem,
            patient_colors,
        )

    handles, labels = (
        axes.flat[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        title="Patient",
        frameon=False,
        loc="upper center",
        ncol=len(labels),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    for extension in ("png", "pdf", "svg"):
        kwargs = (
            {"dpi": 300}
            if extension == "png"
            else {}
        )

        figure.savefig(
            OUT_FIG
            / f"figure1_modality_qc_grouped.{extension}",
            **kwargs,
        )

    plt.close(figure)

    print(f"Wrote figures under: {OUT_FIG}")

    print(
        "MSI was not measured for no_molecule "
        "sections. Those entries remain NA and "
        "are not plotted as zero."
    )


def main() -> None:
    table = collect_qc_table()
    plot_qc(table)


if __name__ == "__main__":
    main()

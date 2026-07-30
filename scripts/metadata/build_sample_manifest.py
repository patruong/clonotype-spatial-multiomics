#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

LEGACY_METADATA = REPO_ROOT / "metadata" / "metadata_short.tsv"
OUTPUT = REPO_ROOT / "metadata" / "sample_manifest.tsv"

ST_ROOT = REPO_ROOT / "data" / "processed" / "st" / "space_ranger_outs"
MORPHOLOGY_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "st"
    / "morphology_annotations"
    / "all_annotations"
)

MSI_ROOT = REPO_ROOT / "data" / "processed" / "msi" / "fake_spaceranger"
MSI_BW_ROOT = MSI_ROOT / "msi_in_bw"
MSI_HE_ROOT = MSI_ROOT / "msi_in_he"

SVDJ_READS = REPO_ROOT / "data" / "raw" / "svdj" / "sma_vdj" / "reads"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def read_legacy_metadata() -> dict[str, dict[str, str]]:
    if not LEGACY_METADATA.is_file():
        raise FileNotFoundError(f"Missing metadata file: {LEGACY_METADATA}")

    with LEGACY_METADATA.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        if "capture_area_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"'capture_area_id' is missing from {LEGACY_METADATA}"
            )

        return {
            row["capture_area_id"].strip(): row
            for row in reader
            if row.get("capture_area_id", "").strip()
        }


def morphology_file(section_id: str) -> Path:
    # V13Y10-061_A1 -> Morphology_061_A1.csv
    section_suffix = section_id.split("-", maxsplit=1)[-1]
    return MORPHOLOGY_ROOT / f"Morphology_{section_suffix}.csv"


def main() -> None:
    legacy = read_legacy_metadata()

    st_sections = {
        path.name
        for path in ST_ROOT.iterdir()
        if path.is_dir()
    }

    section_ids = sorted(st_sections | set(legacy))

    fieldnames = [
        "capture_area_id",
        "barcode",
        "patient",
        "subtype",
        "matrix",
        "msi_target",
        "st_available",
        "filtered_h5_available",
        "spatial_directory_available",
        "msi_available",
        "msi_bw_available",
        "msi_he_available",
        "morphology_available",
        "svdj_raw_available",
        "in_metadata_short",
        "include_main_analysis",
        "used_fig1",
        "used_fig2",
        "used_fig3",
        "exclusion_reason",
        "notes",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for section_id in section_ids:
            metadata = legacy.get(section_id, {})
            barcode = metadata.get("barcode", "").strip()

            section_root = ST_ROOT / section_id
            filtered_h5 = section_root / "filtered_feature_bc_matrix.h5"
            spatial_directory = section_root / "spatial"

            msi_bw = MSI_BW_ROOT / section_id
            msi_he = MSI_HE_ROOT / section_id

            svdj_fasta = (
                SVDJ_READS / f"{barcode}.fasta"
                if barcode
                else None
            )

            in_legacy = section_id in legacy

            writer.writerow(
                {
                    "capture_area_id": section_id,
                    "barcode": barcode,
                    "patient": metadata.get("patient", ""),
                    "subtype": metadata.get("subtype", ""),
                    "matrix": metadata.get("matrix", ""),
                    "msi_target": metadata.get("MSI target", ""),
                    "st_available": yes_no(
                        filtered_h5.is_file()
                        and spatial_directory.is_dir()
                    ),
                    "filtered_h5_available": yes_no(
                        filtered_h5.is_file()
                    ),
                    "spatial_directory_available": yes_no(
                        spatial_directory.is_dir()
                    ),
                    "msi_available": yes_no(
                        msi_bw.is_dir() or msi_he.is_dir()
                    ),
                    "msi_bw_available": yes_no(msi_bw.is_dir()),
                    "msi_he_available": yes_no(msi_he.is_dir()),
                    "morphology_available": yes_no(
                        morphology_file(section_id).is_file()
                    ),
                    "svdj_raw_available": yes_no(
                        svdj_fasta is not None
                        and svdj_fasta.is_file()
                    ),
                    "in_metadata_short": yes_no(in_legacy),
                    "include_main_analysis": "",
                    "used_fig1": "",
                    "used_fig2": "",
                    "used_fig3": "",
                    "exclusion_reason": "",
                    "notes": (
                        ""
                        if in_legacy
                        else "Space Ranger data present but absent from metadata_short.tsv"
                    ),
                }
            )

    print(f"Wrote: {OUTPUT}")
    print(f"Sections: {len(section_ids)}")


if __name__ == "__main__":
    main()

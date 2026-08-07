#!/usr/bin/env python3
from __future__ import annotations

"""Figure 1f: clonotype sharing, abundance-retention curves and OLGA P(gen).

Inputs
------
metadata/metadata_short.tsv

data/processed/svdj/igdiscover/sma_vdj/{CHAIN}_clonotypes_members.tsv

Outputs
-------
results/qc/figure1_clonotype_sharing.tsv
results/qc/figure1_clonotype_retention.tsv
results/qc/figure1_pgen_values.tsv
results/qc/figure1_pgen_stats.tsv
results/figures/figure1/figure1f_bcr_sharing.{png,pdf,svg}
results/figures/figure1/figure1f_tcr_sharing.{png,pdf,svg}
results/figures/figure1/figure1f_clonotype_retention.{png,pdf,svg}
results/figures/figure1/figure1f_pgen_{IGH,IGK,IGL}.{png,pdf,svg}
results/figures/figure1/figure1f_preview.{png,pdf,svg}

Definitions
-----------
* BCR = IGH + IGK + IGL; TCR = TRA + TRB + TRD + TRG.
* Sharing Venns use chain-prefixed clonotype IDs (for example IGH:42), so
  clone_id values from different chains can never collide.
* Retention curves use total clonotype abundance = sum(count) across the
  members table for that chain. The curve reports the percentage of unique
  clonotypes retained at each minimum total-count threshold.
* P(gen) uses CDR3 amino-acid sequences (junction_aa). "Shared" means present
  in all cohort patients; "Private" means present in exactly one cohort patient.
  This matches the final legacy supp_olga_plot.py definition rather than the
  older clone_id-first Venn-region implementation.
* The legacy-compatible anchored-CDR3 filter is retained: C...W for IGH and
  C...F for the other chains.
* Main P(gen) statistics are computed for IGH, IGK and IGL using two-sided
  Mann-Whitney U tests, Benjamini-Hochberg correction across the three chains,
  and Cliff's delta derived from the Mann-Whitney U statistic.
"""

import os
import re
import sys
import time
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    import olga
    import olga.load_model as load_model
    import olga.generation_probability as generation_probability
except ImportError as exc:
    raise SystemExit(
        "OLGA is not importable. Install it in the sVDJ analysis environment."
    ) from exc

try:
    from venn import venn as venn_plot
except ImportError as exc:
    raise SystemExit(
        "The Python package 'venn' is not importable. Install it in the sVDJ analysis environment."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "metadata" / "metadata_short.tsv"
VDJ_ROOT = ROOT / "data" / "processed" / "svdj" / "igdiscover" / "sma_vdj"
QC_ROOT = ROOT / "results" / "qc"
FIG_ROOT = ROOT / "results" / "figures" / "figure1"

BCR_CHAINS = ("IGH", "IGK", "IGL")
TCR_CHAINS = ("TRA", "TRB", "TRD", "TRG")
ALL_CHAINS = BCR_CHAINS + TCR_CHAINS
PGEN_CHAINS = BCR_CHAINS

RETENTION_THRESHOLDS = tuple(range(1, 11))
REFERENCE_THRESHOLD = 3
REQUIRE_ANCHORED_CDR3 = True
RANDOM_SEED = 0


def available_cpus() -> int:
    """Return CPUs available to this process, respecting affinity where possible."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# OLGA is CPU-bound and does not use the GPU.
# Default to 16 worker processes, capped by available CPUs.
# Override at runtime with:
#   SMA_VDJ_PGEN_WORKERS=8 python ...
PGEN_WORKERS = max(
    1,
    min(
        int(os.environ.get("SMA_VDJ_PGEN_WORKERS", "16")),
        available_cpus(),
    ),
)

PGEN_CHUNKSIZE = 32
PGEN_PROGRESS_EVERY = 500

# Read/process different receptor-chain tables concurrently.
# There are seven chains, so at most seven preprocessing workers are useful.
# Override with SMA_VDJ_PREPROCESS_WORKERS=<N>.
PREPROCESS_WORKERS = max(
    1,
    min(
        int(os.environ.get("SMA_VDJ_PREPROCESS_WORKERS", "7")),
        available_cpus(),
        len(ALL_CHAINS),
    ),
)

# Intentionally large: panels are scaled down during Illustrator assembly.
TITLE_FONTSIZE = 19
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 13
ANNOTATION_FONTSIZE = 13


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)


def load_metadata() -> pd.DataFrame:
    md = pd.read_csv(METADATA, sep="\t", dtype=str)
    required = {"barcode", "patient", "subtype", "capture_area_id"}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    md = md[["capture_area_id", "barcode", "patient", "subtype"]].copy()
    for col in md.columns:
        md[col] = md[col].astype(str).str.strip()

    if md["barcode"].duplicated().any():
        raise ValueError("metadata_short.tsv must contain one row per barcode")

    return md


def extract_barcode(sequence_id: pd.Series) -> pd.Series:
    return sequence_id.astype(str).str.extract(r"^(bc\d{4})", expand=False)


def load_members(chain: str) -> pd.DataFrame:
    path = VDJ_ROOT / f"{chain}_clonotypes_members.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)

    table = pd.read_csv(
        path,
        sep="\t",
        usecols=["sequence_id", "clone_id", "count", "junction_aa"],
        dtype={"sequence_id": str, "clone_id": str, "junction_aa": str},
        low_memory=False,
    )
    table["barcode"] = extract_barcode(table["sequence_id"])
    table["count"] = pd.to_numeric(table["count"], errors="coerce").fillna(0.0)
    table = table.dropna(subset=["barcode", "clone_id"])
    return table


def cohort_members(chain: str, metadata: pd.DataFrame) -> pd.DataFrame:
    table = load_members(chain)
    table = table[table["barcode"].isin(set(metadata["barcode"]))].copy()
    return table.merge(
        metadata[["barcode", "patient"]],
        on="barcode",
        how="inner",
        validate="many_to_one",
    )


# -----------------------------------------------------------------------------
# Sharing Venns
# -----------------------------------------------------------------------------
def patient_clonotype_sets(
    metadata: pd.DataFrame,
    chains: Iterable[str],
    preprocessed: dict[str, dict[str, object]],
) -> dict[str, set[str]]:
    """Combine precomputed chain-specific clonotype sets."""
    patients = sorted(
        metadata["patient"].unique(),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    sets_by_patient = {patient: set() for patient in patients}

    for chain in chains:
        chain_sets = preprocessed[chain]["patient_clonotypes"]
        for patient in patients:
            sets_by_patient[patient].update(
                chain_sets.get(patient, set())
            )

    return sets_by_patient



def sharing_region_table(
    receptor: str,
    sets_by_patient: dict[str, set[str]],
) -> pd.DataFrame:
    patients = sorted(sets_by_patient, key=lambda x: int(x) if x.isdigit() else x)
    rows: list[dict[str, object]] = []

    # Every non-empty exact membership pattern among the cohort patients.
    union = set().union(*(sets_by_patient[p] for p in patients))
    for feature in union:
        present = tuple(p for p in patients if feature in sets_by_patient[p])
        rows.append(
            {
                "receptor": receptor,
                "membership": "&".join(present),
                "n_patients": len(present),
                "feature_id": feature,
            }
        )

    return pd.DataFrame(rows)


def plot_venn(
    sets_by_patient: dict[str, set[str]],
    receptor: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(7.0, 6.4))
    else:
        fig = ax.figure

    data = {f"Patient {p}": sets_by_patient[p] for p in sorted(sets_by_patient, key=lambda x: int(x) if x.isdigit() else x)}
    venn_plot(data, ax=ax)
    ax.set_title(f"{receptor} clonotype sharing", fontsize=TITLE_FONTSIZE)

    # The venn package creates text artists internally; enlarge them for panel use.
    for text in ax.texts:
        text.set_fontsize(TICK_FONTSIZE)

    if own_figure:
        fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Retention curves
# -----------------------------------------------------------------------------
def calculate_retention(
    preprocessed: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Combine retention summaries calculated during one-pass preprocessing."""
    frames = [
        pd.DataFrame(preprocessed[chain]["retention_rows"])
        for chain in ALL_CHAINS
    ]
    return pd.concat(frames, ignore_index=True)



def draw_retention(ax: plt.Axes, retention: pd.DataFrame) -> None:
    for chain in ALL_CHAINS:
        sub = retention[retention["chain"] == chain].sort_values("threshold")
        ref = sub.loc[sub["threshold"] == REFERENCE_THRESHOLD, "percent_retained"]
        ref_value = float(ref.iloc[0]) if len(ref) else np.nan
        ax.plot(
            sub["threshold"],
            sub["percent_retained"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=f"{chain} (ret@{REFERENCE_THRESHOLD}={ref_value:.1f}%)",
        )

    ax.axvline(REFERENCE_THRESHOLD, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Minimum total clonotype count", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Clonotypes retained (%)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Clonotype retention", fontsize=TITLE_FONTSIZE)
    ax.set_xlim(min(RETENTION_THRESHOLDS), max(RETENTION_THRESHOLDS))
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_retention(retention: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    draw_retention(ax, retention)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# P(gen)
# -----------------------------------------------------------------------------
_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def clean_aa(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    sequence = re.sub(r"[.\-\s*]", "", str(value)).upper()
    if not sequence:
        return None

    # Legacy-compatible behavior: unexpected alphabetic residues become X;
    # non-alphabetic characters invalidate the sequence.
    cleaned: list[str] = []
    for residue in sequence:
        if residue in _AA:
            cleaned.append(residue)
        elif "A" <= residue <= "Z":
            cleaned.append("X")
        else:
            return None
    return "".join(cleaned)


def anchored_ok(sequence: str, chain: str) -> bool:
    if not sequence or not sequence.startswith("C"):
        return False
    return sequence.endswith("W") if chain == "IGH" else sequence.endswith("F")


def preprocess_chain(
    chain: str,
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Read one members table once and derive all Figure 1f summaries.

    This function intentionally performs sharing, retention and P(gen)
    preprocessing in the same pass so the large members TSV is not repeatedly
    read and parsed.

    CDR3 cleaning is performed once per unique raw junction_aa rather than once
    per members-table row.
    """
    table = load_members(chain)

    cohort_barcodes = set(metadata["barcode"])
    table = table[table["barcode"].isin(cohort_barcodes)].copy()

    table = table.merge(
        metadata[["barcode", "patient"]],
        on="barcode",
        how="inner",
        validate="many_to_one",
    )

    patients = sorted(
        metadata["patient"].unique(),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    # --------------------------------------------------------------
    # Clonotype-sharing summary
    # --------------------------------------------------------------
    patient_clonotypes: dict[str, set[str]] = {
        patient: set() for patient in patients
    }

    for patient, group in table.groupby(
        "patient",
        sort=False,
        observed=True,
    ):
        feature_ids = (
            chain + ":" + group["clone_id"].astype(str)
        ).unique()
        patient_clonotypes[str(patient)] = set(feature_ids)

    # --------------------------------------------------------------
    # Retention summary
    # --------------------------------------------------------------
    clone_counts = (
        table.groupby("clone_id", observed=True)["count"]
        .sum()
    )
    n_total = int(len(clone_counts))

    retention_rows: list[dict[str, object]] = []
    for threshold in RETENTION_THRESHOLDS:
        n_retained = int((clone_counts >= threshold).sum())
        percent = (
            100.0 * n_retained / n_total
            if n_total
            else np.nan
        )

        retention_rows.append(
            {
                "chain": chain,
                "threshold": threshold,
                "n_total_clonotypes": n_total,
                "n_retained": n_retained,
                "percent_retained": percent,
            }
        )

    # --------------------------------------------------------------
    # P(gen) preprocessing
    # --------------------------------------------------------------
    pgen_patient_sets: dict[str, set[str]] = {
        patient: set() for patient in patients
    }

    n_unique_raw_cdr3 = 0
    n_valid_cdr3 = 0

    if chain in PGEN_CHAINS:
        # First collapse millions of members rows down to unique
        # patient/raw-CDR3 combinations.
        pairs = (
            table[["patient", "junction_aa"]]
            .dropna(subset=["junction_aa"])
            .drop_duplicates()
            .copy()
        )

        # clean_aa() is Python-level work.  Calling it once per UNIQUE
        # junction rather than once per members-table row is substantially
        # faster for large repertoires.
        unique_raw = pairs["junction_aa"].drop_duplicates().tolist()
        n_unique_raw_cdr3 = len(unique_raw)

        clean_map = {
            raw: clean_aa(raw)
            for raw in unique_raw
        }

        pairs["cdr3_aa"] = pairs["junction_aa"].map(clean_map)
        pairs = pairs.dropna(subset=["cdr3_aa"])

        if REQUIRE_ANCHORED_CDR3:
            valid_sequences = {
                sequence
                for sequence in pairs["cdr3_aa"].unique()
                if anchored_ok(str(sequence), chain)
            }
            pairs = pairs[
                pairs["cdr3_aa"].isin(valid_sequences)
            ].copy()

        # Different raw representations can clean to the same AA sequence.
        pairs = pairs.drop_duplicates(
            subset=["patient", "cdr3_aa"]
        )

        n_valid_cdr3 = int(pairs["cdr3_aa"].nunique())

        for patient, group in pairs.groupby(
            "patient",
            sort=False,
            observed=True,
        ):
            pgen_patient_sets[str(patient)] = set(
                group["cdr3_aa"].astype(str)
            )

    return {
        "chain": chain,
        "n_member_rows": int(len(table)),
        "n_clonotypes": n_total,
        "n_unique_raw_cdr3": n_unique_raw_cdr3,
        "n_valid_cdr3": n_valid_cdr3,
        "patient_clonotypes": patient_clonotypes,
        "retention_rows": retention_rows,
        "pgen_patient_sets": pgen_patient_sets,
    }


def preprocess_all_chains(
    metadata: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Preprocess all receptor chains concurrently before plotting/OLGA."""
    workers = min(
        PREPROCESS_WORKERS,
        len(ALL_CHAINS),
    )

    print(
        f"Preprocessing {len(ALL_CHAINS)} receptor chains "
        f"with {workers} workers...",
        flush=True,
    )

    started = time.perf_counter()
    results: dict[str, dict[str, object]] = {}

    # spawn is portable and keeps this repository independent of
    # Linux-specific fork semantics.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                preprocess_chain,
                chain,
                metadata,
            ): chain
            for chain in ALL_CHAINS
        }

        for future in as_completed(futures):
            chain = futures[future]
            result = future.result()
            results[chain] = result

            message = (
                f"  {chain}: "
                f"{result['n_member_rows']:,} cohort rows, "
                f"{result['n_clonotypes']:,} clonotypes"
            )

            if chain in PGEN_CHAINS:
                message += (
                    f", {result['n_unique_raw_cdr3']:,} unique raw CDR3s, "
                    f"{result['n_valid_cdr3']:,} valid CDR3s"
                )

            print(message, flush=True)

    elapsed = time.perf_counter() - started

    print(
        f"Preprocessing complete in {elapsed:.1f} s",
        flush=True,
    )

    return results



def shared_private(sets_by_patient: dict[str, set[str]]) -> tuple[set[str], set[str]]:
    patients = list(sets_by_patient)
    if not patients:
        return set(), set()

    shared = set.intersection(*(sets_by_patient[p] for p in patients))
    counts: Counter[str] = Counter()
    for patient in patients:
        counts.update(sets_by_patient[patient])
    private = {sequence for sequence, n in counts.items() if n == 1}
    return shared, private


def find_olga_models() -> Path:
    candidates: list[Path] = []

    env_path = os.environ.get("OLGA_DEFAULT_MODELS")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path(olga.__file__).resolve().parent / "default_models")
    candidates.append(ROOT / "data" / "references" / "olga" / "default_models")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate OLGA default_models. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_olga_model(chain: str):
    model_map = {
        "IGH": "human_B_heavy",
        "IGK": "human_B_kappa",
        "IGL": "human_B_lambda",
        "TRA": "human_T_alpha",
        "TRB": "human_T_beta",
        "TRD": "human_T_delta",
        "TRG": "human_T_gamma",
    }
    model_dir = find_olga_models() / model_map[chain]

    params = model_dir / "model_params.txt"
    marginals = model_dir / "model_marginals.txt"

    def anchors(stem: str) -> Path:
        for suffix in (".txt", ".csv"):
            candidate = model_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing {stem}.txt/.csv in {model_dir}")

    v_anchor = anchors("V_gene_CDR3_anchors")
    j_anchor = anchors("J_gene_CDR3_anchors")

    if chain in {"IGH", "TRB", "TRD"}:
        genomic = load_model.GenomicDataVDJ()
        genomic.load_igor_genomic_data(str(params), str(v_anchor), str(j_anchor))
        generative = load_model.GenerativeModelVDJ()
        generative.load_and_process_igor_model(str(marginals))
        return generation_probability.GenerationProbabilityVDJ(generative, genomic)

    genomic = load_model.GenomicDataVJ()
    genomic.load_igor_genomic_data(str(params), str(v_anchor), str(j_anchor))
    generative = load_model.GenerativeModelVJ()
    generative.load_and_process_igor_model(str(marginals))
    return generation_probability.GenerationProbabilityVJ(generative, genomic)


_WORKER_OLGA_MODEL = None


def init_pgen_worker(chain: str) -> None:
    """Load one OLGA model per worker process."""
    global _WORKER_OLGA_MODEL
    _WORKER_OLGA_MODEL = load_olga_model(chain)


def score_one_sequence(sequence: str) -> tuple[str, float, float, str]:
    """Score one CDR3 amino-acid sequence using the worker-local OLGA model."""
    if _WORKER_OLGA_MODEL is None:
        raise RuntimeError("OLGA worker model has not been initialized")

    try:
        probability = float(
            _WORKER_OLGA_MODEL.compute_aa_CDR3_pgen(sequence)
        )
        log10_probability = (
            np.log10(probability) if probability > 0 else np.nan
        )
        note = "ok" if probability > 0 else "zero_pgen"
    except Exception as exc:
        probability = np.nan
        log10_probability = np.nan
        note = f"error:{type(exc).__name__}"

    return sequence, probability, log10_probability, note


def score_sequences(
    chain: str,
    group: str,
    sequences: Iterable[str],
) -> pd.DataFrame:
    sequences = sorted(set(sequences))
    n_sequences = len(sequences)

    if n_sequences == 0:
        return pd.DataFrame(
            columns=[
                "chain",
                "group",
                "sequence",
                "pgen",
                "log10_pgen",
                "note",
            ]
        )

    # Avoid multiprocessing overhead for very small shared sets.
    workers = (
        1
        if n_sequences < 250
        else min(PGEN_WORKERS, n_sequences)
    )

    print(
        f"{chain} {group}: scoring {n_sequences:,} CDR3s "
        f"with {workers} worker{'s' if workers != 1 else ''}",
        flush=True,
    )

    rows: list[dict[str, object]] = []

    if workers == 1:
        model = load_olga_model(chain)

        for index, sequence in enumerate(sequences, start=1):
            try:
                probability = float(
                    model.compute_aa_CDR3_pgen(sequence)
                )
                log10_probability = (
                    np.log10(probability)
                    if probability > 0
                    else np.nan
                )
                note = "ok" if probability > 0 else "zero_pgen"
            except Exception as exc:
                probability = np.nan
                log10_probability = np.nan
                note = f"error:{type(exc).__name__}"

            rows.append(
                {
                    "chain": chain,
                    "group": group,
                    "sequence": sequence,
                    "pgen": probability,
                    "log10_pgen": log10_probability,
                    "note": note,
                }
            )

            if (
                index % PGEN_PROGRESS_EVERY == 0
                or index == n_sequences
            ):
                print(
                    f"  {chain} {group}: "
                    f"{index:,}/{n_sequences:,}",
                    flush=True,
                )

    else:
        # "spawn" is portable across Linux, macOS and Windows and avoids
        # depending on host-specific fork behaviour.
        context = mp.get_context("spawn")

        with context.Pool(
            processes=workers,
            initializer=init_pgen_worker,
            initargs=(chain,),
        ) as pool:
            iterator = pool.imap(
                score_one_sequence,
                sequences,
                chunksize=PGEN_CHUNKSIZE,
            )

            for index, result in enumerate(iterator, start=1):
                sequence, probability, log10_probability, note = result

                rows.append(
                    {
                        "chain": chain,
                        "group": group,
                        "sequence": sequence,
                        "pgen": probability,
                        "log10_pgen": log10_probability,
                        "note": note,
                    }
                )

                if (
                    index % PGEN_PROGRESS_EVERY == 0
                    or index == n_sequences
                ):
                    print(
                        f"  {chain} {group}: "
                        f"{index:,}/{n_sequences:,}",
                        flush=True,
                    )

    return pd.DataFrame(rows)


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def calculate_pgen(
    metadata: pd.DataFrame,
    preprocessed: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_tables: list[pd.DataFrame] = []
    stats_rows: list[dict[str, object]] = []

    for chain in PGEN_CHAINS:
        sets_by_patient = preprocessed[chain]["pgen_patient_sets"]
        shared, private = shared_private(sets_by_patient)
        print(f"{chain}: shared={len(shared)} private={len(private)}")

        shared_scores = score_sequences(chain, "Shared", shared)
        private_scores = score_sequences(chain, "Private", private)
        values = pd.concat([shared_scores, private_scores], ignore_index=True)
        value_tables.append(values)

        shared_values = shared_scores["log10_pgen"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        private_values = private_scores["log10_pgen"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if len(shared_values) and len(private_values):
            test = mannwhitneyu(
                shared_values,
                private_values,
                alternative="two-sided",
                method="asymptotic",
            )
            u_stat = float(test.statistic)
            p_value = float(test.pvalue)
            cliff_delta = 2.0 * u_stat / (len(shared_values) * len(private_values)) - 1.0
        else:
            u_stat = np.nan
            p_value = np.nan
            cliff_delta = np.nan

        stats_rows.append(
            {
                "chain": chain,
                "n_patients": len(sets_by_patient),
                "patients": ",".join(sorted(sets_by_patient, key=lambda x: int(x) if x.isdigit() else x)),
                "n_shared_sequences": len(shared),
                "n_private_sequences": len(private),
                "n_shared_scored": len(shared_values),
                "n_private_scored": len(private_values),
                "shared_median_log10_pgen": float(np.median(shared_values)) if len(shared_values) else np.nan,
                "private_median_log10_pgen": float(np.median(private_values)) if len(private_values) else np.nan,
                "mannwhitney_u": u_stat,
                "p_value": p_value,
                "cliffs_delta": cliff_delta,
            }
        )

    values = pd.concat(value_tables, ignore_index=True)
    stats = pd.DataFrame(stats_rows)

    valid = stats["p_value"].notna()
    stats["q_value_bh"] = np.nan
    if valid.any():
        stats.loc[valid, "q_value_bh"] = benjamini_hochberg(stats.loc[valid, "p_value"])

    return values, stats


def format_q(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def draw_pgen(
    ax: plt.Axes,
    chain: str,
    values: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    sub = values[values["chain"] == chain]
    shared = sub.loc[sub["group"] == "Shared", "log10_pgen"].dropna().to_numpy()
    private = sub.loc[sub["group"] == "Private", "log10_pgen"].dropna().to_numpy()

    ax.boxplot(
        [shared, private],
        labels=["Shared", "Private"],
        showfliers=False,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    for index, array in enumerate((shared, private), start=1):
        if len(array):
            x = rng.normal(index, 0.045, len(array))
            ax.scatter(x, array, s=9, alpha=0.35)

    stat = stats.loc[stats["chain"] == chain].iloc[0]
    ax.set_title(f"{chain} P(gen)", fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(r"$\log_{10}$ P(gen)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.03,
        0.97,
        f"q={format_q(float(stat['q_value_bh']))}\nδ={float(stat['cliffs_delta']):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
    )


def plot_pgen_chain(chain: str, values: pd.DataFrame, stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    draw_pgen(ax, chain, values, stats)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    print("Cohort patients:", ", ".join(sorted(metadata["patient"].unique())))
    print("Cohort sections:", len(metadata))

    # One parallel preprocessing pass over all seven members tables.
    # Everything below reuses these summaries.
    preprocessed = preprocess_all_chains(metadata)

    bcr_sets = patient_clonotype_sets(
        metadata,
        BCR_CHAINS,
        preprocessed,
    )
    tcr_sets = patient_clonotype_sets(
        metadata,
        TCR_CHAINS,
        preprocessed,
    )

    sharing = pd.concat(
        [
            sharing_region_table("BCR", bcr_sets),
            sharing_region_table("TCR", tcr_sets),
        ],
        ignore_index=True,
    )
    sharing.to_csv(QC_ROOT / "figure1_clonotype_sharing.tsv", sep="\t", index=False)

    bcr_fig = plot_venn(bcr_sets, "BCR")
    save_figure(bcr_fig, FIG_ROOT / "figure1f_bcr_sharing")
    plt.close(bcr_fig)

    tcr_fig = plot_venn(tcr_sets, "TCR")
    save_figure(tcr_fig, FIG_ROOT / "figure1f_tcr_sharing")
    plt.close(tcr_fig)

    retention = calculate_retention(preprocessed)
    retention.to_csv(QC_ROOT / "figure1_clonotype_retention.tsv", sep="\t", index=False)
    retention_fig = plot_retention(retention)
    save_figure(retention_fig, FIG_ROOT / "figure1f_clonotype_retention")
    plt.close(retention_fig)

    pgen_values, pgen_stats = calculate_pgen(
        metadata,
        preprocessed,
    )
    pgen_values.to_csv(QC_ROOT / "figure1_pgen_values.tsv", sep="\t", index=False)
    pgen_stats.to_csv(QC_ROOT / "figure1_pgen_stats.tsv", sep="\t", index=False)

    for chain in PGEN_CHAINS:
        fig = plot_pgen_chain(chain, pgen_values, pgen_stats)
        save_figure(fig, FIG_ROOT / f"figure1f_pgen_{chain}")
        plt.close(fig)

    # Current manuscript-style preview: two sharing Venns + IGH on top;
    # retention + IGL + IGK on bottom.
    preview, axes = plt.subplots(2, 3, figsize=(19.5, 11.5))
    plot_venn(bcr_sets, "BCR", ax=axes[0, 0])
    plot_venn(tcr_sets, "TCR", ax=axes[0, 1])
    draw_pgen(axes[0, 2], "IGH", pgen_values, pgen_stats)
    draw_retention(axes[1, 0], retention)
    draw_pgen(axes[1, 1], "IGL", pgen_values, pgen_stats)
    draw_pgen(axes[1, 2], "IGK", pgen_values, pgen_stats)
    preview.tight_layout()
    save_figure(preview, FIG_ROOT / "figure1f_preview")
    plt.close(preview)

    print(f"Wrote QC tables under: {QC_ROOT}")
    print(f"Wrote Figure 1f panels under: {FIG_ROOT}")
    print("\nP(gen) statistics:")
    print(
        pgen_stats[
            [
                "chain",
                "n_shared_scored",
                "n_private_scored",
                "shared_median_log10_pgen",
                "private_median_log10_pgen",
                "p_value",
                "q_value_bh",
                "cliffs_delta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
METADATA = ROOT / "metadata" / "metadata_short.tsv"
MSI_ROOT = ROOT / "data" / "processed" / "msi" / "fake_spaceranger" / "msi_in_he"
BLACKLIST_DIR = ROOT / "config" / "msi" / "blacklists"
METID_DIR = ROOT / "metadata" / "msi" / "metid"
DATA_ROOT = ROOT / "data" / "intermediate" / "msi" / "nmf"
OUT_ROOT = ROOT / "results" / "msi" / "nmf_selected"

RESIDUAL_9AA_TECH_MZ = [
    230.037769, 230.050789,
    458.088403, 459.114962, 461.111910, 462.081750,
    496.095095, 498.091876,
    556.050378, 557.044272,
    266.033787, 268.030862, 269.024547, 287.012383,
    323.992377, 325.989519,
    384.941755, 423.138002,
    184.863671,
    144.869692, 146.866733, 128.892164, 130.889226,
]

BAD_NAME_PATTERNS = [
    "eprosartan", "hydrochloride", "cefepime", "cefpirome", "kbt",
    "nafimidone", "aminoanthracene", "dichloro", "chloromaleyl",
    "acid green", "pesticide", "insecticide", "drug", "quinalizarin",
]

GOOD_NAME_PATTERNS = [
    "glutathione", "inositol", "fructose", "phosphate", "adenosine",
    "amp", "taurine", "citrate", "gluconic", "diketogulonate",
    "ascorb", "pi(", "pg(", "phosphatidyl", "phosphohexose",
]


def infer_matrix(row):
    text = " ".join(str(row.get(c, "")) for c in ["matrix", "MSI target"]).upper()
    if "9AA" in text or "9-AA" in text:
        return "9AA"
    if "DHB" in text:
        return "DHBA"
    return "UNKNOWN"


def resolve_blacklist(matrix_name: str, override: str | None):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    patterns = [
        f"{matrix_name}*nmf*informed*.txt",
        f"{matrix_name}*exclude*mz*.txt",
        f"{matrix_name}*blacklist*mz*.txt",
        f"{matrix_name}*.txt",
    ]
    for pattern in patterns:
        hits = sorted(p for p in BLACKLIST_DIR.glob(pattern) if p.is_file())
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"No blacklist found for {matrix_name} under {BLACKLIST_DIR}")


def read_mz_list(path: Path | None):
    if path is None:
        return []
    vals = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        for token in re.split(r"[,\s]+", line):
            if not token:
                continue
            try:
                vals.append(float(token))
            except ValueError:
                pass
    return sorted(set(vals))


def read_positions(sample_dir: Path, barcodes):
    path = sample_dir / "spatial" / "tissue_positions.csv"
    try:
        pos = pd.read_csv(path)
        if "barcode" not in pos.columns:
            raise ValueError
    except Exception:
        pos = pd.read_csv(
            path,
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col",
                   "pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    pos["barcode"] = pos["barcode"].astype(str)
    return pos.set_index("barcode").reindex(pd.Index(barcodes).astype(str))


def load_hires(sample_dir: Path, barcodes):
    pos = read_positions(sample_dir, barcodes)
    with open(sample_dir / "spatial" / "scalefactors_json.json") as f:
        sf = json.load(f)
    scale = float(sf["tissue_hires_scalef"])
    spot_diam = float(sf.get("spot_diameter_fullres", 100.0)) * scale
    x = pd.to_numeric(pos["pxl_col_in_fullres"], errors="coerce").to_numpy() * scale
    y = pd.to_numeric(pos["pxl_row_in_fullres"], errors="coerce").to_numpy() * scale
    img = plt.imread(sample_dir / "spatial" / "tissue_hires_image.png")
    return x, y, img, spot_diam


def robust01(v, p_low=1, p_high=99, gamma=0.7):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanpercentile(v, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0, 1) ** gamma


def savefig(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def near_any(mz, targets, ppm):
    if not np.isfinite(mz):
        return False
    for t in targets:
        if t > 0 and abs(mz - t) / t * 1e6 <= ppm:
            return True
    return False


EXPECTED_METID_COLUMNS = [
    "Observed m/z", "Adjusted m/z", "Matched name", "Adduct", "Formula",
    "Matched delta mass (Da)", "Matched difference (ppm)",
    "Matched theoretical mass", "Coverage",
]


def read_metid_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        try:
            next(reader)
        except StopIteration:
            return pd.DataFrame(columns=EXPECTED_METID_COLUMNS)

        for row in reader:
            if not row or all(str(x).strip() == "" for x in row):
                continue
            if len(row) == 9:
                vals = row
            elif len(row) > 9:
                vals = [row[0], row[1], ";".join(row[2:len(row)-6])] + row[len(row)-6:]
            else:
                vals = row + [""] * (9 - len(row))
            rows.append(vals)

    df = pd.DataFrame(rows, columns=EXPECTED_METID_COLUMNS)
    df["Observed m/z"] = pd.to_numeric(df["Observed m/z"], errors="coerce")
    return df


def load_metid(matrix_name: str):
    if not METID_DIR.is_dir():
        return pd.DataFrame(columns=EXPECTED_METID_COLUMNS)

    parts = []
    for path in sorted(METID_DIR.glob("*2ppm*.csv")):
        name = path.name.upper()
        if matrix_name == "9AA" and not name.startswith("9AA"):
            continue
        if matrix_name == "DHBA" and not (name.startswith("DHBA") or name.startswith("DHB")):
            continue
        try:
            df = read_metid_csv(path)
            df["source_file"] = path.name
            parts.append(df)
        except Exception as exc:
            print(f"[WARN] could not read Met-ID {path.name}: {exc}", flush=True)

    if not parts:
        return pd.DataFrame(columns=EXPECTED_METID_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def annotate_mz(mz, metid: pd.DataFrame, ppm=2.0):
    if metid.empty or not np.isfinite(mz):
        return "", ""
    obs = pd.to_numeric(metid["Observed m/z"], errors="coerce").to_numpy()
    ok = np.isfinite(obs)
    if not ok.any():
        return "", ""

    ppm_err = np.full(len(obs), np.inf)
    ppm_err[ok] = np.abs(obs[ok] - mz) / np.maximum(mz, 1e-12) * 1e6
    hits = metid.loc[ppm_err <= ppm]
    if hits.empty:
        return "", ""

    names = "; ".join(dict.fromkeys(hits["Matched name"].astype(str)))
    adducts = "; ".join(dict.fromkeys(hits["Adduct"].astype(str)))
    return names, adducts


def score_k(H: pd.DataFrame, kept: pd.DataFrame, technical_mz, metid,
            top_n=25, technical_ppm=15.0, metid_ppm=2.0):
    mz_lookup = dict(zip(kept["feature_name"].astype(str),
                         pd.to_numeric(kept["mz"], errors="coerce")))
    rows = []

    for factor in H.index.astype(str):
        vals = pd.to_numeric(H.loc[factor], errors="coerce").fillna(0.0)
        top = vals.nlargest(min(top_n, len(vals)))
        denom = float(top.sum()) if float(top.sum()) > 0 else 1.0

        tech_w = 0.0
        bio_w = 0.0
        tech_hits, bio_hits = [], []

        for feature_name, loading in top.items():
            mz = float(mz_lookup.get(str(feature_name), np.nan))
            names, adducts = annotate_mz(mz, metid, ppm=metid_ppm)
            text = f"{names} {adducts}".lower()

            reasons = []
            is_tech = False
            if near_any(mz, technical_mz, technical_ppm):
                is_tech = True
                reasons.append("technical_mz")
            if "[m+cl]" in text or "[m+k-h2]" in text or "[m+na-h2]" in text:
                is_tech = True
                reasons.append("adduct")
            if any(p in text for p in BAD_NAME_PATTERNS):
                is_tech = True
                reasons.append("suspicious_name")

            is_bio = any(p in text for p in GOOD_NAME_PATTERNS)

            if is_tech:
                tech_w += float(loading)
                tech_hits.append(f"{mz:.6f}:{'/'.join(reasons)}")
            elif is_bio:
                bio_w += float(loading)
                bio_hits.append(f"{mz:.6f}")

        rows.append({
            "factor": factor,
            "technical_score": tech_w / denom,
            "bio_score": bio_w / denom,
            "n_top": int(len(top)),
            "technical_hits": ";".join(tech_hits[:20]),
            "bio_hits": ";".join(bio_hits[:20]),
        })

    return pd.DataFrame(rows).sort_values(
        ["technical_score", "bio_score"], ascending=[True, False]
    )


def choose_factors(scores, threshold):
    clean = scores[scores["technical_score"] <= threshold].copy()
    if len(clean) >= 3:
        chosen = clean.sort_values(
            ["technical_score", "bio_score"], ascending=[True, False]
        ).head(3)
        status = "PASS"
    else:
        chosen = scores.sort_values(
            ["technical_score", "bio_score"], ascending=[True, False]
        ).head(3)
        status = "FALLBACK_best3_not_all_clean"

    return status, chosen["factor"].tolist(), {
        "n_clean_factors": int(len(clean)),
        "selected_max_technical_score": float(chosen["technical_score"].max()),
        "selected_mean_technical_score": float(chosen["technical_score"].mean()),
        "selected_mean_bio_score": float(chosen["bio_score"].mean()),
    }


def plot_selected_rgb(sample_dir, W, selected, stem):
    x, y, img, spot_diam = load_hires(sample_dir, W.index)
    valid = np.isfinite(x) & np.isfinite(y)
    rgb = np.column_stack([robust01(W[f].to_numpy()) for f in selected])
    size = max((spot_diam * 0.20) ** 2, 3.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(img)
    ax.scatter(x[valid], y[valid], c=rgb[valid], s=size,
               linewidths=0, edgecolors="none", alpha=0.95)
    ax.set_axis_off()
    ax.set_title("MSI NMF selected RGB", fontsize=16)
    fig.text(0.75, 0.79, f"R: {selected[0]}", fontsize=12)
    fig.text(0.75, 0.75, f"G: {selected[1]}", fontsize=12)
    fig.text(0.75, 0.71, f"B: {selected[2]}", fontsize=12)
    savefig(fig, stem)


def plot_selected_factor(sample_dir, W, factor, stem):
    x, y, img, spot_diam = load_hires(sample_dir, W.index)
    valid = np.isfinite(x) & np.isfinite(y)
    values = W[factor].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(values[valid], [1, 99])
    size = max((spot_diam * 0.20) ** 2, 3.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(img)
    m = ax.scatter(
        x[valid], y[valid], c=values[valid], s=size, cmap="viridis",
        vmin=lo, vmax=hi, linewidths=0, edgecolors="none", alpha=0.95,
    )
    ax.set_axis_off()
    ax.set_title(factor, fontsize=16)
    cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=11)
    savefig(fig, stem)


def process_sample(row, args):
    sample = str(row["capture_area_id"])
    matrix_name = infer_matrix(row)
    if matrix_name == "UNKNOWN":
        raise ValueError(f"{sample}: could not infer matrix")

    sample_dir = MSI_ROOT / sample
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)

    data_sample = DATA_ROOT / sample
    if not data_sample.is_dir():
        raise FileNotFoundError(f"No MSI NMF output for {sample}: {data_sample}")

    override = args.blacklist_9aa if matrix_name == "9AA" else args.blacklist_dhba
    blacklist_path = resolve_blacklist(matrix_name, override)
    technical_mz = read_mz_list(blacklist_path)
    if matrix_name == "9AA":
        technical_mz = sorted(set(technical_mz + RESIDUAL_9AA_TECH_MZ))
    if args.technical_mz_file:
        technical_mz = sorted(set(technical_mz + read_mz_list(Path(args.technical_mz_file))))

    metid = load_metid(matrix_name)
    if metid.empty:
        print(f"[WARN] {sample}: no 2-ppm Met-ID tables found; scoring uses technical m/z only", flush=True)

    score_tables = []
    decisions = []

    for k_dir in sorted(data_sample.glob("k*"), key=lambda p: int(p.name[1:])):
        try:
            k = int(k_dir.name[1:])
        except ValueError:
            continue
        if args.ks and k not in args.ks:
            continue

        W_path = k_dir / "W.tsv.gz"
        H_path = k_dir / "H.tsv.gz"
        kept_path = k_dir / "kept_features.tsv.gz"
        if not (W_path.is_file() and H_path.is_file() and kept_path.is_file()):
            continue

        H = pd.read_csv(H_path, sep="\t", index_col=0)
        kept = pd.read_csv(kept_path, sep="\t")
        scores = score_k(
            H, kept, technical_mz, metid,
            top_n=args.top_n,
            technical_ppm=args.technical_ppm,
            metid_ppm=args.metid_ppm,
        )
        scores.insert(0, "K", k)
        score_tables.append(scores)

        status, selected, stats = choose_factors(scores, args.max_technical_score)
        decisions.append({
            "K": k,
            "status": status,
            "selected_factors": ",".join(selected),
            **stats,
        })

    if not decisions:
        raise RuntimeError(f"{sample}: no usable K outputs found")

    scores_all = pd.concat(score_tables, ignore_index=True)
    decisions_df = pd.DataFrame(decisions)

    passed = decisions_df[decisions_df["status"] == "PASS"].copy()
    if not passed.empty:
        best = passed.sort_values(
            ["K", "selected_mean_technical_score"], ascending=[True, True]
        ).iloc[0]
    else:
        best = decisions_df.sort_values(
            ["selected_max_technical_score", "selected_mean_technical_score", "K"],
            ascending=[True, True, True],
        ).iloc[0]

    best_k = int(best["K"])
    selected = str(best["selected_factors"]).split(",")
    W = pd.read_csv(data_sample / f"k{best_k}" / "W.tsv.gz", sep="\t", index_col=0)

    out = OUT_ROOT / sample
    out.mkdir(parents=True, exist_ok=True)
    scores_all.to_csv(out / "factor_scores.tsv", sep="\t", index=False)
    decisions_df.to_csv(out / "k_decisions.tsv", sep="\t", index=False)

    selected_txt = (
        f"sample: {sample}\n"
        f"matrix: {matrix_name}\n"
        f"best_K: {best_k}\n"
        f"status: {best['status']}\n"
        f"selected_factors: R={selected[0]}, G={selected[1]}, B={selected[2]}\n"
        f"selected_max_technical_score: {best['selected_max_technical_score']}\n"
        f"selected_mean_technical_score: {best['selected_mean_technical_score']}\n"
        f"selected_mean_bio_score: {best['selected_mean_bio_score']}\n"
        f"blacklist: {blacklist_path}\n"
    )
    (out / "selected_factors.txt").write_text(selected_txt)

    plot_selected_rgb(sample_dir, W, selected, out / "rgb_selected")
    for factor in selected:
        plot_selected_factor(sample_dir, W, factor, out / factor)

    print(
        f"[OK] {sample}: best K={best_k}; "
        f"R={selected[0]} G={selected[1]} B={selected[2]}",
        flush=True,
    )

    return {
        "sample": sample,
        "matrix": matrix_name,
        "best_K": best_k,
        "status": best["status"],
        "R": selected[0],
        "G": selected[1],
        "B": selected[2],
        "selected_max_technical_score": best["selected_max_technical_score"],
        "selected_mean_technical_score": best["selected_mean_technical_score"],
        "selected_mean_bio_score": best["selected_mean_bio_score"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--technical-ppm", type=float, default=15.0)
    ap.add_argument("--metid-ppm", type=float, default=2.0)
    ap.add_argument("--max-technical-score", type=float, default=0.35)
    ap.add_argument("--blacklist-9aa", default=None)
    ap.add_argument("--blacklist-dhba", default=None)
    ap.add_argument("--technical-mz-file", default=None)
    args = ap.parse_args()

    md = pd.read_csv(METADATA, sep="\t", dtype=str)
    target = md["MSI target"].fillna("").str.lower()
    md = md[~target.isin({"", "nan", "no_molecule", "none", "na"})].copy()

    if args.samples:
        wanted = set(args.samples)
        md = md[md["capture_area_id"].isin(wanted)].copy()
        missing = wanted - set(md["capture_area_id"])
        if missing:
            raise ValueError(f"Samples not in MSI manuscript cohort: {sorted(missing)}")

    rows = []
    for _, row in md.iterrows():
        sample = str(row["capture_area_id"])
        try:
            rows.append(process_sample(row, args))
        except Exception as exc:
            print(f"[FAILED] {sample}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"sample": sample, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_ROOT / "msi_nmf_selection_summary.tsv", sep="\t", index=False)
    print(f"\nDone. Selected MSI NMF results: {OUT_ROOT}")


if __name__ == "__main__":
    main()

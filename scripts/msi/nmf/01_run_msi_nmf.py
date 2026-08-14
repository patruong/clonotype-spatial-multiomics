#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parents[3]
METADATA = ROOT / "metadata" / "metadata_short.tsv"
MSI_ROOT = ROOT / "data" / "processed" / "msi" / "fake_spaceranger" / "msi_in_he"
BLACKLIST_DIR = ROOT / "config" / "msi" / "blacklists"
DATA_OUT = ROOT / "data" / "intermediate" / "msi" / "nmf"
FIG_OUT = ROOT / "results" / "msi" / "nmf"


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


def load_msi(sample_dir: Path):
    adata = sc.read_10x_mtx(
        sample_dir / "filtered_feature_bc_matrix",
        var_names="gene_symbols",
        make_unique=True,
    )
    adata.obs_names = adata.obs_names.astype(str)
    pos = read_positions(sample_dir, adata.obs_names)
    adata.obs = adata.obs.join(pos)
    return adata


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
            if len(hits) > 1:
                print(f"[WARN] multiple {matrix_name} blacklist matches for '{pattern}'; using {hits[-1].name}",
                      flush=True)
            return hits[-1]

    raise FileNotFoundError(
        f"No blacklist found for {matrix_name} under {BLACKLIST_DIR}. "
        f"Pass --blacklist-9aa or --blacklist-dhba explicitly."
    )


def read_mz_list(path: Path):
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


def extract_mz(value, mz_min=50.0, mz_max=3000.0):
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
    candidates = []
    for token in tokens:
        try:
            val = float(token)
        except ValueError:
            continue
        if mz_min <= val <= mz_max:
            candidates.append((val, "." in token or "e" in token.lower()))
    if not candidates:
        return np.nan
    decimals = [v for v, is_decimal in candidates if is_decimal]
    return float(decimals[0] if decimals else candidates[0][0])


def matches_blacklist(mz, targets, ppm):
    if not np.isfinite(mz):
        return False
    for target in targets:
        if target > 0 and abs(mz - target) / target * 1e6 <= ppm:
            return True
    return False


def background_mask(sample_dir: Path, pos: pd.DataFrame,
                    max_background_fraction=0.75,
                    border_frac=0.06,
                    patch_radius_frac=0.45):
    spatial_dir = sample_dir / "spatial"
    img = plt.imread(spatial_dir / "tissue_hires_image.png")
    arr = np.asarray(img, dtype=float)
    if arr.max() > 1.5:
        arr /= 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3]

    with open(spatial_dir / "scalefactors_json.json") as f:
        sf = json.load(f)
    scale = float(sf["tissue_hires_scalef"])
    spot_diam = float(sf.get("spot_diameter_fullres", 100.0)) * scale

    x = pd.to_numeric(pos["pxl_col_in_fullres"], errors="coerce").to_numpy() * scale
    y = pd.to_numeric(pos["pxl_row_in_fullres"], errors="coerce").to_numpy() * scale

    h, w = arr.shape[:2]
    ym = max(int(round(h * border_frac)), 1)
    xm = max(int(round(w * border_frac)), 1)
    border = np.zeros((h, w), dtype=bool)
    border[:ym, :] = True
    border[-ym:, :] = True
    border[:, :xm] = True
    border[:, -xm:] = True

    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    tissue_score = (maxc - minc) / np.maximum(maxc, 1e-12) + 0.5 * (1.0 - maxc)

    border_scores = tissue_score[border]
    score_cut = np.nanpercentile(border_scores, 70)
    bg_candidate = border & (tissue_score <= score_cut)
    bg_pixels = arr[bg_candidate]
    if len(bg_pixels) < 100:
        bg_pixels = arr[border]
    bg_rgb = np.nanmedian(bg_pixels, axis=0)

    dist = np.sqrt(np.sum((arr - bg_rgb[None, None, :]) ** 2, axis=-1))
    bg_dist = dist[bg_candidate]
    med = float(np.nanmedian(bg_dist))
    mad = float(np.nanmedian(np.abs(bg_dist - med)))
    dist_threshold = float(np.clip(med + 4.0 * 1.4826 * mad, 0.035, 0.18))

    r = max(int(round(spot_diam * patch_radius_frac)), 1)
    bg_frac = np.full(len(x), np.nan, dtype=float)
    for i, (xx, yy) in enumerate(zip(x, y)):
        if not np.isfinite(xx) or not np.isfinite(yy):
            continue
        xi, yi = int(round(xx)), int(round(yy))
        x0, x1 = max(0, xi-r), min(w, xi+r+1)
        y0, y1 = max(0, yi-r), min(h, yi+r+1)
        if x0 < x1 and y0 < y1:
            bg_frac[i] = float(np.mean(dist[y0:y1, x0:x1] <= dist_threshold))

    mask = (
        np.isfinite(x) & np.isfinite(y) &
        np.isfinite(bg_frac) &
        (bg_frac <= max_background_fraction)
    )
    stats = {
        "n_spots_before_mask": int(len(mask)),
        "n_spots_after_mask": int(mask.sum()),
        "background_rgb": [float(v) for v in bg_rgb],
        "background_dist_threshold": dist_threshold,
        "max_background_fraction": float(max_background_fraction),
    }
    return mask, stats, x, y, img, spot_diam


def filter_features(X, feature_names, blacklist, ppm, min_spot_presence):
    X = X.tocsr().astype(np.float64, copy=True) if sparse.issparse(X) else sparse.csr_matrix(
        np.asarray(X, dtype=np.float64)
    )

    if X.nnz:
        min_val = float(X.data.min())
        if min_val < -1e-9:
            raise ValueError(f"MSI intensities contain negative values, min={min_val}")
        X.data[X.data < 0] = 0.0
        X.eliminate_zeros()

    feature_names = np.asarray(feature_names).astype(str)
    mz = np.asarray([extract_mz(v) for v in feature_names], dtype=float)
    presence = np.asarray(X.getnnz(axis=0)).ravel()
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    blacklisted = np.asarray([matches_blacklist(v, blacklist, ppm) for v in mz], dtype=bool)

    keep = (presence >= min_spot_presence) & (variance > 0) & ~blacklisted

    report = pd.DataFrame({
        "feature_name": feature_names,
        "mz": mz,
        "spot_presence": presence,
        "variance": variance,
        "blacklisted": blacklisted,
        "kept": keep,
    })

    Xf = X[:, keep].tocsr()
    Xf.eliminate_zeros()
    return Xf, feature_names[keep], mz[keep], report


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


def plot_mask(img, x, y, mask, stem):
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(img)
    ax.scatter(x[~mask], y[~mask], s=8, alpha=0.20, linewidths=0, label="removed")
    ax.scatter(x[mask], y[mask], s=10, alpha=0.85, linewidths=0, label="kept")
    ax.set_axis_off()
    ax.set_title("MSI tissue mask", fontsize=16)
    ax.legend(frameon=False, fontsize=11)
    savefig(fig, stem)


def plot_factor(img, x, y, spot_diam, values, title, stem):
    valid = np.isfinite(x) & np.isfinite(y)
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values[valid], [1, 99])
    size = max((spot_diam * 0.25) ** 2, 3.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(img)
    m = ax.scatter(
        x[valid], y[valid], c=values[valid], s=size, cmap="viridis",
        vmin=lo, vmax=hi, linewidths=0, edgecolors="none", alpha=0.95,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=16)
    cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=11)
    savefig(fig, stem)


def plot_rgb(img, x, y, spot_diam, W, factor_names, stem):
    valid = np.isfinite(x) & np.isfinite(y)
    rgb = np.column_stack([robust01(W[:, i]) for i in range(3)])
    size = max((spot_diam * 0.25) ** 2, 3.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.imshow(img)
    ax.scatter(x[valid], y[valid], c=rgb[valid], s=size,
               linewidths=0, edgecolors="none", alpha=0.95)
    ax.set_axis_off()
    ax.set_title("MSI NMF RGB (first 3)", fontsize=16)
    fig.text(0.77, 0.79, f"R: {factor_names[0]}", fontsize=12)
    fig.text(0.77, 0.75, f"G: {factor_names[1]}", fontsize=12)
    fig.text(0.77, 0.71, f"B: {factor_names[2]}", fontsize=12)
    savefig(fig, stem)


def plot_loadings(H, names, stem, top_n):
    n = H.shape[0]
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, max(5, 3.4 * nrows)), squeeze=False)
    axes = axes.ravel()
    names = np.asarray(names).astype(str)

    for k in range(n):
        vals = np.asarray(H[k], dtype=float)
        idx = np.argsort(vals)[-top_n:]
        idx = idx[np.argsort(vals[idx])]
        labels = [s if len(s) <= 24 else s[:23] + "…" for s in names[idx]]
        ax = axes[k]
        ax.barh(np.arange(len(idx)), vals[idx])
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlabel("Loading", fontsize=11)
        ax.set_title(f"NMF{k+1}", fontsize=14, fontweight="bold", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, stem)


def run_sample(row, args):
    sample = str(row["capture_area_id"])
    barcode = str(row["barcode"])
    matrix_name = infer_matrix(row)

    if matrix_name == "UNKNOWN":
        raise ValueError(f"{sample}: could not infer MSI matrix from metadata")

    sample_dir = MSI_ROOT / sample
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)

    override = args.blacklist_9aa if matrix_name == "9AA" else args.blacklist_dhba
    blacklist_path = resolve_blacklist(matrix_name, override)
    blacklist = read_mz_list(blacklist_path)
    print(f"[{sample}] matrix={matrix_name} blacklist={blacklist_path.name} n={len(blacklist)}", flush=True)

    adata = load_msi(sample_dir)
    mask, mask_stats, x0, y0, img, spot_diam = background_mask(
        sample_dir,
        adata.obs,
        max_background_fraction=args.max_background_fraction,
    )

    data_dir = DATA_OUT / sample
    fig_dir = FIG_OUT / sample
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        plot_mask(img, x0, y0, mask, fig_dir / "spot_mask")

    adata = adata[mask, :].copy()
    x, y = x0[mask], y0[mask]

    X, feature_names, mz, report = filter_features(
        adata.X,
        adata.var_names.to_numpy(),
        blacklist=blacklist,
        ppm=args.blacklist_ppm,
        min_spot_presence=args.min_spot_presence,
    )
    report.to_csv(data_dir / "feature_filter_report.tsv.gz", sep="\t", index=False, compression="gzip")

    if X.shape[1] < max(args.ks):
        raise ValueError(f"{sample}: only {X.shape[1]} features remain for max K={max(args.ks)}")

    rows = []
    for k in args.ks:
        model = NMF(
            n_components=k,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            random_state=args.random_state,
            max_iter=args.max_iter,
            tol=1e-4,
        )

        t0 = time.perf_counter()
        W = model.fit_transform(X)
        H = model.components_
        runtime = time.perf_counter() - t0

        k_data = data_dir / f"k{k}"
        k_fig = fig_dir / f"k{k}"
        k_data.mkdir(parents=True, exist_ok=True)
        k_fig.mkdir(parents=True, exist_ok=True)

        cols = [f"NMF{i+1}" for i in range(k)]
        pd.DataFrame(W, index=adata.obs_names, columns=cols).to_csv(
            k_data / "W.tsv.gz", sep="\t", compression="gzip"
        )
        pd.DataFrame(H, index=cols, columns=feature_names).to_csv(
            k_data / "H.tsv.gz", sep="\t", compression="gzip"
        )
        pd.DataFrame({"feature_name": feature_names, "mz": mz}).to_csv(
            k_data / "kept_features.tsv.gz", sep="\t", index=False, compression="gzip"
        )

        metrics = {
            "sample": sample,
            "barcode": barcode,
            "matrix": matrix_name,
            "method": "rms_euclidean",
            "k": int(k),
            "random_state": int(args.random_state),
            "n_spots_before_mask": mask_stats["n_spots_before_mask"],
            "n_spots_after_mask": int(adata.n_obs),
            "n_features_after_filter": int(X.shape[1]),
            "blacklist_file": str(blacklist_path),
            "blacklist_n_mz": int(len(blacklist)),
            "blacklist_ppm": float(args.blacklist_ppm),
            "reconstruction_err": float(model.reconstruction_err_),
            "n_iter": int(model.n_iter_),
            "runtime_sec": float(runtime),
        }
        (k_data / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append(metrics)

        if not args.skip_plots:
            for i in range(k):
                plot_factor(img, x, y, spot_diam, W[:, i], f"NMF{i+1}", k_fig / f"NMF{i+1}")
            if k >= 3:
                plot_rgb(img, x, y, spot_diam, W, cols[:3], k_fig / "rgb_first3")
            plot_loadings(H, feature_names, k_fig / "top_loadings", args.top_n)

        print(
            f"[OK] {sample} K={k}: spots={adata.n_obs} features={X.shape[1]} runtime={runtime:.1f}s",
            flush=True,
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None, help="Optional capture_area_id subset")
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--min-spot-presence", type=int, default=3)
    ap.add_argument("--blacklist-ppm", type=float, default=15.0)
    ap.add_argument("--blacklist-9aa", default=None)
    ap.add_argument("--blacklist-dhba", default=None)
    ap.add_argument("--max-background-fraction", type=float, default=0.75)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    md = pd.read_csv(METADATA, sep="\t", dtype=str)
    required = {"capture_area_id", "barcode", "MSI target"}
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"metadata_short.tsv missing: {sorted(missing)}")

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
            rows.extend(run_sample(row, args))
        except Exception as exc:
            print(f"[FAILED] {sample}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"sample": sample, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FIG_OUT / "msi_nmf_summary.tsv", sep="\t", index=False)
    print(f"\nDone. Results: {FIG_OUT}")


if __name__ == "__main__":
    main()

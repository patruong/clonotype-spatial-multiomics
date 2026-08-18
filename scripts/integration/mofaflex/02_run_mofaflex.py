#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import mofaflex as mfl
import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "data/intermediate/integration/mofaflex/inputs"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/integration/mofaflex/f12"
)

DEFAULT_SAMPLES = [
    "bc2059_V13Y10-038_B1",
    "bc2004_V13Y10-060_B1",
]

VIEWS = ["ST", "MSI", "VDJ"]


def to_numpy(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def get_factors(model, group):
    try:
        out = model.get_factors(group=group)
        return to_numpy(out)
    except Exception:
        pass

    out = model.get_factors()

    if isinstance(out, dict):
        if group in out:
            return to_numpy(out[group])
        return to_numpy(next(iter(out.values())))

    return to_numpy(out)


def get_weights(model, view):
    try:
        return to_numpy(model.get_weights(view=view))
    except Exception:
        pass

    out = model.get_weights()

    if isinstance(out, dict):
        return to_numpy(out[view])

    return to_numpy(out)


def get_model_factor_names(model, n_factors):
    if hasattr(model, "factor_names"):
        try:
            return np.asarray(
                [str(x) for x in model.factor_names],
                dtype=object,
            )
        except Exception:
            pass

    if hasattr(model, "get_factor_names"):
        try:
            return np.asarray(
                [str(x) for x in model.get_factor_names()],
                dtype=object,
            )
        except Exception:
            pass

    return np.asarray(
        [f"Factor_{i + 1}" for i in range(n_factors)],
        dtype=object,
    )


def extract_r2(model, group, views):
    factor_r2 = None

    try:
        out = model.get_r2(total=False)

        if isinstance(out, dict):
            if group in out:
                out = out[group]
            else:
                out = next(iter(out.values()))

        factor_r2 = np.asarray(to_numpy(out), dtype=float)

    except Exception:
        pass

    if factor_r2 is None:
        factor_r2 = np.full(
            (0, len(views)),
            np.nan,
            dtype=float,
        )

    view_r2 = None

    try:
        out = model.get_r2(total=True)

        if (
            hasattr(out, "columns")
            and group in out.columns
        ):
            out = out[group]

        elif (
            hasattr(out, "index")
            and group in out.index
        ):
            out = out.loc[group]

        view_r2 = np.asarray(
            to_numpy(out),
            dtype=float,
        )

    except Exception:
        pass

    if view_r2 is None:
        if factor_r2.size:
            view_r2 = np.nansum(
                factor_r2,
                axis=0,
            )
        else:
            view_r2 = np.full(
                len(views),
                np.nan,
                dtype=float,
            )

    return factor_r2, view_r2


def load_views(sample_dir):
    views = {}

    for view in VIEWS:
        path = sample_dir / f"{view}.h5ad"

        if not path.exists():
            if view == "MSI":
                continue

            raise FileNotFoundError(
                f"Required view missing: {path}"
            )

        obj = ad.read_h5ad(path)

        if obj.obs_names.has_duplicates:
            raise RuntimeError(
                f"{path}: duplicate observation names"
            )

        if obj.var_names.has_duplicates:
            raise RuntimeError(
                f"{path}: duplicate feature names"
            )

        if "spatial" not in obj.obsm:
            raise RuntimeError(
                f"{path}: missing .obsm['spatial']"
            )

        views[view] = obj

    if "ST" not in views or "VDJ" not in views:
        raise RuntimeError(
            f"{sample_dir}: Figure 2 requires ST and VDJ views."
        )

    reference = views["ST"]

    for view, obj in views.items():
        if view == "ST":
            continue

        if obj.n_obs != reference.n_obs:
            raise RuntimeError(
                f"{view}: {obj.n_obs} observations; "
                f"ST has {reference.n_obs}"
            )

        if not np.array_equal(
            np.asarray(obj.obs_names, dtype=str),
            np.asarray(reference.obs_names, dtype=str),
        ):
            raise RuntimeError(
                f"{view}: observation ordering differs from ST"
            )

    return views


def run_sample(
    sample,
    input_root,
    output_root,
    n_factors,
    seed,
    max_epochs,
    device,
    overwrite,
):
    sample_dir = input_root / sample
    out_dir = output_root / sample

    if not sample_dir.exists():
        raise FileNotFoundError(sample_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = out_dir / "mofaflex_results.npz"
    model_path = out_dir / "mofaflex_model.pt"

    if result_path.exists() and not overwrite:
        print(f"[SKIP] {sample}")
        return

    views = load_views(sample_dir)

    print("\n" + "=" * 72)
    print(sample)

    active_views = list(views.keys())

    print("active views:", ", ".join(active_views))

    for view in active_views:
        print(
            f"{view}: "
            f"{views[view].n_obs} spots x "
            f"{views[view].n_vars} features"
        )

    # One group per tissue section.
    data = {
        sample: views
    }

    likelihoods = {
        view: "Normal"
        for view in active_views
    }

    weight_prior = {
        view: "Horseshoe"
        for view in active_views
    }

    # Historical Figure 2 configuration:
    # free factors only; no gene-program annotations.
    data_opts = mfl.DataOptions(
        layer=None,
        scale_per_group=True,
        covariates_obsm_key="spatial",
        covariates_obs_key=None,
        guiding_vars_obs_keys=None,
        use_obs="union",
        use_var="union",
        subset_var=None,
        plot_data_overview=False,
        remove_constant_features=True,
        annotations_varm_key=None,
    )

    model_opts = mfl.ModelOptions(
        n_factors=n_factors,
        likelihoods=likelihoods,

        # Spatial Gaussian-process factor prior.
        # This is unrelated to gene-program annotations.
        factor_prior="GP",

        weight_prior=weight_prior,
        nonnegative_weights=False,
        nonnegative_factors=False,
        annotation_confidence=0.90,
        init_factors="random",
        init_scale=0.1,
    )

    n_obs = views["ST"].n_obs

    train_opts = mfl.TrainingOptions(
        device=device,
        batch_size=min(n_obs, 1024),
        max_epochs=max_epochs,
        lr=1e-3,
        early_stopper_patience=300,
        save_path=str(model_path),
        mofa_compat=False,
        seed=seed,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    smooth_opts = mfl.SmoothOptions(
        n_inducing=min(n_obs, 200),
        kernel="RBF",
        mefisto_kernel=False,
        independent_lengthscales=False,
        group_covar_rank=1,
        warp_groups=[],
    )

    # In MOFA-FLEX 0.1.0.post1 training occurs during construction.
    model = mfl.MOFAFLEX(
        data,
        data_opts,
        model_opts,
        train_opts,
        smooth_opts,
    )

    Z = np.asarray(
        get_factors(model, sample)
    )

    if Z.ndim != 2:
        raise RuntimeError(
            f"Unexpected factor matrix shape: {Z.shape}"
        )

    if Z.shape[1] != n_factors:
        raise RuntimeError(
            f"Requested {n_factors} factors but extracted "
            f"{Z.shape[1]}"
        )

    model_factor_names = get_model_factor_names(
        model,
        Z.shape[1],
    )

    # No annotations -> every factor is a free factor.
    factor_names = np.asarray(
        [
            f"Free_{i + 1:02d}"
            for i in range(Z.shape[1])
        ],
        dtype=object,
    )

    result = {
        "Z": Z,
        "samples": np.asarray(
            views["ST"].obs_names,
            dtype=object,
        ),
        "factor_names": factor_names,
        "factor_names_model": model_factor_names,
        "view_names": np.asarray(
            active_views,
            dtype=object,
        ),
    }

    for view in active_views:
        result[f"W_{view}"] = np.asarray(
            get_weights(model, view)
        )
        result[f"features_{view}"] = np.asarray(
            views[view].var_names,
            dtype=object,
        )

    factor_r2, view_r2 = extract_r2(
        model,
        sample,
        active_views,
    )

    result["r2_factor_view"] = factor_r2
    result["r2_view"] = view_r2

    np.savez_compressed(
        result_path,
        **result,
    )

    print(f"[OK] {result_path}")
    print("Z:", Z.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--n-factors",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("MOFA-FLEX Figure 2")
    print(f"factors: {args.n_factors}")
    print(f"seed: {args.seed}")
    print("views:", ", ".join(VIEWS))
    print("gene-program annotations: disabled")

    for sample in args.samples:
        run_sample(
            sample=sample,
            input_root=args.input_root,
            output_root=args.output_root,
            n_factors=args.n_factors,
            seed=args.seed,
            max_epochs=args.max_epochs,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

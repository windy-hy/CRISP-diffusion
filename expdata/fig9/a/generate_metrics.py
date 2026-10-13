"""
Figure 8a - core generator ablation: generate evaluation tables.

Compare three Stage-II generators under the same DiffCRISP front-end:
    1) Deterministic latent predictor
    2) Conditional Gaussian predictor
    3) Conditional latent diffusion

The script reuses DIFFCRISP.eval.evaluate() so R2 / PrDelta / Sinkhorn are
calculated with the same code path as the existing benchmark, and extends
the per-perturbation evaluation with:
    - centered Sinkhorn distance
    - variance log-MAE
    - responder-proportion error

Outputs
-------
fig8a_ablation_metrics_group.csv
    One row per model / split / training seed / perturbation group.

fig8a_ablation_metrics_split_seed.csv
    Mean across perturbation groups for each model / split / training seed.

Important
---------
For the cleanest "generator-only" ablation, GUIDANCE_SCALE is intentionally
set to 1.0. This removes DMSO guidance at inference so the comparison isolates
the Stage-II distribution model itself. Main-model results can still use 1.5.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


# ============================================================================
# Paths
# ============================================================================
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/nips.yaml"

# EDIT THESE THREE PATHS if your checkpoint directories use different names.
MODEL_ROOTS = {
    "Deterministic": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128_deterministic",
    "Gaussian": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128_gaussian",
    "Diffusion": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128",
}

OUTPUT_GROUP = BASE / "ablation_metrics_group.csv"
OUTPUT_SPLIT_SEED = BASE / "ablation_metrics_split_seed.csv"


# ============================================================================
# Evaluation settings
# ============================================================================
SPLITS = [
    "split",
    "split2",
    "split3"
]
SEEDS = [
    0,
    42,
    123
]

# You trained 0...300 (301 epochs), therefore final checkpoint is epoch 300.
TARGET_EPOCH = 300

FM_KEY = "X_scGPT"

# IMPORTANT:
# 1.0 intentionally turns off DMSO guidance for the controlled generator
# ablation. This avoids mixing "diffusion effect" with "CFG effect".
GUIDANCE_SCALE = 1.0

# Responder definition: exactly the same conceptual definition used in Fig. 4.
RESPONDER_TOP_GENES = 50
RESPONDER_Q = 0.95

# Fix only the stochastic sampling used during evaluation.
# Training-seed differences remain in checkpoint weights.
EVAL_SAMPLE_SEED = 20260831

METRICS = [
    "r2score_de",
    "pearson_delta_de",
    "sinkhorn_de",
    "centered_sinkhorn_de",
    "variance_log_mae_de",
    "responder_error",
]


# ============================================================================
# Utilities
# ============================================================================
def dense(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def set_eval_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_checkpoint(model_root: Path, split: str, seed: int) -> Path:
    """
    Prefer model_300*.pt. If it is absent, fall back to the highest numbered
    checkpoint so the script does not silently select an early model.
    """
    folder = model_root / split / f"seed{seed}"
    if not folder.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {folder}")

    exact = sorted(folder.glob(f"model_{TARGET_EPOCH}*.pt"))
    if exact:
        return exact[0]

    candidates = list(folder.glob("model_*_split.pt"))
    if not candidates:
        candidates = list(folder.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found in: {folder}")

    def epoch_number(path):
        try:
            return int(path.stem.split("_")[1])
        except Exception:
            return -1

    candidates = sorted(candidates, key=epoch_number)
    chosen = candidates[-1]
    print(
        f"[warning] {folder}: model_{TARGET_EPOCH} not found; "
        f"using {chosen.name}"
    )
    return chosen


def responder_metrics(ctrl_np, true_np, pred_np):
    """
    Reference-defined responder analysis used in the manuscript:

        delta = mean(true) - mean(control)
        top 50 genes by |delta|
        v = normalized delta on these genes
        score(x) = (x - mean(control))^T v
        threshold = 95th percentile of control scores
        error = |predicted responder proportion - true responder proportion|
    """
    ctrl_np = np.asarray(ctrl_np, dtype=np.float32)
    true_np = np.asarray(true_np, dtype=np.float32)
    pred_np = np.asarray(pred_np, dtype=np.float32)

    ctrl_mean = ctrl_np.mean(axis=0)
    true_mean = true_np.mean(axis=0)

    delta = true_mean - ctrl_mean
    n_genes = min(RESPONDER_TOP_GENES, delta.size)

    gene_idx = np.argsort(np.abs(delta))[-n_genes:]
    direction = delta[gene_idx].astype(np.float64)

    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        return {
            "responder_true": np.nan,
            "responder_pred": np.nan,
            "responder_threshold": np.nan,
            "responder_error": np.nan,
        }

    direction /= norm

    ctrl_scores = (
        ctrl_np[:, gene_idx] - ctrl_mean[gene_idx]
    ) @ direction
    true_scores = (
        true_np[:, gene_idx] - ctrl_mean[gene_idx]
    ) @ direction
    pred_scores = (
        pred_np[:, gene_idx] - ctrl_mean[gene_idx]
    ) @ direction

    threshold = float(np.quantile(ctrl_scores, RESPONDER_Q))
    p_true = float(np.mean(true_scores > threshold))
    p_pred = float(np.mean(pred_scores > threshold))

    return {
        "responder_true": p_true,
        "responder_pred": p_pred,
        "responder_threshold": threshold,
        "responder_error": abs(p_pred - p_true),
    }


# ============================================================================
# Single model/split/seed evaluation
# ============================================================================
def evaluate_one(model_name: str, model_root_str: str, split: str, seed: int):
    """
    Run in an isolated spawned process. This releases GPU/CPU memory after
    each checkpoint, which is useful on Windows.
    """
    import DIFFCRISP.eval as deval
    import DIFFCRISP.model as model_module
    from DIFFCRISP.losses import sinkhorn_dist
    from DIFFCRISP.trainer import Trainer
    from DIFFCRISP.utils import load_config

    model_root = Path(model_root_str)
    checkpoint = find_checkpoint(model_root, split, seed)

    # Old DiffCRISP checkpoints predate generator_type in init_args.
    # Force the module fallback to diffusion only while loading those old files.
    if model_name == "Diffusion":
        model_module.GENERATOR_TYPE = "diffusion"

    # Per-group records are appended by our wrapped calc_metrics.
    records = []

    # evaluate() calls compute_prediction_CRISP immediately before calc_metrics.
    # Store the actual matched control population so responder thresholds can
    # be calculated from the full control distribution, not only ctrl_mean.
    eval_context = {"control": None}

    original_predict = deval.compute_prediction_CRISP
    original_calc = deval.calc_metrics

    def wrapped_predict(
        autoencoder,
        genes,
        cell_embeddings,
        emb_drugs,
        emb_covs=None,
        drugs_pre=None,
        guidance_scale=1.5,
    ):
        eval_context["control"] = dense(genes).astype(np.float32, copy=False)

        return original_predict(
            autoencoder,
            genes,
            cell_embeddings,
            emb_drugs,
            emb_covs=emb_covs,
            drugs_pre=drugs_pre,
            guidance_scale=guidance_scale,
        )

    def wrapped_calc(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
        result = original_calc(
            yt_m,
            yp_m,
            ctrl_m,
            y_true,
            preds,
            idx_de,
        )

        idx = dense(idx_de).astype(int).ravel()
        true_np = dense(y_true).astype(np.float32, copy=False)
        pred_np = dense(preds).astype(np.float32, copy=False)

        if idx.size < 2:
            centered_sinkhorn = np.nan
            variance_log_mae = np.nan
        else:
            truth_de = true_np[:, idx]
            pred_de = pred_np[:, idx]

            # Center each population separately before Sinkhorn.
            truth_centered = (
                truth_de - truth_de.mean(axis=0, keepdims=True)
            ).astype(np.float32)
            pred_centered = (
                pred_de - pred_de.mean(axis=0, keepdims=True)
            ).astype(np.float32)

            centered_sinkhorn = float(
                sinkhorn_dist(
                    torch.from_numpy(truth_centered),
                    torch.from_numpy(pred_centered),
                ).item()
            )

            # Keep exactly the implementation already used by current Fig. 3:
            # mean |log1p(var_pred) - log1p(var_true)|.
            true_var = truth_de.var(axis=0)
            pred_var = pred_de.var(axis=0)
            variance_log_mae = float(
                np.mean(
                    np.abs(
                        np.log1p(pred_var)
                        - np.log1p(true_var)
                    )
                )
            )

        ctrl_np = eval_context["control"]
        if ctrl_np is None:
            raise RuntimeError(
                "Internal evaluation error: control population was not captured."
            )

        responder = responder_metrics(
            ctrl_np=ctrl_np,
            true_np=true_np,
            pred_np=pred_np,
        )

        row = {
            "r2score_de": float(result["r2score_de"]),
            "pearson_delta_de": float(result["pearson_delta_de"]),
            "sinkhorn_de": float(result["sinkhorn_de"]),
            "centered_sinkhorn_de": centered_sinkhorn,
            "variance_log_mae_de": variance_log_mae,
            **responder,
            "n_true_cells": int(true_np.shape[0]),
            "n_pred_cells": int(pred_np.shape[0]),
            "n_control_cells": int(ctrl_np.shape[0]),
            "n_de_genes": int(idx.size),
        }
        records.append(row)

        # Preserve the original evaluate() behavior.
        return result

    deval.compute_prediction_CRISP = wrapped_predict
    deval.calc_metrics = wrapped_calc

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = split
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(**config["dataset"], seed=seed)
    exp.load_model(str(checkpoint))
    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    generator_type = getattr(exp.autoencoder, "generator_type", "legacy_diffusion")

    print("\n" + "=" * 90)
    print(
        f"Evaluating {model_name} | {split} | seed={seed}\n"
        f"checkpoint: {checkpoint}\n"
        f"loaded generator_type: {generator_type}\n"
        f"guidance_scale: {GUIDANCE_SCALE}"
    )
    print("=" * 90)

    set_eval_seed(EVAL_SAMPLE_SEED)

    with torch.inference_mode():
        _, eval_score_dict, _ = deval.evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=exp.datasets["ood_treated"],
            control_dataset=exp.datasets["ood_control"],
            guidance_scale=GUIDANCE_SCALE,
        )

    groups = list(eval_score_dict.keys())
    if len(groups) != len(records):
        raise RuntimeError(
            f"{model_name}/{split}/seed{seed}: "
            f"{len(groups)} evaluated groups but {len(records)} metric records."
        )

    rows = []
    for group, record in zip(groups, records):
        rows.append(
            {
                "model": model_name,
                "generator_type": generator_type,
                "split": split,
                "seed": seed,
                "group": str(group),
                "checkpoint": str(checkpoint),
                "guidance_scale": GUIDANCE_SCALE,
                **record,
            }
        )

    return rows


# ============================================================================
# Main
# ============================================================================
def main():
    all_rows = []
    ctx = mp.get_context("spawn")

    jobs = [
        (model_name, str(model_root), split, seed)
        for model_name, model_root in MODEL_ROOTS.items()
        for split in SPLITS
        for seed in SEEDS
    ]

    for i, job in enumerate(jobs, 1):
        model_name, _, split, seed = job
        print(
            f"\n[{i}/{len(jobs)}] "
            f"{model_name} | {split} | seed={seed}"
        )

        # One child process at a time: avoids GPU memory accumulation on Windows.
        with ctx.Pool(1) as pool:
            rows = pool.apply(evaluate_one, job)
        all_rows.extend(rows)

    group_df = pd.DataFrame(all_rows)

    if group_df.empty:
        raise RuntimeError("No evaluation rows were generated.")

    # Keep only finite rows for reported metric means; individual NaNs remain
    # in the detailed table so failures are visible rather than silently hidden.
    group_df.to_csv(OUTPUT_GROUP, index=False)

    summary_df = (
        group_df
        .groupby(["model", "split", "seed"], as_index=False)[METRICS]
        .mean()
    )
    summary_df["n_groups"] = (
        group_df
        .groupby(["model", "split", "seed"])
        .size()
        .to_numpy()
    )
    summary_df.to_csv(OUTPUT_SPLIT_SEED, index=False)

    print("\n" + "=" * 90)
    print("Split/seed summary")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print("  ", OUTPUT_GROUP)
    print("  ", OUTPUT_SPLIT_SEED)


if __name__ == "__main__":
    mp.freeze_support()
    main()

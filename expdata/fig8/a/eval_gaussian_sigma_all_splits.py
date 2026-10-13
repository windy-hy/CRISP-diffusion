import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/nips.yaml"
MODEL_ROOT = ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128_gaussian"

# Run all three splits.
SPLITS = ["split", "split2", "split3"]
SEEDS = [0, 42, 123]

# Sweep inference-time Gaussian standard deviation.
# Original baseline: sigma = 0.05.
SIGMAS = [0.025, 0.05, 0.10, 0.20]

TARGET_EPOCH = 300
FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.0
EVAL_SAMPLE_SEED = 20260831

RESPONDER_TOP_GENES = 50
RESPONDER_Q = 0.95

METRICS = [
    "r2score_de",
    "pearson_delta_de",
    "sinkhorn_de",
    "centered_sinkhorn_de",
    "variance_log_mae_de",
    "responder_error",
]

# Exact same column layout as ablation_metrics_split_seed.csv.
OUTPUT_CSV = BASE / "gaussian_sigma_sensitivity_split_seed.csv"
OUTPUT_COLUMNS = [
    "model",
    "split",
    "seed",
    "r2score_de",
    "pearson_delta_de",
    "sinkhorn_de",
    "centered_sinkhorn_de",
    "variance_log_mae_de",
    "responder_error",
    "n_groups",
]


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
        torch.cuda.manual_seed_all(seed)


def find_checkpoint(split, seed):
    folder = MODEL_ROOT / split / f"seed{seed}"

    exact = sorted(folder.glob(f"model_{TARGET_EPOCH}*.pt"))
    if exact:
        return exact[0]

    candidates = (
        list(folder.glob("model_*_split.pt"))
        or list(folder.glob("model_*.pt"))
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found: {folder}")

    def epoch_num(p):
        try:
            return int(p.stem.split("_")[1])
        except Exception:
            return -1

    return max(candidates, key=epoch_num)


def override_gaussian_sigma(autoencoder, sigma):
    """
    Override only inference-time Gaussian sampling std after loading checkpoint.
    This keeps the trained checkpoint / predicted mean network unchanged.
    """
    if not hasattr(autoencoder, "gaussian_predictor"):
        raise AttributeError(
            "Loaded model has no gaussian_predictor. "
            f"generator_type={getattr(autoencoder, 'generator_type', 'unknown')}"
        )

    predictor = autoencoder.gaussian_predictor

    if not hasattr(predictor, "fixed_sigma"):
        raise AttributeError(
            "GaussianLatentPredictor has no attribute 'fixed_sigma'. "
            "Please inspect GaussianPredictor.py before running the sweep."
        )

    current = predictor.fixed_sigma

    # Support both a Python float and a scalar Tensor/buffer.
    if torch.is_tensor(current):
        with torch.no_grad():
            current.fill_(float(sigma))
    else:
        predictor.fixed_sigma = float(sigma)

    actual = predictor.fixed_sigma
    if torch.is_tensor(actual):
        actual = float(actual.detach().cpu().item())
    else:
        actual = float(actual)

    if not np.isclose(actual, float(sigma)):
        raise RuntimeError(
            f"Failed to set Gaussian sigma: requested={sigma}, actual={actual}"
        )

    return actual


def responder_metrics(ctrl_np, true_np, pred_np):
    ctrl_np = np.asarray(ctrl_np, np.float32)
    true_np = np.asarray(true_np, np.float32)
    pred_np = np.asarray(pred_np, np.float32)

    ctrl_mean = ctrl_np.mean(axis=0)
    delta = true_np.mean(axis=0) - ctrl_mean

    idx = np.argsort(np.abs(delta))[
        -min(RESPONDER_TOP_GENES, delta.size):
    ]
    direction = delta[idx].astype(np.float64)
    norm = np.linalg.norm(direction)

    if not np.isfinite(norm) or norm <= 1e-12:
        return {
            "responder_true": np.nan,
            "responder_pred": np.nan,
            "responder_threshold": np.nan,
            "responder_error": np.nan,
        }

    direction /= norm

    ctrl_score = (ctrl_np[:, idx] - ctrl_mean[idx]) @ direction
    true_score = (true_np[:, idx] - ctrl_mean[idx]) @ direction
    pred_score = (pred_np[:, idx] - ctrl_mean[idx]) @ direction

    threshold = float(np.quantile(ctrl_score, RESPONDER_Q))
    p_true = float(np.mean(true_score > threshold))
    p_pred = float(np.mean(pred_score > threshold))

    return {
        "responder_true": p_true,
        "responder_pred": p_pred,
        "responder_threshold": threshold,
        "responder_error": abs(p_pred - p_true),
    }


def evaluate_one(split, seed, sigma):
    import DIFFCRISP.eval as deval
    import DIFFCRISP.model as model_module
    from DIFFCRISP.losses import sinkhorn_dist
    from DIFFCRISP.trainer import Trainer
    from DIFFCRISP.utils import load_config

    # Ensure legacy checkpoint fallback constructs Gaussian generator.
    model_module.GENERATOR_TYPE = "gaussian"

    checkpoint = find_checkpoint(split, seed)

    records = []
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
        guidance_scale=1.0,
    ):
        eval_context["control"] = dense(genes).astype(
            np.float32, copy=False
        )
        return original_predict(
            autoencoder,
            genes,
            cell_embeddings,
            emb_drugs,
            emb_covs=emb_covs,
            drugs_pre=drugs_pre,
            guidance_scale=guidance_scale,
        )

    def wrapped_calc(
        yt_m,
        yp_m,
        ctrl_m,
        y_true,
        preds,
        idx_de,
    ):
        base = original_calc(
            yt_m,
            yp_m,
            ctrl_m,
            y_true,
            preds,
            idx_de,
        )

        idx = dense(idx_de).astype(int).ravel()
        true_np = dense(y_true).astype(
            np.float32, copy=False
        )
        pred_np = dense(preds).astype(
            np.float32, copy=False
        )
        ctrl_np = eval_context["control"]

        if idx.size < 2:
            centered_sinkhorn = np.nan
            variance_log_mae = np.nan
        else:
            truth_de = true_np[:, idx]
            pred_de = pred_np[:, idx]

            truth_centered = (
                truth_de
                - truth_de.mean(axis=0, keepdims=True)
            ).astype(np.float32)

            pred_centered = (
                pred_de
                - pred_de.mean(axis=0, keepdims=True)
            ).astype(np.float32)

            centered_sinkhorn = float(
                sinkhorn_dist(
                    torch.from_numpy(truth_centered),
                    torch.from_numpy(pred_centered),
                ).item()
            )

            variance_log_mae = float(
                np.mean(
                    np.abs(
                        np.log1p(pred_de.var(axis=0))
                        - np.log1p(truth_de.var(axis=0))
                    )
                )
            )

        responder = responder_metrics(
            ctrl_np,
            true_np,
            pred_np,
        )

        records.append(
            {
                "r2score_de": float(
                    base["r2score_de"]
                ),
                "pearson_delta_de": float(
                    base["pearson_delta_de"]
                ),
                "sinkhorn_de": float(
                    base["sinkhorn_de"]
                ),
                "centered_sinkhorn_de":
                    centered_sinkhorn,
                "variance_log_mae_de":
                    variance_log_mae,
                **responder,
                "n_true_cells":
                    int(true_np.shape[0]),
                "n_pred_cells":
                    int(pred_np.shape[0]),
                "n_control_cells":
                    int(ctrl_np.shape[0]),
                "n_de_genes":
                    int(idx.size),
            }
        )

        return base

    deval.compute_prediction_CRISP = wrapped_predict
    deval.calc_metrics = wrapped_calc

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = split
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(
        **config["dataset"],
        seed=seed,
    )
    exp.load_model(str(checkpoint))
    exp.autoencoder.to(exp.device).eval()

    generator_type = getattr(
        exp.autoencoder,
        "generator_type",
        "unknown",
    )

    # IMPORTANT:
    # Override sigma AFTER checkpoint loading.
    actual_sigma = override_gaussian_sigma(
        exp.autoencoder,
        sigma,
    )

    print("\n" + "=" * 100)
    print(
        f"Gaussian sigma sweep | "
        f"sigma={actual_sigma:g} | "
        f"{split} | seed={seed}\n"
        f"checkpoint: {checkpoint}\n"
        f"generator_type: {generator_type}\n"
        f"guidance_scale: {GUIDANCE_SCALE}"
    )
    print("=" * 100)

    # Same evaluation random seed for every sigma.
    # This makes different sigma values use the same
    # underlying stochastic sequence as far as possible.
    set_eval_seed(EVAL_SAMPLE_SEED)

    with torch.inference_mode():
        _, score_dict, _ = deval.evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=
                exp.datasets["ood_treated"],
            control_dataset=
                exp.datasets["ood_control"],
            guidance_scale=GUIDANCE_SCALE,
        )

    groups = list(score_dict.keys())

    if len(groups) != len(records):
        raise RuntimeError(
            f"sigma={sigma}/{split}/seed{seed}: "
            f"{len(groups)} groups != "
            f"{len(records)} records"
        )

    return records


def main():
    ctx = mp.get_context("spawn")
    summary_rows = []

    jobs = [
        (sigma, split, seed)
        for sigma in SIGMAS
        for split in SPLITS
        for seed in SEEDS
    ]

    print(
        "\nGaussian fixed-sigma sensitivity"
        "\nCFG = 1.0"
        f"\nTotal jobs = {len(jobs)}"
    )
    print("=" * 100)

    for i, (sigma, split, seed) in enumerate(
        jobs, 1
    ):
        print(
            f"\n[{i}/{len(jobs)}] "
            f"sigma={sigma:g} | "
            f"{split} | seed={seed}"
        )

        # One spawned child per checkpoint evaluation,
        # consistent with the previous evaluation script.
        with ctx.Pool(1) as pool:
            rows = pool.apply(
                evaluate_one,
                (split, seed, sigma),
            )

        df = pd.DataFrame(rows)

        vals = df[METRICS].mean()

        # Model label carries sigma while preserving
        # the exact CSV schema used before.
        model_label = f"Gaussian_sigma{sigma:g}"

        summary_rows.append(
            {
                "model": model_label,
                "split": split,
                "seed": seed,
                "r2score_de":
                    float(vals["r2score_de"]),
                "pearson_delta_de":
                    float(vals["pearson_delta_de"]),
                "sinkhorn_de":
                    float(vals["sinkhorn_de"]),
                "centered_sinkhorn_de":
                    float(vals["centered_sinkhorn_de"]),
                "variance_log_mae_de":
                    float(vals["variance_log_mae_de"]),
                "responder_error":
                    float(vals["responder_error"]),
                "n_groups": int(len(df)),
            }
        )

        print(
            f"sigma={sigma:g} | "
            f"{split} | seed={seed} | "
            f"n_groups={len(df)} | "
            f"R2={vals['r2score_de']:.6f} | "
            f"PrDelta="
            f"{vals['pearson_delta_de']:.6f} | "
            f"Sinkhorn="
            f"{vals['sinkhorn_de']:.6f} | "
            f"CenteredSinkhorn="
            f"{vals['centered_sinkhorn_de']:.6f} | "
            f"VarianceLogMAE="
            f"{vals['variance_log_mae_de']:.6f} | "
            f"ResponderError="
            f"{vals['responder_error']:.6f}"
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=OUTPUT_COLUMNS,
    )

    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 100)
    print("Gaussian sigma sensitivity summary")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    mp.freeze_support()
    main()

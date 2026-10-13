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

MODEL_ROOTS = {
    "Deterministic": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128_deterministic",
    "Gaussian": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128_gaussian",
    "Diffusion": ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128",
}

SPLITS = ["split", "split2", "split3"]
SEEDS = [0, 42, 123]
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

OUTPUT_GROUP = BASE / "ablation_metrics_group.csv"
OUTPUT_SPLIT_SEED = BASE / "ablation_metrics_split_seed.csv"


def dense(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def mean_rows(X, mask):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    sub = X[idx]
    if sp.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, dtype=np.float32).mean(axis=0)


def responder_metrics(ctrl, true, pred):
    ctrl = np.asarray(ctrl, np.float32)
    true = np.asarray(true, np.float32)
    pred = np.asarray(pred, np.float32)

    ctrl_mean = ctrl.mean(0)
    delta = true.mean(0) - ctrl_mean
    idx = np.argsort(np.abs(delta))[-min(RESPONDER_TOP_GENES, delta.size):]
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
    ctrl_score = (ctrl[:, idx] - ctrl_mean[idx]) @ direction
    true_score = (true[:, idx] - ctrl_mean[idx]) @ direction
    pred_score = (pred[:, idx] - ctrl_mean[idx]) @ direction
    threshold = float(np.quantile(ctrl_score, RESPONDER_Q))

    p_true = float(np.mean(true_score > threshold))
    p_pred = float(np.mean(pred_score > threshold))
    return {
        "responder_true": p_true,
        "responder_pred": p_pred,
        "responder_threshold": threshold,
        "responder_error": abs(p_pred - p_true),
    }


def extra_metrics(true_np, pred_np, ctrl_np, idx_de):
    from DIFFCRISP.losses import sinkhorn_dist

    idx = np.asarray(idx_de, dtype=int)
    if idx.size < 2:
        centered = np.nan
        var_mae = np.nan
    else:
        t = true_np[:, idx].astype(np.float32)
        p = pred_np[:, idx].astype(np.float32)

        tc = t - t.mean(0, keepdims=True)
        pc = p - p.mean(0, keepdims=True)
        centered = float(
            sinkhorn_dist(torch.from_numpy(tc), torch.from_numpy(pc)).item()
        )

        var_mae = float(
            np.mean(np.abs(np.log1p(p.var(0)) - np.log1p(t.var(0))))
        )

    return {
        "centered_sinkhorn_de": centered,
        "variance_log_mae_de": var_mae,
        **responder_metrics(ctrl_np, true_np, pred_np),
        "n_true_cells": int(true_np.shape[0]),
        "n_pred_cells": int(pred_np.shape[0]),
        "n_control_cells": int(ctrl_np.shape[0]),
        "n_de_genes": int(idx.size),
    }


# ----------------------------------------------------------------------
# Training-free mean-shift: NO Trainer / Dataset initialization.
# Δ(drug,dose) is estimated only from seen TRAIN cell types.
# OOD controls are used only as the population translated at inference.
# ----------------------------------------------------------------------
def build_shift_map(adata, cfg, split_key):
    obs = adata.obs
    drug_key = cfg["perturbation_key"]
    dose_key = cfg["dose_key"]
    ct_key = cfg["celltype_key"]
    ctrl_key = cfg["control_key"]

    split = obs[split_key].astype(str).to_numpy()
    drug = obs[drug_key].astype(str).to_numpy()
    dose = obs[dose_key].astype(float).to_numpy()
    ct = obs[ct_key].astype(str).to_numpy()
    ctrl = np.asarray(obs[ctrl_key]).astype(int)

    train_treated = (split == "train") & (ctrl != 1)
    out = {}

    pairs = sorted({
        (drug[i], round(float(dose[i]), 6))
        for i in np.flatnonzero(train_treated)
    })

    for d, s in pairs:
        pair = train_treated & (drug == d) & np.isclose(dose, s, atol=1e-6, rtol=0)
        deltas = []

        for celltype in np.unique(ct[pair]):
            tmask = pair & (ct == celltype)
            cmask = (split == "train") & (ctrl == 1) & (ct == celltype)

            tm = mean_rows(adata.X, tmask)
            cm = mean_rows(adata.X, cmask)
            if tm is not None and cm is not None:
                deltas.append(tm - cm)

        if deltas:
            out[(d, s)] = np.mean(np.stack(deltas), axis=0).astype(np.float32)

    return out


def get_shift(shift_map, drug, dose):
    key = (str(drug), round(float(dose), 6))
    if key in shift_map:
        return shift_map[key]

    candidates = [
        (abs(float(k[1]) - float(dose)), k)
        for k in shift_map
        if k[0] == str(drug)
    ]
    if not candidates:
        raise KeyError(f"No TRAIN mean shift for drug={drug}")

    dist, key = min(candidates)
    if dist > 1e-5:
        raise KeyError(f"No TRAIN mean shift for drug={drug}, dose={dose}")
    return shift_map[key]


def evaluate_meanshift_all():
    import scanpy as sc
    import DIFFCRISP.eval as deval
    from DIFFCRISP.utils import load_config

    config = load_config(str(CONFIG))
    cfg = config["dataset"]
    adata = sc.read_h5ad(str(DATA))

    drug_key = cfg["perturbation_key"]
    dose_key = cfg["dose_key"]
    ct_key = cfg["celltype_key"]
    ctrl_key = cfg["control_key"]
    group_key = cfg.get("pert_category", "cov_drug_name")
    degs_key = cfg.get("degs_key", "rank_genes_groups_cov")

    de_genes = adata.uns[degs_key]
    var_names = adata.var_names
    rows = []

    for split_key in SPLITS:
        print(f"\nMeanShift | {split_key}")
        shift_map = build_shift_map(adata, cfg, split_key)

        split = adata.obs[split_key].astype(str).to_numpy()
        ctrl_flag = np.asarray(adata.obs[ctrl_key]).astype(int)
        ct_arr = adata.obs[ct_key].astype(str).to_numpy()
        group_arr = adata.obs[group_key].astype(str).to_numpy()

        ood_treated = (split == "ood") & (ctrl_flag != 1)
        ood_control = (split == "ood") & (ctrl_flag == 1)

        groups, counts = np.unique(group_arr[ood_treated], return_counts=True)

        for group, count in zip(groups, counts):
            if count <= 5 or "dmso" in group.lower() or "control" in group.lower():
                continue

            gmask = ood_treated & (group_arr == group)
            first = np.flatnonzero(gmask)[0]
            celltype = ct_arr[first]
            drug = str(adata.obs.iloc[first][drug_key])
            dose = float(adata.obs.iloc[first][dose_key])

            cmask = ood_control & (ct_arr == celltype)
            if cmask.sum() < 5:
                continue

            # Match DEG lookup exactly to current eval.py behavior.
            if len(str(next(iter(de_genes.keys()))).split("_")) == 2:
                parts = group.split("_")
                deg_key = parts[0] + "_" + parts[1]
            else:
                deg_key = group

            if deg_key not in de_genes:
                continue

            idx_de = np.where(var_names.isin(np.asarray(de_genes[deg_key])))[0]
            if len(idx_de) < 2:
                continue

            ctrl_np = dense(adata.X[cmask]).astype(np.float32)
            true_np = dense(adata.X[gmask]).astype(np.float32)
            shift_vec = get_shift(shift_map, drug, dose)

            pred_np = np.maximum(ctrl_np + shift_vec[None, :], 0).astype(np.float32)

            ctrl_t = torch.from_numpy(ctrl_np)
            true_t = torch.from_numpy(true_np)
            pred_t = torch.from_numpy(pred_np)

            ctrl_m = ctrl_t.mean(0)
            true_m = true_t.mean(0)
            pred_m = pred_t.mean(0)

            base = deval.calc_metrics(
                true_m, pred_m, ctrl_m, true_t, pred_t, idx_de
            )
            extra = extra_metrics(true_np, pred_np, ctrl_np, idx_de)

            rows.append({
                "model": "MeanShift",
                "generator_type": "mean_shift",
                "split": split_key,
                "seed": -1,
                "group": group,
                "checkpoint": "",
                "guidance_scale": 1.0,
                "r2score_de": float(base["r2score_de"]),
                "pearson_delta_de": float(base["pearson_delta_de"]),
                "sinkhorn_de": float(base["sinkhorn_de"]),
                **extra,
            })

        print(f"  groups: {sum(r['split'] == split_key for r in rows)}")

    del adata
    return rows


def find_checkpoint(root, split, seed):
    folder = root / split / f"seed{seed}"
    exact = sorted(folder.glob(f"model_{TARGET_EPOCH}*.pt"))
    if exact:
        return exact[0]

    candidates = list(folder.glob("model_*_split.pt")) or list(folder.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(folder)

    def epoch_num(p):
        try:
            return int(p.stem.split("_")[1])
        except Exception:
            return -1

    return max(candidates, key=epoch_num)


def set_eval_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Learned generators: keep the original Trainer/evaluate path unchanged.
# ----------------------------------------------------------------------
def evaluate_learned(model_name, model_root_str, split, seed):
    import DIFFCRISP.eval as deval
    import DIFFCRISP.model as model_module
    from DIFFCRISP.trainer import Trainer
    from DIFFCRISP.utils import load_config

    root = Path(model_root_str)
    checkpoint = find_checkpoint(root, split, seed)

    if model_name == "Diffusion":
        model_module.GENERATOR_TYPE = "diffusion"

    records = []
    ctx = {"control": None}
    original_predict = deval.compute_prediction_CRISP
    original_calc = deval.calc_metrics

    def wrapped_predict(autoencoder, genes, cell_embeddings, emb_drugs,
                        emb_covs=None, drugs_pre=None, guidance_scale=1.5):
        ctx["control"] = dense(genes).astype(np.float32, copy=False)
        return original_predict(
            autoencoder, genes, cell_embeddings, emb_drugs,
            emb_covs=emb_covs, drugs_pre=drugs_pre,
            guidance_scale=guidance_scale,
        )

    def wrapped_calc(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
        base = original_calc(yt_m, yp_m, ctrl_m, y_true, preds, idx_de)

        true_np = dense(y_true).astype(np.float32, copy=False)
        pred_np = dense(preds).astype(np.float32, copy=False)
        ctrl_np = ctx["control"]

        records.append({
            "r2score_de": float(base["r2score_de"]),
            "pearson_delta_de": float(base["pearson_delta_de"]),
            "sinkhorn_de": float(base["sinkhorn_de"]),
            **extra_metrics(true_np, pred_np, ctrl_np, idx_de),
        })
        return base

    deval.compute_prediction_CRISP = wrapped_predict
    deval.calc_metrics = wrapped_calc

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = split
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(**config["dataset"], seed=seed)
    exp.load_model(str(checkpoint))
    exp.autoencoder.to(exp.device).eval()

    set_eval_seed(EVAL_SAMPLE_SEED)

    with torch.inference_mode():
        _, score_dict, _ = deval.evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=exp.datasets["ood_treated"],
            control_dataset=exp.datasets["ood_control"],
            guidance_scale=GUIDANCE_SCALE,
        )

    groups = list(score_dict.keys())
    if len(groups) != len(records):
        raise RuntimeError(
            f"{model_name}/{split}/seed{seed}: "
            f"{len(groups)} groups != {len(records)} records"
        )

    generator_type = getattr(exp.autoencoder, "generator_type", "legacy_diffusion")
    return [{
        "model": model_name,
        "generator_type": generator_type,
        "split": split,
        "seed": seed,
        "group": str(group),
        "checkpoint": str(checkpoint),
        "guidance_scale": GUIDANCE_SCALE,
        **record,
    } for group, record in zip(groups, records)]


def main():
    # MeanShift: load h5ad ONCE, no Trainer/Dataset preprocessing.
    all_rows = evaluate_meanshift_all()

    jobs = [
        (name, str(root), split, seed)
        for name, root in MODEL_ROOTS.items()
        for split in SPLITS
        for seed in SEEDS
    ]

    ctx = mp.get_context("spawn")
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {job[0]} | {job[2]} | seed={job[3]}")
        with ctx.Pool(1) as pool:
            all_rows.extend(pool.apply(evaluate_learned, job))

    group_df = pd.DataFrame(all_rows)
    group_df.to_csv(OUTPUT_GROUP, index=False)

    summary = (
        group_df.groupby(["model", "split", "seed"], as_index=False)[METRICS]
        .mean()
    )
    counts = (
        group_df.groupby(["model", "split", "seed"])
        .size()
        .rename("n_groups")
        .reset_index()
    )
    summary = summary.merge(counts, on=["model", "split", "seed"])
    summary.to_csv(OUTPUT_SPLIT_SEED, index=False)

    print("\n", summary.to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_GROUP)
    print(OUTPUT_SPLIT_SEED)


if __name__ == "__main__":
    mp.freeze_support()
    main()

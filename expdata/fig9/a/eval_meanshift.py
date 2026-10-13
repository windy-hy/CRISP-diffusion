import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import scanpy as sc

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/nips.yaml"
SPLITS = ["split", "split2", "split3"]

RESPONDER_TOP_GENES = 50
RESPONDER_Q = 0.95


def dense(x):
    if sp.issparse(x):
        return x.toarray()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_rows(X, mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    sub = X[idx]
    if sp.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, np.float32).mean(axis=0)


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

    train_ctrl = (split == "train") & (ctrl == 1)
    train_treated = (split == "train") & (ctrl != 1)

    shift_map = {}
    pairs = sorted({
        (drug[i], round(float(dose[i]), 6))
        for i in np.flatnonzero(train_treated)
    })

    for drug_name, dose_value in pairs:
        pair_mask = (
            train_treated
            & (drug == drug_name)
            & np.isclose(dose, dose_value, atol=1e-6, rtol=0)
        )

        deltas = []
        for celltype in np.unique(ct[pair_mask]):
            t_mask = pair_mask & (ct == celltype)
            c_mask = train_ctrl & (ct == celltype)

            t_mean = mean_rows(adata.X, t_mask)
            c_mean = mean_rows(adata.X, c_mask)

            if t_mean is not None and c_mean is not None:
                deltas.append(t_mean - c_mean)

        if deltas:
            shift_map[(drug_name, dose_value)] = np.mean(
                np.stack(deltas), axis=0
            ).astype(np.float32)

    return shift_map


def get_shift(shift_map, drug_name, dose):
    key = (str(drug_name), round(float(dose), 6))
    if key in shift_map:
        return shift_map[key]

    candidates = [
        (abs(float(k[1]) - float(dose)), k)
        for k in shift_map
        if k[0] == str(drug_name)
    ]
    if not candidates:
        raise KeyError(f"No TRAIN shift for drug={drug_name}")

    dist, key = min(candidates)
    if dist > 1e-5:
        raise KeyError(f"No TRAIN shift for drug={drug_name}, dose={dose}")

    return shift_map[key]


def responder_error(ctrl_np, true_np, pred_np):
    ctrl_mean = ctrl_np.mean(0)
    delta = true_np.mean(0) - ctrl_mean

    idx = np.argsort(np.abs(delta))[-min(RESPONDER_TOP_GENES, delta.size):]
    direction = delta[idx].astype(np.float64)
    norm = np.linalg.norm(direction)

    if not np.isfinite(norm) or norm <= 1e-12:
        return np.nan

    direction /= norm
    ctrl_score = (ctrl_np[:, idx] - ctrl_mean[idx]) @ direction
    true_score = (true_np[:, idx] - ctrl_mean[idx]) @ direction
    pred_score = (pred_np[:, idx] - ctrl_mean[idx]) @ direction

    threshold = np.quantile(ctrl_score, RESPONDER_Q)
    p_true = np.mean(true_score > threshold)
    p_pred = np.mean(pred_score > threshold)
    return float(abs(p_pred - p_true))


def evaluate_split(adata, cfg, split_key):
    import DIFFCRISP.eval as deval
    from DIFFCRISP.losses import sinkhorn_dist

    print(f"\nBuilding MeanShift for {split_key}...")
    shift_map = build_shift_map(adata, cfg, split_key)
    print(f"Built {len(shift_map)} drug-dose shifts")

    obs = adata.obs
    drug_key = cfg["perturbation_key"]
    dose_key = cfg["dose_key"]
    ct_key = cfg["celltype_key"]
    ctrl_key = cfg["control_key"]
    group_key = cfg.get("pert_category", "cov_drug_name")
    degs_key = cfg.get("degs_key", "rank_genes_groups_cov")

    split = obs[split_key].astype(str).to_numpy()
    ctrl_flag = np.asarray(obs[ctrl_key]).astype(int)
    ct_arr = obs[ct_key].astype(str).to_numpy()
    group_arr = obs[group_key].astype(str).to_numpy()

    ood_treated = (split == "ood") & (ctrl_flag != 1)
    ood_control = (split == "ood") & (ctrl_flag == 1)

    de_genes = adata.uns[degs_key]
    var_names = adata.var_names

    results = []
    groups, counts = np.unique(group_arr[ood_treated], return_counts=True)

    for group, count in zip(groups, counts):
        if count <= 5 or "dmso" in group.lower() or "control" in group.lower():
            continue

        gmask = ood_treated & (group_arr == group)
        first = np.flatnonzero(gmask)[0]

        celltype = ct_arr[first]
        drug_name = str(obs.iloc[first][drug_key])
        dose = float(obs.iloc[first][dose_key])

        cmask = ood_control & (ct_arr == celltype)
        if cmask.sum() < 5:
            continue

        first_deg_key = str(next(iter(de_genes.keys())))
        if len(first_deg_key.split("_")) == 2:
            parts = group.split("_")
            deg_key = parts[0] + "_" + parts[1]
        else:
            deg_key = group

        if deg_key not in de_genes:
            continue

        idx_de = np.where(var_names.isin(np.asarray(de_genes[deg_key])))[0]
        if idx_de.size < 2:
            continue

        ctrl_np = dense(adata.X[cmask]).astype(np.float32)
        true_np = dense(adata.X[gmask]).astype(np.float32)

        shift = get_shift(shift_map, drug_name, dose)
        pred_np = np.maximum(ctrl_np + shift[None, :], 0).astype(np.float32)

        ctrl_t = torch.from_numpy(ctrl_np)
        true_t = torch.from_numpy(true_np)
        pred_t = torch.from_numpy(pred_np)

        base = deval.calc_metrics(
            true_t.mean(0), pred_t.mean(0), ctrl_t.mean(0),
            true_t, pred_t, idx_de
        )

        truth_de = true_np[:, idx_de]
        pred_de = pred_np[:, idx_de]
        truth_centered = truth_de - truth_de.mean(0, keepdims=True)
        pred_centered = pred_de - pred_de.mean(0, keepdims=True)

        centered_sinkhorn = float(
            sinkhorn_dist(
                torch.from_numpy(truth_centered.astype(np.float32)),
                torch.from_numpy(pred_centered.astype(np.float32)),
            ).item()
        )

        variance_log_mae = float(
            np.mean(np.abs(
                np.log1p(pred_de.var(0)) - np.log1p(truth_de.var(0))
            ))
        )

        results.append({
            "r2": float(base["r2score_de"]),
            "prdelta": float(base["pearson_delta_de"]),
            "sinkhorn": float(base["sinkhorn_de"]),
            "centered_sinkhorn": centered_sinkhorn,
            "variance_log_mae": variance_log_mae,
            "responder_error": responder_error(ctrl_np, true_np, pred_np),
        })

    if not results:
        raise RuntimeError(f"No OOD groups evaluated for {split_key}")

    summary = {
        k: float(np.nanmean([r[k] for r in results]))
        for k in results[0]
    }

    print(f"\nMeanShift | {split_key} | n_groups={len(results)}")
    print(
        f"R2={summary['r2']:.6f} | "
        f"PrDelta={summary['prdelta']:.6f} | "
        f"Sinkhorn={summary['sinkhorn']:.6f} | "
        f"CenteredSinkhorn={summary['centered_sinkhorn']:.6f} | "
        f"VarianceLogMAE={summary['variance_log_mae']:.6f} | "
        f"ResponderError={summary['responder_error']:.6f}"
    )
    return summary


def main():
    from DIFFCRISP.utils import load_config

    cfg = load_config(str(CONFIG))["dataset"]
    adata = sc.read_h5ad(str(DATA))  # 只读一次

    summaries = {
        split: evaluate_split(adata, cfg, split)
        for split in SPLITS
    }

    print("\n" + "=" * 100)
    print("MeanShift | 3-split summary")
    print("=" * 100)

    for split in SPLITS:
        s = summaries[split]
        print(
            f"{split}: "
            f"R2={s['r2']:.6f} | "
            f"PrDelta={s['prdelta']:.6f} | "
            f"Sinkhorn={s['sinkhorn']:.6f} | "
            f"CenteredSinkhorn={s['centered_sinkhorn']:.6f} | "
            f"VarianceLogMAE={s['variance_log_mae']:.6f} | "
            f"ResponderError={s['responder_error']:.6f}"
        )

    avg = {
        k: float(np.mean([summaries[s][k] for s in SPLITS]))
        for k in summaries[SPLITS[0]]
    }

    print("\nAverage across 3 splits:")
    print(
        f"R2={avg['r2']:.6f} | "
        f"PrDelta={avg['prdelta']:.6f} | "
        f"Sinkhorn={avg['sinkhorn']:.6f} | "
        f"CenteredSinkhorn={avg['centered_sinkhorn']:.6f} | "
        f"VarianceLogMAE={avg['variance_log_mae']:.6f} | "
        f"ResponderError={avg['responder_error']:.6f}"
    )


if __name__ == "__main__":
    main()

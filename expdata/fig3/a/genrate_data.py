import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = (BASE / "../../..").resolve()
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/nips.yaml"
MODEL_ROOT = ROOT / "model/nips"

OUTPUT = BASE / "diffcrisp_distribution_metrics.csv"

SPLITS = ["split", "split2", "split3"]
METRICS = [
    "sinkhorn_de",
    "centered_sinkhorn_de",
    "variance_log_mae_de"
]

SEED = 0
EPOCH = 150
FM_KEY = "X_scGPT"
GUIDANCE = 1.5


def evaluate_one(split):
    import numpy as np
    import scipy.sparse as sp
    import torch

    import DIFFCRISP.eval as deval
    from DIFFCRISP.losses import sinkhorn_dist
    from DIFFCRISP.trainer import Trainer
    from DIFFCRISP.utils import load_config

    records = []

    def dense(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if sp.issparse(x):
            return x.toarray()
        return np.asarray(x)

    original_calc = deval.calc_metrics

    def calc_metrics(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
        result = original_calc(
            yt_m, yp_m, ctrl_m,
            y_true, preds, idx_de
        )

        idx = dense(idx_de).astype(int).ravel()

        truth_np = dense(y_true)[:, idx].astype("float32")
        pred_np = dense(preds)[:, idx].astype("float32")

        truth = torch.from_numpy(truth_np)
        pred = torch.from_numpy(pred_np)

        truth_c = truth - truth.mean(0, keepdim=True)
        pred_c = pred - pred.mean(0, keepdim=True)

        result["centered_sinkhorn_de"] = float(
            sinkhorn_dist(truth_c, pred_c).item()
        )

        truth_var = truth_np.var(axis=0)
        pred_var = pred_np.var(axis=0)

        result["variance_log_mae_de"] = float(
            np.mean(
                np.abs(
                    np.log1p(pred_var)
                    - np.log1p(truth_var)
                )
            )
        )

        records.append({
            name: float(result[name])
            for name in METRICS
        })

        return result

    deval.calc_metrics = calc_metrics

    folder = MODEL_ROOT / split / f"seed{SEED}"
    models = sorted(folder.glob(f"model_{EPOCH}*.pt"))

    if not models:
        raise FileNotFoundError(
            f"未找到模型：{folder}/model_{EPOCH}*.pt"
        )

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = split
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(**config["dataset"], seed=SEED)
    exp.load_model(str(models[0]))
    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    print(f"\n运行 {split}：{models[0]}")

    with torch.inference_mode():
        metrics, groups, _ = deval.evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=exp.datasets["ood_treated"],
            control_dataset=exp.datasets["ood_control"],
            guidance_scale=GUIDANCE
        )

    if len(records) != len(groups):
        raise RuntimeError(
            f"{split}：指标数 {len(records)}，组合数 {len(groups)}"
        )

    rows = []

    for group, values in zip(groups, records):
        if isinstance(group, (list, tuple, np.ndarray)):
            group = " | ".join(map(str, group))

        rows.append({
            "model": "DiffCRISP",
            "split": split,
            "row_type": "group",
            "group": str(group),
            "n_groups": 1,
            **values
        })

    rows.append({
        "model": "DiffCRISP",
        "split": split,
        "row_type": "split_mean",
        "group": "ALL",
        "n_groups": len(groups),
        **{
            name: float(metrics[name])
            for name in METRICS
        }
    })

    return rows


def main():
    all_rows = []
    ctx = mp.get_context("spawn")

    # 每个split单独进程，结束后释放内存
    for split in SPLITS:
        with ctx.Pool(1) as pool:
            all_rows.extend(
                pool.apply(evaluate_one, (split,))
            )

    result = pd.DataFrame(all_rows)

    split_mean = result[
        result["row_type"] == "split_mean"
    ]

    mean_row = {
        "model": "DiffCRISP",
        "split": "three_split_mean",
        "row_type": "three_split_mean",
        "group": "ALL",
        "n_groups": int(split_mean["n_groups"].sum())
    }

    for name in METRICS:
        mean_row[name] = split_mean[name].mean()

    result = pd.concat(
        [result, pd.DataFrame([mean_row])],
        ignore_index=True
    )

    result.to_csv(OUTPUT, index=False)

    print("\n汇总结果：")
    print(
        result[result["row_type"] != "group"]
        .to_string(index=False)
    )
    print("\n已保存：", OUTPUT)


if __name__ == "__main__":
    mp.freeze_support()
    main()
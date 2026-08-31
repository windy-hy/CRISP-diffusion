import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


# =========================
# 设置
# =========================
BASE = Path(__file__).resolve().parent
ROOT = (BASE / "../../..").resolve()
sys.path.insert(0, str(ROOT))

MODEL = "DiffCRISP"  # 跑完改成 CRISP
PACKAGE = "DIFFCRISP" if MODEL == "DiffCRISP" else "CRISP"

SPLIT_KEY = "split2"
TARGET_CELL = "K562"
TARGET_DOSE = 1.0
CHECK_GROUP = "K562_Belinostat_1.0"

TRAIN_SEED = 42
SAMPLE_SEEDS = list(range(10))
MAX_SAVE_PRED_CELLS = 100

MIN_CELLS = 50
MIN_VARIABLE_DE_GENES = 5

DATA = ROOT / "data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/sci.yaml"
MODEL_FILE = (
    ROOT / "model/sci" / SPLIT_KEY
    / f"seed{TRAIN_SEED}"
    / "model_300_split.pt"
)

FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.5

OUT_FILE = BASE / f"all_sampling_{MODEL.lower()}.pt"
QC_FILE = BASE / "all_sampling_qc.csv"


# =========================
# 工具
# =========================
def dense(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def get_rows(x, idx):
    try:
        return dense(x[idx]).astype(np.float32)
    except Exception:
        return np.stack([
            dense(x[int(i)]).reshape(-1)
            for i in idx
        ]).astype(np.float32)


def resolve_de(x, n_genes):
    x = dense(x).reshape(-1)

    if x.size == n_genes and np.all(
        np.isin(np.unique(x), [0, 1])
    ):
        return np.flatnonzero(x)

    x = x.astype(int)
    return x[(x >= 0) & (x < n_genes)]


def group_name(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " | ".join(map(str, x))
    return str(x)


# =========================
# 主程序
# =========================
def main():
    deval = importlib.import_module(f"{PACKAGE}.eval")
    Trainer = importlib.import_module(f"{PACKAGE}.trainer").Trainer
    load_config = importlib.import_module(f"{PACKAGE}.utils").load_config

    print("MODEL：", MODEL)
    print("模型文件：", MODEL_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = SPLIT_KEY
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(**config["dataset"], seed=TRAIN_SEED)

    exp.load_model(str(MODEL_FILE))
    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    treated_all = exp.datasets["ood_treated"]
    control_all = exp.datasets["ood_control"]

    cells = np.asarray(treated_all.celltype).astype(str)
    groups = np.asarray(treated_all.pert_categories).astype(str)
    doses = dense(treated_all.dosages).reshape(-1)

    print("\n全部OOD treated：", len(treated_all))
    print("genes形状：", treated_all.genes.shape)
    print("cell数组：", cells.shape)
    print("group数组：", groups.shape)
    print("dose数组：", doses.shape)

    if not (
        len(treated_all)
        == len(cells)
        == len(groups)
        == len(doses)
    ):
        raise RuntimeError("OOD treated字段行数不一致")

    # =========================
    # 真实细胞与QC
    # =========================
    candidate_mask = (
        (cells == TARGET_CELL)
        & np.isclose(doses, TARGET_DOSE)
    )

    candidate_groups = np.unique(groups[candidate_mask])

    print("\n候选组合数：", len(candidate_groups))
    print("候选细胞数：", int(candidate_mask.sum()))

    metadata = {}
    qc_rows = []

    for name in candidate_groups:
        idx = np.flatnonzero(
            candidate_mask & (groups == name)
        )

        truth_full = get_rows(
            treated_all.genes,
            idx
        )

        de_idx = resolve_de(
            treated_all.degs[idx[0]],
            truth_full.shape[1]
        )

        truth_de = truth_full[:, de_idx]

        n_unique = len(
            np.unique(truth_de, axis=0)
        )

        n_variable = int(
            np.sum(
                truth_de.var(axis=0) > 1e-12
            )
        )

        valid = (
            len(idx) >= MIN_CELLS
            and len(de_idx) > 0
            and n_unique > 1
            and n_variable >= MIN_VARIABLE_DE_GENES
            and np.isfinite(truth_de).all()
        )

        qc_rows.append({
            "group": name,
            "n_cells": len(idx),
            "n_de_genes": len(de_idx),
            "n_unique_rows_de": n_unique,
            "n_variable_genes_de": n_variable,
            "finite": np.isfinite(truth_de).all(),
            "valid": valid
        })

        if valid:
            metadata[name] = {
                "truth": truth_de,
                "de_idx": de_idx,
                "predictions": [],
                "fidelity": [],
                "row_idx": None,
                "n_pred_full": None
            }

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(QC_FILE, index=False)

    valid_names = sorted(metadata)

    print("保留组合数：", len(valid_names))
    print("过滤组合数：", len(candidate_groups) - len(valid_names))
    print("真实细胞数范围：",
          qc.loc[qc["valid"], "n_cells"].min(),
          qc.loc[qc["valid"], "n_cells"].max())
    print("DE基因数范围：",
          qc.loc[qc["valid"], "n_de_genes"].min(),
          qc.loc[qc["valid"], "n_de_genes"].max())

    if CHECK_GROUP not in metadata:
        print(qc[qc["group"] == CHECK_GROUP])
        raise RuntimeError(
            f"{CHECK_GROUP}不存在或未通过QC"
        )

    case = metadata[CHECK_GROUP]
    print("\n检查组合：", CHECK_GROUP)
    print("真实DE矩阵：", case["truth"].shape)
    print("DE基因数：", len(case["de_idx"]))
    print("真实唯一行：",
          len(np.unique(case["truth"], axis=0)))
    print("真实可变基因：",
          int(np.sum(case["truth"].var(axis=0) > 1e-12)))

    # =========================
    # 捕获evaluate预测
    # =========================
    original_calc = deval.calc_metrics
    captured = []

    def calc_metrics(
        yt_m,
        yp_m,
        ctrl_m,
        y_true,
        preds,
        idx_de
    ):
        result = original_calc(
            yt_m,
            yp_m,
            ctrl_m,
            y_true,
            preds,
            idx_de
        )

        pred = dense(preds).astype(np.float32)
        de_idx = resolve_de(
            idx_de,
            pred.shape[1]
        )

        captured.append({
            "pred": pred,
            "de_idx": de_idx,
            "y_true_shape": dense(y_true).shape,
            "fidelity": float(
                result.get("sinkhorn_de", np.nan)
            )
        })

        return result

    deval.calc_metrics = calc_metrics

    kwargs = {
        "autoencoder": exp.autoencoder,
        "treated_dataset": treated_all,
        "control_dataset": control_all
    }

    if MODEL == "DiffCRISP":
        kwargs["guidance_scale"] = GUIDANCE_SCALE

    try:
        for seed in SAMPLE_SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            captured.clear()

            print(f"\n{MODEL} | seed={seed}")

            with torch.inference_mode():
                _, eval_groups, _ = deval.evaluate(
                    **kwargs
                )

            eval_names = [
                group_name(x)
                for x in eval_groups
            ]

            print("evaluate组合数：", len(eval_names))
            print("捕获预测数：", len(captured))

            if len(eval_names) != len(captured):
                raise RuntimeError(
                    "eval_groups与预测数量不一致"
                )

            if len(set(eval_names)) != len(eval_names):
                raise RuntimeError(
                    "evaluate返回重复组合名"
                )

            lookup = dict(
                zip(eval_names, captured)
            )

            missing = [
                name for name in valid_names
                if name not in lookup
            ]

            print("缺失目标组合数：", len(missing))

            if missing:
                print("前10个缺失组合：", missing[:10])
                raise RuntimeError(
                    "部分QC组合未被evaluate返回"
                )

            n_pred_values = []
            n_saved_values = []
            de_mismatch = 0
            nonfinite = 0

            for position, name in enumerate(valid_names):
                item = lookup[name]
                pred_full = item["pred"]
                eval_de = item["de_idx"]
                expected_de = metadata[name]["de_idx"]

                if pred_full.ndim != 2:
                    raise RuntimeError(
                        f"{name}预测不是二维矩阵"
                    )

                if not np.isfinite(pred_full).all():
                    nonfinite += 1
                    continue

                if not np.array_equal(
                    eval_de,
                    expected_de
                ):
                    de_mismatch += 1
                    continue

                pred_de = pred_full[:, expected_de]
                n_pred = len(pred_de)

                # 第一个seed确定固定行索引
                if seed == SAMPLE_SEEDS[0]:
                    keep = min(
                        MAX_SAVE_PRED_CELLS,
                        n_pred
                    )

                    rng = np.random.default_rng(
                        10000 + position
                    )

                    metadata[name]["row_idx"] = (
                        rng.choice(
                            n_pred,
                            keep,
                            replace=False
                        )
                    )

                    metadata[name]["n_pred_full"] = (
                        n_pred
                    )

                else:
                    if (
                        n_pred
                        != metadata[name]["n_pred_full"]
                    ):
                        raise RuntimeError(
                            f"{name}不同seed预测细胞数不一致："
                            f"{metadata[name]['n_pred_full']} vs {n_pred}"
                        )

                row_idx = metadata[name]["row_idx"]
                pred_save = pred_de[row_idx]

                metadata[name]["predictions"].append(
                    pred_save.copy()
                )

                metadata[name]["fidelity"].append(
                    item["fidelity"]
                )

                n_pred_values.append(n_pred)
                n_saved_values.append(
                    len(pred_save)
                )

            print("DE索引不一致数：", de_mismatch)
            print("非有限预测数：", nonfinite)
            print("完整预测细胞数范围：",
                  min(n_pred_values),
                  max(n_pred_values))
            print("保存预测细胞数范围：",
                  min(n_saved_values),
                  max(n_saved_values))

            if de_mismatch > 0:
                raise RuntimeError(
                    "存在DE索引不一致"
                )

            if nonfinite > 0:
                raise RuntimeError(
                    "存在NaN或Inf预测"
                )

    finally:
        deval.calc_metrics = original_calc

    # =========================
    # 整理并保存
    # =========================
    saved_groups = {}

    for name in valid_names:
        item = metadata[name]

        if len(item["predictions"]) != len(SAMPLE_SEEDS):
            raise RuntimeError(
                f"{name}预测seed数不是10"
            )

        shapes = [
            x.shape
            for x in item["predictions"]
        ]

        if len(set(shapes)) != 1:
            raise RuntimeError(
                f"{name}不同seed形状不一致：{shapes}"
            )

        predictions = np.stack(
            item["predictions"]
        )

        saved_groups[name] = {
            "predictions": predictions,
            "truth": item["truth"],
            "de_idx": item["de_idx"],
            "fidelity": np.asarray(
                item["fidelity"],
                dtype=np.float32
            ),
            "saved_row_idx": item["row_idx"],
            "n_pred_full": item["n_pred_full"]
        }

    case = saved_groups[CHECK_GROUP]

    print("\n最终组合数：", len(saved_groups))
    print("Belinostat预测矩阵：",
          case["predictions"].shape)
    print("Belinostat真实矩阵：",
          case["truth"].shape)
    print("Belinostat DE基因：",
          len(case["de_idx"]))

    seed_change = np.mean([
        np.mean(
            np.abs(
                case["predictions"][i]
                - case["predictions"][0]
            )
        )
        for i in range(1, len(SAMPLE_SEEDS))
    ])

    print("Belinostat跨seed平均绝对变化：",
          seed_change)

    torch.save(
        {
            "model": MODEL,
            "split": SPLIT_KEY,
            "cell": TARGET_CELL,
            "dose": TARGET_DOSE,
            "sample_seeds": SAMPLE_SEEDS,
            "max_saved_pred_cells":
                MAX_SAVE_PRED_CELLS,
            "groups": saved_groups
        },
        OUT_FILE
    )

    print("\n已保存：")
    print(OUT_FILE)
    print(QC_FILE)


if __name__ == "__main__":
    main()
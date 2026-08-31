import importlib
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


BASE = Path(__file__).resolve().parent
ROOT = (BASE / "../../..").resolve()
sys.path.insert(0, str(ROOT))

MODEL = "DiffCRISP"  # 跑完改成 CRISP
PACKAGE = "DIFFCRISP" if MODEL == "DiffCRISP" else "CRISP"

SPLIT_KEY = "split2"
TRAIN_SEED = 42
SAMPLE_SEEDS = list(range(10))
SAVE_CELLS = 100

DATA = ROOT / "data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
CONFIG = ROOT / "experiments/configs/sci.yaml"
MODEL_FILE = (
    ROOT / "model/sci" / SPLIT_KEY
    / f"seed{TRAIN_SEED}"
    / "model_300_split.pt"
)

FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.5

OUT_FILE = BASE / f"selected_allgenes_{MODEL.lower()}.pt"


SELECTED_GROUPS = [
    "K562_Meprednisone_1.0",
    "K562_ENMD-2076_1.0",
    "K562_Thiotepa_1.0",
    "K562_Triamcinolone_1.0",
    "K562_Carmofur_1.0",
    "K562_Fluorouracil_1.0",
    "K562_Sirtinol_1.0",
    "K562_Tranylcypromine_1.0",
    "K562_Cyclocytidine_1.0",
    "K562_A-366_1.0",
    "K562_Filgotinib_1.0",
    "K562_Linifanib_1.0",
    "K562_JNJ-26854165_1.0",
    "K562_Resveratrol_1.0",
    "K562_GSK-LSD1_1.0",
    "K562_Cerdulatinib_1.0",
    "K562_CEP-33779_1.0",
    "K562_AMG-900_1.0",
    "K562_PF-573228_1.0",
    "K562_Raltitrexed_1.0"
]


def dense(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def group_name(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " | ".join(map(str, x))
    return str(x)


def main():
    deval = importlib.import_module(f"{PACKAGE}.eval")
    Trainer = importlib.import_module(
        f"{PACKAGE}.trainer"
    ).Trainer
    load_config = importlib.import_module(
        f"{PACKAGE}.utils"
    ).load_config

    print("MODEL：", MODEL)
    print("模型文件：", MODEL_FILE)
    print("待保存组合数：", len(SELECTED_GROUPS))

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    config = load_config(str(CONFIG))
    config["dataset"]["adata_obj"] = str(DATA)
    config["dataset"]["split_key"] = SPLIT_KEY
    config["dataset"]["FM_key"] = FM_KEY

    exp = Trainer()
    exp.init_dataset(
        **config["dataset"],
        seed=TRAIN_SEED
    )

    exp.load_model(str(MODEL_FILE))
    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    treated_all = exp.datasets["ood_treated"]
    control_all = exp.datasets["ood_control"]

    all_groups = set(
        np.asarray(
            treated_all.pert_categories
        ).astype(str)
    )

    missing_data = [
        name for name in SELECTED_GROUPS
        if name not in all_groups
    ]

    print("数据集中缺失组合：", missing_data)

    if missing_data:
        raise RuntimeError("部分组合不在OOD treated中")

    saved = {
        name: {
            "predictions": [],
            "row_idx": None,
            "n_pred_full": None
        }
        for name in SELECTED_GROUPS
    }

    original_calc = deval.calc_metrics
    captured = []

    def patched_calc(
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

        captured.append(
            dense(preds).astype(np.float32)
        )

        return result

    deval.calc_metrics = patched_calc

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

            names = [
                group_name(x)
                for x in eval_groups
            ]

            print("evaluate组合数：", len(names))
            print("捕获预测数：", len(captured))

            if len(names) != len(captured):
                raise RuntimeError(
                    "eval_groups与预测数量不一致"
                )

            if len(set(names)) != len(names):
                raise RuntimeError(
                    "evaluate返回重复组合名"
                )

            lookup = dict(zip(names, captured))

            missing = [
                name for name in SELECTED_GROUPS
                if name not in lookup
            ]

            print("本次缺失组合：", missing)

            if missing:
                raise RuntimeError(
                    "部分目标组合未被evaluate返回"
                )

            for group_id, name in enumerate(SELECTED_GROUPS):
                pred = lookup[name]

                if pred.ndim != 2:
                    raise RuntimeError(
                        f"{name}预测不是二维：{pred.shape}"
                    )

                if pred.shape[1] != 5000:
                    raise RuntimeError(
                        f"{name}不是5000基因：{pred.shape}"
                    )

                if not np.isfinite(pred).all():
                    raise RuntimeError(
                        f"{name}包含NaN或Inf"
                    )

                item = saved[name]

                if seed == SAMPLE_SEEDS[0]:
                    keep = min(
                        SAVE_CELLS,
                        len(pred)
                    )

                    item["row_idx"] = (
                        np.random.default_rng(
                            10000 + group_id
                        ).choice(
                            len(pred),
                            keep,
                            replace=False
                        )
                    )

                    item["n_pred_full"] = len(pred)

                elif len(pred) != item["n_pred_full"]:
                    raise RuntimeError(
                        f"{name}不同seed预测细胞数不一致"
                    )

                pred_save = pred[
                    item["row_idx"]
                ]

                item["predictions"].append(
                    pred_save.copy()
                )

            check = saved[
                "K562_Cerdulatinib_1.0"
            ]["predictions"][-1]

            print(
                "Cerdulatinib本次保存：",
                check.shape
            )

    finally:
        deval.calc_metrics = original_calc

    final_groups = {}

    for name in SELECTED_GROUPS:
        item = saved[name]

        if len(item["predictions"]) != 10:
            raise RuntimeError(
                f"{name}预测seed数不是10"
            )

        shapes = [
            x.shape for x in item["predictions"]
        ]

        if len(set(shapes)) != 1:
            raise RuntimeError(
                f"{name}不同seed形状不一致：{shapes}"
            )

        predictions = np.stack(
            item["predictions"]
        )

        if predictions.shape[2] != 5000:
            raise RuntimeError(
                f"{name}最终不是5000基因"
            )

        final_groups[name] = {
            "predictions": predictions,
            "saved_row_idx": item["row_idx"],
            "n_pred_full": item["n_pred_full"]
        }

        print(
            name,
            "|",
            predictions.shape
        )

    torch.save(
        {
            "model": MODEL,
            "split": SPLIT_KEY,
            "sample_seeds": SAMPLE_SEEDS,
            "n_genes": 5000,
            "saved_cells": SAVE_CELLS,
            "groups": final_groups
        },
        OUT_FILE
    )

    print("\n最终保存组合数：", len(final_groups))
    print("保存：", OUT_FILE)


if __name__ == "__main__":
    main()
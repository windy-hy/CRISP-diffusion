import random
import numpy as np
import pandas as pd
import torch

from DIFFCRISP.utils import load_config
from DIFFCRISP.trainer import Trainer
from DIFFCRISP.eval import evaluate


CONFIG_PATH = "../../../experiments/configs/sci.yaml"

DATA_PATH = (
    "../../../data/sciplex3/"
    "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

MODEL_ROOT = "../../../model/sci"

# 改成实际使用的模型文件
MODEL_FILE = "model.pt"

SPLIT_CELL = {
    "zero_split_drugs": "MCF7",
    "zero_split_drugs2": "A549",
    "zero_split_drugs3": "K562",
}

SEEDS = [0]
# 完整复现时改成：
# SEEDS = [0, 1, 2]

DRUGS = [
    "Dacinostat",
    "Givinostat",
    "Belinostat",
    "Hesperadin",
    "Quisinostat",
    "Alvespimycin",
    "Tanespimycin",
    "TAK-901",
    "Flavopiridol",
]

DOSES = [0.001, 0.01, 0.1, 1.0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_combination(comb):
    parts = str(comb).split("_")

    cell_type = parts[0]
    dose_val = float(parts[-1])
    drug = "_".join(parts[1:-1])

    return cell_type, drug, dose_val


rows = []


for split, target_cell in SPLIT_CELL.items():

    for seed in SEEDS:

        print("\n正在评估：", split, target_cell, "seed =", seed)

        args = load_config(CONFIG_PATH)

        args["dataset"]["adata_obj"] = DATA_PATH
        args["dataset"]["split_key"] = split

        exp = Trainer()

        exp.init_dataset(
            **args["dataset"],
            seed=seed,
        )

        model_path = (
            f"{MODEL_ROOT}/"
            f"{split}/"
            f"seed{seed}/"
            f"{MODEL_FILE}"
        )

        exp.load_model(model_path)
        exp.autoencoder.eval()

        # zero split中目标细胞的control通常放在test
        control_dataset = exp.datasets["test_control"]

        control_count = np.sum(
            control_dataset.celltype == target_cell
        )

        # test中没有足够control时，尝试ood_control
        if control_count < 5:

            control_dataset = exp.datasets["ood_control"]

            control_count = np.sum(
                control_dataset.celltype == target_cell
            )

        if control_count < 5:
            raise ValueError(
                f"{split} 中找不到足够的 "
                f"{target_cell} control细胞"
            )

        print(
            "使用control细胞数：",
            control_count,
        )

        set_seed(seed)

        with torch.no_grad():

            _, condition_scores, _ = evaluate(
                exp.autoencoder,
                exp.datasets["ood_treated"],
                control_dataset,
            )

        condition_count = 0

        for comb, scores in condition_scores.items():

            cell_type, drug, dose_val = (
                parse_combination(comb)
            )

            if cell_type != target_cell:
                continue

            if drug not in DRUGS:
                continue

            if not any(
                np.isclose(dose_val, dose)
                for dose in DOSES
            ):
                continue

            rows.append({
                "split": split,
                "seed": seed,
                "cell_type": cell_type,
                "drug": drug,
                "dose_val": dose_val,
                "pearson_delta_de": (
                    scores["pearson_delta_de"]
                ),
            })

            condition_count += 1

        print(
            "提取到的条件数量：",
            condition_count,
        )

        if condition_count != 36:
            print(
                "警告：正常应为36个条件，"
                "请检查缺失组合或细胞数量。"
            )


# =========================
# 每个条件的详细结果
# =========================

detail = pd.DataFrame(rows)

detail = detail.sort_values(
    [
        "cell_type",
        "seed",
        "drug",
        "dose_val",
    ]
)

detail.to_csv(
    "fig3e_pearson_conditions.csv",
    index=False,
)


# =========================
# 每个细胞系、每个seed：
# 对36个条件求平均
# =========================

per_seed = (
    detail.groupby(
        [
            "cell_type",
            "split",
            "seed",
        ],
        as_index=False,
    )
    .agg(
        pearson_delta_de=(
            "pearson_delta_de",
            "mean",
        ),
        n_conditions=(
            "pearson_delta_de",
            "count",
        ),
    )
)

per_seed.to_csv(
    "fig3e_pearson_per_seed.csv",
    index=False,
)


# =========================
# 多个seed的柱状图统计
# =========================

summary = (
    per_seed.groupby(
        "cell_type",
        as_index=False,
    )
    .agg(
        mean_pearson=(
            "pearson_delta_de",
            "mean",
        ),
        std_pearson=(
            "pearson_delta_de",
            "std",
        ),
        n_seeds=(
            "seed",
            "nunique",
        ),
    )
)

summary.to_csv(
    "fig3e_pearson_summary.csv",
    index=False,
)


print("\n每个seed的结果：")
print(per_seed)

print("\n最终汇总：")
print(summary)
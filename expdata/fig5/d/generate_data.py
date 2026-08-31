import os

import numpy as np
import pandas as pd
import torch

from DIFFCRISP.utils import load_config
from DIFFCRISP.trainer import Trainer
from DIFFCRISP.eval import evaluate

# =========================
# 基本设置
# =========================

SPLIT_KEY = "split_drugs"
TARGET_CELL = ""
if SPLIT_KEY == "split_drugs":
    TARGET_CELL = "MCF7"
elif SPLIT_KEY == "split_drugs2":
    TARGET_CELL = "A549"
elif SPLIT_KEY == "split_drugs3":
    TARGET_CELL = "K562"


SEED = 0
GUIDANCE_SCALE = 1.5
CONFIG_PATH = "../../../experiments/configs/sci.yaml"

DATA_PATH = (
    "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

MODEL_PATH = f"../../../model/sci/{SPLIT_KEY}/seed0/model_250_split.pt"
OUTPUT_PATH =f"SciPlex3_{TARGET_CELL}_DIFFDRISP.csv"

UNSEEN_DRUGS = [
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


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()

    return float(x)


# =========================
# 加载配置
# =========================

args = load_config(CONFIG_PATH)

args["dataset"]["split_key"] = SPLIT_KEY
args["dataset"]["adata_obj"] = DATA_PATH
args["model"]["seed"] = SEED


# =========================
# 初始化数据集
# =========================

exp = Trainer()

exp.init_dataset(
    **args["dataset"],
    seed=SEED,
)

treated_dataset = exp.datasets["ood_treated"]
control_dataset = exp.datasets["ood_control"]


# =========================
# 找到K562的36个组合
# =========================

pert_categories = np.asarray(
    treated_dataset.pert_categories
).astype(str)

cell_types = np.asarray(
    treated_dataset.celltype
).astype(str)

drug_names = np.asarray(
    treated_dataset.drugs_names
).astype(str)

doses = to_numpy(
    treated_dataset.dosages
)

if doses.ndim > 1:
    doses = doses[:, 0]


condition_table = pd.DataFrame({
    "pert_category": pert_categories,
    "cell_type": cell_types,
    "condition": drug_names,
    "dose_val": doses.astype(float),
})


condition_table = condition_table[
    (condition_table["cell_type"] == TARGET_CELL)
    & condition_table["condition"].isin(UNSEEN_DRUGS)
].copy()


condition_table = (
    condition_table
    .drop_duplicates("pert_category")
    .sort_values(
        [
            "condition",
            "dose_val",
        ]
    )
    .reset_index(drop=True)
)


print("药物数量：", condition_table["condition"].nunique())
print("组合数量：", len(condition_table))

print(
    condition_table[
        [
            "condition",
            "dose_val",
            "pert_category",
        ]
    ].to_string(index=False)
)


if condition_table["condition"].nunique() != 9:
    raise ValueError(
        "未见药物数量不是9种，实际为："
        + str(condition_table["condition"].nunique())
    )


if len(condition_table) != 36:
    raise ValueError(
        "药物-剂量组合不是36个，实际为："
        + str(len(condition_table))
    )


# =========================
# 加载模型
# =========================

exp.load_model(MODEL_PATH)

exp.autoencoder.to(exp.device)
exp.autoencoder.eval()


# 固定推理随机数
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# =========================
# 计算指标
# =========================

with torch.no_grad():

    metrics_all, eval_score_dict, pred_dict = evaluate(
        autoencoder=exp.autoencoder,
        treated_dataset=treated_dataset,
        control_dataset=control_dataset,
        guidance_scale=GUIDANCE_SCALE,
    )


# =========================
# 只提取36个组合
# =========================

results = []


for _, row in condition_table.iterrows():

    pert_category = row["pert_category"]

    if pert_category not in eval_score_dict:
        raise KeyError(
            "evaluate结果中没有找到："
            + pert_category
        )

    score = eval_score_dict[pert_category]

    results.append({
        "condition": row["condition"],
        "dose_val": row["dose_val"],

        "CRISP_R2_DE": to_float(
            score["r2score_de"]
        ),

        "CRISP_Pearson_Delta_DE": to_float(
            score["pearson_delta_de"]
        ),

        "CRISP_Sinkhorn_DE": to_float(
            score["sinkhorn_de"]
        ),
    })


result_df = pd.DataFrame(results)


if len(result_df) != 36:
    raise ValueError(
        "最终保存结果不是36行，实际为："
        + str(len(result_df))
    )


# =========================
# 保存一个CSV
# =========================

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n保存完成：", OUTPUT_PATH)
print("保存行数：", len(result_df))
print(result_df.to_string(index=False))

print("\n三个指标均值：")

print(
    "R2_DE：",
    result_df["CRISP_R2_DE"].mean()
)

print(
    "Pearson_Delta_DE：",
    result_df["CRISP_Pearson_Delta_DE"].mean()
)

print(
    "Sinkhorn_DE：",
    result_df["CRISP_Sinkhorn_DE"].mean()
)
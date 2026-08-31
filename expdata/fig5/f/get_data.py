import numpy as np
import pandas as pd
import torch

from DIFFCRISP.utils import load_config
from DIFFCRISP.trainer import Trainer
from DIFFCRISP.eval import evaluate


SPLIT_KEY = "zero_split_drugs"
TARGET_CELL = ""
if SPLIT_KEY == "zero_split_drugs":
    TARGET_CELL = "MCF7"
elif SPLIT_KEY == "zero_split_drugs2":
    TARGET_CELL = "A549"
elif SPLIT_KEY == "zero_split_drugs3":
    TARGET_CELL = "K562"

SEED = 0
GUIDANCE_SCALE = 1.5
EPOCHS = [100, 150, 200,250,300]

CONFIG_PATH = "../../../experiments/configs/sci.yaml"

DATA_PATH = (
    "../../../data/sciplex3/"
    "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

MODEL_DIR = (
    f"../../../model/sci/{SPLIT_KEY}/seed0"
)

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


# =========================
# 初始化数据，只执行一次
# =========================

args = load_config(CONFIG_PATH)

args["dataset"]["split_key"] = SPLIT_KEY
args["dataset"]["adata_obj"] = DATA_PATH
args["model"]["seed"] = SEED

exp = Trainer()

exp.init_dataset(
    **args["dataset"],
    seed=SEED,
)

treated_dataset = exp.datasets["ood_treated"]
control_dataset = exp.datasets["ood_control"]


# =========================
# 找到MCF7的36个组合
# 与Fig.2h完全相同
# =========================

doses = to_numpy(treated_dataset.dosages)

if doses.ndim > 1:
    doses = doses[:, 0]

condition_table = pd.DataFrame({
    "pert_category": np.asarray(
        treated_dataset.pert_categories
    ).astype(str),

    "cell_type": np.asarray(
        treated_dataset.celltype
    ).astype(str),

    "condition": np.asarray(
        treated_dataset.drugs_names
    ).astype(str),

    "dose_val": doses.astype(float),
})

condition_table = condition_table[
    (condition_table["cell_type"] == TARGET_CELL)
    & condition_table["condition"].isin(UNSEEN_DRUGS)
]

condition_table = (
    condition_table
    .drop_duplicates("pert_category")
    .sort_values(["condition", "dose_val"])
    .reset_index(drop=True)
)

print("组合数量：", len(condition_table))

if len(condition_table) != 36:
    raise ValueError(
        "组合数量不是36，实际为："
        + str(len(condition_table))
    )


# =========================
# 依次评估不同训练轮数
# =========================

summary = []

for epoch in EPOCHS:

    model_path = (
        MODEL_DIR
        + f"/model_{epoch}_split.pt"
    )

    print("\n评估模型：", model_path)

    exp.load_model(model_path)

    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    # 保证每个checkpoint使用相同推理随机数
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    with torch.no_grad():

        _, eval_score_dict, _ = evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=treated_dataset,
            control_dataset=control_dataset,
            guidance_scale=GUIDANCE_SCALE
        )

    pearsons = []

    for pert_category in condition_table["pert_category"]:

        if pert_category not in eval_score_dict:
            raise KeyError(
                "没有找到组合："
                + pert_category
            )

        pearsons.append(
            float(
                eval_score_dict[
                    pert_category
                ]["pearson_delta_de"]
            )
        )

    mean_pearson = np.mean(pearsons)

    summary.append({
        "epoch": epoch,
        "n_conditions": len(pearsons),
        "pearson_delta_de_mean": mean_pearson,
    })

    print(
        f"epoch={epoch:2d}  "
        f"组合数={len(pearsons):2d}  "
        f"Pearson均值={mean_pearson:.6f}"
    )


# =========================
# 保存结果
# =========================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    f"{TARGET_CELL}.csv",
    index=False,
)

print("\n最终结果：")
print(summary_df.to_string(index=False))
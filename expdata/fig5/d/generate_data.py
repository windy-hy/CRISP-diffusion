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

SPLITS = [
    "split_drugs",
    "split_drugs2",
    "split_drugs3",
]

SEEDS = [
    0,
    42,
    123,
]
# SEEDS = [
#     1327,
#     1337,
#     1347,
# ]


CELL_MAP = {
    "split_drugs": "MCF7",
    "split_drugs2": "A549",
    "split_drugs3": "K562",
}


CONFIG_PATH = "../../../experiments/configs/sci.yaml"

DATA_PATH = (
    "../../../data/sciplex3/"
    "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
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


def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)



# =========================
# 保存所有结果
# =========================

all_results = []



# =========================
# 9个实验循环
# =========================

for SPLIT_KEY in SPLITS:

    TARGET_CELL = CELL_MAP[SPLIT_KEY]


    for SEED in SEEDS:


        print("\n==============================")
        print("Running:")
        print("split =", SPLIT_KEY)
        print("seed =", SEED)
        print("cell =", TARGET_CELL)
        print("==============================\n")


        MODEL_PATH = (
            f"../../../model/sci/fcfp4_mmd0.0_loss0.5l1l2+0.2_64_unseendrugs/"
            f"{SPLIT_KEY}/seed{SEED}/model_300_split.pt"
        )


        # -------------------------
        # 加载配置
        # -------------------------

        args = load_config(CONFIG_PATH)

        args["dataset"]["split_key"] = SPLIT_KEY
        args["dataset"]["adata_obj"] = DATA_PATH
        args["model"]["seed"] = SEED



        # -------------------------
        # 初始化数据
        # -------------------------

        exp = Trainer()


        exp.init_dataset(
            **args["dataset"],
            seed=SEED,
        )


        treated_dataset = exp.datasets["ood_treated"]
        control_dataset = exp.datasets["ood_control"]



        # -------------------------
        # 找目标药物
        # -------------------------

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
            doses = doses[:,0]


        condition_table = pd.DataFrame({

            "pert_category": pert_categories,
            "cell_type": cell_types,
            "condition": drug_names,
            "dose_val": doses.astype(float),

        })


        condition_table = condition_table[
            (condition_table["cell_type"] == TARGET_CELL)
            &
            (condition_table["condition"].isin(UNSEEN_DRUGS))
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



        print(
            "drug:",
            condition_table["condition"].nunique(),
            "conditions:",
            len(condition_table)
        )


        if condition_table["condition"].nunique()!=9:
            raise ValueError(
                "药物数量错误"
            )


        if len(condition_table)!=36:
            raise ValueError(
                "组合数量错误"
            )



        # -------------------------
        # 加载模型
        # -------------------------

        exp.load_model(
            MODEL_PATH
        )


        exp.autoencoder.to(
            exp.device
        )

        exp.autoencoder.eval()



        np.random.seed(SEED)
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)



        # -------------------------
        # evaluate
        # -------------------------

        with torch.no_grad():

            metrics_all, eval_score_dict, pred_dict = evaluate(
                autoencoder=exp.autoencoder,
                treated_dataset=treated_dataset,
                control_dataset=control_dataset,
            )



        # -------------------------
        # 保存36个结果
        # -------------------------

        for _, row in condition_table.iterrows():

            pert_category = row["pert_category"]


            score = eval_score_dict[pert_category]


            all_results.append({

                "split": SPLIT_KEY,

                "seed": SEED,

                "cell_type": TARGET_CELL,

                "condition": row["condition"],

                "dose_val": row["dose_val"],


                "CRISP_R2_DE":
                    to_float(
                        score["r2score_de"]
                    ),


                "CRISP_Pearson_Delta_DE":
                    to_float(
                        score["pearson_delta_de"]
                    ),


                "CRISP_Sinkhorn_DE":
                    to_float(
                        score["sinkhorn_de"]
                    ),

            })



# =========================
# 保存总结果
# =========================

result_df = pd.DataFrame(
    all_results
)


OUTPUT_PATH = (
    "SciPlex3_DIFFCRISP_all_split_seed.csv"
)


result_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n完成")
print(
    "总行数:",
    len(result_df)
)


print(
    result_df.groupby(
        [
            "split",
            "seed",
        ]
    )[
        [
            "CRISP_R2_DE",
            "CRISP_Pearson_Delta_DE",
            "CRISP_Sinkhorn_DE",
        ]
    ].mean()
)
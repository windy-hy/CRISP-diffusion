import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../../..").resolve()
sys.path.insert(0, str(ROOT))

from DIFFCRISP.trainer import Trainer


DATA_PATH = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
DRUG_PATH = ROOT / "data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet"
MODEL_ROOT = ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2"
OUTPUT_PATH = HERE / "DIFFCRISP_Figure2e.csv"

SPLITS = ["split", "split2", "split3"]
SEEDS = [0, 42, 123]

MODEL_FILE = "model_200_split.pt"
FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.5


def mean_de(matrix, idx_de):
    matrix = matrix[:, idx_de]

    if sp.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).reshape(-1)

    return np.asarray(matrix).mean(axis=0).reshape(-1)


def to_dense(adata):
    adata = adata.copy()

    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata


print("正在读取NIPS数据集……")

adata = sc.read_h5ad(DATA_PATH)
smile_df = pd.read_parquet(DRUG_PATH)

if "SMILES" in smile_df.columns:
    smile_df = smile_df.set_index("SMILES")

smile_df.index = smile_df.index.astype(str)

obs = adata.obs
var_names = pd.Index(adata.var_names.astype(str))
de_dict = adata.uns["rank_genes_groups_cov"]

cell_types = obs["cell_type"].astype(str).to_numpy()
cov_drugs = obs["cov_drug_name"].astype(str).to_numpy()

control_mask = (
    pd.to_numeric(
        obs["neg_control"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
    .to_numpy()
    == 1
)

rows = []
processed = set()


for split in SPLITS:

    print("\n" + "=" * 70)
    print("当前划分：", split)
    print("=" * 70)

    split_values = (
        obs[split]
        .astype(str)
        .str.lower()
        .to_numpy()
    )

    ood_ctrl_mask = (
        control_mask
        & (split_values == "ood")
    )

    print("\nOOD对照组数量：")

    ctrl_counts = (
        obs.loc[ood_ctrl_mask, "cell_type"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    print(ctrl_counts.to_string())
    print("OOD对照组总数：", int(ood_ctrl_mask.sum()))

    ood_treated_mask = (
        (~control_mask)
        & (split_values == "ood")
    )

    current_groups = (
        obs.loc[ood_treated_mask, "cov_drug_name"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    combinations = []
    ctrl_cache = {}

    for cov_drug in current_groups:

        if cov_drug in processed:
            continue

        true_idx = np.where(
            ood_treated_mask
            & (cov_drugs == cov_drug)
        )[0]

        if len(true_idx) == 0:
            continue

        cell_type = cell_types[true_idx[0]]

        ctrl_idx = np.where(
            ood_ctrl_mask
            & (cell_types == cell_type)
        )[0]

        if len(ctrl_idx) == 0:
            print("没有OOD对照组，跳过：", cov_drug)
            continue

        if cov_drug not in de_dict:
            print("没有DE基因，跳过：", cov_drug)
            continue

        # 字典中已经是50个，直接全部使用
        de_genes = np.asarray(
            de_dict[cov_drug]
        ).astype(str)

        idx_de = np.where(
            var_names.isin(de_genes)
        )[0]

        if len(idx_de) != 50:
            raise ValueError(
                cov_drug
                + " 匹配到的DE基因数量为 "
                + str(len(idx_de))
            )

        true_obs = obs.iloc[true_idx]

        smiles = (
            true_obs["SMILES"]
            .dropna()
            .astype(str)
            .unique()
        )

        doses = (
            pd.to_numeric(
                true_obs["dose_val"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        if len(smiles) != 1 or len(doses) != 1:
            raise ValueError(
                cov_drug
                + " 的SMILES或剂量不唯一"
            )

        if cell_type not in ctrl_cache:
            ctrl_cache[cell_type] = to_dense(
                adata[ctrl_idx]
            )

        ctrl_mean = mean_de(
            adata.X[ctrl_idx],
            idx_de,
        )

        true_mean = mean_de(
            adata.X[true_idx],
            idx_de,
        )

        combinations.append({
            "cov_drug": cov_drug,
            "cell_type": cell_type,
            "smile": smiles[0],
            "dose": float(doses[0]),
            "idx_de": idx_de,
            "ctrl_mean": ctrl_mean,
            "true_delta": true_mean - ctrl_mean,
            "adata_ctrl": ctrl_cache[cell_type],
        })

        processed.add(cov_drug)

    print("\n需要评价的组合数量：", len(combinations))

    for seed in SEEDS:

        model_path = (
            MODEL_ROOT
            / split
            / ("seed" + str(seed))
            / MODEL_FILE
        )

        print("\n加载模型：", model_path)

        exp = Trainer()
        exp.load_model(str(model_path))
        exp.autoencoder.eval()

        for number, item in enumerate(combinations, 1):

            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            with torch.no_grad():
                result = exp.get_prediction(
                    adata_ctrl=item["adata_ctrl"],
                    dose=item["dose"],
                    smile=item["smile"],
                    smile_df=smile_df,
                    FM_emb=FM_KEY,
                    guidance_scale=GUIDANCE_SCALE,
                )

            adata_pred = result[0] if isinstance(result, tuple) else result

            pred_mean = mean_de(
                adata_pred.X,
                item["idx_de"],
            )

            pred_delta = (
                pred_mean
                - item["ctrl_mean"]
            )

            pos_ratio = float(
                np.mean(
                    item["true_delta"]
                    * pred_delta
                    > 0
                )
            )

            print(
                "[" + str(number) + "/" + str(len(combinations)) + "]",
                item["cov_drug"],
                "seed=" + str(seed),
                "pos_ratio=" + str(round(pos_ratio, 4)),
            )

            rows.append({
                "split": split,
                "seed": seed,
                "pos_ratio": pos_ratio,
                "cov_drug": item["cov_drug"],
                "cell_type": item["cell_type"],
            })

            del result
            del adata_pred

        del exp
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


result_df = (
    pd.DataFrame(rows)
    .groupby(
        ["cov_drug", "cell_type"],
        as_index=False,
    )["pos_ratio"]
    .mean()
)

result_df["model"] = "DIFFCRISP"

result_df = result_df[
    [
        "pos_ratio",
        "model",
        "cov_drug",
        "cell_type",
    ]
]

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n生成完成：", OUTPUT_PATH)
print("最终组合数量：", len(result_df))

print("\n各细胞类型组合数量：")
print(result_df["cell_type"].value_counts().sort_index())
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from DIFFCRISP.trainer import Trainer


# =========================================================
# 设置
# =========================================================

DATA = "../../../data/nips/nips_pp_scFM_resplit.h5ad"
MODEL_ROOT = "../../../model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2/"
EMB = "../../../data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet"

OUT = Path("./nips_predictions_diffcrisp")
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["split", "split2", "split3"]

MODEL_FILE = "model_200_split.pt"

FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.5

MAX_CELLS = 1000
SEED = 42


# =========================================================
# 工具函数
# =========================================================

def dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def sample_indices(indices, n=1000):
    indices = np.asarray(indices)

    if len(indices) <= n:
        return indices

    rng = np.random.default_rng(SEED)

    return np.sort(
        rng.choice(
            indices,
            size=n,
            replace=False
        )
    )


# =========================================================
# 读取数据
# =========================================================

adata = sc.read_h5ad(DATA)

smile_df = pd.read_parquet(EMB)
smile_df.index = smile_df.index.astype(str)


# =========================================================
# control
# =========================================================

if "control" in adata.obs.columns:

    control_mask = (
        pd.to_numeric(
            adata.obs["control"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy()
        == 1
    )

else:

    control_mask = (
        pd.to_numeric(
            adata.obs["neg_control"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy()
        == 1
    )


cell_types = (
    adata.obs["cell_type"]
    .astype(str)
    .to_numpy()
)

cov_drugs = (
    adata.obs["cov_drug_name"]
    .astype(str)
    .to_numpy()
)


manifest = []


# =========================================================
# 三个 split
# =========================================================

for split in SPLITS:

    print("\n==========================")
    print("Running:", split)
    print("==========================")

    model_path = (
        Path(MODEL_ROOT)
        / split
        / "seed0"
        / MODEL_FILE
    )

    exp = Trainer()
    exp.load_model(str(model_path))

    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()


    split_label = (
        adata.obs[split]
        .astype(str)
        .str.lower()
        .to_numpy()
    )


    # OOD treated
    ood_mask = (
        (split_label == "ood")
        & (~control_mask)
    )


    combinations = (
        adata.obs.loc[
            ood_mask,
            "cov_drug_name"
        ]
        .astype(str)
        .unique()
    )


    split_dir = OUT / split
    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for i, combo in enumerate(
        combinations,
        1
    ):

        # =============================================
        # treated
        # =============================================

        treated_idx = np.where(
            ood_mask
            & (cov_drugs == combo)
        )[0]

        if len(treated_idx) < 6:
            continue


        treated_obs = adata.obs.iloc[
            treated_idx
        ]


        cell_type = str(
            treated_obs["cell_type"].iloc[0]
        )

        drug = str(
            treated_obs["condition"].iloc[0]
        )

        dose = float(
            treated_obs["dose_val"].iloc[0]
        )

        smile = str(
            treated_obs["SMILES"].iloc[0]
        )


        if smile not in smile_df.index:
            print("缺少 embedding:", drug)
            continue


        # =============================================
        # 同 cell type control
        # =============================================

        ctrl_idx = np.where(
            control_mask
            & (cell_types == cell_type)
        )[0]

        if len(ctrl_idx) < 6:
            continue


        # 固定抽样
        treated_idx = sample_indices(
            treated_idx,
            MAX_CELLS
        )

        ctrl_idx = sample_indices(
            ctrl_idx,
            MAX_CELLS
        )


        ctrl = adata[
            ctrl_idx
        ].copy()

        ctrl.X = dense(
            ctrl.X
        ).astype(np.float32)


        print(
            f"[{i}/{len(combinations)}]",
            split,
            "|",
            cell_type,
            "|",
            drug,
            "| dose =",
            dose,
            "| treated =",
            len(treated_idx),
            "| ctrl =",
            len(ctrl_idx),
        )


        # =============================================
        # Prediction
        # =============================================

        with torch.inference_mode():

            pred = exp.get_prediction(
                ctrl,
                dose=dose,
                smile=smile,
                smile_df=smile_df,
                FM_emb=FM_KEY,
                guidance_scale=GUIDANCE_SCALE
            )


        if isinstance(pred, tuple):
            pred = pred[0]


        pred_matrix = dense(
            pred.X
        ).astype(np.float32)


        # =============================================
        # 保存
        #
        # 只保存：
        # 1. prediction
        # 2. treated index
        # 3. control index
        #
        # true/control以后从原始h5ad重新读取
        # =============================================

        file_name = (
            f"{i:04d}.npz"
        )

        output_file = (
            split_dir
            / file_name
        )


        np.savez(
            output_file,

            pred=pred_matrix,

            treated_idx=treated_idx,

            ctrl_idx=ctrl_idx,
        )


        manifest.append({
            "split": split,
            "file": str(output_file),

            "cov_drug": combo,

            "cell_type": cell_type,

            "drug": drug,

            "dose": dose,

            "smile": smile,

            "n_treated": len(treated_idx),

            "n_ctrl": len(ctrl_idx),

            "n_pred": pred_matrix.shape[0],
        })


        # 每次保存一次，防止中断
        pd.DataFrame(
            manifest
        ).to_csv(
            OUT / "manifest.csv",
            index=False
        )


    del exp

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n完成")

print(
    "预测目录：",
    OUT
)

print(
    "组合数：",
    len(manifest)
)
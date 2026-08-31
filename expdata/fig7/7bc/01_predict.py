from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from DIFFCRISP.trainer import Trainer


# =========================================================
# 路径
# =========================================================

DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"

MODEL = "../../../model/sci/split3/seed0/model_300_split.pt"

EMB = "../../../data/drug_embeddings/fcfp4_1024_embedding_lincs_sciplex3.parquet"


ROOT = Path("./mcf7_screening")

PRED_DIR = ROOT / "predictions"

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 设置
# =========================================================

SPLIT = "split3"

CELL = "MCF7"

DOSE = 1.0

MAX_CELLS = 1000

SEED = 42


# =========================================================
# 工具
# =========================================================

def dense(x):

    if sp.issparse(x):
        return x.toarray()

    return np.asarray(x)


def sample_idx(idx):

    idx = np.asarray(idx)

    if len(idx) <= MAX_CELLS:
        return idx

    rng = np.random.default_rng(
        SEED
    )

    idx = rng.choice(
        idx,
        size=MAX_CELLS,
        replace=False
    )

    return np.sort(idx)


# =========================================================
# 数据
# =========================================================

adata = sc.read_h5ad(
    DATA
)


smile_df = pd.read_parquet(
    EMB
)

smile_df.index = (
    smile_df.index
    .astype(str)
)


# =========================================================
# 加载 DiffCRISP
# =========================================================

exp = Trainer()

exp.load_model(
    MODEL
)


# =========================================================
# MCF7 OOD control
# =========================================================

ctrl_mask = (

    (adata.obs["control"].to_numpy() == 1)

    &

    (
        adata.obs[SPLIT]
        .astype(str)
        .to_numpy()
        == "ood"
    )

    &

    (
        adata.obs["cell_type"]
        .astype(str)
        .to_numpy()
        == CELL
    )
)


ctrl_idx = np.where(
    ctrl_mask
)[0]


ctrl_idx = sample_idx(
    ctrl_idx
)


print(
    "Control cells:",
    len(ctrl_idx)
)


ctrl = adata[
    ctrl_idx
].copy()


ctrl.X = dense(
    ctrl.X
).astype(
    np.float32
)


# =========================================================
# 所有 split3 OOD MCF7 药物
# =========================================================

drug_mask = (

    (
        adata.obs[SPLIT]
        .astype(str)
        == "ood"
    )

    &

    (
        adata.obs["cell_type"]
        .astype(str)
        == CELL
    )

    &

    (
        adata.obs["dose_val"]
        == DOSE
    )

    &

    (
        adata.obs["control"]
        == 0
    )
)


drugs = (

    adata.obs.loc[
        drug_mask,
        "condition"
    ]

    .dropna()

    .astype(str)

    .unique()
)


print(
    "OOD drugs:",
    len(drugs)
)


# =========================================================
# Prediction
# =========================================================

manifest = []


for i, drug in enumerate(
    drugs,
    1
):

    print(
        f"[{i}/{len(drugs)}]",
        drug
    )


    mask = (

        (
            adata.obs["condition"]
            .astype(str)
            == drug
        )

        &

        (
            adata.obs[SPLIT]
            .astype(str)
            == "ood"
        )

        &

        (
            adata.obs["cell_type"]
            .astype(str)
            == CELL
        )

        &

        (
            adata.obs["dose_val"]
            == DOSE
        )

        &

        (
            adata.obs["control"]
            == 0
        )
    )


    treated_idx = np.where(
        mask
    )[0]


    if len(treated_idx) <= 5:

        print("  treated cells too few")
        continue


    treated_idx = sample_idx(
        treated_idx
    )


    treated_obs = (
        adata.obs
        .iloc[treated_idx]
    )


    smiles = (

        treated_obs[
            "SMILES"
        ]

        .dropna()

        .astype(str)
    )


    if smiles.empty:

        print("  no SMILES")
        continue


    smile = smiles.iloc[0]


    if smile not in smile_df.index:

        print("  SMILES not found")
        continue


    # =====================================================
    # DiffCRISP inference
    # =====================================================

    with torch.inference_mode():

        pred = exp.get_prediction(

            ctrl,

            dose=DOSE,

            smile=smile,

            smile_df=smile_df,

            FM_emb="X_scGPT",

            guidance_scale=1.5
        )


    if isinstance(
        pred,
        tuple
    ):

        pred = pred[0]


    file = f"{i:04d}.npz"


    np.savez(

        PRED_DIR / file,

        pred=dense(
            pred.X
        ).astype(
            np.float32
        ),

        treated_idx=treated_idx,

        ctrl_idx=ctrl_idx
    )


    manifest.append({

        "drug": drug,

        "smile": smile,

        "file": file,

        "n_true": len(
            treated_idx
        ),

        "n_ctrl": len(
            ctrl_idx
        )
    })


    pd.DataFrame(
        manifest
    ).to_csv(

        ROOT / "manifest.csv",

        index=False
    )


print(
    "\nPrediction 完成：",
    len(manifest)
)
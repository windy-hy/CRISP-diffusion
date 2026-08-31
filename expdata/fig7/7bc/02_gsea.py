from pathlib import Path

import re

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gseapy as gp


# =========================================================
# 路径
# =========================================================

DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"

GMT = "../../../data/gsea/c2.cp.v2024.1.Hs.symbols.gmt"


ROOT = Path("./mcf7_screening")

PRED_DIR = ROOT / "predictions"

GSEA_DIR = ROOT / "gsea"


GSEA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 设置
# =========================================================

SEED = 42

PERMUTATION_NUM = 1000


# =========================================================
# 工具
# =========================================================

def dense(x):

    if sp.issparse(x):
        return x.toarray()

    return np.asarray(x)


def safe_name(name):

    return re.sub(

        r'[<>:"/\\|?*]',

        "_",

        str(name)

    ).rstrip(
        ". "
    )


def run_gsea(
    x,
    ctrl,
    genes
):

    x_df = pd.DataFrame(

        dense(x).T,

        index=genes
    )


    ctrl_df = pd.DataFrame(

        dense(ctrl).T,

        index=genes
    )


    x_df.columns = (
        ["Perturb"]
        * x_df.shape[1]
    )


    ctrl_df.columns = (
        ["Ctrl"]
        * ctrl_df.shape[1]
    )


    df = pd.concat(

        [
            x_df,
            ctrl_df
        ],

        axis=1
    )


    result = gp.gsea(

        data=df,

        gene_sets=GMT,

        cls=list(
            df.columns
        ),

        min_size=10,

        permutation_type="phenotype",

        permutation_num=PERMUTATION_NUM,

        method="signal_to_noise",

        outdir=None,

        threads=4,

        seed=SEED

    ).res2d.copy()


    result["NES"] = pd.to_numeric(

        result["NES"],

        errors="coerce"
    )


    result[
        "FDR q-val"
    ] = pd.to_numeric(

        result[
            "FDR q-val"
        ],

        errors="coerce"
    )


    return result


# =========================================================
# 数据
# =========================================================

adata = sc.read_h5ad(
    DATA
)


genes = (

    adata.var_names

    .astype(str)

    .to_numpy()
)


manifest = pd.read_csv(

    ROOT / "manifest.csv"
)


completed = []


# =========================================================
# 每个药 GSEA
# =========================================================

for i, row in manifest.iterrows():

    drug = row[
        "drug"
    ]


    print(
        f"[{i+1}/{len(manifest)}]",
        drug
    )


    drug_file = safe_name(
        drug
    )


    truth_file = (

        GSEA_DIR
        / f"{drug_file}_gsea_truth.csv"
    )


    pred_file = (

        GSEA_DIR
        / f"{drug_file}_gsea_pred.csv"
    )


    # 已完成直接跳过
    if (
        truth_file.exists()
        and pred_file.exists()
    ):

        print(
            "  已存在，跳过"
        )

        completed.append(
            drug
        )

        continue


    saved = np.load(

        PRED_DIR
        / row["file"]
    )


    pred = saved[
        "pred"
    ]


    true = dense(

        adata[
            saved[
                "treated_idx"
            ]
        ].X
    )


    ctrl = dense(

        adata[
            saved[
                "ctrl_idx"
            ]
        ].X
    )


    # =====================================================
    # GSEA
    # =====================================================

    true_gsea = run_gsea(

        true,

        ctrl,

        genes
    )


    pred_gsea = run_gsea(

        pred,

        ctrl,

        genes
    )


    # =====================================================
    # 显著结果
    # =====================================================

    true_sig = true_gsea[

        (
            true_gsea[
                "NES"
            ].abs()
            > 1
        )

        &

        (
            true_gsea[
                "FDR q-val"
            ]
            < 0.25
        )

    ].copy()


    pred_sig = pred_gsea[

        (
            pred_gsea[
                "NES"
            ].abs()
            > 1
        )

        &

        (
            pred_gsea[
                "FDR q-val"
            ]
            < 0.05
        )

    ].copy()


    # 和原 CRISP 一样：
    # truth 没有任何显著 pathway 的药不进入 screening
    if len(
        true_sig
    ) == 0:

        print(
            "  no significant truth pathway"
        )

        continue


    true_sig.to_csv(

        truth_file,

        index=False
    )


    pred_sig.to_csv(

        pred_file,

        index=False
    )


    completed.append(
        drug
    )


    pd.DataFrame({

        "drug": completed

    }).to_csv(

        ROOT
        / "gsea_manifest.csv",

        index=False
    )


print(
    "\nGSEA 完成：",
    len(completed)
)
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from DIFFCRISP.trainer import Trainer


# =========================
# 路径与参数
# =========================

NIPS_PATH = "../../../data/nips/nips_pp_scFM_resplit.h5ad"
PBMC_PATH = "../../../data/pbmc/pbmc_bench_pp_all.h5ad"

DRUG_EMB_PATH = (
    "../../../data/drug_embeddings/"
    "fcfp4_1024_embedding_lincs_nips.parquet"
)

MODEL_PATH = "../../../model/nips/split/seed0/model_200_split.pt"
OUTPUT_FILE = "fig4c_diffcrisp.csv"

FM_KEY = "X_scGPT_blood"
GUIDANCE_SCALE = 1.5


NIPS_CELL = "NK cells"
PBMC_CELL = "Natural killer cell"


# 运行NIPS模型时，predict中的CFG溶剂设置必须是：
# DMSO_INDEX = 43
# DMSO_DOSE = 1.0


# =========================
# 原文图4c的13种seen drugs
# =========================

SEEN_DRUGS = [
    "Belinostat",
    "CHIR99021",
    "Crizotinib",
    "Dabrafenib",
    "Dactolisib",
    "Foretinib",
    "Idelalisib",
    "LDN193189",
    "Linagliptin",
    "ODemethylatedAdapalene",
    "Palbociclib",
    "Penfluridol",
    "PorcnInhibitorIII",
]


# PBMC-Bench中的8个原始测序批次
RAW_BATCHES = [
    "Drop-seq",
    "Smart-seq2",
    "inDrops",
    "10x Chromium (v2)",
    "10x Chromium (v2) A",
    "10x Chromium (v2) B",
    "10x Chromium (v3)",
    "CEL-Seq2",
]


# 画图时，把3个10x v2批次归为同一横坐标
BATCH_GROUP = {
    "Drop-seq": "Drop-seq",
    "Smart-seq2": "Smart-seq2",
    "inDrops": "inDrops",
    "10x Chromium (v2)": "10x Chromium (v2)",
    "10x Chromium (v2) A": "10x Chromium (v2)",
    "10x Chromium (v2) B": "10x Chromium (v2)",
    "10x Chromium (v3)": "10x Chromium (v3)",
    "CEL-Seq2": "CEL-Seq2",
}


def mean_expression(adata):
    """计算每个基因的细胞平均表达。"""
    return np.asarray(
        adata.X.mean(axis=0)
    ).reshape(-1)


# =========================
# 读取数据
# =========================

nips = sc.read_h5ad(NIPS_PATH)
pbmc = sc.read_h5ad(PBMC_PATH)




drug_emb = pd.read_parquet(DRUG_EMB_PATH)
drug_emb.index = drug_emb.index.astype(str)

# 防止SMILES索引重复
drug_emb = drug_emb[
    ~drug_emb.index.duplicated(keep="first")
]

drug_to_smile = dict(
    zip(
        nips.obs["condition"].astype(str),
        nips.obs["SMILES"].astype(str),
    )
)


# =========================
# 统一两个数据集的基因名称
# =========================

if "gene_id" in nips.var.columns:
    nips.var_names = (
        nips.var["gene_id"]
        .astype(str)
        .to_numpy()
    )

if "gene_name" in pbmc.var.columns:
    pbmc.var_names = (
        pbmc.var["gene_name"]
        .astype(str)
        .to_numpy()
    )

nips.var_names_make_unique()
pbmc.var_names_make_unique()

# 顺序跟随NIPS
overlap = nips.var_names.intersection(
    pbmc.var_names
)

print("NIPS细胞数：", nips.n_obs)
print("PBMC细胞数：", pbmc.n_obs)
print("重叠基因数：", len(overlap))


# =========================
# 检查药物和PBMC批次
# =========================

nips_drugs = set(
    nips.obs["condition"].astype(str)
)

missing_drugs = [
    drug for drug in SEEN_DRUGS
    if drug not in nips_drugs
]

if missing_drugs:
    raise ValueError(
        f"NIPS中缺少药物：{missing_drugs}"
    )

available_batches = set(
    pbmc.obs["Method"].astype(str)
)

missing_batches = [
    batch for batch in RAW_BATCHES
    if batch not in available_batches
]

if missing_batches:
    raise ValueError(
        f"PBMC中缺少批次：{missing_batches}"
    )

print("\n图4c seen drugs数量：", len(SEEN_DRUGS))
print(SEEN_DRUGS)


# =========================
# 加载DiffCRISP模型
# =========================

exp = Trainer()
exp.load_model(MODEL_PATH)

exp.autoencoder.to(exp.device)
exp.autoencoder.eval()


# =========================
# NIPS真实NK对照均值
# =========================

neg_control = pd.to_numeric(
    nips.obs["neg_control"],
    errors="coerce",
)

dose_values = pd.to_numeric(
    nips.obs["dose_val"],
    errors="coerce",
)

nips_control = nips[
    (nips.obs["cell_type"].astype(str) == NIPS_CELL)
    & (neg_control == 1)
].copy()

if nips_control.n_obs == 0:
    raise ValueError("没有找到NIPS NK对照细胞")

ctrl_nips_mean = mean_expression(
    nips_control[:, overlap]
)

print("\nNIPS NK对照细胞数：", nips_control.n_obs)


# =========================
# 提前准备PBMC的8个NK批次
# =========================

for drug in SEEN_DRUGS:
    mask = (
        (nips.obs["cell_type"].astype(str) == NIPS_CELL)
        & (nips.obs["condition"].astype(str) == drug)
        & (neg_control == 0)
    )

    doses = sorted(
        pd.to_numeric(
            nips.obs.loc[mask, "dose_val"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    print(f"{drug:25s}: {doses}")


pbmc_batches = {}
pbmc_control_means = {}

for batch in RAW_BATCHES:

    adata_batch = pbmc[
        (pbmc.obs["Method"].astype(str) == batch)
        & (
            pbmc.obs["cell_type"].astype(str)
            == PBMC_CELL
        )
    ].copy()

    if adata_batch.n_obs == 0:
        raise ValueError(
            f"{batch}中没有找到NK细胞"
        )

    pbmc_batches[batch] = adata_batch

    pbmc_control_means[batch] = mean_expression(
        adata_batch[:, overlap]
    )

    print(
        f"{batch:24s} "
        f"NK细胞数={adata_batch.n_obs}"
    )


# =========================
# 逐药物、逐批次预测
# =========================

de_dict = nips.uns["rank_genes_groups_cov"]
results = []

for drug_index, drug in enumerate(SEEN_DRUGS, start=1):

    print(
        f"\n[{drug_index}/{len(SEEN_DRUGS)}] "
        f"处理药物：{drug}"
    )

    # 当前药物在NIPS中的真实处理细胞
    treated = nips[
        (nips.obs["cell_type"].astype(str) == NIPS_CELL)
        & (nips.obs["condition"].astype(str) == drug)
        & (neg_control == 0)
        ].copy()

    if treated.n_obs == 0:
        print("跳过：没有真实处理细胞")
        continue

    # 自动读取当前药物唯一的剂量
    drug_doses = (
        pd.to_numeric(
            treated.obs["dose_val"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(drug_doses) != 1:
        raise ValueError(
            f"{drug} 的有效剂量数量不是1：{drug_doses}"
        )

    dose = float(drug_doses[0])

    print(
        f"真实处理细胞数={treated.n_obs}，"
        f"剂量={dose}"
    )

    # NIPS中的真实药物响应
    true_delta = (
        mean_expression(treated[:, overlap])
        - ctrl_nips_mean
    )

    # 当前细胞类型和药物对应的Top 50 DE基因
    de_key = f"{NIPS_CELL}_{drug}"

    if de_key not in de_dict:
        print("跳过：没有DE基因键", de_key)
        continue

    top_de = [
        str(gene)
        for gene in de_dict[de_key][:50]
        if str(gene) in overlap
    ]

    if len(top_de) == 0:
        print("跳过：Top DE基因不在重叠基因中")
        continue

    de_index = overlap.get_indexer(top_de)

    # 药物SMILES
    smile = drug_to_smile.get(drug)

    if smile is None:
        print("跳过：没有SMILES", drug)
        continue

    if smile not in drug_emb.index:
        print("跳过：药物嵌入中没有SMILES", drug)
        continue

    # 分别对8个PBMC批次预测
    for batch in RAW_BATCHES:

        pbmc_control = pbmc_batches[batch]

        with torch.inference_mode():

            pred = exp.get_prediction(
                pbmc_control,
                dose=dose,
                smile=smile,
                smile_df=drug_emb,
                FM_emb=FM_KEY,
                guidance_scale=GUIDANCE_SCALE,
            )

        if pred.n_vars != nips.n_vars:
            raise ValueError(
                f"预测基因数={pred.n_vars}，"
                f"NIPS基因数={nips.n_vars}，两者不一致"
            )

        # DiffCRISP解码器输出顺序与NIPS训练基因顺序一致
        pred.var_names = nips.var_names.copy()

        # PBMC预测响应
        pred_delta = (
            mean_expression(pred[:, overlap])
            - pbmc_control_means[batch]
        )

        true_de = true_delta[de_index]
        pred_de = pred_delta[de_index]

        valid = (
            np.isfinite(true_de)
            & np.isfinite(pred_de)
            & (true_de != 0)
        )

        if valid.sum() == 0:
            print(
                f"跳过：{drug} {batch} "
                f"没有有效DE基因"
            )
            continue

        # 真实和预测变化方向一致的基因比例
        direction_acc = np.mean(
            np.sign(true_de[valid])
            == np.sign(pred_de[valid])
        )

        results.append({
            "direction_acc": float(direction_acc),
            "drug": drug,
            "dose": dose,
            "cell_type": NIPS_CELL,
            "batch_original": batch,
            "batch_group": BATCH_GROUP[batch],
            "n_cells": int(pbmc_control.n_obs),
            "n_de_genes": int(valid.sum()),
            "method": "DiffCRISP",
        })

        print(
            f"{batch:24s} "
            f"n={pbmc_control.n_obs:4d}  "
            f"DE={valid.sum():2d}  "
            f"acc={direction_acc:.4f}"
        )

        del pred

    # 只覆盖保存同一个文件，不再产生partial文件
    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# 最终保存和检查
# =========================

result_df = pd.DataFrame(results)

if result_df.empty:
    raise ValueError("没有成功生成任何结果")

result_df = result_df.sort_values(
    [
        "batch_group",
        "batch_original",
        "drug",
    ]
).reset_index(drop=True)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n=========================")
print("全部完成")
print("=========================")

print("保存文件：", OUTPUT_FILE)
print("最终行数：", len(result_df))
print("理论最大行数：", len(SEEN_DRUGS) * len(RAW_BATCHES))

print("\n每个原始批次的结果数量：")
print(
    result_df.groupby("batch_original")
    .size()
    .to_string()
)

print("\n画图分组后的结果数量：")
print(
    result_df.groupby("batch_group")
    .size()
    .to_string()
)

print("\n各批次方向准确率均值：")
print(
    result_df.groupby("batch_group")[
        "direction_acc"
    ]
    .mean()
    .round(4)
    .to_string()
)
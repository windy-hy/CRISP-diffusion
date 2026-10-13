import gc
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from DIFFCRISP.trainer import Trainer


# =========================
# 路径和参数
# =========================

NIPS_PATH = "../../../data/nips/nips_pp_scFM_resplit.h5ad"
PBMC_PATH = "../../../data/pbmc/pbmc_bench_pp_all.h5ad"

DRUG_EMB_PATH = (
    "../../../data/drug_embeddings/"
    "fcfp4_1024_embedding_lincs_nips.parquet"
)

# 图4f使用原始split模型，不是unseen模型
MODEL_PATH = (
    "../../../model/nips/split/seed0/"
    "model_150_split.pt"
)

OUTPUT_FILE = "figure4f_diffcrisp.csv"

DRUG = "Dactolisib"
NIPS_CELL = "NK cells"
PBMC_CELL = "Natural killer cell"
PBMC_BATCH = "10x Chromium (v3)"

FM_KEY = "X_scGPT_blood"
GUIDANCE_SCALE = 2


# 使用NIPS模型预测时，predict中的CFG溶剂设置应为：
# DMSO_INDEX = 43
# DMSO_DOSE = 1.0


def mean_expression(adata):
    """计算每个基因的细胞平均表达。"""
    return np.asarray(
        adata.X.mean(axis=0)
    ).reshape(-1)


# =========================
# 基本检查
# =========================

for path in [
    NIPS_PATH,
    PBMC_PATH,
    DRUG_EMB_PATH,
    MODEL_PATH,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


# =========================
# 读取数据
# =========================

nips = sc.read_h5ad(NIPS_PATH)
pbmc = sc.read_h5ad(PBMC_PATH)

print("NIPS原始形状：", nips.shape)
print("PBMC原始形状：", pbmc.shape)


# =========================
# 设置基因名称
# =========================

if "gene_id" in nips.var.columns:
    nips.var_names = pd.Index(
        nips.var["gene_id"].astype(str)
    )

if "gene_name" in pbmc.var.columns:
    pbmc.var_names = pd.Index(
        pbmc.var["gene_name"].astype(str)
    )


# NIPS预测输出对应NIPS原始基因顺序，因此不能随意去重或重排
if nips.var_names.has_duplicates:
    duplicated = nips.var_names[
        nips.var_names.duplicated()
    ].unique()

    raise ValueError(
        "NIPS基因名存在重复，不能直接对齐预测输出："
        f"{duplicated[:10].tolist()}"
    )


# PBMC可能存在重复gene_name，保留第一次出现
pbmc_keep = ~pbmc.var_names.duplicated(
    keep="first"
)

print(
    "PBMC重复基因数量：",
    int((~pbmc_keep).sum())
)

pbmc = pbmc[:, pbmc_keep].copy()


# 共同基因顺序严格跟随NIPS
pbmc_gene_set = set(pbmc.var_names)

overlap = pd.Index([
    gene
    for gene in nips.var_names
    if gene in pbmc_gene_set
])

if len(overlap) == 0:
    raise ValueError("NIPS和PBMC没有共同基因")

print("NIPS基因数：", nips.n_vars)
print("PBMC去重后基因数：", pbmc.n_vars)
print("共同基因数：", len(overlap))


# =========================
# 检查FM
# =========================

if FM_KEY not in pbmc.obsm:
    raise KeyError(
        f"PBMC的obsm中没有 {FM_KEY}，"
        f"当前可用：{list(pbmc.obsm.keys())}"
    )


# =========================
# 药物信息
# =========================

drug_emb = pd.read_parquet(
    DRUG_EMB_PATH
)

drug_emb.index = (
    drug_emb.index
    .astype(str)
    .str.strip()
)

drug_emb = drug_emb[
    ~drug_emb.index.duplicated(keep="first")
]


drug_info = (
    nips.obs[
        ["condition", "SMILES"]
    ]
    .astype(str)
    .drop_duplicates()
)

drug_rows = drug_info[
    drug_info["condition"] == DRUG
]

if drug_rows.empty:
    raise ValueError(
        f"NIPS中没有药物：{DRUG}"
    )

smiles = drug_rows["SMILES"].iloc[0]

if smiles not in drug_emb.index:
    raise KeyError(
        f"药物嵌入文件中没有该SMILES：{smiles}"
    )

print("药物：", DRUG)
print("SMILES：", smiles)
print("药物嵌入维度：", drug_emb.shape[1])


# =========================
# 筛选NIPS真实细胞
# =========================

neg_control = pd.to_numeric(
    nips.obs["neg_control"],
    errors="coerce"
)

nips_cell_type = (
    nips.obs["cell_type"]
    .astype(str)
)

nips_condition = (
    nips.obs["condition"]
    .astype(str)
)


# NIPS NK对照细胞
nips_control_mask = (
    nips_cell_type.eq(NIPS_CELL)
    & neg_control.eq(1)
)

nips_control = nips[
    nips_control_mask
].copy()

if nips_control.n_obs == 0:
    raise ValueError(
        f"NIPS中没有找到{NIPS_CELL}对照细胞"
    )


# NIPS NK Dactolisib真实处理细胞
nips_treated_mask = (
    nips_cell_type.eq(NIPS_CELL)
    & nips_condition.eq(DRUG)
    & neg_control.eq(0)
)

nips_treated = nips[
    nips_treated_mask
].copy()

if nips_treated.n_obs == 0:
    raise ValueError(
        f"NIPS中没有找到{NIPS_CELL}的{DRUG}处理细胞"
    )


# 自动读取剂量
drug_doses = (
    pd.to_numeric(
        nips_treated.obs["dose_val"],
        errors="coerce"
    )
    .dropna()
    .unique()
)

if len(drug_doses) != 1:
    raise ValueError(
        f"{DRUG}存在多个有效剂量：{drug_doses}"
    )

dose = float(drug_doses[0])

print("\nNIPS对照细胞数：", nips_control.n_obs)
print("NIPS真实处理细胞数：", nips_treated.n_obs)
print("剂量：", dose)


# =========================
# 筛选PBMC的10x v3 NK细胞
# =========================

pbmc_mask = (
    pbmc.obs["Method"]
    .astype(str)
    .eq(PBMC_BATCH)
    &
    pbmc.obs["cell_type"]
    .astype(str)
    .eq(PBMC_CELL)
)

pbmc_control = pbmc[
    pbmc_mask
].copy()

if pbmc_control.n_obs == 0:
    raise ValueError(
        f"PBMC中没有找到："
        f"{PBMC_BATCH}，{PBMC_CELL}"
    )

print("PBMC预测输入细胞数：", pbmc_control.n_obs)


# =========================
# 真实药物响应
# =========================

nips_control_mean = mean_expression(
    nips_control[:, overlap]
)

nips_treated_mean = mean_expression(
    nips_treated[:, overlap]
)

# Ground truth：
# NIPS真实处理均值 - NIPS真实对照均值
gt = (
    nips_treated_mean
    - nips_control_mean
)


# PBMC原始对照均值
pbmc_control_mean = mean_expression(
    pbmc_control[:, overlap]
)


# =========================
# 加载DiffCRISP模型
# =========================

exp = Trainer()
exp.load_model(MODEL_PATH)

exp.autoencoder.to(exp.device)
exp.autoencoder.eval()


# =========================
# 预测PBMC药物处理结果
# =========================

with torch.inference_mode():

    pred_adata = exp.get_prediction(
        pbmc_control,
        dose=dose,
        smile=smiles,
        smile_df=drug_emb,
        FM_emb=FM_KEY,
        guidance_scale=GUIDANCE_SCALE,
    )


# DiffCRISP解码器输出应与NIPS训练基因完全一致
if pred_adata.n_vars != nips.n_vars:
    raise ValueError(
        f"预测基因数={pred_adata.n_vars}，"
        f"NIPS基因数={nips.n_vars}，"
        "两者不一致"
    )

pred_adata.var_names = nips.var_names.copy()


# PBMC预测处理均值
pred_treated_mean = mean_expression(
    pred_adata[:, overlap]
)

# Prediction：
# PBMC预测处理均值 - PBMC原始对照均值
pred = (
    pred_treated_mean
    - pbmc_control_mean
)


# =========================
# 获取Top 50 DE基因
# =========================

if "rank_genes_groups_cov" not in nips.uns:
    raise KeyError(
        "nips.uns中没有rank_genes_groups_cov"
    )

de_dict = nips.uns[
    "rank_genes_groups_cov"
]

de_key = f"{NIPS_CELL}_{DRUG}"

if de_key not in de_dict:
    print("找不到DE键：", de_key)
    print("部分可用键：")
    print(list(de_dict.keys())[:30])

    raise KeyError(de_key)


# Top 50中只保留NIPS和PBMC的共同基因
top_de = list(dict.fromkeys([
    str(gene)
    for gene in de_dict[de_key][:50]
    if str(gene) in overlap
]))

if len(top_de) == 0:
    raise ValueError(
        "Top 50 DE基因中没有共同基因"
    )


# 给每个共同基因标记是否属于DE基因
de_mask = overlap.isin(top_de)


# =========================
# 生成图4f表格
# =========================

result_df = pd.DataFrame({
    "gt": gt.astype(float),
    "pred": pred.astype(float),
    "de": de_mask.astype(bool),
})


# 删除异常值，正常情况下不应有
finite_mask = (
    np.isfinite(result_df["gt"])
    & np.isfinite(result_df["pred"])
)

if not finite_mask.all():
    print(
        "删除非有限数值行数：",
        int((~finite_mask).sum())
    )

    result_df = result_df[
        finite_mask
    ].reset_index(drop=True)


# 保存为与原始figure4f_CRISP.csv相同的三列
result_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================
# 结果检查
# =========================

de_rows = result_df[
    result_df["de"]
]

if len(de_rows) >= 2:
    de_pearson = np.corrcoef(
        de_rows["gt"],
        de_rows["pred"]
    )[0, 1]
else:
    de_pearson = np.nan


valid_direction = (
    np.isfinite(de_rows["gt"])
    & np.isfinite(de_rows["pred"])
    & de_rows["gt"].ne(0)
)

if valid_direction.sum() > 0:
    direction_acc = np.mean(
        np.sign(
            de_rows.loc[valid_direction, "gt"]
        )
        ==
        np.sign(
            de_rows.loc[valid_direction, "pred"]
        )
    )
else:
    direction_acc = np.nan


print("\n=========================")
print("图4f表格生成完成")
print("=========================")

print("保存位置：", OUTPUT_FILE)
print("总基因数：", len(result_df))
print("DE基因数：", int(result_df["de"].sum()))
print("非DE基因数：", int((~result_df["de"]).sum()))
print("DE基因Pearson：", round(float(de_pearson), 4))
print("DE基因方向准确率：", round(float(direction_acc), 4))

print("\n前5行：")
print(result_df.head().to_string(index=False))


del pred_adata
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
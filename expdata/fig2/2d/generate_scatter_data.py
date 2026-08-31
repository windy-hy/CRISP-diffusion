import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import DIFFCRISP.trainer as ct

data_path = "../../../data/nips/nips_pp_scFM_resplit.h5ad"
drug_embedding_path = (
    "../../../data/drug_embeddings/"
    "fcfp4_1024_embedding_lincs_nips.parquet"
)
model_path = (
    "../../../model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2/split2/seed0/"
    "model_200_split.pt"
)

# 数据划分
split_key = "split2"
split_value = "ood"

# 目标细胞类型
cell_type = "B cells"

# 目标药物和剂量
# drug = 'Palbociclib'
drug = "Idelalisib"
dose = 0.07092198581560284


# 如果有多个剂量，应改成真实列名，例如"dose_val"
dose_key = None

fm_key = "X_scGPT"
guidance_scale = 1.5
output_csv = f"DIFFCRISP_{drug}_Bcells.csv"

def mean_vector(matrix):
    """
    将稀疏矩阵或稠密矩阵按细胞维度求均值，
    返回一维numpy数组。
    """

    if sp.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).reshape(-1)

    return np.asarray(matrix).mean(axis=0).reshape(-1)


print("正在读取NIPS数据集……")
nips = sc.read(data_path)
var_names = np.asarray(nips.var_names)
print("数据形状：", nips.shape)


drug_smile_table = (
    nips.obs[["condition", "SMILES"]]
    .drop_duplicates(subset=["condition"])
    .set_index("condition")
)

if drug not in drug_smile_table.index:
    raise ValueError(
        "数据集中找不到药物："
        + drug
    )

smile = drug_smile_table.loc[
    drug,
    "SMILES",
]

print("药物：", drug)
print("SMILES：", smile)
print("剂量：", dose)


smile_df = pd.read_parquet(
    drug_embedding_path
)

control_mask = (
    (nips.obs["neg_control"] == 1)
    & (nips.obs[split_key] == split_value)
    & (nips.obs["cell_type"] == cell_type)
)

adata_ctrl = nips[control_mask].copy()

if adata_ctrl.n_obs == 0:
    raise ValueError(
        "没有找到对应的对照组细胞。"
    )

print("对照组细胞数量：", adata_ctrl.n_obs)


true_mask = (
    (nips.obs[split_key] == split_value)
    & (nips.obs["cell_type"] == cell_type)
    & (nips.obs["condition"] == drug)
)

# 如果一种药物具有多个剂量，必须增加剂量筛选
if dose_key is not None:
    if dose_key not in nips.obs.columns:
        raise ValueError(
            "数据中不存在剂量列："
            + dose_key
        )

    true_mask = (
        true_mask
        & np.isclose(
            nips.obs[dose_key].astype(float),
            dose,
        )
    )

adata_true = nips[true_mask].copy()

if adata_true.n_obs == 0:
    raise ValueError(
        "没有找到对应的真实药物处理组。"
    )

print("真实处理组细胞数量：", adata_true.n_obs)

print("正在加载模型……")

exp = ct.Trainer()
exp.load_model(model_path)

print("正在预测……")

prediction_result  = exp.get_prediction(
    adata_ctrl,
    dose=dose,
    smile=smile,
    smile_df=smile_df,
    FM_emb=fm_key,
    guidance_scale=guidance_scale,
)

if isinstance(prediction_result, tuple):
    adata_pred = prediction_result[0]
else:
    adata_pred = prediction_result

print("预测细胞数量：", adata_pred.n_obs)



de_genes_dict = nips.uns[
    "rank_genes_groups_cov"
]

de_key = cell_type + "_" + drug

if de_key not in de_genes_dict:
    raise ValueError(
        "差异基因字典中找不到："
        + de_key
    )

de_genes = np.asarray(
    de_genes_dict[de_key]
)

de_bool = np.isin(
    var_names,
    de_genes,
)

print("匹配到的DE基因数量：", de_bool.sum())

true_mean = mean_vector(
    adata_true.X
)

pred_mean = mean_vector(
    adata_pred.X
)

control_mean = mean_vector(
    adata_ctrl.X
)

true_delta = true_mean - control_mean
pred_delta = pred_mean - control_mean


data_df = pd.DataFrame(
    {
        "gene_index": np.arange(len(var_names)),
        "gt": true_delta,
        "pred": pred_delta,
        "de": de_bool,
    }
)

# 先画普通基因，后画DE基因
data_df = data_df.sort_values(
    by="de",
    ascending=True,
)

data_df.to_csv(
    output_csv,
    index=False,
)

print("CSV数据已保存：", output_csv)

print("\n前10行：")
print(data_df.head(10))
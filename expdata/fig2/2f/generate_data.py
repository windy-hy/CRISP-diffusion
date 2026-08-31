import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from DIFFCRISP.trainer import Trainer


DATA_PATH = ROOT / "data/nips/nips_pp_scFM_resplit.h5ad"
DRUG_PATH = ROOT / "data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet"
MODEL_ROOT = ROOT / "model/nips"
OUTPUT_DIR = Path(__file__).resolve().parent

SPLIT_KEY = "split2"
SEED = 0
MODEL_FILE = "model_150_split.pt"

CELL_TYPE = "T cells CD4+"
DRUG_NAME = "Dactolisib"
GENES = ["ZFAS1", "SNHG6"]

FM_KEY = "X_scGPT"
GUIDANCE_SCALE = 1.5


def gene_values(matrix, index):
    values = matrix[:, index]

    if sp.issparse(values):
        values = values.toarray()

    return np.asarray(values).reshape(-1)


def get_2sd_mask(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()
    std = values.std(ddof=0)

    if std == 0:
        return np.ones(len(values), dtype=bool)

    return np.abs(values - mean) <= 2 * std

print("正在读取数据……")

adata = sc.read_h5ad(DATA_PATH)
obs = adata.obs

print("数据形状：", adata.shape)


# =========================================================
# 基本字段
# =========================================================
neg_control = (
    pd.to_numeric(
        obs["neg_control"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

split_values = (
    obs[SPLIT_KEY]
    .astype(str)
    .str.strip()
    .str.lower()
)

cell_values = obs["cell_type"].astype(str)
drug_values = obs["condition"].astype(str)


# =========================================================
# OOD真实处理组
# =========================================================
treated_mask = (
    (neg_control == 0)
    & (split_values == "ood")
    & (cell_values == CELL_TYPE)
    & (drug_values == DRUG_NAME)
)

adata_true = adata[treated_mask].copy()

if adata_true.n_obs == 0:
    raise ValueError("没有找到真实OOD处理组")


# =========================================================
# OOD真实对照组
# =========================================================
control_mask = (
    (neg_control == 1)
    & (split_values == "ood")
    & (cell_values == CELL_TYPE)
)

adata_ctrl = adata[control_mask].copy()

if adata_ctrl.n_obs == 0:
    raise ValueError("没有找到OOD对照组")


# =========================================================
# 药物信息
# =========================================================
doses = (
    pd.to_numeric(
        adata_true.obs["dose_val"],
        errors="coerce",
    )
    .dropna()
    .unique()
)

smiles = (
    adata_true.obs["SMILES"]
    .dropna()
    .astype(str)
    .unique()
)

if len(doses) != 1:
    raise ValueError("真实处理组包含多个剂量：" + str(doses))

if len(smiles) != 1:
    raise ValueError("真实处理组包含多个SMILES：" + str(smiles))

dose = float(doses[0])
smile = smiles[0]


print("\n" + "=" * 55)
print("Split：", SPLIT_KEY)
print("Cell type：", CELL_TYPE)
print("Drug：", DRUG_NAME)
print("Dose：", dose)
print("真实处理数量 True：", adata_true.n_obs)
print("真实OOD对照数量 Ctrl：", adata_ctrl.n_obs)
print("=" * 55)


# =========================================================
# 药物指纹
# =========================================================
smile_df = pd.read_parquet(DRUG_PATH)

if "SMILES" in smile_df.columns:
    smile_df = smile_df.set_index("SMILES")

smile_df.index = smile_df.index.astype(str)
smile_df = smile_df.select_dtypes(include=[np.number])

if smile not in smile_df.index:
    raise KeyError("药物指纹中找不到SMILES：" + smile)


# =========================================================
# 加载单个seed模型
# =========================================================
model_path = (
    MODEL_ROOT
    / SPLIT_KEY
    / ("seed" + str(SEED))
    / MODEL_FILE
)

if not model_path.exists():
    raise FileNotFoundError("模型不存在：" + str(model_path))

print("\n加载模型：", model_path)

exp = Trainer()
exp.load_model(str(model_path))

exp.autoencoder.to(exp.device)
exp.autoencoder.eval()


# =========================================================
# 模型输入转稠密
# =========================================================
adata_ctrl_model = adata_ctrl.copy()

if sp.issparse(adata_ctrl_model.X):
    adata_ctrl_model.X = (
        adata_ctrl_model.X
        .toarray()
        .astype(np.float32)
    )
else:
    adata_ctrl_model.X = np.asarray(
        adata_ctrl_model.X,
        dtype=np.float32,
    )


np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 只预测一次
# =========================================================
print("\n正在预测……")

with torch.no_grad():
    result = exp.get_prediction(
        adata_ctrl=adata_ctrl_model,
        dose=dose,
        smile=smile,
        smile_df=smile_df,
        FM_emb=FM_KEY,
        guidance_scale=GUIDANCE_SCALE,
    )

adata_pred = result[0] if isinstance(result, tuple) else result


print("\n最终细胞数量：")
print("DIFFCRISP预测数量：", adata_pred.n_obs)
print("True真实处理数量：", adata_true.n_obs)
print("Ctrl真实OOD对照数量：", adata_ctrl.n_obs)

if adata_pred.n_obs != adata_ctrl.n_obs:
    raise ValueError(
        "预测数量与输入对照数量不一致："
        + str(adata_pred.n_obs)
        + " vs "
        + str(adata_ctrl.n_obs)
    )


# =========================================================
# 保存两个基因的分布数据
# =========================================================
var_names = pd.Index(adata.var_names.astype(str))

for gene in GENES:

    if gene not in var_names:
        raise KeyError("找不到基因：" + gene)

    gene_index = var_names.get_loc(gene)

    pred_raw = gene_values(
        adata_pred.X,
        gene_index,
    )

    true_raw = gene_values(
        adata_true.X,
        gene_index,
    )

    ctrl_raw = gene_values(
        adata_ctrl.X,
        gene_index,
    )

    # Ctrl决定Ctrl和Pred使用哪些输入细胞
    ctrl_keep = get_2sd_mask(ctrl_raw)
    ctrl = ctrl_raw[ctrl_keep]
    # pred = pred_raw[ctrl_keep]
    # 全部预测细胞用于绘图
    pred = pred_raw.copy()

    # True按照自身分布过滤
    true_keep = get_2sd_mask(true_raw)
    true = true_raw[true_keep]

    print("\n基因：", gene)
    print("Ctrl：", len(ctrl_raw), "→", len(ctrl))
    print("DIFFCRISP：", len(pred_raw), "→", len(pred))
    print("True：", len(true_raw), "→", len(true))

    result_df = pd.concat(
        [
            pd.DataFrame({
                "exp": pred,
                "label": "DIFFCRISP",
            }),
            pd.DataFrame({
                "exp": true,
                "label": "True",
            }),
            pd.DataFrame({
                "exp": ctrl,
                "label": "Ctrl",
            }),
        ],
        ignore_index=True,
    )

    output_path = OUTPUT_DIR / (
        "figure2f_"
        + gene
        + "_DIFFCRISP.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\n基因：", gene)
    print(result_df.groupby("label")["exp"].agg(["count", "mean", "std"]))
    print("已保存：", output_path)


print("\nFigure 2f数据生成完成。")
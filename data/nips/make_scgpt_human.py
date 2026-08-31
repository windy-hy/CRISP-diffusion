import numpy as np
import pandas as pd
import scanpy as sc


FULL_PATH = "nips_pp_scFM_resplit.h5ad"
CTRL_PATH = "nips_pp_scFM_resplit_ctrl_scgpt.h5ad"
OUTPUT_PATH = "nips_pp_scFM_resplit.h5ad"

FM_KEY = "X_scGPT_human"


full = sc.read_h5ad(FULL_PATH)
ctrl = sc.read_h5ad(CTRL_PATH)

print("完整数据：", full.shape)
print("对照数据：", ctrl.shape)
print("对照文件obsm：", list(ctrl.obsm.keys()))

if FM_KEY not in ctrl.obsm:
    raise KeyError(
        f"对照文件没有 {FM_KEY}，"
        f"当前有：{list(ctrl.obsm.keys())}"
    )

# 检查对照文件中的细胞是否全部存在于完整数据
missing_in_full = ctrl.obs_names.difference(
    full.obs_names
)

if len(missing_in_full) > 0:
    raise ValueError(
        f"有{len(missing_in_full)}个对照细胞无法在完整数据中匹配"
    )

# 检查完整数据的所有对照细胞是否都能在对照文件找到
full_control = pd.to_numeric(
    full.obs["neg_control"],
    errors="coerce"
).eq(1)

missing_control = full.obs_names[
    full_control
].difference(ctrl.obs_names)

if len(missing_control) > 0:
    raise ValueError(
        f"完整数据有{len(missing_control)}个对照细胞缺少scGPT-human嵌入"
    )

dim = ctrl.obsm[FM_KEY].shape[1]

# 完整n_obs行；处理细胞暂时为0
full_embedding = np.zeros(
    (full.n_obs, dim),
    dtype=np.float32
)

positions = full.obs_names.get_indexer(
    ctrl.obs_names
)

full_embedding[positions] = np.asarray(
    ctrl.obsm[FM_KEY],
    dtype=np.float32
)

full.obsm[FM_KEY] = full_embedding

print("写入后的嵌入形状：", full.obsm[FM_KEY].shape)
print("完整数据细胞数：", full.n_obs)

full.write_h5ad(OUTPUT_PATH)

print("保存完成：", OUTPUT_PATH)
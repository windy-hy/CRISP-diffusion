import scanpy as sc
import pandas as pd

data_path = (
    "../../../data/sciplex3/"
    "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

adata = sc.read_h5ad(data_path)

print([
    column
    for column in adata.obs.columns
    if "split" in column.lower()
])

split_key = "split_drugs"
target_cell = "K562"

obs = adata.obs.copy()

obs["_split"] = (
    obs[split_key]
    .astype(str)
    .str.lower()
)

print(
    pd.crosstab(
        [
            obs["cell_type"],
            obs["control"],
        ],
        obs["_split"],
    )
)

train_mask = obs["_split"].isin([
    "train",
    "valid",
    "validation",
])

target_in_train = obs[
    train_mask
    & (obs["cell_type"].astype(str) == target_cell)
]

print("训练中K562总数：", len(target_in_train))
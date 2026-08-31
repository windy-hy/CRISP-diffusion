import scanpy as sc
import pandas as pd


data_path = (
    "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

adata = sc.read_h5ad(data_path)
obs = adata.obs


settings = [
    ("split_drugs", "zero_split_drugs", "MCF7"),
    ("split_drugs2", "zero_split_drugs2", "A549"),
    ("split_drugs3", "zero_split_drugs3", "K562"),
]


for source_split, zero_split, target_cell in settings:

    obs[zero_split] = (
        obs[source_split]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_control_train = (
        (obs["cell_type"].astype(str) == target_cell)
        & (obs["control"].astype(int) == 1)
        & obs[zero_split].isin([
            "train",
            "valid",
            "validation",
        ])
    )

    # 从训练中丢弃，加入iid
    obs.loc[
        target_control_train,
        zero_split,
    ] = "test"

    target_train = (
        (obs["cell_type"].astype(str) == target_cell)
        & obs[zero_split].isin([
            "train",
            "valid",
            "validation",
        ])
    )

    print("\n", zero_split, "目标细胞系：", target_cell)
    print("移除的训练对照细胞数：", target_control_train.sum())
    print("训练中目标细胞系总数：", target_train.sum())

    print(
        pd.crosstab(
            [
                obs["cell_type"],
                obs["control"],
            ],
            obs[zero_split],
        )
    )

    # 覆盖原来的h5ad文件
    adata.write_h5ad(data_path)
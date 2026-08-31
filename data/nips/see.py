import pandas as pd
import scanpy as sc


DATA_PATH = "nips_pp_scFM_resplit.h5ad"
SPLIT_KEYS = ["split", "split2", "split3"]

# backed="r"只读取obs，不把完整表达矩阵加载进内存
adata = sc.read_h5ad(DATA_PATH, backed="r")
obs = adata.obs.copy()

cell_type = obs["cell_type"].astype(str)
condition = obs["condition"].astype(str)

neg_control = pd.to_numeric(
    obs["neg_control"],
    errors="coerce"
)

treated = neg_control.eq(0)

summary = []

for split_key in SPLIT_KEYS:

    split = (
        obs[split_key]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\n" + "=" * 60)
    print("划分列：", split_key)
    print("划分标签：", split.unique().tolist())

    # 只看药物处理细胞
    table = pd.crosstab(
        cell_type[treated],
        split[treated]
    )

    print("\n各细胞类型的处理细胞数量：")
    print(table)

    ood_mask = treated & split.eq("ood")
    train_mask = treated & split.eq("train")

    ood_types = sorted(
        cell_type[ood_mask].unique()
    )

    train_types = sorted(
        cell_type[train_mask].unique()
    )

    print("\nOOD留出的细胞类型：")
    print(ood_types)

    print("\n训练中的细胞类型：")
    print(train_types)

    print("\nOOD细胞类型数量：", len(ood_types))

    if len(ood_types) != 2:
        print("注意：该split留出的细胞类型数量不是2。")

    for ct in sorted(cell_type[treated].unique()):
        ct_train = train_mask & cell_type.eq(ct)
        ct_ood = ood_mask & cell_type.eq(ct)

        summary.append({
            "split": split_key,
            "cell_type": ct,
            "train_treated_cells": int(ct_train.sum()),
            "ood_treated_cells": int(ct_ood.sum()),
            "train_drugs": int(
                condition[ct_train].nunique()
            ),
            "ood_drugs": int(
                condition[ct_ood].nunique()
            ),
        })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "nips_split_celltype_check.csv",
    index=False
)

print("\n" + "=" * 60)
print("汇总结果：")
print(summary_df.to_string(index=False))

adata.file.close()
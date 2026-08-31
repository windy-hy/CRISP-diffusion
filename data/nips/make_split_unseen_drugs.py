import scanpy as sc
import pandas as pd


INPUT_FILE = "nips_pp_scFM_resplit.h5ad"
OUTPUT_FILE = "nips_pp_scFM_resplit_unseen.h5ad"

SPLIT_KEYS = ["split", "split2", "split3"]

UNSEEN_DRUGS = [
    "CHIR99021",
    "Crizotinib",
    "Foretinib",
    "Idelalisib",
    "Linagliptin",
    "Palbociclib",
    "Penfluridol",
    "PorcnInhibitorIII",
    "R428",
]


adata = sc.read_h5ad(INPUT_FILE)

condition = adata.obs["condition"].astype(str)

neg_control = pd.to_numeric(
    adata.obs["neg_control"],
    errors="coerce"
)

treated = neg_control.eq(0)


for split_key in SPLIT_KEYS:

    original = (
        adata.obs[split_key]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    new_split = original.copy()

    # 只移动原训练集中的9种药物处理细胞
    move_mask = (
        original.eq("train")
        & treated
        & condition.isin(UNSEEN_DRUGS)
    )

    new_split.loc[move_mask] = "ood"

    new_key = f"{split_key}_unseen_drugs"

    adata.obs[new_key] = pd.Categorical(
        new_split,
        categories=["train", "test", "ood"]
    )

    # 检查原来的OOD有没有被改变
    original_ood = original.eq("ood")
    preserved = (
        adata.obs.loc[original_ood, new_key]
        .astype(str)
        .eq("ood")
        .all()
    )

    # 检查test是否保持不变
    original_test = original.eq("test")
    test_preserved = (
        adata.obs.loc[original_test, new_key]
        .astype(str)
        .eq("test")
        .all()
    )

    # 检查训练集中是否还有未见药物泄漏
    leakage = (
        adata.obs[new_key].astype(str).eq("train")
        & treated
        & condition.isin(UNSEEN_DRUGS)
    ).sum()

    print("\n", new_key)
    print("原OOD数量：", original_ood.sum())
    print("新增OOD数量：", move_mask.sum())
    print(
        "新OOD总数：",
        adata.obs[new_key].astype(str).eq("ood").sum()
    )
    print("原OOD是否全部保留：", preserved)
    print("原test是否全部保留：", test_preserved)
    print("训练集未见药物泄漏数：", leakage)


adata.write_h5ad(
    INPUT_FILE
)

print("\n保存完成：", INPUT_FILE)




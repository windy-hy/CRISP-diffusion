import scanpy as sc
import pandas as pd


DATA_PATH = "nips_pp_scFM_resplit.h5ad"
SPLIT_KEY = "split_unseen_drugs"

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


adata = sc.read_h5ad(DATA_PATH, backed="r")
obs = adata.obs

split = (
    obs[SPLIT_KEY]
    .astype(str)
    .str.strip()
    .str.lower()
)

condition = (
    obs["condition"]
    .astype(str)
    .str.strip()
)

treated = pd.to_numeric(
    obs["neg_control"],
    errors="coerce"
).eq(0)


# 新split的train中，9种未见药物的处理细胞
leak_mask = (
    split.eq("train")
    & treated
    & condition.isin(UNSEEN_DRUGS)
)

print("检查列：", SPLIT_KEY)
print("train中未见药物处理细胞总数：", int(leak_mask.sum()))

print("\n每种药物在train中的细胞数：")
counts = (
    condition[leak_mask]
    .value_counts()
    .reindex(UNSEEN_DRUGS, fill_value=0)
)

print(counts.to_string())

if leak_mask.sum() == 0:
    print("\n检查通过：train中没有这9种药物的处理细胞。")
else:
    print("\n存在数据泄漏：train中仍有未见药物处理细胞。")

adata.file.close()
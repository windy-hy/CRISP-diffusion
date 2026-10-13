import pandas as pd


FILE1 = "SciPlex3_MCF7_CRISP.csv"
FILE2 = "SciPlex3_MCF7_DIFFDRISP.csv"

OUTPUT_FILE = "fig2h.csv"

df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)


# 统一药物名称和剂量格式，避免浮点精度导致无法匹配
for df in [df1, df2]:
    df["condition"] = df["condition"].astype(str).str.strip()
    df["dose_val"] = pd.to_numeric(
        df["dose_val"],
        errors="coerce",
    ).round(6)


# 第一张表保留CRISP名称
df1 = df1.rename(columns={
    "CRISP_R2_DE": "CRISP_R2_DE",
    "CRISP_Pearson_Delta_DE": "CRISP_Pearson_Delta_DE",
    "CRISP_Sinkhorn_DE": "CRISP_Sinkhorn_DE",
})


# 第二张表改成DIFFCRISP名称
df2 = df2.rename(columns={
    "CRISP_R2_DE": "DIFFCRISP_R2_DE",
    "CRISP_Pearson_Delta_DE": "DIFFCRISP_Pearson_Delta_DE",
    "CRISP_Sinkhorn_DE": "DIFFCRISP_Sinkhorn_DE",
})


# 按药物和剂量合并
merged = pd.merge(
    df1,
    df2,
    on=[
        "condition",
        "dose_val",
    ],
    how="inner",
    validate="one_to_one",
)


# 排列最终列顺序
merged = merged[
    [
        "condition",
        "dose_val",

        "CRISP_R2_DE",
        "DIFFCRISP_R2_DE",

        "CRISP_Pearson_Delta_DE",
        "DIFFCRISP_Pearson_Delta_DE",

        "CRISP_Sinkhorn_DE",
        "DIFFCRISP_Sinkhorn_DE",
    ]
]


merged.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("第一张表行数：", len(df1))
print("第二张表行数：", len(df2))
print("合并后行数：", len(merged))
print("保存完成：", OUTPUT_FILE)

print(merged.to_string(index=False))
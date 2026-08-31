from pathlib import Path
import pandas as pd


ROOT = Path(".")

DIFF_FILE = ROOT / "DIFFCRISP_Figure2e.csv"
CRISP_FILE = ROOT / "CRISP_Figure2e.csv"

OUT_FILE = ROOT / "fig2e.csv"


# =========================
# 读取
# =========================

diff = pd.read_csv(DIFF_FILE)
crisp = pd.read_csv(CRISP_FILE)


# 删除多余索引列
diff = diff.loc[:, ~diff.columns.str.startswith("Unnamed")]
crisp = crisp.loc[:, ~crisp.columns.str.startswith("Unnamed")]


# =========================
# 统一模型名称
# =========================

diff["model"] = "DIFFCRISP"
crisp["model"] = "CRISP"


# =========================
# 保留需要的列
# =========================

columns = [
    "pos_ratio",
    "model",
    "cov_drug",
    "cell_type",
]

diff = diff[columns]
crisp = crisp[columns]


# =========================
# 合并
# =========================

result = pd.concat(
    [
        diff,
        crisp,
    ],
    ignore_index=True,
)


# =========================
# 排序
# =========================

model_order = {
    "DIFFCRISP": 0,
    "CRISP": 1,
}

result["_model_order"] = result["model"].map(model_order)

result = (
    result
    .sort_values(
        [
            "cell_type",
            "_model_order",
            "cov_drug",
        ]
    )
    .drop(columns="_model_order")
    .reset_index(drop=True)
)


# =========================
# 保存
# =========================

result.to_csv(
    OUT_FILE,
    index=False,
)

print("生成完成：", OUT_FILE)

print("\n各模型数据量：")
print(
    result["model"]
    .value_counts()
)

print("\n各细胞类型 × 模型：")
print(
    result.groupby(
        ["cell_type", "model"]
    )
    .size()
    .unstack(fill_value=0)
)
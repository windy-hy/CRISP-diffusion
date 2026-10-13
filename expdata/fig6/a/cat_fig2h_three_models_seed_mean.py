import pandas as pd
from pathlib import Path

# ============================================================
# 三个模型的 all-split × seed 结果
# 每个文件应包含 3 splits × 3 seeds × 36 conditions
# 本脚本先对 3 个 seed 求均值，再按 split + condition + dose 合并
# ============================================================

FILE_CRISP = "SciPlex3_CRISP_all_split_seed.csv"
FILE_DIFFCRISP = "SciPlex3_DIFFCRISP_all_split_seed.csv"
FILE_CHEMCPA = "SciPlex3_chemCPA_all_split_seed.csv"

OUTPUT_FILE = "fig2h.csv"

METRIC_SUFFIXES = [
    "R2_DE",
    "Pearson_Delta_DE",
    "Sinkhorn_DE",
]

# split -> held-out cell type，仅用于文件里缺少 cell_type 时补全
SPLIT_TO_CELL = {
    "split_drugs": "MCF7",
    "split_drugs2": "A549",
    "split_drugs3": "K562",
}


def find_metric_column(df, suffix):
    """
    在不同前缀/大小写下寻找指定 metric。
    例如：
      CRISP_R2_DE
      DIFFCRISP_R2_DE
      chemCPA_R2_DE
    """
    exact = [
        c for c in df.columns
        if c.lower().endswith(suffix.lower())
    ]

    if len(exact) != 1:
        raise ValueError(
            f"无法唯一定位指标 {suffix}。\n"
            f"匹配列: {exact}\n"
            f"全部列: {list(df.columns)}"
        )

    return exact[0]


def prepare_model(file_path, model_name):
    print("\n" + "=" * 80)
    print(f"读取 {model_name}: {file_path}")
    print("=" * 80)

    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

    required = ["split", "seed", "condition", "dose_val"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name} 缺少必要列: {missing}\n"
            f"当前列: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # 统一 key
    # --------------------------------------------------------
    df["split"] = df["split"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()

    df["dose_val"] = pd.to_numeric(
        df["dose_val"],
        errors="coerce",
    ).round(6)

    df["seed"] = pd.to_numeric(
        df["seed"],
        errors="coerce",
    )

    # cell_type 如果没有，则由 split 自动补
    if "cell_type" not in df.columns:
        df["cell_type"] = df["split"].map(SPLIT_TO_CELL)
    else:
        df["cell_type"] = df["cell_type"].astype(str).str.strip()

    # --------------------------------------------------------
    # 自动寻找三个指标，并统一成 MODEL_xxx
    # --------------------------------------------------------
    rename_map = {}

    for suffix in METRIC_SUFFIXES:
        old_col = find_metric_column(df, suffix)
        new_col = f"{model_name}_{suffix}"
        rename_map[old_col] = new_col

    df = df.rename(columns=rename_map)

    metric_cols = [
        f"{model_name}_{suffix}"
        for suffix in METRIC_SUFFIXES
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 检查每个 condition 应该有多少 seed
    # --------------------------------------------------------
    key_cols = [
        "split",
        "cell_type",
        "condition",
        "dose_val",
    ]

    seed_counts = (
        df.groupby(key_cols)["seed"]
        .nunique()
    )

    bad = seed_counts[seed_counts != 3]

    if len(bad) > 0:
        print(
            f"\n[警告] {model_name} 有 {len(bad)} 个 condition "
            "不是恰好 3 个 seed："
        )
        print(bad.head(20))
    else:
        print(
            f"{model_name}: 所有 condition 均为 3 seeds。"
        )

    # --------------------------------------------------------
    # 核心：3 个 seed 直接取均值
    # 最终每个 split × drug × dose 只剩 1 行
    # --------------------------------------------------------
    mean_df = (
        df
        .groupby(
            key_cols,
            as_index=False,
        )[metric_cols]
        .mean()
    )

    mean_df = (
        mean_df
        .sort_values(
            [
                "split",
                "condition",
                "dose_val",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"{model_name}: 原始行数={len(df)}, "
        f"3-seed 均值后行数={len(mean_df)}"
    )

    print(
        f"{model_name}: split 数="
        f"{mean_df['split'].nunique()}, "
        f"condition 数={len(mean_df)}"
    )

    return mean_df


# ============================================================
# 读取并对 3 seeds 求均值
# ============================================================

crisp = prepare_model(
    FILE_CRISP,
    "CRISP",
)

diffcrisp = prepare_model(
    FILE_DIFFCRISP,
    "DIFFCRISP",
)

chemcpa = prepare_model(
    FILE_CHEMCPA,
    "ChemCPA",
)


# ============================================================
# 合并三个模型
# ============================================================

MERGE_KEYS = [
    "split",
    "cell_type",
    "condition",
    "dose_val",
]

merged = pd.merge(
    crisp,
    diffcrisp,
    on=MERGE_KEYS,
    how="inner",
    validate="one_to_one",
)

merged = pd.merge(
    merged,
    chemcpa,
    on=MERGE_KEYS,
    how="inner",
    validate="one_to_one",
)


# ============================================================
# 最终列顺序
# ============================================================

merged = merged[
    [
        "split",
        "cell_type",
        "condition",
        "dose_val",

        "CRISP_R2_DE",
        "DIFFCRISP_R2_DE",
        "ChemCPA_R2_DE",

        "CRISP_Pearson_Delta_DE",
        "DIFFCRISP_Pearson_Delta_DE",
        "ChemCPA_Pearson_Delta_DE",

        "CRISP_Sinkhorn_DE",
        "DIFFCRISP_Sinkhorn_DE",
        "ChemCPA_Sinkhorn_DE",
    ]
]


# split 顺序固定
split_order = {
    "split_drugs": 0,
    "split_drugs2": 1,
    "split_drugs3": 2,
}

merged["_split_order"] = (
    merged["split"]
    .map(split_order)
    .fillna(99)
)

merged = (
    merged
    .sort_values(
        [
            "_split_order",
            "condition",
            "dose_val",
        ]
    )
    .drop(columns="_split_order")
    .reset_index(drop=True)
)


# ============================================================
# 保存
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 检查
# ============================================================

print("\n" + "=" * 80)
print("合并完成")
print("=" * 80)

print("CRISP 3-seed均值后行数：", len(crisp))
print("DiffCRISP 3-seed均值后行数：", len(diffcrisp))
print("ChemCPA 3-seed均值后行数：", len(chemcpa))
print("最终合并行数：", len(merged))
print("保存完成：", OUTPUT_FILE)

print("\n每个 split 行数：")
print(
    merged.groupby("split")
    .size()
)

# 正常情况下：
# 3 splits × 36 conditions = 108 rows
if len(merged) != 108:
    print(
        f"\n[警告] 正常预期 108 行，当前为 {len(merged)} 行。"
        "请检查三个文件是否存在缺失 condition。"
    )
else:
    print("\n[OK] 最终正好 108 行 = 3 splits × 36 conditions。")


# ============================================================
# 顺手打印三个模型的 overall mean
# 注意：这里是 3-seed 均值后的 108 个 condition 的总体均值
# ============================================================

print("\n=== Overall mean across 3 splits ===")

for model in [
    "CRISP",
    "DIFFCRISP",
    "ChemCPA",
]:
    print(
        f"\n{model}:"
    )
    print(
        merged[
            [
                f"{model}_R2_DE",
                f"{model}_Pearson_Delta_DE",
                f"{model}_Sinkhorn_DE",
            ]
        ].mean()
    )


print("\n前 20 行：")
print(
    merged.head(20).to_string(index=False)
)

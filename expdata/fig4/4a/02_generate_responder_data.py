from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# =========================================================
# 1. 路径
# =========================================================

DATA = "../../../data/nips/nips_pp_scFM_resplit.h5ad"

DIFF_DIR = Path("./nips_predictions_diffcrisp")
CRISP_DIR = Path("./nips_predictions_crisp")

OUT = Path("./response_analysis")
OUT.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. 参数
# =========================================================

# 用真实treated-control变化最大的前50个基因
# 定义drug-response direction
TOP_GENES = 50

# control response score的95%分位数
# 作为responder threshold
RESPONDER_Q = 0.95


# =========================================================
# 3. 工具函数
# =========================================================

def dense(x):

    if sp.issparse(x):
        return x.toarray()

    return np.asarray(x)


def load_npz(root, file_value, split):

    p = Path(file_value)

    if not p.exists():

        p = (
            root
            / str(split)
            / p.name
        )

    if not p.exists():

        raise FileNotFoundError(
            "找不到文件：" + str(p)
        )

    return np.load(p)


def response_score(
    x,
    gene_idx,
    ctrl_mean,
    direction
):

    x = dense(x)

    return (
        (
            x[:, gene_idx]
            - ctrl_mean[gene_idx]
        )
        @ direction
    )


# =========================================================
# 4. 读取数据
# =========================================================

print("读取 h5ad ...")

adata = sc.read_h5ad(DATA)

diff_manifest = pd.read_csv(
    DIFF_DIR / "manifest.csv"
)

crisp_manifest = pd.read_csv(
    CRISP_DIR / "manifest.csv"
)


# =========================================================
# 5. 两个模型的perturbation严格配对
# =========================================================

candidate_keys = [
    "split",
    "cov_drug",
    "cell_type",
    "drug",
    "dose",
]

KEYS = [
    x
    for x in candidate_keys
    if (
        x in diff_manifest.columns
        and x in crisp_manifest.columns
    )
]

print(
    "配对字段：",
    KEYS
)

manifest = diff_manifest.merge(
    crisp_manifest,
    on=KEYS,
    suffixes=(
        "_diff",
        "_crisp"
    )
)

print(
    "成功配对组合数：",
    len(manifest)
)


# =========================================================
# 6. 遍历每个perturbation
# =========================================================

results = []

for i, row in manifest.iterrows():

    print(
        f"[{i + 1}/{len(manifest)}]",
        row["split"],
        "|",
        row["cell_type"],
        "|",
        row["drug"]
    )


    # -----------------------------------------------------
    # DiffCRISP预测
    # -----------------------------------------------------

    diff_saved = load_npz(
        DIFF_DIR,
        row["file_diff"],
        row["split"],
    )

    diff_pred = dense(
        diff_saved["pred"]
    )


    # -----------------------------------------------------
    # CRISP预测
    # -----------------------------------------------------

    crisp_saved = load_npz(
        CRISP_DIR,
        row["file_crisp"],
        row["split"],
    )

    crisp_pred = dense(
        crisp_saved["pred"]
    )


    # -----------------------------------------------------
    # 使用DiffCRISP保存的treated / ctrl索引
    # 作为统一reference
    # -----------------------------------------------------

    treated_idx = diff_saved[
        "treated_idx"
    ].astype(int)

    ctrl_idx = diff_saved[
        "ctrl_idx"
    ].astype(int)


    true = dense(
        adata[
            treated_idx
        ].X
    )

    ctrl = dense(
        adata[
            ctrl_idx
        ].X
    )


    # =====================================================
    # 7. 定义真实drug-response direction
    # =====================================================

    ctrl_mean = ctrl.mean(
        axis=0
    )

    true_mean = true.mean(
        axis=0
    )

    delta = (
        true_mean
        - ctrl_mean
    )


    # 真实变化最大的Top50 genes
    gene_idx = np.argsort(
        np.abs(delta)
    )[-TOP_GENES:]


    direction = delta[
        gene_idx
    ]

    norm = np.linalg.norm(
        direction
    )

    if norm < 1e-8:

        print(
            "跳过：response direction接近0"
        )

        continue


    direction = (
        direction
        / norm
    )


    # =====================================================
    # 8. 单细胞response score
    # =====================================================

    ctrl_score = response_score(
        ctrl,
        gene_idx,
        ctrl_mean,
        direction,
    )

    true_score = response_score(
        true,
        gene_idx,
        ctrl_mean,
        direction,
    )

    crisp_score = response_score(
        crisp_pred,
        gene_idx,
        ctrl_mean,
        direction,
    )

    diff_score = response_score(
        diff_pred,
        gene_idx,
        ctrl_mean,
        direction,
    )


    # =====================================================
    # 9. Control的95%分位定义responder
    # =====================================================

    threshold = np.quantile(
        ctrl_score,
        RESPONDER_Q
    )


    # =====================================================
    # 10. Responder proportion
    # =====================================================

    ctrl_prop = np.mean(
        ctrl_score > threshold
    )

    true_prop = np.mean(
        true_score > threshold
    )

    crisp_prop = np.mean(
        crisp_score > threshold
    )

    diff_prop = np.mean(
        diff_score > threshold
    )


    crisp_error = abs(
        crisp_prop
        - true_prop
    )

    diff_error = abs(
        diff_prop
        - true_prop
    )


    # DiffCRISP相对CRISP改善多少
    improvement = (
        crisp_error
        - diff_error
    )


    result = {

        "split":
            row["split"],

        "cell_type":
            row["cell_type"],

        "drug":
            row["drug"],

        "dose":
            row["dose"],

        "true_prop":
            true_prop,

        "ctrl_prop":
            ctrl_prop,

        "crisp_prop":
            crisp_prop,

        "diffcrisp_prop":
            diff_prop,

        "crisp_error":
            crisp_error,

        "diffcrisp_error":
            diff_error,

        "improvement":
            improvement,

        "diffcrisp_win":
            int(
                diff_error
                < crisp_error
            ),

        "threshold":
            threshold,

        "n_true":
            len(true_score),

        "n_ctrl":
            len(ctrl_score),

        "n_crisp":
            len(crisp_score),

        "n_diffcrisp":
            len(diff_score),
    }


    if "cov_drug" in row.index:
        result["cov_drug"] = row[
            "cov_drug"
        ]


    results.append(
        result
    )


# =========================================================
# 11. 保存总体结果
# =========================================================

df = pd.DataFrame(
    results
)

summary_file = (
    OUT
    / "responder_proportion.csv"
)

df.to_csv(
    summary_file,
    index=False
)


print("\n==============================")
print("Overall")
print("==============================")

print(
    "组合数：",
    len(df)
)

print(
    "CRISP MAE：",
    df["crisp_error"].mean()
)

print(
    "DiffCRISP MAE：",
    df["diffcrisp_error"].mean()
)

print(
    "DiffCRISP win rate：",
    df["diffcrisp_win"].mean()
)


print("\n==============================")
print("By split")
print("==============================")


for split in [
    "split",
    "split2",
    "split3",
]:

    x = df[
        df["split"]
        == split
    ]

    if len(x) == 0:
        continue

    print(
        "\n",
        split
    )

    print(
        "n =",
        len(x)
    )

    print(
        "CRISP MAE =",
        round(
            x["crisp_error"].mean(),
            4
        )
    )

    print(
        "DiffCRISP MAE =",
        round(
            x[
                "diffcrisp_error"
            ].mean(),
            4
        )
    )

    print(
        "DiffCRISP win rate =",
        round(
            x[
                "diffcrisp_win"
            ].mean(),
            4
        )
    )


# =========================================================
# 12. 保存case候选排序
# =========================================================

case_candidates = (
    df
    .sort_values(
        "improvement",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

case_candidates.to_csv(
    OUT
    / "case_candidates.csv",
    index=False
)


print("\n改善最大的前20个case：")

show_columns = [
    "split",
    "cell_type",
    "drug",
    "dose",
    "true_prop",
    "crisp_prop",
    "diffcrisp_prop",
    "crisp_error",
    "diffcrisp_error",
    "improvement",
]

print(
    case_candidates[
        show_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# 13. 自动选择一个代表性case
#
# 不选提升最大的极端case
# 在DiffCRISP优于CRISP的组合中，
# 选择improvement最接近正改善中位数的case
# =========================================================

better = df[
    df["improvement"] > 0
].copy()

if len(better) == 0:

    raise RuntimeError(
        "没有找到DiffCRISP优于CRISP的case"
    )


median_improvement = better[
    "improvement"
].median()

better[
    "_distance_to_median"
] = np.abs(
    better["improvement"]
    - median_improvement
)

case_row = (
    better
    .sort_values(
        "_distance_to_median"
    )
    .iloc[0]
)


print("\n==============================")
print("Representative case")
print("==============================")

print(
    case_row[
        show_columns
    ]
)


# =========================================================
# 14. 重新读取该case，保存单细胞score
# =========================================================

case_match = manifest[
    (manifest["split"]
        == case_row["split"])
    &
    (manifest["cell_type"]
        == case_row["cell_type"])
    &
    (manifest["drug"]
        == case_row["drug"])
    &
    np.isclose(
        manifest["dose"].astype(float),
        float(case_row["dose"])
    )
]

if len(case_match) != 1:

    raise RuntimeError(
        "代表case匹配数量不是1："
        + str(len(case_match))
    )


row = case_match.iloc[0]


diff_saved = load_npz(
    DIFF_DIR,
    row["file_diff"],
    row["split"],
)

crisp_saved = load_npz(
    CRISP_DIR,
    row["file_crisp"],
    row["split"],
)


diff_pred = dense(
    diff_saved["pred"]
)

crisp_pred = dense(
    crisp_saved["pred"]
)


treated_idx = diff_saved[
    "treated_idx"
].astype(int)

ctrl_idx = diff_saved[
    "ctrl_idx"
].astype(int)


true = dense(
    adata[
        treated_idx
    ].X
)

ctrl = dense(
    adata[
        ctrl_idx
    ].X
)


ctrl_mean = ctrl.mean(
    axis=0
)

true_mean = true.mean(
    axis=0
)

delta = (
    true_mean
    - ctrl_mean
)

gene_idx = np.argsort(
    np.abs(delta)
)[-TOP_GENES:]

direction = delta[
    gene_idx
]

direction = (
    direction
    /
    np.linalg.norm(
        direction
    )
)


ctrl_score = response_score(
    ctrl,
    gene_idx,
    ctrl_mean,
    direction,
)

true_score = response_score(
    true,
    gene_idx,
    ctrl_mean,
    direction,
)

crisp_score = response_score(
    crisp_pred,
    gene_idx,
    ctrl_mean,
    direction,
)

diff_score = response_score(
    diff_pred,
    gene_idx,
    ctrl_mean,
    direction,
)


threshold = np.quantile(
    ctrl_score,
    RESPONDER_Q
)


case_scores = pd.concat(
    [
        pd.DataFrame({
            "score":
                ctrl_score,
            "group":
                "Control",
        }),

        pd.DataFrame({
            "score":
                true_score,
            "group":
                "Observed",
        }),

        pd.DataFrame({
            "score":
                crisp_score,
            "group":
                "CRISP",
        }),

        pd.DataFrame({
            "score":
                diff_score,
            "group":
                "DiffCRISP",
        }),
    ],
    ignore_index=True
)


case_scores[
    "threshold"
] = threshold

case_scores[
    "split"
] = case_row[
    "split"
]

case_scores[
    "cell_type"
] = case_row[
    "cell_type"
]

case_scores[
    "drug"
] = case_row[
    "drug"
]

case_scores[
    "dose"
] = case_row[
    "dose"
]


case_file = (
    OUT
    / "representative_case_scores.csv"
)

case_scores.to_csv(
    case_file,
    index=False
)


print("\n保存：")
print(summary_file)
print(
    OUT
    / "case_candidates.csv"
)
print(case_file)

print(
    "\n这三个CSV确认无误后，"
    "nips_predictions和"
    "nips_predictions_crisp可以删除。"
)
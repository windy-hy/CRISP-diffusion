from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 设置
# =========================================================

GSEA_DIR = Path("./mcf7_pathway_check/gsea")
OUT = Path("./mcf7_pathway_check")

DRUGS = [
    "Toremifene",
    "Fulvestrant",
    "Lapatinib",
]


# =========================================================
# 读取结果
# =========================================================

pos_results = {}
neg_results = {}


for drug in DRUGS:

    pos_file = (
        GSEA_DIR
        / f"{drug}_gsea_truth_pos.csv"
    )

    neg_file = (
        GSEA_DIR
        / f"{drug}_gsea_truth_neg.csv"
    )

    pos = pd.read_csv(pos_file)
    neg = pd.read_csv(neg_file)

    pos["NES"] = pd.to_numeric(
        pos["NES"],
        errors="coerce"
    )

    neg["NES"] = pd.to_numeric(
        neg["NES"],
        errors="coerce"
    )

    pos_results[drug] = pos.set_index("Term")
    neg_results[drug] = neg.set_index("Term")


# =========================================================
# 通用汇总函数
# =========================================================

def summarize(results):

    all_terms = set()

    for df in results.values():
        all_terms.update(df.index)

    rows = []

    for term in all_terms:

        row = {
            "Term": term
        }

        values = []

        for drug in DRUGS:

            if term in results[drug].index:

                nes = results[
                    drug
                ].loc[
                    term,
                    "NES"
                ]

                row[f"{drug}_NES"] = nes

                values.append(nes)

            else:

                row[f"{drug}_NES"] = np.nan


        row["n_drugs"] = len(values)

        row["Mean_NES"] = (
            np.mean(values)
            if values
            else np.nan
        )

        rows.append(row)


    return pd.DataFrame(rows)


# =========================================================
# 正负分别统计
# =========================================================

pos_summary = summarize(
    pos_results
)

neg_summary = summarize(
    neg_results
)


# 正富集：平均 NES 越大越强
pos_summary = pos_summary.sort_values(
    [
        "n_drugs",
        "Mean_NES"
    ],
    ascending=[
        False,
        False
    ]
)


# 负富集：平均 NES 越负越强
neg_summary = neg_summary.sort_values(
    [
        "n_drugs",
        "Mean_NES"
    ],
    ascending=[
        False,
        True
    ]
)


# =========================================================
# 3/3共同正富集
# =========================================================

common_pos_3 = pos_summary[
    pos_summary["n_drugs"] == 3
].copy()


print(
    "\n======================================"
)
print(
    "3/3 共同正富集 pathways:",
    len(common_pos_3)
)
print(
    "======================================\n"
)

print(
    common_pos_3[
        [
            "Term",
            "Toremifene_NES",
            "Fulvestrant_NES",
            "Lapatinib_NES",
            "Mean_NES",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# 3/3共同负富集
# =========================================================

common_neg_3 = neg_summary[
    neg_summary["n_drugs"] == 3
].copy()


print(
    "\n======================================"
)
print(
    "3/3 共同负富集 pathways:",
    len(common_neg_3)
)
print(
    "======================================\n"
)

print(
    common_neg_3[
        [
            "Term",
            "Toremifene_NES",
            "Fulvestrant_NES",
            "Lapatinib_NES",
            "Mean_NES",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# >=2/3 正富集
# =========================================================

common_pos_2 = pos_summary[
    pos_summary["n_drugs"] >= 2
].copy()


print(
    "\n======================================"
)
print(
    ">=2/3 共同正富集 pathways:",
    len(common_pos_2)
)
print(
    "======================================\n"
)

print(
    common_pos_2[
        [
            "Term",
            "n_drugs",
            "Toremifene_NES",
            "Fulvestrant_NES",
            "Lapatinib_NES",
            "Mean_NES",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
)


# =========================================================
# >=2/3 负富集
# =========================================================

common_neg_2 = neg_summary[
    neg_summary["n_drugs"] >= 2
].copy()


print(
    "\n======================================"
)
print(
    ">=2/3 共同负富集 pathways:",
    len(common_neg_2)
)
print(
    "======================================\n"
)

print(
    common_neg_2[
        [
            "Term",
            "n_drugs",
            "Toremifene_NES",
            "Fulvestrant_NES",
            "Lapatinib_NES",
            "Mean_NES",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
)


# =========================================================
# 保存
# =========================================================

common_pos_3.to_csv(
    OUT / "three_drugs_common_positive_3of3.csv",
    index=False
)

common_neg_3.to_csv(
    OUT / "three_drugs_common_negative_3of3.csv",
    index=False
)

common_pos_2.to_csv(
    OUT / "three_drugs_common_positive_2of3.csv",
    index=False
)

common_neg_2.to_csv(
    OUT / "three_drugs_common_negative_2of3.csv",
    index=False
)


print(
    "\n完成"
)
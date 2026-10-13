from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gseapy as gp


# =========================================================
# 基本设置
# =========================================================

DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
GMT = "../../../data/gsea/c2.cp.v2024.1.Hs.symbols.gmt"

OUT = Path("./mcf7_pathway_check")
GSEA_DIR = OUT / "gsea"

OUT.mkdir(parents=True, exist_ok=True)
GSEA_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = "split3"
CELL = "MCF7"
DOSE = 1.0

MAX_CELLS = 1000
SEED = 42
PERMUTATION_NUM = 1000


# NCI approved breast cancer drugs
DRUGS = [
    "Capecitabine",
    "Fluorouracil",
    "Toremifene",
    "Fulvestrant",
    "Lapatinib",
    "Thiotepa",
]


# =========================================================
# 工具
# =========================================================

def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def sample_adata(x, max_cells=1000):

    if x.n_obs <= max_cells:
        return x.copy()

    rng = np.random.default_rng(SEED)

    idx = rng.choice(
        x.n_obs,
        size=max_cells,
        replace=False
    )

    return x[idx].copy()


def run_gsea(treated, ctrl, genes):

    treated_df = pd.DataFrame(
        dense(treated.X).T,
        index=genes
    )

    ctrl_df = pd.DataFrame(
        dense(ctrl.X).T,
        index=genes
    )

    treated_df.columns = (
        ["Perturb"] * treated_df.shape[1]
    )

    ctrl_df.columns = (
        ["Ctrl"] * ctrl_df.shape[1]
    )

    df = pd.concat(
        [treated_df, ctrl_df],
        axis=1
    )

    result = gp.gsea(
        data=df,
        gene_sets=GMT,
        cls=list(df.columns),
        min_size=10,
        permutation_type="phenotype",
        permutation_num=PERMUTATION_NUM,
        method="signal_to_noise",
        outdir=None,
        threads=4,
        seed=SEED
    ).res2d.copy()

    result["NES"] = pd.to_numeric(
        result["NES"],
        errors="coerce"
    )

    result["FDR q-val"] = pd.to_numeric(
        result["FDR q-val"],
        errors="coerce"
    )

    return result


# =========================================================
# pathway 汇总函数
# =========================================================

def summarize_pathways(drug_results, direction):

    all_terms = set()

    for df in drug_results.values():
        all_terms.update(df.index)

    rows = []

    for term in all_terms:

        row = {
            "Term": term
        }

        nes_values = []

        for drug in DRUGS:

            if (
                drug in drug_results
                and term in drug_results[drug].index
            ):

                nes = drug_results[
                    drug
                ].loc[
                    term,
                    "NES"
                ]

                row[f"{drug}_NES"] = nes
                nes_values.append(nes)

            else:

                row[f"{drug}_NES"] = np.nan


        row["n_drugs"] = len(nes_values)

        row["Mean_NES"] = (
            np.mean(nes_values)
            if nes_values
            else np.nan
        )

        # 最弱富集强度
        if nes_values:

            if direction == "negative":
                # 负富集：越接近0越弱
                row["Weakest_NES"] = np.max(
                    nes_values
                )

            else:
                # 正富集：越接近0越弱
                row["Weakest_NES"] = np.min(
                    nes_values
                )

        else:
            row["Weakest_NES"] = np.nan

        rows.append(row)


    if len(rows) == 0:

        return pd.DataFrame(
            columns=[
                "Term",
                "n_drugs",
                "Mean_NES",
                "Weakest_NES",
            ]
        )


    summary = pd.DataFrame(rows)

    if direction == "negative":

        summary = summary.sort_values(
            [
                "n_drugs",
                "Mean_NES"
            ],
            ascending=[
                False,
                True
            ]
        )

    else:

        summary = summary.sort_values(
            [
                "n_drugs",
                "Mean_NES"
            ],
            ascending=[
                False,
                False
            ]
        )

    return summary


# =========================================================
# NES threshold 统计函数
# =========================================================

def threshold_summary(
    drug_results,
    direction
):

    if direction == "negative":

        thresholds = [
            -1.00,
            -1.25,
            -1.50,
            -1.75,
            -2.00,
        ]

    else:

        thresholds = [
            1.00,
            1.25,
            1.50,
            1.75,
            2.00,
        ]


    all_terms = set()

    for df in drug_results.values():
        all_terms.update(df.index)


    rows = []


    for threshold in thresholds:

        counts = {}


        for term in all_terms:

            n = 0


            for drug in DRUGS:

                if drug not in drug_results:
                    continue

                if term not in drug_results[drug].index:
                    continue


                nes = drug_results[
                    drug
                ].loc[
                    term,
                    "NES"
                ]


                if direction == "negative":

                    if nes <= threshold:
                        n += 1

                else:

                    if nes >= threshold:
                        n += 1


            counts[term] = n


        row = {
            "NES_threshold": threshold
        }


        for k in [
            3,
            4,
            5,
            6
        ]:

            row[f">={k}/6"] = sum(
                n >= k
                for n in counts.values()
            )


        rows.append(row)


    return pd.DataFrame(rows)


# =========================================================
# 读取数据
# =========================================================

adata = sc.read_h5ad(DATA)

genes = (
    adata.var_names
    .astype(str)
    .to_numpy()
)


# =========================================================
# MCF7 control
# split3 OOD
# =========================================================

ctrl = adata[
    (adata.obs["control"] == 1)
    &
    (adata.obs[SPLIT].astype(str) == "ood")
    &
    (adata.obs["cell_type"].astype(str) == CELL)
].copy()


print(
    "MCF7 control cells:",
    ctrl.n_obs
)


ctrl = sample_adata(
    ctrl,
    MAX_CELLS
)


print(
    "MCF7 sampled control cells:",
    ctrl.n_obs
)


# =========================================================
# 六个乳腺癌药真实 GSEA
# =========================================================

positive_results = {}
negative_results = {}


for i, drug in enumerate(
    DRUGS,
    1
):

    print(
        f"\n[{i}/{len(DRUGS)}] {drug}"
    )


    treated = adata[
        (adata.obs["condition"].astype(str) == drug)
        &
        (adata.obs[SPLIT].astype(str) == "ood")
        &
        (adata.obs["cell_type"].astype(str) == CELL)
        &
        (adata.obs["dose_val"] == DOSE)
    ].copy()


    print(
        "treated cells:",
        treated.n_obs
    )


    if treated.n_obs <= 5:

        print(
            "细胞太少，跳过"
        )

        continue


    treated = sample_adata(
        treated,
        MAX_CELLS
    )


    print(
        "sampled treated cells:",
        treated.n_obs
    )


    # =====================================================
    # GSEA
    # =====================================================

    gsea = run_gsea(
        treated,
        ctrl,
        genes
    )


    # =====================================================
    # 显著 pathway
    #
    # 和原 CRISP truth threshold 一致：
    # |NES| > 1
    # FDR < 0.25
    # =====================================================

    sig = gsea[
        (gsea["NES"].abs() > 1)
        &
        (gsea["FDR q-val"] < 0.25)
    ].copy()


    pos = sig[
        sig["NES"] > 0
    ].copy()


    neg = sig[
        sig["NES"] < 0
    ].copy()


    print(
        "significant positive pathways:",
        len(pos)
    )

    print(
        "significant negative pathways:",
        len(neg)
    )


    # =====================================================
    # 保存
    # =====================================================

    sig.to_csv(
        GSEA_DIR
        / f"{drug}_gsea_truth_all.csv",
        index=False
    )


    pos.to_csv(
        GSEA_DIR
        / f"{drug}_gsea_truth_pos.csv",
        index=False
    )


    neg.to_csv(
        GSEA_DIR
        / f"{drug}_gsea_truth_neg.csv",
        index=False
    )


    # =====================================================
    # 保存到字典用于后面统计
    # =====================================================

    positive_results[
        drug
    ] = pos.set_index(
        "Term"
    )


    negative_results[
        drug
    ] = neg.set_index(
        "Term"
    )


# =========================================================
# 正负 pathway 分别汇总
# =========================================================

positive_summary = summarize_pathways(
    positive_results,
    "positive"
)


negative_summary = summarize_pathways(
    negative_results,
    "negative"
)


positive_summary.to_csv(
    OUT
    / "mcf7_pathway_summary_positive.csv",
    index=False
)


negative_summary.to_csv(
    OUT
    / "mcf7_pathway_summary_negative.csv",
    index=False
)


# =========================================================
# recurrence
# =========================================================

print(
    "\n\n=========================================="
)
print(
    "共同显著正富集 pathway"
)
print(
    "=========================================="
)


for k in [
    6,
    5,
    4,
    3
]:

    n = (
        positive_summary[
            "n_drugs"
        ]
        >= k
    ).sum()

    print(
        f">= {k}/6 drugs: {n} 条"
    )


print(
    "\n=========================================="
)
print(
    "共同显著负富集 pathway"
)
print(
    "=========================================="
)


for k in [
    6,
    5,
    4,
    3
]:

    n = (
        negative_summary[
            "n_drugs"
        ]
        >= k
    ).sum()

    print(
        f">= {k}/6 drugs: {n} 条"
    )


# =========================================================
# 不同 NES threshold
# =========================================================

positive_threshold_df = threshold_summary(
    positive_results,
    "positive"
)


negative_threshold_df = threshold_summary(
    negative_results,
    "negative"
)


print(
    "\n\n=========================================="
)
print(
    "正富集：不同 NES 阈值"
)
print(
    "=========================================="
)

print(
    positive_threshold_df.to_string(
        index=False
    )
)


print(
    "\n=========================================="
)
print(
    "负富集：不同 NES 阈值"
)
print(
    "=========================================="
)

print(
    negative_threshold_df.to_string(
        index=False
    )
)


positive_threshold_df.to_csv(
    OUT
    / "nes_threshold_summary_positive.csv",
    index=False
)


negative_threshold_df.to_csv(
    OUT
    / "nes_threshold_summary_negative.csv",
    index=False
)


# =========================================================
# Top recurrent positive
# =========================================================

print(
    "\n\n=========================================="
)
print(
    "Top 50 recurrent POSITIVE pathways"
)
print(
    "==========================================\n"
)


cols = [
    "Term",
    "n_drugs",
    "Mean_NES",
    "Weakest_NES",
]


print(
    positive_summary[
        cols
    ]
    .head(50)
    .to_string(
        index=False
    )
)


# =========================================================
# Top recurrent negative
# =========================================================

print(
    "\n\n=========================================="
)
print(
    "Top 50 recurrent NEGATIVE pathways"
)
print(
    "==========================================\n"
)


print(
    negative_summary[
        cols
    ]
    .head(50)
    .to_string(
        index=False
    )
)


# =========================================================
# Breast / cancer 相关 pathway
# =========================================================

KEYWORDS = [
    "BREAST",
    "CANCER",
    "ESTROGEN",
    "ESR",
    "ERBB",
    "HER2",
    "EGFR",
    "PI3K",
    "AKT",
    "MTOR",
    "CELL_CYCLE",
    "APOPTOSIS",
    "P53",
]


def get_related(
    summary,
    direction
):

    mask = summary[
        "Term"
    ].astype(str).str.upper().apply(
        lambda x: any(
            keyword in x
            for keyword in KEYWORDS
        )
    )


    related = summary[
        mask
    ].copy()


    if direction == "negative":

        related = related.sort_values(
            [
                "n_drugs",
                "Mean_NES"
            ],
            ascending=[
                False,
                True
            ]
        )

    else:

        related = related.sort_values(
            [
                "n_drugs",
                "Mean_NES"
            ],
            ascending=[
                False,
                False
            ]
        )


    return related


positive_related = get_related(
    positive_summary,
    "positive"
)


negative_related = get_related(
    negative_summary,
    "negative"
)


positive_related.to_csv(
    OUT
    / "breast_related_candidates_positive.csv",
    index=False
)


negative_related.to_csv(
    OUT
    / "breast_related_candidates_negative.csv",
    index=False
)


# =========================================================
# 打印 breast-related positive
# =========================================================

print(
    "\n\n=========================================="
)
print(
    "Breast / cancer related POSITIVE candidates"
)
print(
    "==========================================\n"
)


print(
    positive_related[
        cols
    ]
    .head(100)
    .to_string(
        index=False
    )
)


# =========================================================
# 打印 breast-related negative
# =========================================================

print(
    "\n\n=========================================="
)
print(
    "Breast / cancer related NEGATIVE candidates"
)
print(
    "==========================================\n"
)


print(
    negative_related[
        cols
    ]
    .head(100)
    .to_string(
        index=False
    )
)


# =========================================================
# 合并正负候选
# =========================================================

positive_related_out = positive_related.copy()
positive_related_out["Direction"] = "Positive"

negative_related_out = negative_related.copy()
negative_related_out["Direction"] = "Negative"


related_all = pd.concat(
    [
        positive_related_out,
        negative_related_out
    ],
    ignore_index=True
)


related_all.to_csv(
    OUT
    / "breast_related_candidates_all.csv",
    index=False
)


print(
    "\n=========================================="
)
print(
    "全部完成"
)
print(
    "=========================================="
)
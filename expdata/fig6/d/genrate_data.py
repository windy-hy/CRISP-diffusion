import gc
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from DIFFCRISP.trainer import Trainer


# =========================
# 路径与参数
# =========================

NIPS_PATH = "../../../data/nips/nips_pp_scFM_resplit.h5ad"
PBMC_PATH = "../../../data/pbmc/pbmc_bench_pp_all.h5ad"

DRUG_EMB_PATH = (
    "../../../data/drug_embeddings/"
    "fcfp4_1024_embedding_lincs_nips.parquet"
)

# Seen drugs使用原始split模型
SEEN_MODEL_PATH = (
    "../../../model/nips/split/seed0/"
    "model_300_split.pt"
)

# Unseen drugs使用删除9种药物后重新训练的模型
UNSEEN_MODEL_PATH = (
    "../../../model/nips/split_unseen_drugs/seed0/"
    "model_200_split.pt"
)

OUTPUT_RAW = "fig4d_diffcrisp_raw.csv"
OUTPUT_SUMMARY = "fig4d_diffcrisp_summary.csv"

METHOD_NAME = "DiffCRISP"

FM_KEY = "X_scGPT_blood"
GUIDANCE_SCALE = 1.5

# NIPS模型内部CFG溶剂设置应为：
# DMSO_INDEX = 43
# DMSO_DOSE = 1.0


# =========================
# Seen drugs：三个细胞类型都是13种
# =========================

SEEN_DRUGS = [
    "Belinostat",
    "CHIR99021",
    "Crizotinib",
    "Dabrafenib",
    "Dactolisib",
    "Foretinib",
    "Idelalisib",
    "LDN193189",
    "Linagliptin",
    "ODemethylatedAdapalene",
    "Palbociclib",
    "Penfluridol",
    "PorcnInhibitorIII",
]


# =========================
# Unseen drugs
# B和NK为9种；CD4额外包含MLN2238
# =========================

UNSEEN_9 = [
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

UNSEEN_CD4 = UNSEEN_9 + [
    "MLN2238",
]


# =========================
# 细胞类型名称
# =========================

CELL_CONFIG = {
    "B cells": {
        "pbmc_candidates": [
            "B cell",
            "B cells",
        ],
        "expected_batches": 9,
    },

    "NK cells": {
        "pbmc_candidates": [
            "Natural killer cell",
            "NK cell",
            "NK cells",
        ],
        "expected_batches": 8,
    },

    "T cells CD4+": {
        "pbmc_candidates": [
            "CD4-positive, alpha-beta T cell",
            "CD4 T cell",
            "CD4+ T cell",
            "T cells CD4+",
        ],
        "expected_batches": 9,
    },
}


# PBMC-Bench的9个原始测序批次
# NK中Seq-Well没有有效细胞，因此只有8个
PAPER_BATCHES = [
    "Drop-seq",
    "Smart-seq2",
    "inDrops",
    "Seq-Well",
    "10x Chromium (v2)",
    "10x Chromium (v2) A",
    "10x Chromium (v2) B",
    "10x Chromium (v3)",
    "CEL-Seq2",
]


def mean_expression(adata):
    return np.asarray(
        adata.X.mean(axis=0)
    ).reshape(-1)


def resolve_pbmc_cell_type(pbmc, candidates):
    available = set(
        pbmc.obs["cell_type"].astype(str)
    )

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        "PBMC中找不到对应细胞类型。\n"
        f"候选名称：{candidates}\n"
        f"PBMC可用名称：{sorted(available)}"
    )


def get_drug_smile_map(nips):
    table = (
        nips.obs[
            ["condition", "SMILES"]
        ]
        .astype(str)
        .drop_duplicates()
    )

    result = {}

    for drug, group in table.groupby("condition"):
        smiles = group["SMILES"].unique()

        if len(smiles) != 1:
            raise ValueError(
                f"{drug}对应多个SMILES：{smiles}"
            )

        result[str(drug)] = str(smiles[0])

    return result


def save_results(results):
    if not results:
        return

    pd.DataFrame(results).to_csv(
        OUTPUT_RAW,
        index=False,
    )


# =========================
# 检查文件
# =========================

for path in [
    NIPS_PATH,
    PBMC_PATH,
    DRUG_EMB_PATH,
    SEEN_MODEL_PATH,
    UNSEEN_MODEL_PATH,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


# =========================
# 读取数据
# =========================

nips = sc.read_h5ad(NIPS_PATH)
pbmc = sc.read_h5ad(PBMC_PATH)

print("NIPS原始形状：", nips.shape)
print("PBMC原始形状：", pbmc.shape)


# =========================
# 设置基因名称
# =========================

if "gene_id" in nips.var.columns:
    nips.var_names = pd.Index(
        nips.var["gene_id"].astype(str)
    )

if "gene_name" in pbmc.var.columns:
    pbmc.var_names = pd.Index(
        pbmc.var["gene_name"].astype(str)
    )

if nips.var_names.has_duplicates:
    duplicated = nips.var_names[
        nips.var_names.duplicated()
    ].unique()

    raise ValueError(
        "NIPS基因名称存在重复："
        f"{duplicated[:10].tolist()}"
    )

# PBMC重复基因保留第一次出现
pbmc_keep = ~pbmc.var_names.duplicated(
    keep="first"
)

print(
    "PBMC重复基因数：",
    int((~pbmc_keep).sum())
)

pbmc = pbmc[:, pbmc_keep].copy()

# 共同基因顺序严格跟随NIPS
pbmc_gene_set = set(pbmc.var_names)

overlap = pd.Index([
    gene
    for gene in nips.var_names
    if gene in pbmc_gene_set
])

print("NIPS基因数：", nips.n_vars)
print("PBMC去重后基因数：", pbmc.n_vars)
print("共同基因数：", len(overlap))

if len(overlap) == 0:
    raise ValueError("NIPS和PBMC没有共同基因")


# =========================
# 检查FM
# =========================

if FM_KEY not in pbmc.obsm:
    raise KeyError(
        f"PBMC的obsm中没有 {FM_KEY}\n"
        f"当前可用：{list(pbmc.obsm.keys())}"
    )


# =========================
# 药物嵌入
# =========================

drug_emb = pd.read_parquet(
    DRUG_EMB_PATH
)

drug_emb.index = (
    drug_emb.index
    .astype(str)
    .str.strip()
)

drug_emb = drug_emb[
    ~drug_emb.index.duplicated(keep="first")
]

drug_to_smile = get_drug_smile_map(nips)


# =========================
# NIPS公共信息
# =========================

nips_cell_type = (
    nips.obs["cell_type"]
    .astype(str)
)

nips_condition = (
    nips.obs["condition"]
    .astype(str)
)

neg_control = pd.to_numeric(
    nips.obs["neg_control"],
    errors="coerce",
)

de_dict = nips.uns[
    "rank_genes_groups_cov"
]


# =========================
# 预先准备三个细胞类型的数据
# =========================

cell_data = {}

for nips_cell, config in CELL_CONFIG.items():

    pbmc_cell = resolve_pbmc_cell_type(
        pbmc,
        config["pbmc_candidates"],
    )

    # NIPS真实对照
    nips_control = nips[
        nips_cell_type.eq(nips_cell)
        & neg_control.eq(1)
    ].copy()

    if nips_control.n_obs == 0:
        raise ValueError(
            f"NIPS中没有{nips_cell}对照细胞"
        )

    nips_control_mean = mean_expression(
        nips_control[:, overlap]
    )

    # PBMC各原始批次
    pbmc_batches = {}
    pbmc_control_means = {}

    for batch in PAPER_BATCHES:

        mask = (
            pbmc.obs["Method"]
            .astype(str)
            .eq(batch)
            &
            pbmc.obs["cell_type"]
            .astype(str)
            .eq(pbmc_cell)
        )

        adata_batch = pbmc[mask].copy()

        # 没有该细胞类型时跳过，例如NK的Seq-Well
        if adata_batch.n_obs == 0:
            continue

        pbmc_batches[batch] = adata_batch

        pbmc_control_means[batch] = (
            mean_expression(
                adata_batch[:, overlap]
            )
        )

    print("\n=========================")
    print("NIPS细胞类型：", nips_cell)
    print("PBMC细胞类型：", pbmc_cell)
    print("NIPS对照细胞数：", nips_control.n_obs)
    print("有效PBMC批次数：", len(pbmc_batches))

    for batch, adata_batch in pbmc_batches.items():
        print(
            f"{batch:24s} "
            f"n={adata_batch.n_obs}"
        )

    expected = config["expected_batches"]

    if len(pbmc_batches) != expected:
        raise ValueError(
            f"{nips_cell}应有{expected}个批次，"
            f"实际得到{len(pbmc_batches)}个："
            f"{list(pbmc_batches.keys())}"
        )

    cell_data[nips_cell] = {
        "pbmc_cell": pbmc_cell,
        "nips_control_mean": nips_control_mean,
        "pbmc_batches": pbmc_batches,
        "pbmc_control_means": pbmc_control_means,
    }


# =========================
# 单个实验设置的运行函数
# =========================

def run_setting(
    setting_name,
    model_path,
    drug_map,
    results,
):

    print("\n\n################################")
    print("开始运行：", setting_name)
    print("模型：", model_path)
    print("################################")

    exp = Trainer()
    exp.load_model(model_path)

    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    for nips_cell, drugs in drug_map.items():

        current_cell_data = cell_data[nips_cell]

        pbmc_cell = current_cell_data[
            "pbmc_cell"
        ]

        nips_control_mean = current_cell_data[
            "nips_control_mean"
        ]

        pbmc_batches = current_cell_data[
            "pbmc_batches"
        ]

        pbmc_control_means = current_cell_data[
            "pbmc_control_means"
        ]

        print("\n-------------------------")
        print("细胞类型：", nips_cell)
        print("药物数量：", len(drugs))
        print("-------------------------")

        for drug_index, drug in enumerate(
            drugs,
            start=1,
        ):

            print(
                f"\n[{drug_index}/{len(drugs)}] "
                f"{setting_name} | "
                f"{nips_cell} | {drug}"
            )

            # NIPS真实处理细胞
            treated_mask = (
                nips_cell_type.eq(nips_cell)
                & nips_condition.eq(drug)
                & neg_control.eq(0)
            )

            treated = nips[
                treated_mask
            ].copy()

            if treated.n_obs == 0:
                print("跳过：没有真实处理细胞")
                continue

            # 自动读取唯一剂量
            drug_doses = (
                pd.to_numeric(
                    treated.obs["dose_val"],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if len(drug_doses) != 1:
                raise ValueError(
                    f"{nips_cell} / {drug} "
                    f"存在多个剂量：{drug_doses}"
                )

            dose = float(drug_doses[0])

            # NIPS真实响应
            true_delta = (
                mean_expression(
                    treated[:, overlap]
                )
                - nips_control_mean
            )

            # Top 50 DE基因
            de_key = f"{nips_cell}_{drug}"

            if de_key not in de_dict:
                print(
                    "跳过：没有DE键",
                    de_key,
                )
                continue

            top_de = list(dict.fromkeys([
                str(gene)
                for gene in de_dict[de_key][:50]
                if str(gene) in overlap
            ]))

            if len(top_de) == 0:
                print(
                    "跳过：没有共同DE基因"
                )
                continue

            de_index = overlap.get_indexer(
                top_de
            )

            if np.any(de_index < 0):
                raise ValueError(
                    f"{de_key}基因索引对齐失败"
                )

            # 药物SMILES
            if drug not in drug_to_smile:
                print(
                    "跳过：没有SMILES",
                    drug,
                )
                continue

            smile = drug_to_smile[drug]

            if smile not in drug_emb.index:
                print(
                    "跳过：嵌入文件无SMILES",
                    drug,
                )
                continue

            # 遍历当前细胞类型的PBMC批次
            for batch, pbmc_control in (
                pbmc_batches.items()
            ):

                with torch.inference_mode():

                    pred = exp.get_prediction(
                        pbmc_control,
                        dose=dose,
                        smile=smile,
                        smile_df=drug_emb,
                        FM_emb=FM_KEY,
                        guidance_scale=(
                            GUIDANCE_SCALE
                        ),
                    )

                if pred.n_vars != nips.n_vars:
                    raise ValueError(
                        f"预测基因数={pred.n_vars}，"
                        f"NIPS基因数={nips.n_vars}"
                    )

                # 解码输出顺序对应NIPS训练基因
                pred.var_names = (
                    nips.var_names.copy()
                )

                pred_delta = (
                    mean_expression(
                        pred[:, overlap]
                    )
                    - pbmc_control_means[batch]
                )

                true_de = true_delta[
                    de_index
                ]

                pred_de = pred_delta[
                    de_index
                ]

                valid = (
                    np.isfinite(true_de)
                    & np.isfinite(pred_de)
                    & (true_de != 0)
                )

                if valid.sum() == 0:
                    print(
                        f"跳过：{batch} "
                        "没有有效DE基因"
                    )
                    continue

                direction_acc = np.mean(
                    np.sign(true_de[valid])
                    ==
                    np.sign(pred_de[valid])
                )

                results.append({
                    "setting": setting_name,
                    "direction_acc": float(
                        direction_acc
                    ),
                    "drug": drug,
                    "dose": dose,
                    "cell_type": nips_cell,
                    "pbmc_cell_type": pbmc_cell,
                    "batch_original": batch,
                    "n_cells": int(
                        pbmc_control.n_obs
                    ),
                    "n_de_genes": int(
                        valid.sum()
                    ),
                    "method": METHOD_NAME,
                })

                print(
                    f"{batch:24s} "
                    f"n={pbmc_control.n_obs:4d} "
                    f"DE={valid.sum():2d} "
                    f"acc={direction_acc:.4f}"
                )

                del pred

            # 每个药物完成后保存一次
            save_results(results)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del exp

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# 运行Seen drugs
# =========================

results = []

seen_drug_map = {
    "B cells": SEEN_DRUGS,
    "NK cells": SEEN_DRUGS,
    "T cells CD4+": SEEN_DRUGS,
}

run_setting(
    setting_name="Seen drugs",
    model_path=SEEN_MODEL_PATH,
    drug_map=seen_drug_map,
    results=results,
)


# =========================
# 运行Unseen drugs
# =========================

unseen_drug_map = {
    "B cells": UNSEEN_9,
    "NK cells": UNSEEN_9,
    "T cells CD4+": UNSEEN_CD4,
}

run_setting(
    setting_name="Unseen drugs",
    model_path=UNSEEN_MODEL_PATH,
    drug_map=unseen_drug_map,
    results=results,
)


# =========================
# 最终保存原始结果
# =========================

result_df = pd.DataFrame(results)

if result_df.empty:
    raise ValueError("没有生成任何结果")

cell_order = {
    "B cells": 0,
    "NK cells": 1,
    "T cells CD4+": 2,
}

setting_order = {
    "Seen drugs": 0,
    "Unseen drugs": 1,
}

result_df["_setting_order"] = (
    result_df["setting"]
    .map(setting_order)
)

result_df["_cell_order"] = (
    result_df["cell_type"]
    .map(cell_order)
)

result_df = result_df.sort_values(
    [
        "_setting_order",
        "_cell_order",
        "batch_original",
        "drug",
    ]
).drop(
    columns=[
        "_setting_order",
        "_cell_order",
    ]
).reset_index(drop=True)

result_df.to_csv(
    OUTPUT_RAW,
    index=False,
)


# =========================
# 生成画图汇总表
# =========================

summary_df = (
    result_df
    .groupby(
        [
            "setting",
            "cell_type",
            "method",
        ],
        as_index=False,
    )
    .agg(
        mean_direction_acc=(
            "direction_acc",
            "mean",
        ),
        std_direction_acc=(
            "direction_acc",
            "std",
        ),
        sem_direction_acc=(
            "direction_acc",
            "sem",
        ),
        n=(
            "direction_acc",
            "size",
        ),
    )
)

summary_df["_setting_order"] = (
    summary_df["setting"]
    .map(setting_order)
)

summary_df["_cell_order"] = (
    summary_df["cell_type"]
    .map(cell_order)
)

summary_df = summary_df.sort_values(
    [
        "_setting_order",
        "_cell_order",
    ]
).drop(
    columns=[
        "_setting_order",
        "_cell_order",
    ]
).reset_index(drop=True)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# =========================
# 最终检查
# =========================

print("\n=========================")
print("图4d数据生成完成")
print("=========================")

print("原始结果文件：", OUTPUT_RAW)
print("汇总结果文件：", OUTPUT_SUMMARY)
print("总行数：", len(result_df))

print("\n各组结果数量：")
print(
    result_df.groupby(
        [
            "setting",
            "cell_type",
        ]
    )
    .size()
    .to_string()
)

print("\n汇总结果：")
print(
    summary_df.to_string(
        index=False
    )
)


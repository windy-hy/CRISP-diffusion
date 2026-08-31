from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
GENES = ["ZFAS1", "SNHG6"]


def get_group(df, label):
    mask = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        == label.lower()
    )

    return (
        pd.to_numeric(
            df.loc[mask, "exp"],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )


def same_values(values1, values2):
    if len(values1) != len(values2):
        return False

    return np.allclose(
        np.sort(values1),
        np.sort(values2),
        rtol=1e-6,
        atol=1e-8,
    )


for gene in GENES:

    crisp_path = DATA_DIR / (
        "figure2f_" + gene + "_CRISP.csv"
    )

    diffcrisp_path = DATA_DIR / (
        "figure2f_" + gene + "_DIFFCRISP.csv"
    )

    output_path = DATA_DIR / (
        "figure2f_" + gene + ".csv"
    )

    if not crisp_path.exists():
        raise FileNotFoundError(
            "找不到文件：" + str(crisp_path)
        )

    if not diffcrisp_path.exists():
        raise FileNotFoundError(
            "找不到文件：" + str(diffcrisp_path)
        )

    crisp_df = pd.read_csv(crisp_path)
    diffcrisp_df = pd.read_csv(diffcrisp_path)

    for df, path in [
        (crisp_df, crisp_path),
        (diffcrisp_df, diffcrisp_path),
    ]:
        if "exp" not in df.columns:
            raise ValueError(
                str(path) + " 缺少 exp 列"
            )

        if "label" not in df.columns:
            raise ValueError(
                str(path) + " 缺少 label 列"
            )

    crisp_pred = get_group(
        crisp_df,
        "CRISP",
    )

    diffcrisp_pred = get_group(
        diffcrisp_df,
        "DIFFCRISP",
    )

    crisp_true = get_group(
        crisp_df,
        "True",
    )

    diffcrisp_true = get_group(
        diffcrisp_df,
        "True",
    )

    crisp_ctrl = get_group(
        crisp_df,
        "Ctrl",
    )

    diffcrisp_ctrl = get_group(
        diffcrisp_df,
        "Ctrl",
    )

    if len(crisp_pred) == 0:
        raise ValueError(
            crisp_path.name
            + " 中没有找到 CRISP 数据"
        )

    if len(diffcrisp_pred) == 0:
        raise ValueError(
            diffcrisp_path.name
            + " 中没有找到 DIFFCRISP 数据"
        )

    if not same_values(
        crisp_true,
        diffcrisp_true,
    ):
        raise ValueError(
            gene
            + " 的两个文件中 True 数据不一致"
        )

    if not same_values(
        crisp_ctrl,
        diffcrisp_ctrl,
    ):
        raise ValueError(
            gene
            + " 的两个文件中 Ctrl 数据不一致"
        )

    merged_df = pd.concat(
        [
            pd.DataFrame({
                "exp": crisp_pred,
                "label": "CRISP",
            }),
            pd.DataFrame({
                "exp": diffcrisp_pred,
                "label": "DIFFCRISP",
            }),
            pd.DataFrame({
                "exp": crisp_true,
                "label": "True",
            }),
            pd.DataFrame({
                "exp": crisp_ctrl,
                "label": "Ctrl",
            }),
        ],
        ignore_index=True,
    )

    merged_df.to_csv(
        output_path,
        index=False,
    )

    print("\n基因：", gene)
    print(
        merged_df
        .groupby("label")["exp"]
        .agg(["count", "mean", "std"])
    )
    print("已保存：", output_path)
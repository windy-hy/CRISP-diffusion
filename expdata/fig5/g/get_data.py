import sys
import gc
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =========================
# 路径
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments/configs/sci.yaml"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data/sciplex3/"
      "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

OUTPUT_FILE = SCRIPT_DIR / "fig3e_diffcrisp.csv"


# =========================
# 参数
# =========================

SPLITS = {
    "zero_split_drugs": "MCF7",
    "zero_split_drugs2": "A549",
    "zero_split_drugs3": "K562",
}

EPOCHS = {
    "zero_split_drugs": 100,
    "zero_split_drugs2": 100,
    "zero_split_drugs3": 100,
}

SEED = 0
GUIDANCE_SCALE = 1.5

UNSEEN_DRUGS = [
    "Dacinostat",
    "Givinostat",
    "Belinostat",
    "Hesperadin",
    "Quisinostat",
    "Alvespimycin",
    "Tanespimycin",
    "TAK-901",
    "Flavopiridol",
]


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


# =========================
# 单独运行一个 split
# =========================

def run_one_split(
    split_key,
    target_cell,
    epoch,
    output_path
):
    from DIFFCRISP.utils import load_config
    from DIFFCRISP.trainer import Trainer
    from DIFFCRISP.eval import evaluate

    model_path = (
        PROJECT_ROOT
        / f"model/sci/{split_key}/seed{SEED}/"
          f"model_{epoch}_split.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型不存在：{model_path}"
        )

    print(
        f"\n评估：{split_key} "
        f"{target_cell} "
        f"seed={SEED} "
        f"epoch={epoch}"
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    args = load_config(
        str(CONFIG_PATH)
    )

    args["dataset"]["split_key"] = split_key
    args["dataset"]["adata_obj"] = str(DATA_PATH)
    args["model"]["seed"] = SEED

    exp = Trainer()

    exp.init_dataset(
        **args["dataset"],
        seed=SEED,
    )

    treated_dataset = exp.datasets["ood_treated"]
    control_dataset = exp.datasets["ood_control"]

    doses = to_numpy(
        treated_dataset.dosages
    )

    if doses.ndim > 1:
        doses = doses[:, 0]

    condition_table = pd.DataFrame({
        "pert_category": np.asarray(
            treated_dataset.pert_categories
        ).astype(str),

        "cell_type": np.asarray(
            treated_dataset.celltype
        ).astype(str),

        "drug": np.asarray(
            treated_dataset.drugs_names
        ).astype(str),

        "dose": doses.astype(float),
    })

    condition_table = condition_table[
        (
            condition_table["cell_type"]
            == target_cell
        )
        & condition_table["drug"].isin(
            UNSEEN_DRUGS
        )
    ]

    condition_table = (
        condition_table
        .drop_duplicates("pert_category")
        .sort_values(["drug", "dose"])
        .reset_index(drop=True)
    )

    print(
        "组合数量：",
        len(condition_table)
    )

    if len(condition_table) != 36:
        raise ValueError(
            f"{target_cell} 组合数量不是36，"
            f"实际为 {len(condition_table)}"
        )

    exp.load_model(
        str(model_path)
    )

    exp.autoencoder.to(exp.device)
    exp.autoencoder.eval()

    with torch.inference_mode():
        prediction, eval_score_dict, extra = evaluate(
            autoencoder=exp.autoencoder,
            treated_dataset=treated_dataset,
            control_dataset=control_dataset,
            guidance_scale=GUIDANCE_SCALE,
        )

    rows = []

    for row in condition_table.itertuples(
        index=False
    ):
        if row.pert_category not in eval_score_dict:
            raise KeyError(
                f"没有找到组合："
                f"{row.pert_category}"
            )

        rows.append({
            "method": "DIFFCRISP",
            "seed": SEED,
            "split_key": split_key,
            "cell_type": row.cell_type,
            "drug": row.drug,
            "dose": float(row.dose),
            "pert_category": row.pert_category,
            "pearson_delta_de": float(
                eval_score_dict[
                    row.pert_category
                ]["pearson_delta_de"]
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"{target_cell} 保存完成："
        f"{len(result)} 行"
    )

    del prediction
    del extra
    del eval_score_dict
    del condition_table
    del treated_dataset
    del control_dataset
    del exp
    del args

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# 主进程
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split")
    parser.add_argument("--cell")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output")

    cmd_args = parser.parse_args()

    # 子进程模式：只运行一个split
    if cmd_args.split is not None:
        run_one_split(
            split_key=cmd_args.split,
            target_cell=cmd_args.cell,
            epoch=cmd_args.epoch,
            output_path=cmd_args.output,
        )

        return

    # 主进程模式：依次启动三个独立进程
    temp_files = []

    for split_key, target_cell in SPLITS.items():
        epoch = EPOCHS[split_key]

        temp_file = (
            SCRIPT_DIR
            / f"_temp_{split_key}.csv"
        )

        temp_files.append(temp_file)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--split",
            split_key,
            "--cell",
            target_cell,
            "--epoch",
            str(epoch),
            "--output",
            str(temp_file),
        ]

        print(
            f"\n启动独立进程："
            f"{target_cell}"
        )

        subprocess.run(
            command,
            check=True,
            cwd=str(SCRIPT_DIR),
        )

        print(
            f"{target_cell} 进程结束，"
            f"内存已由系统回收"
        )

    # 合并三个split
    result = pd.concat(
        [
            pd.read_csv(file)
            for file in temp_files
        ],
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "dose",
            "cell_type",
            "drug",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # 删除临时文件
    for file in temp_files:
        if file.exists():
            file.unlink()

    print("\n全部完成")
    print("保存文件：", OUTPUT_FILE)
    print("总行数：", len(result))

    print("\n每个细胞系的组合数量：")
    print(
        result.groupby("cell_type")
        .size()
        .to_string()
    )

    print("\n每个剂量的数据数量：")
    print(
        result.groupby("dose")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

# 固定MMD
MMD_CO = 0.0
arg = f'fcfp4_mmd{MMD_CO}_loss0.5l1l2+0.2_128'
FM_key = 'X_scGPT'
DATASETS = {
    "nips": {
        "data_path": "../data/nips/nips_pp_scFM_resplit.h5ad",
        "drug_path": "../data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet",
        "config_path": "../experiments/configs/nips.yaml",
    },
    # "sci": {
    #     "data_path": "../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad",
    #     "drug_path": "../data/drug_embeddings/fcfp4_1024_embedding_lincs_sciplex3.parquet",
    #     "config_path": "../experiments/configs/sci.yaml",
    # },
}


# SPLITS = ["zero_split_drugs", "zero_split_drugs2", "zero_split_drugs3"]
# SPLITS = [ "split"]
SPLITS = [
    "split",
    "split2",
    "split3"
]
SEEDS = [
    # 0,
    42,
    123
]
# SEEDS = [0]





failed_runs = []

for dataset, config in DATASETS.items():
    for split in SPLITS:
        for seed in SEEDS:

            save_dir = (
                "../model/"
                + dataset
                + f"/{arg}/"
                + split
                + "/seed"
                + str(seed)
            )

            command = [
                sys.executable,
                str(TRAIN_SCRIPT),

                "--config",
                config["config_path"],

                "--split",
                split,

                "--seed",
                str(seed),

                "--savedir",
                save_dir,

                "--mmd_co",
                str(MMD_CO),

                "--drug_emb",
                config["drug_path"],

                "--data_path",
                config["data_path"],

                "--FM_key",
                FM_key
            ]

            print("\n" + "=" * 70)
            print(
                "开始运行：",
                dataset,
                split,
                "seed=" + str(seed),
            )
            print("固定 MMD =", MMD_CO)
            print("保存目录：", save_dir)
            print("=" * 70)

            try:
                subprocess.run(
                    command,
                    cwd=TRAIN_SCRIPT.parent,
                    check=True,
                )

            except subprocess.CalledProcessError as error:
                print(
                    "运行失败：",
                    dataset,
                    split,
                    "seed=" + str(seed),
                )

                failed_runs.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "seed": seed,
                        "returncode": error.returncode,
                    }
                )


print("\n全部实验运行结束。")

if failed_runs:
    print("\n失败的实验：")

    for item in failed_runs:
        print(
            item["dataset"],
            item["split"],
            "seed=" + str(item["seed"]),
            "返回码=" + str(item["returncode"]),
        )
else:
    print("全部 9 组实验运行成功。")
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 你刚才的单组测试脚本
TEST_SCRIPT = PROJECT_ROOT / "test.py"

FM_key = "X_scGPT"
DATASETS = {
    "nips": {
        "data_path": "../data/nips/nips_pp_scFM_resplit.h5ad",
        "config_path": "../experiments/configs/nips.yaml",
        "model_root": "../model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_128",
    },

    # "sci": {
    #     "data_path": "../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad",
    #     "config_path": "../experiments/configs/sci.yaml",
    #     "model_root": "../model/sci/fcfp4_mmd0.0_loss0.5l1l2+0.2_64",
    # },
}


SPLITS = [
    "split",
    "split2",
    "split3",
]

SEEDS = [
    0,
    42,
    123,
]


failed_runs = []


for dataset, config in DATASETS.items():

    for split in SPLITS:

        for seed in SEEDS:

            model_dir = (
                config["model_root"]
                + "/"
                + split
                + "/seed"
                + str(seed)
            )

            command = [
                sys.executable,
                str(TEST_SCRIPT),

                "--config",
                config["config_path"],

                "--split",
                split,

                "--seed",
                str(seed),

                "--data_path",
                config["data_path"],

                "--model_dir",
                model_dir,

                "--FM_key",
                FM_key
            ]

            print("\n" + "=" * 70)

            print(
                "开始测试：",
                dataset,
                split,
                "seed=" + str(seed),
            )

            print(
                "模型目录：",
                model_dir,
            )

            print("=" * 70)

            try:
                subprocess.run(
                    command,
                    cwd=TEST_SCRIPT.parent,
                    check=True,
                )

            except subprocess.CalledProcessError as error:

                print(
                    "测试失败：",
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


print("\n全部测试结束。")


if failed_runs:

    print("\n失败的测试：")

    for item in failed_runs:

        print(
            item["dataset"],
            item["split"],
            "seed=" + str(item["seed"]),
            "返回码=" + str(item["returncode"]),
        )

else:

    print("全部18组测试成功。")
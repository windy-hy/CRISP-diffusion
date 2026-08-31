import os
import re
import gc
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from DIFFCRISP.utils import load_config
from DIFFCRISP.trainer import Trainer
from DIFFCRISP.eval import evaluate

def test_all_epochs():
    guidance_scale = 1.5
    seed = 0
    split = 'split'
    dataset = 'nips'
    arg = '/rdkit'


    data_path = None
    config_path = None
    if dataset == 'nips':
        data_path = '../data/nips/nips_pp_scFM_resplit.h5ad'
        config_path = '../experiments/configs/nips.yaml'
    elif dataset == 'sci':
        data_path = '../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad'
        config_path = '../experiments/configs/sci.yaml'

    # 1. 初始化命令行解析
    parser = argparse.ArgumentParser(description="Batch Test All Epochs")
    parser.add_argument("--config", type=str, default=config_path, help="Path to config file")
    parser.add_argument("--split", type=str, default=split, help="Split key for data")
    parser.add_argument("--seed", type=int, default=seed, help="Seed")
    parser.add_argument("--FM_key", type=str, default=None, help="key of FM embeddings")
    parser.add_argument("--data_path", type=str, default=data_path, help="Path of adata")
    parser.add_argument("--model_dir", type=str, default=f'../model/{dataset}{arg}/{split}/seed{seed}')

    pars_args = parser.parse_args()
    args = load_config(pars_args.config)

    # 2. 更新配置参数
    args['dataset']['split_key'] = pars_args.split
    args["model"]['seed'] = pars_args.seed
    if pars_args.FM_key is not None:
        args['dataset']['FM_key'] = pars_args.FM_key
    if pars_args.data_path is not None:
        args['dataset']['adata_obj'] = pars_args.data_path

    # 3. 获取所有模型文件并排序
    ckpt_dir = pars_args.model_dir
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pt')]

    def get_epoch_num(filename):
        match = re.search(r'model_(\d+)_split\.pt', filename)
        return int(match.group(1)) if match else -1

    ckpt_files.sort(key=get_epoch_num)

    if not ckpt_files:
        print(f"❌ 在 {ckpt_dir} 下没有找到任何模型文件！请检查路径。")
        return

    # 4. 实例化 Trainer 并加载数据 (整个脚本只执行一次，极大节省时间！)
    print(f"🚀 正在初始化 Trainer 并加载数据集 ({pars_args.split}) ...")
    exp = Trainer()
    exp.init_dataset(**args["dataset"], seed=args["model"]['seed'])

    has_ood = 'ood_treated' in exp.datasets
    treated_key = "ood_treated" if has_ood else "test_treated"
    control_key = "ood_control" if has_ood else "test_control"
    print(f"📌 数据加载完毕！测试评估子集: {treated_key} vs {control_key}")

    # 用于记录画图的数据
    epochs = []
    r2_scores = []
    pearsons = []
    sinkhorns = []

    txt_path = os.path.join(
        ckpt_dir,
        f"{pars_args.split}_{guidance_scale}_results.txt"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "Epoch\tR2_DE\tPearson_Delta_DE\tSinkhorn_DE\tModel_File\n"
        )

    print("\n" + "=" * 60)
    print(f" 🔍 开始批量扫描 {len(ckpt_files)} 个模型权重")
    print("=" * 60)

    # 5. 循环加载权重并测试
    for ckpt_filename in ckpt_files:
        model_path = os.path.join(ckpt_dir, ckpt_filename)
        epoch_num = get_epoch_num(ckpt_filename)

        print(f"\n>>> 正在加载并评估 Epoch {epoch_num} ...")

        try:
            # 加载当前 Epoch 权重
            exp.load_model(model_path)
            exp.autoencoder.to(exp.device)
            exp.autoencoder.eval()

            # 所有模型、所有Epoch统一使用相同的推理噪声
            # eval_seed = 0
            # np.random.seed(eval_seed)
            # torch.manual_seed(eval_seed)
            # torch.cuda.manual_seed(eval_seed)
            # torch.cuda.manual_seed_all(eval_seed)


            with torch.no_grad():
                metrics_all, eval_score_dict, pred_dict = evaluate(
                    autoencoder=exp.autoencoder,
                    treated_dataset=exp.datasets[treated_key],
                    control_dataset=exp.datasets[control_key],
                    guidance_scale=guidance_scale
                )

            # 提取核心指标
            r2_de = metrics_all.get('r2score_de', 0)
            pearson_delta_de = metrics_all.get('pearson_delta_de', 0)
            sinkhorn_de = metrics_all.get('sinkhorn_de', 0)

            print(f"   - R² Score:        {r2_de:.8f}")
            print(f"   - Pearson Delta:   {pearson_delta_de:.8f}")
            print(f"   - Sinkhorn Dist:   {sinkhorn_de:.8f}")

            # 记录数据
            epochs.append(epoch_num)
            r2_scores.append(r2_de)
            pearsons.append(pearson_delta_de)
            sinkhorns.append(sinkhorn_de)

            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch_num}\t"
                    f"{r2_de:.8f}\t"
                    f"{pearson_delta_de:.8f}\t"
                    f"{sinkhorn_de:.8f}\t"
                    f"{ckpt_filename}\n"
                )

        except Exception as e:
            print(f"❌ 评估 Epoch {epoch_num} 时报错: {e}")

        finally:
            # 极其重要：每一轮结束强制清空显存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()




if __name__ == "__main__":
    test_all_epochs()
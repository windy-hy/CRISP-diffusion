import os
import pprint
import argparse
import logging
from DIFFCRISP.utils import load_config
from DIFFCRISP.trainer import Trainer
import yaml
import torch


if __name__ == "__main__":

    use_FM = True
    seed = 0
    split = 'split'
    dataset = 'nips'
    FM_key = 'X_scGPT_human'
    arg = '/scgpt_human'

    data_path = None
    drug_path = None
    config_path = None
    mmd_co = 0
    if dataset == 'nips':
        # data_path = '../data/nips/nips_pp_scFM_resplit.h5ad'
        data_path = '../data/nips/nips_pp_scFM_resplit.h5ad'
        drug_path = '../data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet'
        config_path = '../experiments/configs/nips.yaml'
        mmd_co = 0.0001
    elif dataset == 'sci':
        data_path = '../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad'
        # data_path = '../data/sciplex3/sciplex3_pp_hvgenes_scFM_zero_split.h5ad'
        drug_path = '../data/drug_embeddings/fcfp4_1024_embedding_lincs_sciplex3.parquet'
        config_path = '../experiments/configs/sci.yaml'
        mmd_co = 0.0001


    # 步骤1：初始化命令行参数解析器，定义训练所需的命令行参数,删除required=True
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to the config file",
                        default=config_path)
    parser.add_argument("--split", type=str,  help="Split key for data",
                        default=split)
    parser.add_argument("--seed", type=int, help="Seed",
                        default=seed)
    parser.add_argument("--savedir", type=str,help="Path of save model",
                        default=f'../model/{dataset}{arg}/{split}/seed{seed}')
    parser.add_argument("--FM_key",type=str, default=f'{FM_key}',help="key of FM embeddings")
    parser.add_argument("--drug_emb", type=str,  help="Path of drug embeddings",
                        default=drug_path)
    parser.add_argument("--data_path", type=str,  help="Path of adata",
                        default=data_path)
    parser.add_argument("--mmd_co", type=float, default=mmd_co)

    # 解析命令行传入的参数，将参数值存储到pars_args对象中
    pars_args = parser.parse_args()
    # 提取配置文件路径参数  ../experiments/configs/nips.yaml
    config_path = pars_args.config
    # 加载yaml配置文件，返回字典格式的配置参数
    args = load_config(config_path)

    # 将命令行传入的参数更新到配置字典中
    args['dataset']['split_key'] = pars_args.split
    args["model"]['seed'] = pars_args.seed
    args['model']['mmd_co'] = pars_args.mmd_co
    args['training']['save_dir'] = pars_args.savedir

    if pars_args.FM_key is not None:
        args['dataset']['FM_key'] = pars_args.FM_key
    if pars_args.drug_emb is not None:
        args['model']['drug_emb'] = pars_args.drug_emb
    if pars_args.data_path is not None:
        args['dataset']['adata_obj'] = pars_args.data_path

    formatted_str = pprint.pformat(args)

    log_path = args['training']['save_dir']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename= f'{log_path}/log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info(f'Argument setting: {formatted_str}')
    yaml.dump(
        args, open(f"{log_path}/config.yaml", "w"), default_flow_style=False
    )

    exp = Trainer()

    exp.init_dataset(**args["dataset"],seed=args["model"]['seed'])
    logging.info(f'Finish init dataset')

    exp.init_drug_embedding(chem_model='rdkit',chem_df=args["model"]["drug_emb"])
    logging.info(f'Finish init drug embedding')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp.init_model(
        seed=args["model"]['seed'],
        mmd_co=args['model']['mmd_co'],
    )

    exp.load_train()
    logging.info(f'Start training')
    eval_ood = args['dataset']['split_ood']
    exp.train(**args["training"],eval_ood=eval_ood)
    logging.info(f'Finish training')




"""
The Trainer framework is adapted from chemCPA
"""
import logging
import math
from typing import List, Optional, Union
import os
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
import numpy as np
import torch
import pickle
import copy
import scanpy as sc
from tqdm import tqdm

from DIFFCRISP.data import Dataset, custom_collate
from DIFFCRISP.embedding import get_chemical_representation
from DIFFCRISP.model import PertAE
from DIFFCRISP.eval import evaluate, compute_prediction_CRISP


class Trainer:
    """
        CRISP模型训练器核心类
        负责：
        1. 单细胞数据集加载、预处理与子集划分（训练集/测试集/OOD集）
        2. 药物化学嵌入计算与初始化
        3. PertAE模型构建与预训练模型加载
        4. 模型训练（含损失计算、学习率调度、Checkpoint保存）
        5. 药物反应预测（支持对照组+扰动条件输入，返回预测结果）
        6. 训练过程评估（分布内/IID+分布外/OOD评估）
        """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def init_dataset(
        self,
        adata_obj,
        perturbation_key: Union[str, None],
        dose_key: Union[str, None],
        smiles_key: Union[str, None],
        celltype_key='cell_type',
        covariate_keys=None,
        FM_key='X_scGPT',
        control_key: str = 'control',
        pc_cov: str='cell_type',
        degs_key: str = "rank_genes_groups_cov",
        pert_category: str = "cov_drug_dose_name",
        split_ood: bool = True,
        split_key: str = "split",
        seed=0,
        ):
        """
            加载并预处理单细胞数据集，划分训练集/测试集/OOD子集（核心数据初始化步骤）

            参数说明：
                adata_obj: 输入数据（AnnData对象或AnnData文件路径字符串），存储单细胞基因表达及元信息
                perturbation_key: 扰动条件列名（如药物名称），在obsDataFrame中
                dose_key: 药物剂量列名，在obsDataFrame中
                smiles_key: 药物SMILES字符串列名，在obsDataFrame中
                celltype_key: 细胞类型列名（默认'cell_type'），用于区分不同细胞类型
                covariate_keys: 其他协变量列名列表（如供体、处理时间等），可选
                FM_key: 基础模型嵌入的键名（默认'X_scGPT'），在obsm中存储单细胞预训练嵌入
                control_key: 对照组标识值（默认'control'），在perturbation_key列中标记对照组样本
                pc_cov: 配对对照组的分组列名（默认'cell_type'），用于为处理组匹配同条件对照组
                        例：'celltype_donor'表示按细胞类型+供体双重匹配对照组
                degs_key: 差异表达基因（DEGs）的键名（默认'rank_genes_groups_cov'），在uns中存储各细胞类型-扰动组的DEG
                pert_category: 评估分组列名（默认'cov_drug_dose_name'），用于对比学习采样和评估
                              需与uns中DEG字典的键对齐（格式：细胞类型+药物名称+剂量）
                split_ood: 是否划分分布外（OOD）子集（默认True），OOD为模型未见过的细胞类型
                split_key: 数据拆分状态列名（默认'split'），obs中标记样本为'train'/'test'/'ood'
                seed: 随机种子（默认0），保证数据拆分的可重复性
                use_FM: 是否使用基础模型嵌入（默认True），False则使用原始基因表达向量作为细胞特征
            输出：
                实例属性self.datasets: 字典，包含训练集、测试集（处理组/对照组）、OOD集（处理组/对照组）
        """
        dataset = Dataset(
            adata_obj,
            perturbation_key,
            dose_key,
            celltype_key,
            covariate_keys,
            smiles_key,
            FM_key,
            degs_key,
            pert_category,
            control_key,
            split_key,
            pc_cov,
            seed,
        )

        if split_ood:
            self.datasets = {
                # 训练集：split为"train"的所有样本（处理组+对照组）
                "training": dataset.subset("train", "all"),
                # 测试处理组：split为"test"的处理组样本
                "test_treated": dataset.subset("test", "treated"),
                # 测试对照组：split为"test"的对照组样本
                "test_control": dataset.subset('test','control'),
                # OOD处理组：split为"ood"的处理组样本（分布外测试）
                "ood_treated": dataset.subset('ood','treated'),
                # OOD对照组：split为"ood"的对照组样本（分布外测试）
                "ood_control": dataset.subset('ood','control'),
            }

        else:
            self.datasets = {
                "training": dataset.subset("train", "all"),
                "test_treated": dataset.subset("test", "treated"),
                "test_control": dataset.subset('test','control'),
            }
        # 删除原始dataset对象，释放内存
        del dataset


    def init_drug_embedding(self, chem_model: str, chem_df):
        """
        提取训练集中所有药物的化学嵌入，构建药物特征矩阵（用于模型输入）

        参数：
            chem_model: 化学嵌入模型名称（如'rdkit'），指定提取化学特征的方法
            chem_df: 药物信息DataFrame，包含SMILES字符串及预计算的化学特征（可选）

        输出：
            实例属性self.drug_embeddings: 药物嵌入张量（shape: [药物数量, 嵌入维度]）
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drug_embeddings = get_chemical_representation(
            # 输入：训练集中所有唯一药物的标准化SMILES列表
            smiles=self.datasets['training'].canon_smiles_unique_sorted,
            embedding_model=chem_model, # rdkit
            data_df=chem_df,
            device=device,
        )
    
    def init_model(
        self,
        hparams='',
        mmd_co=0.1,
        celltype_co=1,
        seed=42,
    ):
        """
        初始化CRISP核心模型（PertAE：扰动自编码器）

        参数：
            hparams: 模型超参数字典或配置字符串（如网络层数、隐藏层维度、学习率等）
            mmd_co: MMD（最大均值差异）损失权重（默认0.1）
                    用于衡量处理组与对照组的潜在空间分布差异，约束模型生成合理的扰动后分布
            celltype_co: 细胞类型分类损失权重（默认1）
                    用于约束模型学习细胞类型特异性特征，提升不同细胞类型的预测准确性
            seed: 随机种子（默认0），保证模型初始化的可重复性

        输出：
            实例属性self.autoencoder: 初始化后的PertAE模型
        """
        paired_shape = self.datasets["training"].paired_cell_embeddings.shape
        fm_dim = paired_shape[1] if len(paired_shape) > 1 else self.datasets["training"].num_genes

        self.autoencoder = PertAE(
            self.datasets["training"].num_genes, # 基因数量
            self.datasets["training"].num_drugs, # 药物数量
            self.datasets['training'].num_celltypes, # 细胞类型数量
            self.datasets["training"].num_covariates, # 协变量数量
            drug_embeddings=self.drug_embeddings, # 药物嵌入
            mmd_co=mmd_co,
            celltype_co=celltype_co,
            device=self.device,
            seed=seed,
            hparams=hparams,  # 超参数配置（如网络层数、隐藏层维度等）
            # FM嵌入维度：预计算的细胞基础模型嵌入（如scGPT）的维度
            # FM_ndim=self.datasets["training"].paired_cell_embeddings.shape[1]
            # FM_ndim=fm_dim  # <--- 换成刚刚算出来的自适应维度

        )

    def load_model(self,model_path):
        """
        加载预训练模型权重和配置，恢复训练或用于预测

        参数：
            model_path: 预训练模型文件路径（.pt格式），包含模型状态字典、初始化参数、训练历史
        """
        # 加载模型文件（map_location确保CPU也能加载GPU训练的模型）
        mo = torch.load(model_path,map_location=torch.device('cpu'))
        # state_dict, init_args, _ = mo
        self.autoencoder = PertAE(**mo[1],
                                  drug_embeddings=torch.nn.Embedding.from_pretrained(mo[0]['drug_embeddings.weight']),
                                  device=self.device)
        self.autoencoder.load_state_dict(mo[0])

    def get_prediction(self, 
                       adata_ctrl, 
                       drug_name=None, 
                       dose=None, 
                       ref_drug_dict=None, 
                       FM_emb='X_scGPT', 
                       smile=None, 
                       smile_df=None, 
                       return_adata=True,
                       guidance_scale = 1.5
                       ):
        '''
        核心预测函数：给定对照组单细胞数据和扰动条件，预测扰动后的基因表达及潜在特征

        输入要求（二选一，必须提供一组完整信息）：
            1. drug_name + ref_drug_dict：已知药物名称+药物索引映射（与训练集药物匹配）
            2. smile + smile_df：药物SMILES字符串+预计算化学特征的DataFrame（支持新药物）

        参数详细说明：
            adata_ctrl: 对照组单细胞AnnData对象
            drug_name: 药物名称，需与ref_drug_dict中的键一致
            dose: 药物剂量，与训练集剂量格式一致（如μM）
            ref_drug_dict: 药物名称→模型训练时的药物索引映射字典
            FM_emb: 基础模型嵌入的键名（默认'X_scGPT'），对应adata_ctrl.obsm中的嵌入矩阵
            smile: 药物SMILES字符串，用于提取化学嵌入（支持未见过的药物）
            smile_df: 药物化学特征DataFrame，索引为SMILES字符串，值为预计算的化学嵌入
            return_adata: 是否返回AnnData对象（默认True），False返回PyTorch张量

        输出：
            若return_adata=True：
                adata_pred: 预测的扰动后基因表达
                adata_lat: 扰动后的潜在空间嵌入
                adata_mu: 对照组潜在空间均值嵌入
            若return_adata=False：
                preds: 预测的扰动后基因表达张量（shape: [细胞数, 基因数]）
                latent_treated: 扰动后潜在嵌入张量（shape: [细胞数,  latent维度]）
                mu: 对照组潜在均值嵌入张量（shape: [细胞数,  latent维度]）
        '''

        
        self.autoencoder.eval()
        # 提取对照组细胞的基础模型嵌入（shape: [细胞数, 嵌入维度]）
        cell_embs = torch.tensor(adata_ctrl.obsm[FM_emb],device=self.device)
        # 提取对照组基因表达数据（兼容稀疏矩阵和稠密矩阵）
        try:
            genes = torch.tensor(adata_ctrl.X.A,device=self.device)
        except:
            genes = torch.tensor(adata_ctrl.X,device=self.device)

        # 构建批量药物剂量张量（每个细胞对应相同剂量，shape: [细胞数, 1]）
        n_rows = cell_embs.shape[0]
        # 路径1：通过药物名称获取药物嵌入（适用于训练集中已有的药物）
        if (drug_name is not None) and (drug_name in ref_drug_dict.keys()):
            emb_drugs = (
                # 药物索引张量
                torch.tensor([ref_drug_dict[drug_name] for i in range(n_rows)],dtype=torch.long,device=self.device),
                # 药物剂量张量
                torch.tensor([dose for i in range(n_rows)],dtype=torch.float,device=self.device)
                )
            drugs_pre=None
        # 路径2：通过SMILES获取药物嵌入（适用于新药物/训练集外药物）
        else:
            assert (smile is not None) and (smile_df is not None)
            emb_drugs = (
                None,
                torch.tensor([dose for i in range(n_rows)],dtype=torch.float,device=self.device)
            )
            # 提取预计算的药物化学嵌入
            drugs_pre = torch.tensor(np.array([smile_df.loc[smile].values]*n_rows), dtype=torch.float, device=self.device)

        # 调用核心预测函数，计算扰动后结果
        preds = compute_prediction_CRISP(
            self.autoencoder,
            genes, # 对照组基因表达
            cell_embs, # 对照组细胞基础模型嵌入
            emb_drugs, # 药物索引+剂量（路径1）或 None+剂量（路径2）
            emb_covs=None, # 协变量
            drugs_pre=drugs_pre, # 药物化学嵌入（路径2）
            guidance_scale= guidance_scale
        )
        
        if return_adata:
            # 转换为AnnData对象，保留原始obs信息
            adata_pred = sc.AnnData(preds.cpu().numpy())
            # adata_lat = sc.AnnData(latent_treated.cpu().numpy())
            # adata_mu = sc.AnnData(mu.cpu().numpy())

            adata_pred.obs = adata_ctrl.obs.copy()
            adata_pred.obs['condition'] = drug_name # 添加扰动条件标签
            # adata_lat.obs = adata_pred.obs.copy()
            # adata_mu.obs = adata_pred.obs.copy()

            return adata_pred
        else:
            return preds


    def load_train(self):
        """
        初始化训练集数据加载器（DataLoader），构建高效的训练数据管道
        """
        # 往已有字典中添加新的键值对（若键已存在则覆盖，不存在则新增）

        self.datasets.update(
            {
                "loader_tr": torch.utils.data.DataLoader(
                    self.datasets["training"],
                    batch_size=self.autoencoder.hparams["batch_size"],
                    collate_fn=custom_collate,
                    shuffle=True,
                    drop_last=True,
                )
            }
        )

    def train(
            self,
            num_epochs: int,
            max_minutes: int,
            checkpoint_freq: int,
            save_dir: str,
            eval_ood=True,  # whether to conduct ood evaluation
    ):
        assert save_dir is not None
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir()

        start_time = time.time()
        for epoch in tqdm(range(num_epochs)):
            epoch_training_stats = defaultdict(float)

            for data in self.datasets["loader_tr"]:
                genes, paired_cell_embeddings, drugs_idx, dosages, degs, celltype_idx = data[:6]
                neg_genes, neg_paired_cell_embeddings, neg_drugs_idx, neg_dosages, neg_degs, neg_celltype_idx = data[6:12]
                covariates, neg_covariates = data[12], data[13]

                training_stats = self.autoencoder.iter_update(
                    epoch=epoch,
                    genes=genes,
                    cell_embeddings=paired_cell_embeddings,
                    drugs_idx=drugs_idx,
                    dosages=dosages,
                    degs=degs,
                    celltype_idx=celltype_idx,
                    covariates=covariates,
                    neg_genes=neg_genes,
                    neg_cell_embeddings=neg_paired_cell_embeddings,
                    neg_drugs_idx=neg_drugs_idx,
                    neg_dosages=neg_dosages,
                    neg_degs=neg_degs,
                    neg_celltype_idx=neg_celltype_idx,
                    neg_covariates=neg_covariates,
                )

                for key, val in training_stats.items():
                    epoch_training_stats[key] += val

            # ==========================================
            # 【核心修改 1】动态支持多组件的学习率调度
            # ==========================================
            # 更新 VAE 的学习率
            if hasattr(self.autoencoder, 'scheduler_ae'):
                self.autoencoder.scheduler_ae.step()
            # 更新 LDM 扩散模型的学习率
            if hasattr(self.autoencoder, 'scheduler_diff'):
                self.autoencoder.scheduler_diff.step()

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if key not in self.autoencoder.history.keys():
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(val)
            self.autoencoder.history["epoch"].append(epoch)

            epoch_training_stats["epoch"] = epoch
            logging.info("\n%s", pformat(dict(epoch_training_stats), indent=4, width=1))

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history["elapsed_time_min"] = ellapsed_minutes

            # ==========================================
            # 【核心修改 2】同时监控 VAE 和 Diffusion 的崩溃
            # ==========================================
            reconst_loss_is_nan = math.isnan(epoch_training_stats.get("loss_reconstruction", 0.0))
            diff_loss_is_nan = math.isnan(epoch_training_stats.get("loss_diff", 0.0))

            stop = (
                    ellapsed_minutes > max_minutes
                    or (epoch == num_epochs - 1)
                    or reconst_loss_is_nan
                    or diff_loss_is_nan  # 扩散损失 NaN 也停止
            )


            if epoch in [100,200,300,400,500,550,600] or stop:
                file_name = f'model_{epoch}_split.pt'
                torch.save(
                    (
                        self.autoencoder.state_dict(),
                        self.autoencoder.init_args,
                        self.autoencoder.history,
                    ),
                    os.path.join(save_dir, file_name),
                )
                logging.info(f"model_saved: {file_name}")

                stats = {
                    "epoch": epoch,
                    # "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                    "max_minutes_reached": ellapsed_minutes > max_minutes,
                    "max_epochs_reached": epoch == num_epochs - 1,
                }

                logging.info("\n%s", pformat(stats, indent=4, width=1))

            # 如果触发了停止条件，退出训练循环
            if stop:
                break

        results = self.autoencoder.history
        results["total_epochs"] = epoch
        return results



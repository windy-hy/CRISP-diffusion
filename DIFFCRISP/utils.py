import warnings
from typing import Optional

# import dgl
import pandas as pd
import scanpy as sc
from rdkit import Chem
import yaml
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def load_config(config_path):
    # 以只读模式打开配置文件，使用SafeLoader安全解析YAML
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    # config['model']['hparams']["lr"] = float(config['model']['hparams']["lr"])
    # config['model']['hparams']["wd"] = float(config['model']['hparams']["wd"])
    # config['model']['hparams']["cell_wd"] = float(config['model']['hparams']["cell_wd"])
    # config['model']['hparams']["dropout"] = float(config['model']['hparams']["dropout"])

    return config

def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):
    """
        按协变量（如细胞类型、物种）分组，分别计算每组的差异表达基因（DEG）
        核心逻辑：对每个协变量类别，子集化数据后，以对应类别的对照组为参考计算DEG，最终整合结果

        Parameters
        ----------
        adata : sc.AnnData
            存储单细胞/批量转录组数据的AnnData对象（行：样本/细胞，列：基因）
        groupby : str
            obs层的分组列名，需为「协变量_扰动_连续变量」的组合格式（如celltype_drug_dose），
            用于定义DEG计算的分组
        control_group : str
            groupby列中的对照组标识（如Vehicle_0，代表溶媒对照组+0剂量）
        covariate : str
            obs层的协变量列名（如cell_type、species），用于拆分DEG计算的维度
        n_genes : int (default: 50)
            每个分组返回的差异基因数量
        rank_genes_groups_by_cov : bool (default: True)
            若为True，按差异分数的绝对值排序（同时包含上调/下调基因）；
            若为False，仅按正分数排序（仅包含上调基因）
        key_added : str (default: 'rank_genes_groups_cov')
            存储DEG结果到adata.uns的键名
        return_dict : bool (default: False)
            若为True，返回DEG基因列表和logFC的字典；若为False，仅将结果存入adata.uns

        Returns
        -------
        主要效果：在adata.uns中添加两个键：
            - key_added: 字典，键为分组名，值为对应分组的DEG基因列表
            - {key_added}_logfc: 字典，键为分组名，值为对应基因的log2倍变化值列表

        若return_dict=True，额外返回：
            gene_dict : dict
                分组-DEG基因列表的映射字典
            logfc_dict : dict
                分组-logFC值列表的映射字典

        Usage example:
            rank_genes_groups_by_cov(
                adata,
                groupby='cov_product_dose',
                control_group='Vehicle_0',
                covariate='cell_type'
            )
    """
    # 初始化存储DEG和logFC的字典
    gene_dict = {} # 存储每个分组的差异基因列表
    logfc_dict = {} # 存储每个分组的差异基因logFC值
    # 获取协变量的所有唯一类别（如所有细胞类型）
    cov_categories = adata.obs[covariate].unique()
    # 遍历每个协变量类别，单独计算DEG
    for cov_cat in cov_categories:
        # 构建当前协变量类别的对照组名称（如celltypeA_Vehicle_0）
        control_group_cov = "_".join([cov_cat, control_group])

        # 子集化数据：仅保留当前协变量类别的样本/细胞
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # 调用scanpy内置函数计算差异基因
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,  # 以当前类别的对照组为参考
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # 提取DEG结果：基因名和logFC值
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"]) # 差异基因名
        logfc_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['logfoldchanges']) # 对应logFC
        # print(adata_cov.uns["rank_genes_groups"].keys())
        # break
        # 将每个分组的结果存入字典
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()
            logfc_dict[group] = logfc_genes[group].tolist()
    # 将结果存入adata.uns，便于后续调用
    adata.uns[key_added] = gene_dict
    adata.uns[f'{key_added}_logfc'] = logfc_dict
    # 若指定返回字典，则返回DEG和logFC字典
    if return_dict:
        return gene_dict, logfc_dict


def canonicalize_smiles(smiles: Optional[str]):
    """
    规范化SMILES字符串（化学结构的文本表示），确保同一分子的不同SMILES写法统一
    处理空值/无效SMILES的情况，避免RDKit报错

    Args:
        smiles : Optional[str]
            原始SMILES字符串（可为None/空字符串）

    Returns:
        str | None: 规范化后的SMILES字符串；若输入为空/无效，返回None
    """
    if smiles:
        return Chem.CanonSmiles(smiles) # RDKit的规范化函数
    else:
        return None

def sample_neg(adata, split_key, cov_drug_key, condition_key,seed):
    """
        为训练集样本采样负样本索引，用于对比学习（Contrastive Learning）
        核心逻辑：负样本定义为「同一药物条件下，不同协变量-药物组合的样本」，若无则随机选训练集样本

        Parameters
        ----------
        adata : sc.AnnData
            存储转录组数据的AnnData对象
        split_key : str
            obs层的数据集划分列名（如split），值包括'train'/'val'/'test'，用于筛选训练集
        cov_drug_key : str
            obs层的「协变量-药物」组合列名（如celltype_drug），用于定义正样本组
        condition_key : str
            obs层的条件列名（如drug），用于定义同一条件的样本池
        seed : int
            随机种子，保证采样结果可复现

        Returns
        -------
        list[int]
            负样本索引列表，长度与训练集样本数一致，每个元素对应训练集样本的负样本在训练集中的序号
    """
    random.seed(seed)
    # 子集化数据：仅保留训练集样本
    adata_train = adata[adata.obs[split_key]=='train']
    # 步骤1：按「协变量-药物」组合分组，获取每组的样本索引
    grouped_adata = adata_train.obs.groupby(cov_drug_key, observed=False) # observed=False保留所有类别（包括空类别）
    # 构建「训练集样本索引→训练集内序号」的映射（便于后续索引转换）
    index_to_num = dict(zip(adata_train.obs.index,range(len(adata_train))))
    # 转换每组的样本索引为训练集内序号，并存为字典（键：组合名，值：序号列表）
    grouped_idx = grouped_adata.apply(lambda group: [index_to_num[i] for i in group.index])
    grouped_idx = dict(grouped_idx)

    # 步骤2：按条件（如药物）分组，获取每个条件的样本序号池
    grouped_adata = adata_train.obs.groupby(condition_key,observed=False)
    cond_idx = grouped_adata.apply(lambda group: [index_to_num[i] for i in group.index])
    cond_idx = dict(cond_idx)

    # 步骤3：为每个「协变量-药物」组合，计算负样本池（同一条件下的非本组样本）
    grouped_comp_idx = {}
    for k,v in grouped_idx.items():
        # ct = k.split('_')[0]
        # 拆分组合键，提取药物名（如celltypeA_drugX → drugX）
        dg = k.split('_')[1]
        # 负样本池 = 同一药物的所有样本 - 本组样本（集合差集）
        grouped_comp_idx[k] = list(set(cond_idx[dg]) - set(v))

    # positive_indices = []
    # 步骤4：为每个训练集样本采样负样本
    negative_indices = []
    for i in adata_train.obs[cov_drug_key].values:
        # print(i)
        # positive_idx = random.choice(grouped_idx[i])
        # positive_indices.append(positive_idx)
        # 若当前组合有可用负样本池，随机选一个
        if len(grouped_comp_idx[i]) > 0:
            negative_idx = random.choice(grouped_comp_idx[i])
        # 若无可用负样本池，随机选训练集任意样本
        else: 
            negative_idx = random.choice(range(len(adata_train)))
        negative_indices.append(negative_idx)
    # 删除临时字典，释放内存
    del grouped_idx, grouped_comp_idx
    
    # pos_emb = pretrain_embs[positive_indices]
    # neg_emb = pretrain_embs[negative_indices]

    return negative_indices


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    对张量除batch维度外的所有维度求均值（扁平化非batch维度）
    核心用途：计算扩散模型的损失（如离散高斯似然损失），将高维特征的损失聚合为标量

    示例：
    - 输入shape: (batch_size, channels, height, width) → 对后3维求均值
    - 输入shape: (batch_size, gene_num) → 对gene_num维度求均值
    :param tensor: 输入张量，第一维为batch维度
    :return: 按batch维度的均值张量，shape为(batch_size,)
    """
    # 生成非batch维度的列表：range(1, len(tensor.shape)) → 跳过第0维（batch）
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    药物剂量编码器：将药物SMILES字符串转换为融合剂量信息的rFCFP分子指纹
    核心逻辑：
    1. SMILES→FCFP4指纹（基于特征的摩根指纹）
    2. 指纹值 × log10(剂量+1) 融合剂量信息
    3. 支持多药组合（comb_num>1）：多个药物指纹累加

    :param drug_SMILES_list: 药物SMILES字符串列表，len=样本数
           单药：["C1=CC=CC=C1", ...]；多药组合：["C1=CC=CC=C1+CN1C=NC2=C1C(=O)N(C(=O)N2C)C", ...]
    :param dose_list: 药物剂量列表，len=样本数（多药组合时为总剂量）
    :param num_Bits: FCFP4指纹的维度（默认1024维）
    :param comb_num: 药物组合数，1=单药，>1=多药组合
    :return: 融合剂量的分子指纹数组，shape=(样本数, num_Bits)，dtype=np.float32
    """

    drug_len = len(drug_SMILES_list)
    # 初始化指纹数组（全零）
    fcfp4_array = np.zeros((drug_len, num_Bits))
    # 单药处理逻辑

    for i, smiles in enumerate(drug_SMILES_list):
        smi = smiles
        # 1. SMILES字符串→RDKit分子对象
        mol = Chem.MolFromSmiles(smi)
        # 2. 计算FCFP4指纹（半径2，基于特征，1024位）→二进制字符串
        fcfp4 = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, useFeatures=True, nBits=num_Bits
        ).ToBitString()
        # 3. 二进制字符串→浮点数组（0/1）
        fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
        # 4. 融合剂量信息：指纹值 × log10(剂量+1)（+1避免log10(0)）
        fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
        # 5. 存入数组
        fcfp4_array[i] = fcfp4_list

    return fcfp4_array
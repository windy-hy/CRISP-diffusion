"""
The preprocessing parts of perturbation condition and covariates (except cell type) are adapted from chemCPA.
"""

import logging
from typing import List, Optional, Union
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder

from DIFFCRISP.utils import canonicalize_smiles, sample_neg


indx = lambda a, i: a[i] if a is not None else None

def get_group_idx(data, pert_group_key):
    """
    为扰动分组（如细胞类型+药物+剂量）分配唯一索引，用于模型区分不同分组

    参数：
        data: AnnData对象，包含单细胞数据及obs元信息
        pert_group_key: obs中存储扰动分组的列名（如'cov_drug_dose_name'）

    返回：
        unique_group_name_to_idx: 分组名称 → 索引的映射字典（如{'HT29_药物A_10μM': 0}）
        group_idx: 每个样本的分组索引张量（shape: [样本数]）
    """
    group_name = data.obs[pert_group_key].values # 提取所有样本的分组名称
    unique_group_name = np.unique(group_name) # 获取唯一分组名称并排序
    unique_group_name_to_idx = dict(zip(unique_group_name,range(len(unique_group_name)))) # 构建分组名称到索引的映射
    # 为每个样本分配对应的分组索引，转换为PyTorch张量
    group_idx = torch.tensor([unique_group_name_to_idx[i] for i in group_name],dtype=torch.long)

    return unique_group_name_to_idx, group_idx

def get_degs(data, pert_group_key, de_genes,var_names):
    """
    生成差异表达基因（DEG）的掩码矩阵：标记每个样本中哪些基因是该分组的DEG

    参数：
        data: AnnData对象，包含样本分组信息
        pert_group_key: obs中扰动分组列名（需与de_genes的键对齐）
        de_genes: 字典，key=分组名称，value=该分组的DEG基因名列表
        var_names: 基因名称列表（data.var_names）

    返回：
        degs: DEG掩码张量（shape: [样本数, 基因数]），bool类型，True表示该基因为DEG
    """
    number_idx = np.array(range(len(data))) # 生成样本索引
    degs = torch.zeros((data.shape)) # 初始化DEG掩码矩阵（全False）
    degs = degs.bool()
    # 遍历每个分组，标记该分组样本的DEG
    for k,sub_degenes in de_genes.items():
        # 筛选当前分组的样本索引
        adata_sub_index = number_idx[data.obs[pert_group_key]==k]
        # 标记这些样本中属于DEG的基因（True）
        degs[adata_sub_index,:] = torch.tensor(var_names.isin(sub_degenes)).detach().clone()

    return degs

def get_groups(obs_df,covnames: str,split: str,control_key):
    """
    辅助函数：根据协变量和数据拆分状态，分组对照组样本（用于后续计算配对对照均值）

    参数：
        obs_df: 样本元信息DataFrame
        covnames: 协变量列名（如'celltype_donor'，用于匹配同条件对照组）
        split: 数据拆分状态（如'train'，仅对该拆分下的样本分组）
        control_key: obs中标记对照组的列名（1=对照组，0=处理组）

    返回：
        grouped_df_control: 分组后的对照组DataFrame（按'pc_cov_split'分组）
    """
    # 构建分组键：协变量+拆分状态
    obs_df['pc_cov_split'] = obs_df[covnames].astype(str) + '_' + obs_df[split].astype(str)
    obs_control = obs_df[obs_df[control_key]==1] # 筛选对照组样本
    grouped_df_control = obs_control.groupby('pc_cov_split')  # 按分组键对对照组分组（同分组的样本为同条件对照组）
    return grouped_df_control

# def get_paired_mean(obs_df, data, control_key, pc_cov, split_key, calc='mean',keep_ctrl=True):
#     """
#     计算每个样本的配对对照组均值（基因表达或FM嵌入），核心用于模型学习"处理组vs对照组"的差异
#
#     参数：
#         obs_df: 样本元信息DataFrame
#         data: 待计算均值的矩阵（如基因表达、FM嵌入，shape: [样本数, 特征数]）
#         control_key: 标记对照组的列名（1=对照组，0=处理组）
#         pc_cov: 配对对照的协变量列名（如'celltype_donor'，确保匹配同细胞类型+同供体的对照组）
#         split_key: 数据拆分状态列名（如'split'，区分train/test/ood）
#         calc: 计算方式（'mean'=均值，'std'=标准差，默认'mean'）
#         keep_ctrl: 是否保留对照组原始数据（True=对照组样本不变，仅替换处理组；False=仅返回处理组的配对均值）
#
#     返回：
#         paired: 配对对照组均值矩阵（shape与data一致），处理组样本替换为其对照组均值，对照组样本保留原始值（若keep_ctrl=True）
#     """
#     # 按协变量和拆分状态分组对照组
#     grouped_df_control = get_groups(obs_df,pc_cov,split_key,control_key)
#     # 构建样本索引→矩阵行号的映射（确保样本与data行对齐）
#     index_to_num = dict(zip(obs_df.index,range(len(obs_df))))
#
#     # 定义计算分组均值的函数
#     def get_mean(group):
#         idxnum = torch.tensor([index_to_num[i] for i in group.index]) # 获取当前分组样本的矩阵行号
#         sub_x = data[idxnum] # 提取该分组的所有样本
#         return torch.mean(sub_x,dim=0) # 计算该分组的均值（按特征维度，shape: [特征数]）
#
#     # 定义计算分组标准差的函数
#     def get_std(group):
#         idxnum = torch.tensor([index_to_num[i] for i in group.index])
#         sub_x = data[idxnum]
#         return torch.std(sub_x,dim=0)
#
#     if calc=='mean':
#         grouped_mean = grouped_df_control.apply(get_mean)
#     else:
#         grouped_mean = grouped_df_control.apply(get_std)
#
#     # 转换为分组键→统计值的字典
#     group_dict = dict(grouped_mean)
#     keys = group_dict.keys()
#
#     if keep_ctrl:
#         # choice_control_mean = [group_dict[k] for k in obs_df[obs_df[control_key]==0]['pc_cov_split'].values]
#         # 保留对照组原始数据，仅替换处理组样本
#         paired = data.clone()
#         # 筛选处理组样本中存在配对对照组的索引
#         treated_index = [index_to_num[i] for i in obs_df[obs_df[control_key]==0].index if (obs_df.loc[i,'pc_cov_split'] in keys)]
#
#         # # 为每个处理组样本分配其配对对照组的统计值
#         # choice_control_mean = [group_dict[obs_df['pc_cov_split'][i]] for i in treated_index]
#         # # 替换处理组样本为配对对照组统计值
#         # paired[treated_index,:] = torch.tensor(np.stack(choice_control_mean,axis=0))
#
#         # 提前计算好对照组均值，直接挨个填入，避免同时存在多份巨大的拷贝
#         for i in treated_index:
#             # 直接赋值，PyTorch 会自动处理，这样内存峰值只有原来的一小部分
#             paired[i, :] = torch.tensor(group_dict[obs_df['pc_cov_split'][i]])
#     else:
#         # 仅返回处理组的配对对照组统计值（不保留原始对照组数据）
#         choice_control_mean = [group_dict[k] for k in obs_df['pc_cov_split'].values if k in keys]
#         paired = torch.tensor(np.stack(choice_control_mean,axis=0))
#
#     return paired

def get_paired_mean(obs_df, data, control_key, pc_cov, split_key, calc='mean', keep_ctrl=True):
    import scipy.sparse as sp
    import numpy as np
    import torch

    grouped_df_control = get_groups(obs_df, pc_cov, split_key, control_key)
    index_to_num = dict(zip(obs_df.index, range(len(obs_df))))

    # 动态适配：求均值
    def get_mean(group):
        idxnum = [index_to_num[i] for i in group.index]
        sub_x = data[idxnum]
        if sp.issparse(sub_x):
            # scipy 自带的高效稀疏矩阵求均值算法
            mean_val = np.asarray(sub_x.mean(axis=0)).squeeze()
            return torch.tensor(mean_val, dtype=torch.float32)
        else:
            return torch.mean(sub_x, dim=0)

    # 动态适配：求标准差
    def get_std(group):
        idxnum = [index_to_num[i] for i in group.index]
        sub_x = data[idxnum]
        if sp.issparse(sub_x):
            # 稀疏矩阵标准差：平方的均值 - 均值的平方
            mean_sq = np.asarray(sub_x.power(2).mean(axis=0)).squeeze()
            sq_mean = np.asarray(sub_x.mean(axis=0)).squeeze() ** 2
            std_val = np.sqrt(mean_sq - sq_mean)
            return torch.tensor(std_val, dtype=torch.float32)
        else:
            return torch.std(sub_x, dim=0)

    if calc == 'mean':
        grouped_mean = grouped_df_control.apply(get_mean)
    else:
        grouped_mean = grouped_df_control.apply(get_std)

    group_dict = dict(grouped_mean)
    keys = group_dict.keys()

    if keep_ctrl:

        if sp.issparse(data):
            # ⚠️ 终极内存瘦身 (use_FM=False)：使用极轻量的对象引用数组 (仅占几百KB)
            paired = np.empty(data.shape[0], dtype=object)
            pc_cov_array = obs_df['pc_cov_split'].values
            treated_index = [index_to_num[i] for i in obs_df[obs_df[control_key] == 0].index if
                             (obs_df.loc[i, 'pc_cov_split'] in keys)]

            # 填入处理组的均值引用（复用内存，0 额外消耗）
            for i in treated_index:
                paired[i] = group_dict[pc_cov_array[i]]
            # 对照组保持为空(None)，我们会在 __getitem__ 中动态补全

        else:
            # use_FM=True：data是小型的稠密特征（如512维），直接走Tensor很安全
            paired = torch.zeros((data.shape[0], data.shape[1]), dtype=torch.float32)
            pc_cov_array = obs_df['pc_cov_split'].values
            treated_index = [index_to_num[i] for i in obs_df[obs_df[control_key] == 0].index if
                             (obs_df.loc[i, 'pc_cov_split'] in keys)]
            control_index = [index_to_num[i] for i in obs_df[obs_df[control_key] == 1].index]

            for i in treated_index:
                paired[i, :] = group_dict[pc_cov_array[i]]
            paired[control_index, :] = data[control_index]
    else:
        # 原有的 else 逻辑保持不变...
        choice_control_mean = [group_dict[k] for k in obs_df['pc_cov_split'].values if k in keys]
        paired = torch.tensor(np.stack(choice_control_mean, axis=0))

    return paired

def get_cov_gpt(adata, data, control_key, cov_name, split_key, setting='train'):
    """
    计算特定协变量分组下的对照组FM嵌入均值（用于模型初始化或正则化）

    参数：
        adata: AnnData对象
        data: FM嵌入矩阵（shape: [样本数, 嵌入维度]）
        control_key: 标记对照组的列名
        cov_name: 协变量列名（如'cell_type'）
        split_key: 数据拆分状态列名
        setting: 目标拆分状态（如'train'，仅计算训练集的均值）

    返回：
        group_mean_dict: 协变量分组→FM嵌入均值的字典（如{'HT29': 均值向量}）
    """
    # 筛选训练集的对照组样本
    adata_control = adata[(adata.obs[control_key]==1) & (adata.obs[split_key]==setting)]
    grouped_adata_control = adata_control.obs.groupby(cov_name) # 按协变量分组
    index_to_num = dict(zip(adata.obs.index,range(len(adata)))) # 样本索引 → 矩阵行号映射

    # 计算每个分组的FM嵌入均值
    def get_mean(group):
        idxnum = torch.tensor([index_to_num[i] for i in group.index])
        sub_x = data[idxnum]
        return torch.mean(sub_x,axis=0)
    grouped_mean = grouped_adata_control.apply(get_mean)
    group_mean_dict = dict(grouped_mean)

    return group_mean_dict

def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):
    """
    将药物名称列表转换为标准化（canonical）的SMILES字符串列表，确保药物表征一致性

    参数：
        drug_names: 药物名称列表（如['药物A', '药物B']）
        dataset: AnnData对象，obs中包含药物名称和SMILES映射
        perturbation_key: obs中药物名称列名
        smiles_key: obs中SMILES字符串列名

    返回：
        标准化SMILES列表（与输入药物名称列表顺序一致）
    """
    # 构建药物名称→标准化SMILES的映射字典（去重）
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    # 按输入药物名称顺序，返回对应的标准化SMILES
    return [name_to_smiles_map[name] for name in drug_names]

def drug_to_idx(drugs_names):
    """
    将药物名称列表转换为唯一索引

    参数：
        drugs_names: 药物名称列表

    返回：
        drugs_idx: 每个样本的药物索引列表（单个药物→对应索引，组合药物→第一个药物索引）
        drugs_names_unique_sorted: 唯一药物名称排序列表
        _drugs_name_to_idx: 药物名称→索引的映射字典
    """
    # 提取所有唯一药物（处理组合药物，如"药物A+药物B"拆分为两个药物）
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]
    # 排序唯一药物名称（确保索引固定）
    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    # 构建药物名称 → 索引映射
    _drugs_name_to_idx = {
        smiles: idx for idx, smiles in enumerate(drugs_names_unique_sorted)
    }
    # 为每个样本分配药物索引（组合药物取第一个药物的索引）
    drugs_idx = [_drugs_name_to_idx[drug] for drug in drugs_names]

    return drugs_idx, drugs_names_unique_sorted,_drugs_name_to_idx

class Dataset:
    """
    CRISP核心数据集类：加载原始单细胞数据，完成全流程预处理，生成模型输入所需的所有数据

    主要属性（预处理后输出）：
        genes: 基因表达张量（shape: [样本数, 基因数]）
        FM_emb: 基础模型（如scGPT）嵌入张量（shape: [样本数, 嵌入维度]）
        paired_cell_embeddings: 配对对照组FM嵌入均值张量（shape: [样本数, 嵌入维度]）
        drugs_idx: 药物索引张量（shape: [样本数]）
        dosages: 药物剂量张量（shape: [样本数]）
        degs: DEG掩码张量（shape: [样本数, 基因数]）
        celltype: 细胞类型列表
        indices: 数据拆分索引字典（train/test/ood、control/treated）
    """
    # 类属性声明（明确数据类型）
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor  # 药物编码（OneHot+剂量）
    drugs_idx: torch.Tensor  # 药物索引张量
    max_num_perturbations: int  # 每个样本的最大药物数量
    dosages: torch.Tensor  # 药物剂量张量 (dataset_size, max_num_perturbations)
    drugs_names_unique_sorted: np.ndarray  # 数据集中所有药物名称的排序列表

    def __init__(
        self,
        data,
        perturbation_key=None,
        dose_key=None,
        celltype_key='cell_type',
        covariate_keys=None,
        smiles_key='SMILES',
        FM_key="X_scGPT",
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_name",
        control_key = 'control',
        split_key="split",
        pc_cov='type_donor',
        seed=0,
        use_FM=True,
    ):
        """
        初始化Dataset，完成数据加载和预处理

        参数说明：
            data: 输入数据（AnnData对象或文件路径字符串，如'h5ad文件'）
            perturbation_key: obs中扰动条件列名（如药物名称）
            dose_key: obs中药物剂量列名（默认None）
            celltype_key: obs中细胞类型列名（默认'cell_type'）
            covariate_keys: 其他协变量列名列表（如['donor']）
            smiles_key: obs中药物SMILES字符串列名（默认'SMILES'）
            FM_key: obsm中基础模型嵌入列名（默认'X_scGPT'，预计算的单细胞嵌入）
                    提前计算嵌入可避免训练时重复运行基础模型，节省时间和内存
            degs_key: uns中差异表达基因（DEG）字典的键名（默认'rank_genes_groups_cov'）
            pert_category: obs中评估分组列名（默认'cov_drug_name'）
                          格式：细胞类型+药物名称，用于对比学习采样和评估，需与DEG字典键对齐
            control_key: obs中标记对照组的列名（默认'control'，1=对照组，0=处理组）
            split_key: obs中数据拆分状态列名（默认'split'，值为'train'/'test'/'ood'）
            pc_cov: 配对对照的协变量列名（默认'type_donor'）
                    如'celltype_donor'表示按"细胞类型+供体"匹配对照组
            seed: 随机种子（默认0），保证预处理可重复性
            use_FM: 是否使用基础模型嵌入（默认True），False则使用原始基因表达作为细胞特征
        """
        logging.info(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
        else:
            data = sc.read(data)
        logging.info(f"Finished data loading.")



        # try:
        #     self.genes = torch.Tensor(data.X.A)  # 稀疏矩阵（如CSR）转稠密矩阵
        # except:
        #     self.genes = torch.Tensor(data.X)

        # 如果是稀疏矩阵，强转为按行压缩的 CSR 格式
        if sp.issparse(data.X):
            self.genes = data.X.tocsr()
        else:
            # 兼容非稀疏的情况
            self.genes = torch.tensor(data.X, dtype=torch.float32)

        self.var_names = data.var_names # 存储基因名称列表
        if use_FM:
            self.FM_emb = torch.tensor(data.obsm[FM_key],dtype=torch.float)
        else: 
            # 稀疏矩阵用 copy()，稠密张量用 clone()
            self.FM_emb = self.genes.copy() if sp.issparse(self.genes) else self.genes.clone()
        self.control_key = control_key # 存储对照组标记键名
        obs_df = data.obs.copy() # 复制样本元信息（避免修改原始数据）
        
        # data.obs['drug_dose_name'] = adata.obs.condition.astype(str) + '_' + adata.obs.dose_val.astype(str)
        # data.obs['cov_drug_dose_name'] = adata.obs.cell_type.astype(str) + '_' + adata.obs.drug_dose_name.astype(str)

        # 若obs中无'cov_drug_name'列，自动构建（细胞类型+药物名称）
        if 'cov_drug_name' not in data.obs.columns:
            data.obs['cov_drug_name'] = data.obs[celltype_key].astype(str) + '_' + data.obs[perturbation_key].astype(str)

        # 计算每个样本的配对对照组FM嵌入均值（用于模型学习扰动差异）
        self.paired_cell_embeddings = get_paired_mean(obs_df,self.FM_emb,control_key,pc_cov,split_key)
        # self.paired_genes = get_paired_mean(obs_df,self.genes,control_key,celltype_key,split_key,keep_ctrl=True)
        # self.paired_std = get_paired_mean(obs_df,self.genes,control_key,celltype_key,split_key,calc='std',keep_ctrl=True)

        # 采样负样本索引（相同扰动条件但不同细胞类型的样本，用于对比学习）
        self.neg_idx = sample_neg(data, split_key, 'cov_drug_name',perturbation_key,seed)

        # 处理细胞类型信息
        self.celltype = np.array(data.obs[celltype_key].values)

        # 处理药物扰动和协变量信息
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        # 协变量键名统一转为列表（兼容单个字符串输入）
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys

        # 若存在扰动条件（药物），处理药物名称、剂量、SMILES编码
        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            # 存储评估分组名称（细胞类型+药物名称）
            self.pert_categories = np.array(data.obs[pert_category].values)
            # 存储DEG字典（每个分组的差异表达基因）
            self.de_genes = data.uns[degs_key]
            # 提取药物名称和剂量列表
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # 药物名称 → 唯一索引（支持组合药物）
            drugs_idx,self.drugs_names_unique_sorted,_ = drug_to_idx(self.drugs_names)
            # 药物名称 → 标准化SMILES（确保药物表征一致性）
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
            )
            # 转换药物索引为张量
            self.drugs_idx = torch.tensor(
                drugs_idx,
                dtype=torch.long,
            )
            # 转换药物剂量为张量（统一为float类型）
            dosages = [float(dosage) for dosage in self.dose_names]
            self.dosages = torch.tensor(
                dosages,
                dtype=torch.float32,
            )
        # 若无扰动条件，相关属性设为None
        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
        # 处理协变量（如供体、处理时间等），转为独热编码
        if isinstance(covariate_keys, list) and covariate_keys:
            # 检查协变量键名是否重复
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {} # 协变量键名→原始值列表
            self.covariate_names_unique = {} # 协变量键名→唯一值列表
            self.covariates = [] # 协变量独热编码张量列表
            for cov in covariate_keys:
                # 存储协变量原始值
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                # 存储协变量唯一值
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]
                # 协变量独热编码（适配模型输入）
                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                names = self.covariate_names[cov]
                # 转换原始值为独热编码，转为PyTorch张量
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.covariates = None
        # 计算协变量数量、基因数量、药物数量（模型输入维度）
        if self.covariates is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]
        self.num_genes = self.genes.shape[1] # 基因数量（特征维度）
        self.num_drugs = (
            len(self.drugs_names_unique_sorted)
            if self.drugs_names_unique_sorted is not None
            else 0
        ) # 药物数量

        # 生成DEG掩码矩阵（标记每个样本的差异表达基因）
        self.degs = get_degs(data, pert_category, self.de_genes, self.var_names)
        # 生成评估分组索引（用于模型区分不同分组）
        self.unique_group_name_dict, self.group_idxs = get_group_idx(data, pert_category)
        # 构建数据拆分索引字典（快速筛选train/test/ood、control/treated样本）
        self.indices = {
            # "all": list(range(len(self.genes))),
            "all": list(range(self.genes.shape[0])),
            "control": np.where(data.obs[control_key] == 1)[0].tolist(),
            "treated": np.where(data.obs[control_key] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

    def subset(self, split, condition="all"):
        """
        从Dataset中筛选子集，生成SubDataset对象

        参数：
            split: 数据拆分状态（'train'/'test'/'ood'）
            condition: 样本条件（'control'/'treated'/'all'，默认'all'）

        返回：
            SubDataset: 筛选后的子数据集（包含该子集的所有预处理数据）
        """
        #  筛选双条件匹配的样本索引：取split对应的索引与condition对应的索引的交集
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        # 用筛选出的索引生成子数据集SubDataset
        return SubDataset(self, idx, split)

    def __getitem__(self, i):

        # --- 动态解压当前行的稀疏矩阵 ---
        # 直接提取底层数据，避开外层封装的开销
        if sp.issparse(self.genes):
            # csr_matrix 提取单行的最快写法
            row_start = self.genes.indptr[i]
            row_end = self.genes.indptr[i + 1]
            dense_row = np.zeros(self.genes.shape[1], dtype=np.float32)
            dense_row[self.genes.indices[row_start:row_end]] = self.genes.data[row_start:row_end]
            gene_i = torch.from_numpy(dense_row)
        else:
            gene_i = self.genes[i]

        # 动态解析 paired_cell_embeddings
        p = self.paired_cell_embeddings[i]
        # 如果 p 是 Tensor，说明是处理组的均值或 use_FM=True；如果是 None，直接用它自己的基因表达
        paired_emb_i = p if isinstance(p, torch.Tensor) else gene_i
        if self.covariates is None:
            return (
                # self.genes[i], # 基因表达
                gene_i,  # 使用解压后的 tensor
                # self.paired_cell_embeddings[i], # 配对对照组FM嵌入
                paired_emb_i,  # 替换原来的 self.paired_cell_embeddings[i]
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i), # DEG掩码
                indx(self.celltype_idx, i),
                indx(self.group_idxs,i), # 分组索引
                None,
            )
        else:
            return (
                # self.genes[i],
                gene_i,  # 使用解压后的 tensor
                # self.paired_cell_embeddings[i],
                paired_emb_i,  # 替换原来的 self.paired_cell_embeddings[i]
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                indx(self.group_idxs,i),
                *[indx(cov, i) for cov in self.covariates], # 协变量（独热编码）
            )

    def __len__(self):
        # return len(self.genes)
        return self.genes.shape[0]


class SubDataset:
    """
    子数据集类：从Dataset中筛选出的子集（如train/test/ood、control/treated）
    适配模型训练/评估的批量加载，仅包含该子集的必要数据
    """

    def __init__(self, dataset: Dataset, indices, split_set):
        """
        初始化SubDataset，筛选Dataset中的指定索引样本

        参数：
            dataset: 原始Dataset对象
            indices: 筛选的样本索引列表
            split_set: 数据拆分状态（'train'/'test'/'ood'），用于区分是否需要负样本
        """
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.canon_smiles_unique_sorted = dataset.canon_smiles_unique_sorted

        # 筛选核心数据（基因表达、配对FM嵌入）
        self.genes = dataset.genes[indices]
        # self.paired_genes = dataset.paired_genes[indices]
        # self.paired_std = dataset.paired_std[indices]
        self.paired_cell_embeddings = dataset.paired_cell_embeddings[indices]

        self.drugs_idx = indx(dataset.drugs_idx, indices)
        self.dosages = indx(dataset.dosages, indices)
        if dataset.covariates is not None:
            self.covariates = [indx(cov, indices) for cov in dataset.covariates]
        else:
            self.covariates = None
        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}

        if self.covariate_keys is not None:
            for cov in self.covariate_keys:
                self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)
        else:
            self.covariate_names = None

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs
        # 筛选DEG掩码和分组索引
        self.degs = dataset.degs[indices]
        self.group_idxs = indx(dataset.group_idxs, indices)
        # 处理细胞类型：为子集中的唯一细胞类型分配新索引
        self.celltype = indx(dataset.celltype, indices)
        self.unique_celltype = np.array(sorted(set(self.celltype))) # 子集中的唯一细胞类型
        self.num_celltypes = len(self.unique_celltype)  # 子集中的细胞类型数量
        self.celltype_to_idx = {ct:idx for idx,ct in enumerate(self.unique_celltype)}  # 细胞类型 → 索引映射
        celltype_idx = [self.celltype_to_idx[ct] for ct in self.celltype]
        self.celltype_idx = torch.tensor(celltype_idx, dtype=torch.long) # 细胞类型索引张量

        # 仅训练集需要负样本数据（用于对比学习）
        if split_set == 'train':
            neg_idx = dataset.neg_idx
            self.neg_idx = neg_idx
            self.neg_genes = self.genes[neg_idx]
            # self.neg_paired_genes = self.paired_genes[neg_idx]
            # self.neg_paired_std = self.paired_std[neg_idx]
            self.neg_paired_cell_embeddings = self.paired_cell_embeddings[neg_idx]
            self.neg_drugs_idx = indx(self.drugs_idx, neg_idx)
            self.neg_dosages = indx(self.dosages, neg_idx)
            self.neg_degs = self.degs[neg_idx]
            self.neg_celltype_idx = indx(self.celltype_idx, neg_idx)
            if self.covariate_keys is not None:
                for cov in self.covariate_keys:
                    self.neg_covariate_names[cov] = indx(dataset.covariate_names[cov], neg_idx)
            else:
                self.neg_covariate_names = None
        else:
            # 测试集/OOD集不需要负样本
            self.neg_idx = None


    def __getitem__(self, i):

        # --- 动态解压正样本 ---
        if sp.issparse(self.genes):
            # 🚀 极速提取 CSR 稀疏矩阵单行底层数据 (绕过 scipy 缓慢的 toarray)
            row_start = self.genes.indptr[i]
            row_end = self.genes.indptr[i + 1]
            dense_row = np.zeros(self.genes.shape[1], dtype=np.float32)
            dense_row[self.genes.indices[row_start:row_end]] = self.genes.data[row_start:row_end]
            gene_i = torch.from_numpy(dense_row)
        else:
            gene_i = self.genes[i]
            if not isinstance(gene_i, torch.Tensor):
                gene_i = torch.tensor(gene_i, dtype=torch.float32)

        # 🚀 动态解析正样本的 paired_cell_embeddings
        p = self.paired_cell_embeddings[i]
        paired_emb_i = p if isinstance(p, torch.Tensor) else gene_i

        # --- 动态解压负样本 ---
        if hasattr(self, 'neg_genes') and self.neg_genes is not None:
            if sp.issparse(self.neg_genes):
                # 🚀 极速提取负样本 CSR 稀疏矩阵单行底层数据
                row_start = self.neg_genes.indptr[i]
                row_end = self.neg_genes.indptr[i + 1]
                dense_row_neg = np.zeros(self.neg_genes.shape[1], dtype=np.float32)
                dense_row_neg[self.neg_genes.indices[row_start:row_end]] = self.neg_genes.data[row_start:row_end]
                neg_gene_i = torch.from_numpy(dense_row_neg)
            else:
                neg_gene_i = self.neg_genes[i]
                if not isinstance(neg_gene_i, torch.Tensor):
                    neg_gene_i = torch.tensor(neg_gene_i, dtype=torch.float32)

            # 🚀 动态解析负样本的 paired_cell_embeddings
            p_neg = self.neg_paired_cell_embeddings[i] if self.neg_idx is not None else None
            neg_paired_emb_i = p_neg if isinstance(p_neg, torch.Tensor) else neg_gene_i
        else:
            neg_gene_i = None
            neg_paired_emb_i = None

        if (self.covariates is None):
            return (
                gene_i,
                paired_emb_i,  # 替换原 self.paired_cell_embeddings[i]
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                neg_gene_i,
                neg_paired_emb_i,  # 替换原 self.neg_paired_cell_embeddings[i]
                indx(self.neg_drugs_idx, i),
                indx(self.neg_dosages, i),
                indx(self.neg_degs, i),
                indx(self.neg_celltype_idx, i),
                None,
                None,
            )
        else:
            return (
                gene_i,
                paired_emb_i,  # 替换原 self.paired_cell_embeddings[i]
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                indx(self.degs, i),
                indx(self.celltype_idx, i),
                neg_gene_i,
                neg_paired_emb_i,  # 替换原 self.neg_paired_cell_embeddings[i]
                indx(self.neg_drugs_idx, i),
                indx(self.neg_dosages, i),
                indx(self.neg_degs, i),
                indx(self.neg_celltype_idx, i),
                *[indx(cov, i) for cov in self.covariates],
                *[indx(cov, i) for cov in self.neg_covariates],
            )

    def __len__(self):
        # return len(self.genes)
        return self.genes.shape[0]
    
def custom_collate(batch):
    """
    自定义数据整理函数
    将批量样本的每个部分堆叠为张量，处理None值，自动移至GPU

    参数：
        batch: 批量样本列表（每个元素是__getitem__返回的元组）

    返回：
        concat_batch: 整理后的批量数据（每个元素是堆叠后的张量或None）
    """
    # 转置批量样本：从"样本列表→每个样本的属性元组"转为"属性列表→每个属性的样本列表"
    transposed = zip(*batch)
    concat_batch = []
    # 遍历每个属性的样本列表
    for samples in transposed:
        if samples[0] is None:
            # 若该属性第一个样本为None，整个批量该属性设为None
            concat_batch.append(None)
        else:
            concat_batch.append(torch.stack(samples, 0).to("cuda"))
    return concat_batch

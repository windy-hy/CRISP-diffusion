import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics import R2Score
from tqdm.auto import tqdm
from DIFFCRISP.data import SubDataset
from DIFFCRISP.model import PertAE
from DIFFCRISP.losses import sinkhorn_dist,energy_dist,gaussian_mmd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.metrics import r2_score
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


def plot_umap_comparison(genes_control_sub, y_true, preds, condition_name, save_dir="./plots"):
    """
    绘制单细胞扰动的 UMAP 对比图
    绿色：未用药对照组 (Control)
    蓝色：真实用药组 (True Treated)
    橙色：模型预测组 (Predicted)
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 把张量转回 numpy，准备拼装
    ctrl_np = genes_control_sub.cpu().numpy()
    true_np = y_true.cpu().numpy()
    pred_np = preds.cpu().numpy()

    # 2. 纵向拼接所有细胞矩阵
    X_concat = np.concatenate([ctrl_np, true_np, pred_np], axis=0)

    # 3. 制作对应的标签
    labels = (
            ['1. Control (No Drug)'] * ctrl_np.shape[0] +
            ['2. True Treated'] * true_np.shape[0] +
            ['3. Predicted (Diffusion)'] * pred_np.shape[0]
    )

    # 4. 构建 AnnData 对象 (Scanpy 的标准格式)
    adata = ad.AnnData(X=X_concat)
    adata.obs['Condition'] = labels

    # 5. 执行极速降维流水线
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)

    # 6. 画图并保存
    # 我们用经典的配色：对照组灰色/浅绿色，真实蓝色，预测橙色/红色
    palette = {
        '1. Control (No Drug)': '#B0BEC5',  # 灰蓝色
        '2. True Treated': '#1F77B4',  # 深蓝色
        '3. Predicted (Diffusion)': '#FF7F0E'  # 亮橙色
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata,
        color='Condition',
        palette=palette,
        title=f"UMAP: {condition_name}",
        ax=ax,
        show=False,
        frameon=False
    )

    save_path = os.path.join(save_dir, f"UMAP_{condition_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎨 完美！{condition_name} 的 UMAP 图已保存至: {save_path}")


def bool2idx(x):
    """
    辅助函数：将布尔数组转换为True值对应的索引数组
    用于快速筛选满足条件的样本/基因索引

    参数：
        x: 布尔数组（如标记DEG的掩码、标记分组的掩码）
    返回：
        np.ndarray: True值的位置索引（一维数组）
    """
    # np.where(x) 捕获 True 位置的索引并封装为元组
    return np.where(x)[0]


def repeat_n(x, n):
    """
    辅助函数：将张量在第0维（样本维度）重复n次，适配批量预测需求
    核心场景：单个处理组样本的药物/协变量嵌入，需匹配对照组样本数量

    参数：
        x: 待重复的张量（如单个样本的药物索引/剂量/协变量嵌入）
        n: 重复次数（通常为对照组样本数量）
    返回：
        torch.Tensor: 重复后的张量，shape [n, x.shape[-1]]
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def mean(x: list):
    """
    辅助函数：计算列表均值，处理空列表边界情况

    参数：
        x: 数值列表
    返回：
        float: 列表均值；若列表为空，返回-1（标记无效值）
    """
    return np.mean(x) if len(x) else -1


def compute_r2(y_true, y_pred):
    """
    计算R²决定系数（衡量预测值与真实值的拟合程度），处理极端值和NaN

    参数：
        y_true: 真实值张量（如真实基因表达）
        y_pred: 预测值张量（如模型输出的基因表达）
    返回：
        float: R²值（范围[-∞, 1]，越接近1拟合越好）；若预测值含NaN，返回-1
    """
    # 裁剪预测值到合理范围，避免极端值导致计算错误
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    # 初始化torchmetrics的R²指标
    metric = R2Score()
    # 更新指标：传入预测值和真实值（注意顺序：pred, target）
    metric.update(y_pred, y_true)
    # 计算最终R²值并转换为Python标量
    return metric.compute().item()

def compute_cor(y_true, y_pred):
    """
    计算皮尔逊相关系数（衡量变量间线性相关程度），处理极端值

    参数：
        y_true: 真实值张量
        y_pred: 预测值张量
    返回：
        float: 皮尔逊相关系数（范围[-1, 1]，绝对值越大相关性越强）
    """
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    # 计算相关系数矩阵：shape [2, 2]（行：真实值、预测值；列：对应变量）
    cor_mtx = torch.corrcoef(torch.stack([y_true,y_pred],dim=0))
    # 返回真实值与预测值的相关系数（矩阵[0,1]位置）
    return cor_mtx[0,1].item()

def compute_prediction_CRISP(autoencoder, genes, cell_embeddings, emb_drugs, emb_covs=None, drugs_pre=None, guidance_scale=1.5):
    """
    调用CRISP模型（PertAE）进行预测，返回预测结果并分离计算图（detach）

    参数：
        autoencoder: PertAE模型实例（已训练好的模型）
        genes: 对照组基因表达张量（shape [样本数, 基因数]）
        cell_embeddings: 对照组细胞嵌入张量（如scGPT嵌入，shape [样本数, 嵌入维度]）
        emb_drugs: 元组 (药物索引张量, 剂量张量)，均为shape [样本数]
        emb_covs: 协变量嵌入列表（如供体、批次等，可选）
        drugs_pre: 预训练药物嵌入层（可选）
    返回：
        gene_pred: 预测的处理组基因表达张量（detach后，不参与梯度计算）
        latent_treated: 处理组的潜在空间表示张量
        mu: 变分自编码器的均值张量（VAE组件）
    """
    gene_pred = autoencoder.predict(
            genes=genes,
            cell_embeddings=cell_embeddings,
            drugs_idx=emb_drugs[0],
            dosages=emb_drugs[1],
            covariates=emb_covs,
            drugs_pre=drugs_pre,
            guidance_scale=guidance_scale
        )
    gene_pred = gene_pred.detach()

    return gene_pred

def evaluate(autoencoder: PertAE, treated_dataset: SubDataset, control_dataset: SubDataset,guidance_scale=1.5):

    """
    参数：
        autoencoder: 已训练的PertAE模型实例
        treated_dataset: 处理组子数据集（真实药物扰动后的样本）
        control_dataset: 对照组子数据集（未用药的基线样本）
    返回：
        metrics_dict_all: 所有有效分组的指标均值
        eval_score_dict: 每个扰动分组的详细指标
        pred_dict: 每个分组的预测值/真实值/对照组均值（用于后续分析/可视化）
    """
    # 初始化存储结构：分组指标、预测结果
    eval_score_dict = {}
    pred_dict = {}

    genes_control = control_dataset.genes
    genes_true = treated_dataset.genes

    # dataset.pert_categories contains: 'celltype_perturbation_dose' info
    # 构建扰动分组索引（celltype_drug_dose），方便快速筛选分组样本
    pert_categories_index = pd.Index(treated_dataset.pert_categories, dtype="category")
    # 提取DEG字典（键：分组名，值：该分组的差异表达基因列表）
    de_genes = treated_dataset.de_genes
    # 提取基因名称列表（用于匹配DEG）
    var_names = treated_dataset.var_names
    # 遍历每个唯一的扰动分组（细胞类型+药物+剂量），并获取分组样本数

    # 1. 获取所有分组和对应样本数
    categories, counts = np.unique(treated_dataset.pert_categories, return_counts=True)
    # 2. 用tqdm包装原迭代器，指定总长度（确保进度条100%准确）
    # 自定义进度条格式
    pbar = tqdm(
        zip(categories, counts),
        desc="处理分组",
        total=len(categories),  # 总步数=所有分组数
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # 3. 遍历带进度条的所有分组（保留原有过滤逻辑）
    for cell_drug_dose_comb, category_count in pbar:


        # 边界条件1：分组样本数≤5，统计意义不足，跳过该分组
        if category_count <= 5:
            continue

        # 边界条件2：DMSO/control分组是基线，无需作为扰动组评估，跳过
        if (
            "dmso" in cell_drug_dose_comb.lower()
            or "control" in cell_drug_dose_comb.lower()
        ):
            continue

        # 匹配DEG字典的键格式：
        # 若DEG键是“细胞类型+药物”（无剂量），则拆分分组名匹配；否则直接匹配
        if len(list(de_genes.keys())[0].split('_')) == 2:
            cell_drug_comb = cell_drug_dose_comb.split('_')[0]+'_'+cell_drug_dose_comb.split('_')[1]
            # 生成DEG掩码：标记哪些基因是该分组的差异表达基因
            bool_de = var_names.isin(
                np.array(de_genes[cell_drug_comb])
            )
        else:
            bool_de = var_names.isin(
                np.array(de_genes[cell_drug_dose_comb])
            )
        # 转换DEG掩码为基因索引（用于筛选DEG的表达数据）
        idx_de = bool2idx(bool_de)

        # 边界条件3：DEG数量<2，无法计算R²等指标，跳过该分组
        if len(idx_de) < 2:
            continue

        # 生成当前分组的样本掩码（标记哪些样本属于该分组）
        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        # 转换样本掩码为样本索引（筛选该分组的处理组样本）
        idx_all = bool2idx(bool_category)
        # 取该分组第一个样本的索引（用于提取药物/协变量信息）
        idx = idx_all[0]


        # 提取当前分组的细胞类型（分组名第一个部分）
        ct = cell_drug_dose_comb.split('_')[0]

        # 🔥 新增 1：精确计算 m (该细胞类型在对照组的总样本数)
        # ==========================================================
        m_count = np.sum(np.array(control_dataset.celltype) == ct)

        # --- 🚀 修复 1：安全切片对照组基因并解压 ---
        _sub_ctrl = genes_control[control_dataset.celltype == ct]
        if _sub_ctrl.shape[0] < 5:  # 稀疏矩阵用 shape[0] 取长度更安全
            continue

        if sp.issparse(_sub_ctrl):
            genes_control_sub = torch.tensor(_sub_ctrl.toarray(), dtype=torch.float32, device='cuda')
        else:
            genes_control_sub = torch.tensor(_sub_ctrl, dtype=torch.float32, device='cuda')

        n_rows = genes_control_sub.size(0)

        if treated_dataset.covariates is not None:
            emb_covs = [repeat_n(cov[idx], n_rows) for cov in treated_dataset.covariates]
        else:
            emb_covs = None

        emb_drugs = (
            repeat_n(treated_dataset.drugs_idx[idx], n_rows).squeeze(),
            repeat_n(treated_dataset.dosages[idx], n_rows).squeeze(),
        )

        # --- 🚀 修复 2：动态解析并解压对照组特征 (兼容 use_FM=False) ---
        _sub_emb = control_dataset.paired_cell_embeddings[control_dataset.celltype == ct]
        if isinstance(_sub_emb, torch.Tensor):
            cell_embeddings_sub = _sub_emb.to('cuda', dtype=torch.float32)
        elif _sub_emb[0] is None:
            # use_FM=False时，对照组的特征就是它自己的基因表达
            cell_embeddings_sub = genes_control_sub
        else:
            cell_embeddings_sub = torch.tensor(list(_sub_emb), dtype=torch.float32, device='cuda')

        preds = compute_prediction_CRISP(
            autoencoder,
            genes_control_sub,
            cell_embeddings_sub,
            emb_drugs,
            emb_covs,
            guidance_scale=guidance_scale
        )

        # --- 🚀 修复 3：安全切片真实处理组基因并解压 ---
        _y_true = genes_true[idx_all, :]
        if sp.issparse(_y_true):
            y_true = torch.tensor(_y_true.toarray(), dtype=torch.float32)
        else:
            y_true = torch.tensor(_y_true, dtype=torch.float32)

        preds = preds.detach().to('cpu')

        # 计算均值：对照组均值、真实处理组均值、预测处理组均值（降低样本波动）
        ctrl_m = genes_control_sub.mean(dim=0).to('cpu')
        yt_m = y_true.mean(dim=0).to('cpu')
        yp_m = preds.mean(dim=0).to('cpu')

        # 计算该分组的所有评估指标（全基因+DEG）
        metrics_dict=calc_metrics(yt_m, yp_m, ctrl_m, y_true, preds, idx_de)
        # 🔥 新增 2：把计算好的 m 也存进字典，方便外层评估脚本直接读取
        # ==========================================================
        metrics_dict['m_cells'] = int(m_count)
        # 存储该分组的指标
        eval_score_dict[cell_drug_dose_comb] = metrics_dict
        # 存储该分组的真实/预测/对照数据
        pred_dict[cell_drug_dose_comb] = {'true':yt_m,'pred':yp_m,'ctrl':ctrl_m}
    # 汇总所有分组的指标：计算每个指标的均值
    metrics_dict_all = {}
    for k,v in eval_score_dict.items():
        for k_, v_ in v.items():
            if k_ in list(metrics_dict_all.keys()):
                metrics_dict_all[k_] += [v_]
            else:
                metrics_dict_all[k_] = [v_]
    # 对每个指标取均值，得到整体评估分数
    for k,v in metrics_dict_all.items():
        metrics_dict_all[k] = np.mean(v)
    # 返回：整体指标、分组指标、分组预测数据
    return metrics_dict_all, eval_score_dict, pred_dict



def calc_metrics(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
    """
    计算单个扰动分组的详细评估指标，区分全基因和差异表达基因（DEG）：
    - 绝对表达指标：R²、皮尔逊相关、MSE
    - 相对差异指标：处理组-对照组的皮尔逊相关（更反映药物扰动效果）
    - 分布距离指标：Sinkhorn距离（DEG）


    """

    metrics_dict = {}
    # 筛选DEG的真实/预测均值（用于DEG指标计算）
    yt_de_m = yt_m[idx_de]
    yp_de_m = yp_m[idx_de]
    # 边界处理：若DEG均值全为0，添加极小值避免计算错误
    if yt_de_m.sum() == 0:
        yt_de_m[0] = yt_de_m[0] + 1e-6
    if yp_de_m.sum() == 0:
        yp_de_m[0] = yp_de_m[0] + 1e-6
    # 1. 全基因指标
    metrics_dict['r2score'] = max(compute_r2(yt_m, yp_m),0)
    metrics_dict['r2score_de'] = max(compute_r2(yt_m[idx_de], yp_m[idx_de]),0)
    metrics_dict['pearson'] = pearsonr(yt_m, yp_m)[0]
    # 2. DEG指标（差异表达基因）
    metrics_dict['pearson_de'] = pearsonr(yt_de_m, yp_de_m)[0]
    metrics_dict['mse'] = mse(yt_m,yp_m)
    metrics_dict['mse_de'] = mse(yt_m[idx_de],yp_m[idx_de])
    # 3. 相对差异指标（处理组 - 对照组）：更反映药物扰动的预测效果
    metrics_dict['pearson_delta'] = pearsonr(yt_m-ctrl_m,yp_m-ctrl_m)[0]
    metrics_dict['pearson_delta_de'] = pearsonr(yt_m[idx_de]-ctrl_m[idx_de],yp_m[idx_de]-ctrl_m[idx_de])[0]

    # metrics_dict['mmd'] = gaussian_mmd(y_true,preds).item()
    # metrics_dict['mmd'] = 0
    # metrics_dict['mmd_de'] = gaussian_mmd(y_true[:,idx_de],preds[:,idx_de]).item()

    # if (preds.sum()==0) & (y_true.sum()==0):
    #     metrics_dict['sinkhorn'] = 0
    # else:
    #     metrics_dict['sinkhorn'] = 0

    # 4. 分布距离指标：Sinkhorn距离（衡量预测分布与真实分布的差异，仅计算DEG）
    if (preds[:,idx_de].sum()==0) & (y_true[:,idx_de].sum()==0):
        metrics_dict['sinkhorn_de'] = 0
    else:
        metrics_dict['sinkhorn_de'] = sinkhorn_dist(y_true[:,idx_de],preds[:,idx_de]).item()
        
    # metrics_dict['energy'] = energy_dist(y_true,preds).item()
    # metrics_dict['energy'] = 0
    # metrics_dict['energy_de'] = energy_dist(y_true[:,idx_de],preds[:,idx_de]).item()

    # 计算预测变化量 (yp_m - ctrl_m) 的符号是否与真实变化量 (yt_m - ctrl_m) 的符号相同
    metrics_dict['directional_accuracy'] = torch.mean(
        (torch.sign(yt_m[idx_de] - ctrl_m[idx_de]) == torch.sign(yp_m[idx_de] - ctrl_m[idx_de])).float()
    ).item()

    # 🔥 新增这一行：记录该组合下到底有多少个差异基因 (就是 m 的组成部分)
    metrics_dict['idx_de'] = idx_de.tolist() if isinstance(idx_de, np.ndarray) else list(idx_de)

    # deal with nan value
    # 处理NaN值：将NaN替换为0（避免后续均值计算错误）
    if np.isnan(metrics_dict['pearson']):
        metrics_dict['pearson'] = 0
    if np.isnan(metrics_dict['pearson_de']):
        metrics_dict['pearson_de'] = 0
    if np.isnan(metrics_dict['pearson_delta']):
        metrics_dict['pearson_delta'] = 0
    if np.isnan(metrics_dict['pearson_delta_de']):
        metrics_dict['pearson_delta_de'] = 0

    return metrics_dict
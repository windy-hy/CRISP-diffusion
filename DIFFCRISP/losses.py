"""
This module contains various loss functions used for training CRISP models.
It includes specialized losses for drug perturbation prediction tasks, such as
adaptive losses, MMD-based losses, and losses focused on differentially expressed genes.
该模块包含CRISP模型训练所需的各类损失函数，针对药物扰动预测任务定制：
- 自适应损失：聚焦显著表达变化的基因
- MMD类损失：衡量预测分布与真实分布的差异
- DEG/HVG专项损失：针对差异表达基因/高变异基因优化
- 方向损失：确保表达变化方向预测准确
"""

import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss

def loss_adapt(pred,true,mean_ctrl,std_ctrl,thres=2,std_thres=0.01):
    """
    自适应损失函数：通过对照组统计量归一化预测值和真实值，聚焦表达显著变化的基因
    核心逻辑：
        1. 用对照组的均值/标准差标准化处理组的预测/真实表达（消除基线差异）
        2. 筛选出“预测/真实表达变化超过阈值”的基因，仅对这些基因计算损失
        3. 裁剪极端值避免梯度爆炸，最终返回归一化后的MSE

    Args:
        pred: 预测的处理组基因表达值 (shape: [batch_size, num_genes])
        true: 真实的处理组基因表达值 (shape: [batch_size, num_genes])
        mean_ctrl: 对照组基因表达均值 (shape: [num_genes])
        std_ctrl: 对照组基因表达标准差 (shape: [num_genes])
        thres: 判定“显著表达变化”的阈值（默认2倍标准差）
        std_thres: 最小标准差阈值（避免除以0，默认0.01）

    Returns:
        torch.Tensor: 显著变化基因的归一化MSE损失（标量）
    """
    # 防止标准差为0，添加极小值；裁剪标准差到最小阈值以上
    std_ctrl += 1e-8
    std_ctrl = std_ctrl.clamp(min=std_thres)
    # 计算标准化后的表达变化（处理组 - 对照组均值）/ 对照组标准差
    pred_delta = ((pred-mean_ctrl)/std_ctrl)
    true_delta = ((true-mean_ctrl)/std_ctrl)
    # 生成掩码：筛选出“预测变化”或“真实变化”超过阈值的基因（聚焦显著变化基因）
    # torch.logical_or 是专门用于布尔张量的逐元素逻辑或操作
    mask = torch.logical_or((pred_delta**2)>(thres**2),(true_delta**2)>(thres**2))
    # mask = torch.logical_and(mask_p,std_ctrl>std_thres)
    # 计算损失：裁剪极端值（-10到10）避免梯度爆炸，乘以掩码仅计算显著基因，最后归一化
    return torch.sum(((pred_delta-true_delta).clamp(min=-10,max=10)**2) * mask) / pred.shape[0] / pred.shape[1]


class RBF(nn.Module):
    """
    径向基函数（RBF）核实现，用于MMD（最大均值差异）计算
    特点：使用多个带宽值的核组合，提升分布差异度量的鲁棒性

    Args:
        n_kernels: 不同带宽的核数量（默认5）
        mul_factor: 连续带宽值之间的乘法因子（默认2.0，带宽呈指数分布）
        bandwidth: 固定带宽值，若为None则从数据中自适应估计（默认None）
        device: 计算设备（默认'cuda'）
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None,device='cuda'):
        super().__init__()
        # 生成带宽乘数：mul_factor^(i - n_kernels//2)，使带宽覆盖不同尺度
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """
        确定RBF核的带宽参数：
        - 若指定固定带宽，直接返回
        - 否则从数据的成对L2距离中自适应估计

        Args:
            L2_distances: 样本间成对平方欧氏距离矩阵 (shape: [n_samples, n_samples])

        Returns:
            float: 最终带宽值
        """

        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            # 自适应带宽：所有成对距离的均值（排除对角线，即样本自身距离）
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        """
        计算输入数据的RBF核矩阵（多个带宽核的加权和）

        Args:
            X: 输入数据张量 (shape: [n_samples, feature_dim])

        Returns:
            torch.Tensor: 核矩阵 (shape: [n_samples, n_samples])，包含多个带宽的贡献
        """
        # 计算样本间成对平方欧氏距离
        L2_distances = torch.cdist(X, X) ** 2
        # 计算每个带宽的核矩阵，然后求和（dim=0是带宽维度）
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

def MMDloss(X,Y,device='cuda'):
    """
    计算两个分布之间的最大均值差异（MMD）损失：
    MMD是衡量两个分布相似度的指标，值越小表示分布越接近
    公式：MMD(X,Y) = E[K(X,X)] - 2E[K(X,Y)] + E[K(Y,Y)]，其中K是RBF核

    Args:
        X: 第一个分布的样本 (shape: [n_x, feature_dim])
        Y: 第二个分布的样本 (shape: [n_y, feature_dim])
        device: 计算设备（默认'cuda'）

    Returns:
        torch.Tensor: X和Y的MMD距离（标量）
    """


    kernel = RBF(device=device)
    # 拼接两个分布，计算联合核矩阵
    K = kernel(torch.vstack([X, Y]))
    X_size = X.shape[0]
    # 分解核矩阵：XX（X内部）、XY（X-Y交叉）、YY（Y内部）
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()

    return XX - 2 * XY + YY

# def MMD_group(group_idx,true, pred,device='cuda'):
#     """
#     Computes group-wise MMD loss between true and predicted values.
#     
#     Args:
#         group_idx: Group indices for samples
#         true: True values
#         pred: Predicted values
#         device: Computation device (default: 'cuda')
#         
#     Returns:
#         Average MMD loss across groups
#     """
#     l_ = torch.tensor([0.0],device=device)
#     for i in set(group_idx):
#         mask = group_idx==i.item()
#         # l = MMDloss(true[mask,:],pred[mask,:])
#         l = compute_mmd(true[mask,:],pred[mask,:])
#         l_ += l

#     return l_/len(set(group_idx))

def gaussian_mmd(x,y,blur=1):
    """
    使用GeomLoss库计算高斯核MMD损失（简化版，替代自定义RBF核的MMDloss）

    Args:
        x: 第一个分布的样本 (shape: [n_x, feature_dim])
        y: 第二个分布的样本 (shape: [n_y, feature_dim])
        blur: 高斯核带宽参数（默认1）

    Returns:
        torch.Tensor: 高斯MMD距离（标量）
    """
    # 初始化高斯MMD损失函数，移至样本所在设备
    loss_f = SamplesLoss(loss='gaussian',blur=blur).to(x.device)
    return loss_f(x,y)

def sinkhorn_dist(x,y,blur=.05):
    """
    计算Sinkhorn距离（熵正则化的最优传输距离）：
    衡量两个分布之间的“运输成本”，比MMD更适合捕捉分布的全局差异

    Args:
        x: 第一个分布的样本 (shape: [n_x, feature_dim])
        y: 第二个分布的样本 (shape: [n_y, feature_dim])
        blur: 正则化参数（越小越接近真实最优传输，默认0.05）

    Returns:
        torch.Tensor: Sinkhorn距离（标量）
    """
    # 🚀 强制锁定纯 PyTorch 后端，永不调用 KeOps！
    sink = SamplesLoss(loss="sinkhorn",blur=blur,backend='tensorized').to(x.device)
    return sink(x,y)

def energy_dist(x,y):
    """
    计算能量距离：
    衡量两个分布的差异，具有良好的统计性质（如一致性）

    Args:
        x: 第一个分布的样本 (shape: [n_x, feature_dim])
        y: 第二个分布的样本 (shape: [n_y, feature_dim])

    Returns:
        torch.Tensor: 能量距离（标量）
    """
    Edist = SamplesLoss(loss='energy').to(x.device)
    return Edist(x,y)

class AFMSELoss(torch.nn.Module):
    """
    聚焦差异表达基因（DEG）的MSE损失：
    仅对标记为DEG的基因计算MSE，迫使模型优先准确预测药物扰动后的关键基因
    """

    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
        """
        计算DEG的MSE损失

        Args:
            y: 真实基因表达值 (shape: [batch_size, num_genes])
            pred: 预测基因表达值 (shape: [batch_size, num_genes])
            degs: DEG掩码（1=DEG，0=非DEG）(shape: [batch_size, num_genes])

        Returns:
            torch.Tensor: DEG的平均MSE损失（标量）
        """

        degs = degs.float()
        # 仅保留DEG的真实值和预测值
        y_de = y * degs
        pred_de = pred * degs
        mse = (y_de - pred_de)**2
        num_degs = degs.sum(axis=1)
        mse = mse.sum(axis=1)/(num_degs+1e-6)
        mse = sum(mse)/(len(torch.nonzero(num_degs)) + 1e-6)
        return mse
    
def HVGPRLoss(true,pred,hvgs):
    """
    基于高变异基因（HVG）的皮尔逊相关损失：
    目标是最大化预测值与真实值在HVG上的皮尔逊相关系数，损失=1-平均相关系数

    Args:
        true: 真实基因表达值 (shape: [batch_size, num_genes])
        pred: 预测基因表达值 (shape: [batch_size, num_genes])
        hvgs: HVG掩码（1=HVG，0=非HVG）(shape: [batch_size, num_genes])

    Returns:
        torch.Tensor: 1 - 平均皮尔逊相关系数（标量，越小越好）
    """
    hvgs = hvgs.float()
    # 仅保留HVG的真实值和预测值
    y_hvg = true * hvgs
    pred_hvg = pred * hvgs
    # 计算每个样本的皮尔逊相关系数：cov(X,Y)/(std(X)*std(Y))
    cov_xy = (y_hvg*pred_hvg).mean(1)-y_hvg.mean(1)*pred_hvg.mean(1)
    std_x = torch.sqrt(((y_hvg-y_hvg.mean(1).unsqueeze(1))**2).mean(1))
    std_y = torch.sqrt(((pred_hvg-pred_hvg.mean(1).unsqueeze(1))**2).mean(1))      
    pr = cov_xy / ((std_x * std_y) + 1e-8)

    return 1-pr.mean()

class AFMSELoss_wei(torch.nn.Module):
    """
    加权版DEG-MSE损失：
    基于基因表达方差加权，对表达更稳定（方差小）的DEG赋予更高权重，迫使模型优先准确预测这类基因
    """
    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
        """
        计算加权DEG-MSE损失

        Args:
            y: 真实基因表达值 (shape: [batch_size, num_genes])
            degs: DEG掩码（1=DEG，0=非DEG）(shape: [batch_size, num_genes])

        Returns:
            torch.Tensor: 加权DEG-MSE损失（标量）
        """
        # 步骤1：计算基础DEG-MSE（同AFMSELoss）
        y_de = y * degs
        pred_de = pred * degs
        mse = (y_de - pred_de)**2
        num_degs = degs.sum(axis=1)
        mse = mse.sum(axis=1)/(num_degs+1e-6)
        # 步骤2：筛选有DEG的样本
        treat = torch.nonzero(num_degs)
        degs = degs.bool()
        # 提取所有样本的DEG表达值，重塑为[有DEG的样本数, DEG数量]
        selected_elements = y[degs]
        # 每个样本有50个DEG
        split_tensor = selected_elements.reshape((treat.shape[0],50))
        # 计算每个样本DEG的表达方差
        variances_per_row = split_tensor.var(dim=1)
        # 权重=1/方差（方差越小，权重越大），裁剪权重到[0.3,2]避免极端值
        weight_clip = (1/variances_per_row).clip(0.3,2)
        # 初始化权重张量（与批次样本数一致）
        weight = torch.zeros((y.shape[0]),device=mse.device)
        weight[treat[:,0]] = weight_clip # 为有DEG的样本赋值权重
        # 步骤3：计算加权MSE损失
        loss = sum(mse * weight)/treat.shape[0]

        return loss
    
def dir_loss(y, pred, degs, ctrl):
    """
    方向损失：聚焦基因表达变化的方向（上调/下调）是否预测正确
    核心：对比“真实变化方向”和“预测变化方向”的差异，惩罚方向错误的预测

    Args:
        y: 真实处理组表达值 (shape: [batch_size, num_genes])
        pred: 预测处理组表达值 (shape: [batch_size, num_genes])
        degs: DEG掩码（1=DEG，0=非DEG）(shape: [batch_size, num_genes])
        ctrl: 对照组表达值 (shape: [batch_size, num_genes])

    Returns:
        torch.Tensor: 方向错误的平均MSE损失（标量）
    """

    degs = degs.float()
    y_de = y * degs
    pred_de = pred * degs
    ctrl_de = ctrl * degs
    # 计算变化方向：softsign(x) = x/(1+|x|)，输出∈[-1,1]，表示变化方向和幅度
    dir = (F.softsign(y_de - ctrl_de) - F.softsign(pred_de - ctrl_de))**2
    num_degs = degs.sum(axis=1)
    dir = dir.sum(axis=1)/(num_degs+1e-6)


    dir = sum(dir)/(len(torch.nonzero(num_degs))+1e-6)

    return dir






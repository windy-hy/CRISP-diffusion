import enum
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    生成正弦余弦时间步嵌入（Sinusoidal Timestep Embedding）
    扩散模型的核心组件：将离散的时间步（如0~1000）转为高维连续向量，让模型学习时间依赖的去噪规律
    原理参考Transformer的位置编码，支持分数时间步（非整数timesteps）

    :param timesteps: 1D张量，shape为(N,)，N是batch_size，每个元素是该样本的扩散时间步（可浮点）
    :param dim: 输出嵌入向量的维度（需为偶数，若为奇数会自动补零）
    :param max_period: 控制嵌入的最小频率，越大则低频分量越多
    :return: 时间步嵌入张量，shape为(N, dim)
    """
    half = dim // 2
    # 生成频率向量：从max_period^(-2*(i/half))到max_period^0的等比数列
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # 计算时间步×频率的矩阵：shape (N, half)
    args = timesteps[:, None].float() * freqs[None]
    # 拼接余弦和正弦嵌入：shape (N, half) + (N, half) = (N, dim)（若dim为偶数）
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 若dim为奇数，补零到完整维度
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_beta_schedule(schedule_type, timesteps, beta_start=1e-4, beta_end=0.02):
    """
    获取扩散模型的 Beta 噪声表 (Noise Schedule)

    :param schedule_type: 'linear' 或 'cosine'
    :param timesteps: 总去噪步数
    :param beta_start: 线性表的起始噪声率 (默认 1e-4)
    :param beta_end: 线性表的终止噪声率 (默认 0.02)
    :return: shape 为 (timesteps,) 的 betas 张量
    """
    if schedule_type == 'linear':
        # 经典的线性噪声表 (DDPM 原论文标准配置)
        return torch.linspace(beta_start, beta_end, timesteps)

    elif schedule_type == 'cosine':
        # 适用于高维紧密特征的余弦噪声表 (Improved DDPM 配置)
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        s = 0.008  # 微小偏移量，防止 t=0 时除以 0 导致数值崩溃

        # 公式: f(t) = cos((t/T + s)/(1+s) * PI/2)^2
        f_t = torch.cos(((t / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alphas_cumprod_cosine = f_t / f_t[0]

        # 反推 betas
        betas = 1.0 - (alphas_cumprod_cosine[1:] / alphas_cumprod_cosine[:-1])
        # 极其关键的防御性截断：防止在接近 T 时方差无限大导致 NaN
        return torch.clip(betas, 0.0001, 0.999)

    else:
        raise ValueError(f"不支持的噪声表类型: {schedule_type}，请选择 'linear' 或 'cosine'")


class UMLPBlock(nn.Module):
    """专为 1D 潜空间设计的 cfDiffusion 风格 U-Net 基础块 (带 AdaLN)"""

    def __init__(self, in_dim, out_dim, cond_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(out_dim, out_dim)

        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        # 🔥 极其关键：处理 U-Net 通道数变化时的残差匹配
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)

        h = self.fc1(x)
        h = self.layer_norm(h)
        h = h * (1 + gamma) + beta
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)

        return h + self.skip_proj(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim, z_sem_dim, dropout_prob=0.15):  # 增加 dropout 参数
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(time_embed_dim + z_sem_dim, hidden_dim * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, t_emb, z_sem):
        cond = torch.cat([t_emb, z_sem], dim=-1)
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.fc1(x)
        h = self.layer_norm(h)
        h = h * (1 + gamma) + beta
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ConditionalDiffusion(nn.Module):
    def __init__(self,
                 gene_dim,
                 out_dim,
                 hidden_dim,  # 推荐 512
                 z_sem_dim,  # 你的特征拼接后的维度 (如 128细胞 + 128药物 = 256)
                 time_pos_dim=256,
                 time_embed_dim=512,
                 # time_pos_dim=128,
                 # time_embed_dim=256,
                 timesteps=1000,
                 num_layers=2,
                 schedule_type = 'linear',
                 model = 'MLP'):
        super().__init__()
        self.genes_dim = gene_dim
        self.out_dim = out_dim
        self.timesteps = timesteps
        self.time_pos_dim = time_pos_dim
        self.model = model

        betas = get_beta_schedule(schedule_type, timesteps)

        # ================= 扩散系数注册 (原样保留，标准做法) =================
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

        posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).clamp(min=1e-20)
        posterior_variance = torch.cat([torch.tensor([0.0]), posterior_variance[1:]])
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))

        posterior_mean_coef1 = (
                    self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).clamp(min=1e-20))
        posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod).clamp(
                min=1e-20))
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

        # ================= 网络结构定义 =================
        self.time_embed = nn.Sequential(
            nn.Linear(time_pos_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if model == 'UNET':

        # ================= U-Net 宏观结构定义 (cfDiffusion 架构) =================
            cond_dim = time_embed_dim + z_sem_dim
            dim1 = hidden_dim  # 512
            dim2 = hidden_dim * 2  # 比如 512 (最高只到这里，绝不上 1024/2048)

            self.input_layer = nn.Linear(gene_dim, dim1)
            # 1. 极简 Down 阶段 (就 1 层，轻度压缩)
            self.down1 = UMLPBlock(dim1, dim2, cond_dim)
            # 2. 极简 Middle 阶段 (就 1 层，提取核心药效)
            self.mid = UMLPBlock(dim2, dim2, cond_dim)
            # 3. 极简 Up 阶段 (就 1 层，拼接浅层特征)
            # 输入维度: dim2(当前) + dim1(来自 h0 的跳跃连接)
            self.up1 = UMLPBlock(dim2 + dim1, dim1, cond_dim)
            self.output_layer = nn.Sequential(
                nn.LayerNorm(dim1),
                nn.SiLU(),
                nn.Linear(dim1, out_dim)
            )
            nn.init.zeros_(self.output_layer[-1].weight)
            nn.init.zeros_(self.output_layer[-1].bias)

        elif model == 'MLP':

        # ================= MLP 宏观结构定义 =================
            self.input_layer = nn.Linear(gene_dim, hidden_dim)
            # 组装我们强大的 AdaLN-MLPBlocks
            self.mlp_blocks = nn.ModuleList([
                MLPBlock(
                    hidden_dim=hidden_dim,
                    time_embed_dim=time_embed_dim,
                    z_sem_dim=z_sem_dim
                ) for _ in range(num_layers)
            ])
            self.output_layer = nn.Linear(hidden_dim, out_dim)
            # 让网络一开始瞎猜时预测的噪声就是 0，防止初始化时 Loss 爆炸
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

        else:
            raise NotImplementedError

    def forward(self, x0, z_sem):
        """训练前向传播"""
        batch_size = x0.shape[0]
        t = self.time_sample(batch_size, x0.device)
        xt, noise = self.q_sample(x0, t)

        pred_noise = self.predict_noise(xt, t, z_sem)

        # 返回真实噪声和预测噪声
        x0_recon = self._predict_xstart_from_eps(xt, t, pred_noise)
        return x0_recon, noise, pred_noise, t

    def predict_noise(self, x, t, z_sem):
        if self.model == 'UNET':

            t_emb = timestep_embedding(t, self.time_pos_dim)
            t_emb = self.time_embed(t_emb)
            cond = torch.cat([t_emb, z_sem], dim=-1)
            # 0. 进门 (保留 h0 作为跳跃连接的原生血脉)
            h0 = self.input_layer(x)  # Shape: [B, dim1]
            # 1. 下采样
            h1 = self.down1(h0, cond)  # Shape: [B, dim2]
            # 2. 瓶颈层
            h_mid = self.mid(h1, cond)  # Shape: [B, dim2]
            # 3. 上采样 + 跳跃连接绝杀
            # 将最原始、最清晰的 h0 直接跨越过来，和 h_mid 拼接！
            h_up = self.up1(torch.cat([h_mid, h0], dim=-1), cond)  # Shape: [B, dim1]
            # 4. 输出
            out = self.output_layer(h_up)
            return out

        elif self.model == 'MLP':

            t_emb = timestep_embedding(t, self.time_pos_dim)
            t_emb = self.time_embed(t_emb)

            h = self.input_layer(x)
            for mlp_block in self.mlp_blocks:
                h = mlp_block(h, t_emb, z_sem)  # 此时的融合已经靠 AdaLN 完成！
            h = self.output_layer(h)
            return h

        else:
            raise NotImplementedError

    def time_sample(self, batch_size, device):
        """纯 PyTorch 原生采样，避免 CPU/GPU 频繁通信阻塞"""
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def _predict_xstart_from_eps(self, x_t, t, eps):
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return (x_t - sqrt_one_minus_alpha_cumprod_t * eps) / sqrt_alpha_cumprod_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        coef1 = self.posterior_mean_coef1[t].unsqueeze(1)
        coef2 = self.posterior_mean_coef2[t].unsqueeze(1)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = self.posterior_variance[t].unsqueeze(-1)
        posterior_log_variance = self.posterior_log_variance[t].unsqueeze(-1)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t,z_sem_cond, z_sem_uncond=None, guidance_scale=2.0):
        """单步采样，支持 CFG 控制"""
        out = self.p_mean_variance(x,
                                   t,
                                   z_sem_cond=z_sem_cond,
                                   z_sem_uncond=z_sem_uncond,
                                   guidance_scale=guidance_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().unsqueeze(-1).expand_as(x)
        log_var = out["log_variance"].clamp(max=20.0)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(self, x, t,z_sem_cond, z_sem_uncond=None, guidance_scale = 2.0):
        """计算反向均值和方差，融合 CFG 推理逻辑"""

        # 🔥 修复：使用外部传进来的生物学无条件 (z_sem_uncond)，而不是粗暴的全 0
        if guidance_scale > 1.0 and z_sem_uncond is not None:
            # 有条件的预测 (吃药)
            noise_cond = self.predict_noise(x, t, z_sem_cond)
            # 无条件的预测 (溶剂对照)
            noise_uncond = self.predict_noise(x, t, z_sem_uncond)
            # 提取纯粹药效并放大: 无条件 + scale * (有条件 - 无条件)
            model_output = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            model_output = self.predict_noise(x, t, z_sem_cond)


        # 获取方差
        model_variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_log_variance = torch.log(model_variance.clamp(min=1e-20))
        model_variance = model_variance[t].unsqueeze(-1).expand_as(x)
        model_log_variance = model_log_variance[t].unsqueeze(-1).expand_as(x)

        # 预测 x0
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

        # 获取后验均值
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def generate(self, z_sem_cond, z_sem_uncond=None, guidance_scale=2.0, progress=False):
        """标准 DDPM 生成"""
        # 🔥 修复：签名改为接收 z_sem_cond 和 z_sem_uncond
        batch_size = z_sem_cond.shape[0]
        img = torch.randn(batch_size, self.genes_dim, device=z_sem_cond.device)
        indices = list(range(self.timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="DDPM采样", leave=False)

        # 反向去噪循环
        for i in indices:
            t = torch.tensor([i] * batch_size, device=z_sem_cond.device, dtype=torch.long)
            # 🔥 修复：透传条件
            out = self.p_sample(
                x=img,
                t=t,
                z_sem_cond=z_sem_cond,
                z_sem_uncond=z_sem_uncond,
                guidance_scale=guidance_scale
            )
            img = out["sample"]

        return img

    def generate_ddim(self, z_sem_cond, z_sem_uncond=None,guidance_scale=2.0, ddim_steps=20, eta=0.0):
        """
        极速采样通道：利用 DDIM 将 1000 步压缩到 50 步以内
        eta = 0.0 表示确定性采样，这对保留基因表达的稳定性极好
        """
        batch_size = z_sem_cond.shape[0]
        device = z_sem_cond.device
        img = torch.randn(batch_size, self.genes_dim, device=device)

        # 生成等间距的时间步序列，例如 1000 步切成 50 份
        step_size = self.timesteps // ddim_steps
        time_steps = list(range(0, self.timesteps, step_size))[::-1]

        for i, step in enumerate(time_steps):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # 利用溶剂对照进行 CFG 放大
            if guidance_scale > 1.0 and z_sem_uncond is not None:
                noise_cond = self.predict_noise(img, t, z_sem_cond)
                noise_uncond = self.predict_noise(img, t, z_sem_uncond)
                # 减去溶剂背景
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                pred_noise = self.predict_noise(img, t, z_sem_cond)

            # 2. DDIM 核心公式计算
            alpha_bar_t = self.alphas_cumprod[step]
            # 获取上一个时间步的 alpha_bar (如果是最后一步，即到达 t=0，则设为 1)
            prev_step = step - step_size
            alpha_bar_t_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else torch.tensor(1.0, device=device)

            # 预测出最原始的 x0
            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            # 如果不是完全确定性采样 (eta>0)，计算方差 (我们默认 eta=0.0)
            sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            noise = torch.randn_like(img) if prev_step >= 0 else 0.0

            # 指向下一个状态 (更清晰的图像)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * pred_noise
            img = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma * noise

        return img


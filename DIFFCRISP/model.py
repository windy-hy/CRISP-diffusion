import json
import logging
import math
import os
import random
from collections import OrderedDict
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import mse_loss

# 导入你存放扩散模型的包，假设放在了 CRISP/ConditionalDiffusion.py 中
# 如果都在同一个文件，请直接确保 ConditionalDiffusion 类在这个文件内
from DIFFCRISP.ConditionalDiffusion import ConditionalDiffusion
from DIFFCRISP.losses import AFMSELoss, MMDloss

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(
        self,
        sizes,
        dropout,
        batch_norm=True,
        last_layer_act="linear",
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]

        layers = [l for l in layers if l is not None][:-2]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        layers_dict = OrderedDict(
            {str(i): module for i, module in enumerate(layers)}
        )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            return self.relu(x)
        return self.network(x)


def seed_everything(seed=42):
    random.seed(seed)  # 锁死 Python 原生随机（影响 Dataloader 打乱等）
    os.environ['PYTHONHASHSEED'] = str(seed)  # 锁死 Python 字典 hash 顺序
    np.random.seed(seed)  # 锁死 Numpy（也就是你现在写的）
    torch.manual_seed(seed)  # 锁死 PyTorch CPU（也就是你现在写的）
    torch.cuda.manual_seed(seed)  # 锁死 PyTorch 当前 GPU！
    torch.cuda.manual_seed_all(seed)  # 锁死 PyTorch 所有 GPU！


class DBEncoder(nn.Module):
    """移植自 dbdiffusion 的编码器"""
    def __init__(self, n_genes: int, latent_dim: int = 128, hidden_dim=[1024, 1024, 1024], dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(n_genes, hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.PReLU(),
                ))
            else:
                self.network.append(nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.PReLU(),
                ))
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return F.normalize(x, p=2, dim=1)

class DBDecoder(nn.Module):
    """移植自 dbdiffusion 的解码器"""
    def __init__(self, n_genes: int, latent_dim: int = 128, hidden_dim=[1024, 1024, 1024], dropout: float = 0.0):
        super().__init__()
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.PReLU(),
                ))
            else:
                self.network.append(nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.PReLU(),
                ))
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        return x

def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]

    return [mv_input(x) for x in inputs]


class PertAE(torch.nn.Module):
    def __init__(
            self,
            num_genes: int,
            num_drugs: int,
            num_celltypes: int,
            num_covariates: int,
            drug_embeddings=None,
            mmd_co=None,
            celltype_co=None,
            device="cpu",
            seed=42,
            hparams="",
            FM_ndim=512,
    ):
        super(PertAE, self).__init__()

        seed_everything(seed)

        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.device = device
        self.FM_ndim = FM_ndim
        self.mmd_co = mmd_co
        self.set_hparams_(hparams)

        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "num_celltypes": num_celltypes,
            "FM_ndim": FM_ndim,
            "hparams": hparams,
        }

        # ==========================================
        # 1. 第一阶段：基因 VAE (压缩与解压基因表达)
        # ==========================================
        lat_dim = self.hparams["lat_dim"]

        self.gene_encoder = DBEncoder(num_genes, latent_dim=lat_dim, hidden_dim=[2048, 1024, 512])
        self.gene_decoder = DBDecoder(num_genes, latent_dim=lat_dim, hidden_dim=[512, 1024, 2048])

        # ==========================================
        # 2. 第二阶段：条件网络与扩散模型 (Latent Diffusion)
        # ==========================================
        # 药物特征提取
        if drug_embeddings is None:
            self.drug_embeddings = torch.nn.Embedding(self.num_drugs, lat_dim)
        else:
            self.drug_embeddings = drug_embeddings

        drug_emb_dim = self.drug_embeddings.embedding_dim
        self.drug_proj = nn.Linear(drug_emb_dim + 1, lat_dim)

        self.fm_proj = MLP(
            [self.FM_ndim]
            + [self.hparams["encoder_width"]] * self.hparams["encoder_depth"]
            + [self.hparams["lat_dim"]],
            dropout=self.hparams['dropout'],
        )

        # 核心：引入你写的条件扩散模块 (此时作用在 lat_dim 维度上)
        self.diffusion = ConditionalDiffusion(
            gene_dim=lat_dim,
            out_dim=lat_dim,
            hidden_dim=lat_dim * 4,
            z_sem_dim=lat_dim * 2,
            timesteps=1000,
            num_layers=5,
            model='MLP'
        )

        self.loss_mse = torch.nn.MSELoss()
        self.loss_afmse = AFMSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.iteration = 0
        self.to(self.device)

        # 拆分优化器 (严格分离 VAE 和 扩散模型)
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        # AE 优化器 (5e-4)
        ae_params = (get_params(self.gene_encoder, True) +
                     get_params(self.gene_decoder, True)
                     )
        self.opt_ae = torch.optim.AdamW(ae_params, lr=5e-4, weight_decay=0.01)
        self.scheduler_ae = torch.optim.lr_scheduler.StepLR(self.opt_ae, step_size=50, gamma=0.5)

        # LDM 优化器 (1e-4)
        diff_params = (
                get_params(self.diffusion, True) +
                get_params(self.drug_proj, True) +
                get_params(self.fm_proj, True)
        )
        self.opt_diff = torch.optim.AdamW(diff_params, lr=1e-4,weight_decay=1e-3)
        self.scheduler_diff = torch.optim.lr_scheduler.StepLR(self.opt_diff, step_size=50, gamma=0.5)

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        self.hparams = {"lat_dim": 128,
                        "batch_size": 128,
                        "dropout": 0.2,
                        "encoder_width": 256,
                        "encoder_depth": 4,
                        "ED_dropout":0.1,
                        "dosers_width": 64,
                        "dosers_depth": 3,
                        "embedding_encoder_width": 128,
                        "embedding_encoder_depth": 4,
                        }
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)
        return self.hparams

    def compute_drug_embeddings_(self, drugs_idx=None, dosages=None, drugs_pre=None):

        if drugs_idx is not None:

            drugs_idx, dosages = _move_inputs(drugs_idx, dosages, device=self.device)
            latent_drugs = self.drug_embeddings(drugs_idx)
        else:
            # 推理(Zero-Shot)时，有可能直接传入预计算的 193 维特征
            latent_drugs, dosages = _move_inputs(drugs_pre, dosages, device=self.device)

        # 2. 处理剂量维度
        if len(dosages.size()) == 0:
            dosages = dosages.unsqueeze(0)
        # 确保剂量是 [batch_size, 1] 的形状
        dosages = dosages.view(-1, 1)
        dosages = torch.log1p(dosages)

        # 3. 拼接：[batch_size, 193] + [batch_size, 1] -> [batch_size, 194]
        drug_dose_concat = torch.cat([latent_drugs, dosages], dim=1)
        final_drug_emb = self.drug_proj(drug_dose_concat)
        return final_drug_emb



    def get_semantic_condition(self, cell_embeddings, drugs_idx, dosages, drugs_pre):
        # 1. 获取 128 维的药物特征
        drug_emb = self.compute_drug_embeddings_(drugs_idx, dosages, drugs_pre)

        # 2. 获取 128 维的对照组细胞特征
        if cell_embeddings.shape[1] == self.FM_ndim:

            z_control = self.fm_proj(cell_embeddings)
            z_control = F.normalize(z_control, p=2, dim=1)
        else:
            with torch.no_grad():
                z_control = self.gene_encoder(cell_embeddings)

        # 3. 拼接得到 256 维的 z_sem (128 + 128)
        z_sem = torch.cat([z_control, drug_emb], dim=1)
        return z_sem

    @torch.no_grad()
    def predict(self, genes, cell_embeddings, drugs_idx=None, dosages=None, covariates=None, drugs_pre=None,guidance_scale=1.5):
        """
        推理阶段 (Zero-Shot Generation):
        从高斯噪声开始，以 z_sem 为条件，去噪生成潜在表示 z0，最后解码为基因表达
        """
        genes, cell_embeddings, drugs_idx, dosages = _move_inputs(genes, cell_embeddings, drugs_idx, dosages,
                                                                  device=self.device)
        # 真实的语义条件 (吃药组)
        z_sem_cond = self.get_semantic_condition(cell_embeddings, drugs_idx, dosages, drugs_pre)

        z_sem_uncond = None

        # 生物学 CFG - 减去真实的溶剂背景
        if guidance_scale > 1.0:

            # nips
            DMSO_INDEX = 43
            DMSO_DOSE = 1.0

            # sci
            # DMSO_INDEX = 187
            # DMSO_DOSE = 0.0

            batch_size = cell_embeddings.shape[0]

            solvent_idx = torch.full(
                (batch_size,),
                fill_value=DMSO_INDEX,
                dtype=torch.long,
                device=self.device,
            )

            solvent_dosages = torch.full(
                (batch_size,),
                fill_value=DMSO_DOSE,
                dtype=dosages.dtype,
                device=self.device,
            )

            # 无条件分支使用DMSO，而不是当前药物的drugs_pre
            z_sem_uncond = self.get_semantic_condition(
                cell_embeddings,
                solvent_idx,
                solvent_dosages,
                None,
            )

        img = self.diffusion.generate_ddim(
            z_sem_cond=z_sem_cond,
            z_sem_uncond=z_sem_uncond,
            guidance_scale=guidance_scale
        )

        gene_recon = self.gene_decoder(img)
        gene_recon = torch.nn.functional.relu(gene_recon)
        return gene_recon


    def iter_update(self, epoch, genes, cell_embeddings, drugs_idx=None, dosages=None,degs=None, **kwargs):
        genes, cell_embeddings, drugs_idx, dosages = _move_inputs(genes, cell_embeddings, drugs_idx, dosages,
                                                                  device=self.device)

        # 定义阶段分界线
        AE_WARMUP = 40

        if epoch < AE_WARMUP:
            self.gene_encoder.train()
            self.gene_decoder.train()

            z0 = self.gene_encoder(genes)
            genes_recon = self.gene_decoder(z0)
            ## 计算损失：MSE + AFMSE
            afloss = self.loss_afmse(y=genes, pred=genes_recon, degs=degs)
            mseloss = self.loss_mse(genes_recon, genes)
            l1loss = self.l1_loss(genes_recon, genes)
            mmdloss = MMDloss(genes, genes_recon)

            loss_recon = 0.5 * l1loss + 0.5 * mseloss + 0.2 * afloss
            # loss_recon = 0.75 * l1loss + 0.75 * mseloss + 0.25 * afloss
            # loss_recon =  0.75 * mseloss + 0.25 * afloss
            # loss_ae = loss_recon + self.mmd_co * mmdloss
            loss_ae = loss_recon


            self.opt_ae.zero_grad()
            loss_ae.backward()
            self.opt_ae.step()

            # 保持字典输出一致
            loss_dict = {"loss": loss_ae.item(), "loss_af": afloss.item(),"loss_mse": mseloss.item(),"loss_l1": l1loss.item(),"mmdloss": mmdloss.item()}

        else:
            # ==================================
            # 阶段二：冻结 VAE，训练 Latent Diffusion + 融合后排斥解耦
            # ==================================
            self.gene_encoder.eval()
            self.gene_decoder.eval()
            with torch.no_grad():
                z0_target = self.gene_encoder(genes)

            z_sem = self.get_semantic_condition(cell_embeddings, drugs_idx, dosages, None)

            x0_recon, noise, pred_noise, t = self.diffusion(z0_target, z_sem)
            loss_diff = self.loss_mse(pred_noise, noise)

            total_diff_loss = loss_diff

            self.opt_diff.zero_grad()
            total_diff_loss.backward()
            self.opt_diff.step()

            loss_dict = {
                "loss": total_diff_loss.item(),
                "loss_reconstruction": 0.0,
                "loss_diff": loss_diff.item(),
            }

        return loss_dict


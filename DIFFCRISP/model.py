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

# 原 DiffCRISP diffusion
from DIFFCRISP.ConditionalDiffusion import ConditionalDiffusion

# 新增两个 controlled baseline
from DIFFCRISP.DeterministicPredictor import DeterministicLatentPredictor
from DIFFCRISP.GaussianPredictor import GaussianLatentPredictor

from DIFFCRISP.losses import AFMSELoss, MMDloss


# ============================================================
# 只改这一行就可以切换生成器，trainer.py / data.py 不用改
#
# 可选：
#   "diffusion"      -> 原始 DiffCRISP
#   "deterministic"  -> 确定性 latent predictor
#   "gaussian"       -> conditional Gaussian stochastic predictor
# ============================================================
GENERATOR_TYPE = "diffusion"


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
    random.seed(seed)  # 锁死 Python 原生随机
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DBEncoder(nn.Module):
    """移植自 dbdiffusion 的编码器"""

    def __init__(self, n_genes: int, latent_dim: int = 128,
                 hidden_dim=[1024, 1024, 1024], dropout: float = 0.0):
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

    def __init__(self, n_genes: int, latent_dim: int = 128,
                 hidden_dim=[1024, 1024, 1024], dropout: float = 0.0):
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
            generator_type=None,
    ):
        super(PertAE, self).__init__()

        seed_everything(seed)


        # ============================================================
        if generator_type is None:
            generator_type = GENERATOR_TYPE

        self.generator_type = str(generator_type).lower()

        if self.generator_type not in ["diffusion", "deterministic", "gaussian"]:
            raise ValueError(
                "generator_type must be one of: "
                "'diffusion', 'deterministic', 'gaussian'"
            )

        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.device = device
        self.FM_ndim = FM_ndim
        self.mmd_co = mmd_co
        self.dmso_index = None
        self.dmso_dose = None
        self.set_hparams_(hparams)

        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "num_celltypes": num_celltypes,
            "FM_ndim": FM_ndim,
            "hparams": hparams,
            # 新增：保证保存后的 checkpoint 能恢复正确生成器结构
            "generator_type": self.generator_type,
        }

        # ==========================================
        # 1. 第一阶段：基因 Autoencoder
        # ==========================================
        lat_dim = self.hparams["lat_dim"]

        self.gene_encoder = DBEncoder(
            num_genes,
            latent_dim=lat_dim,
            hidden_dim=[2048, 1024, 512]
        )
        self.gene_decoder = DBDecoder(
            num_genes,
            latent_dim=lat_dim,
            hidden_dim=[512, 1024, 2048]
        )

        # ==========================================
        # 2. 第二阶段：公共条件网络
        # ==========================================
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

        # ============================================================
        # 3. 隔离三种 Stage-II generator
        # ============================================================
        if self.generator_type == "diffusion":
            # 原 DiffCRISP：完全保持你的 ConditionalDiffusion 配置
            self.diffusion = ConditionalDiffusion(
                gene_dim=lat_dim,
                out_dim=lat_dim,
                hidden_dim=lat_dim * 4,
                z_sem_dim=lat_dim * 2,
                timesteps=1000,
                num_layers=5,
                model='MLP'
            )

        elif self.generator_type == "deterministic":
            self.deterministic_predictor = DeterministicLatentPredictor(
                z_sem_dim=lat_dim * 2,
                latent_dim=lat_dim,
                hidden_dim=lat_dim * 4,
                num_layers=5,
                dropout_prob=0.15,
            )

        elif self.generator_type == "gaussian":
            self.gaussian_predictor = GaussianLatentPredictor(
                z_sem_dim=lat_dim * 2,
                latent_dim=lat_dim,
                hidden_dim=lat_dim * 4,
                num_layers=5,
                dropout_prob=0.15,
                fixed_sigma=0.1,
            )

        self.loss_mse = torch.nn.MSELoss()
        self.loss_afmse = AFMSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.iteration = 0
        self.to(self.device)

        # 拆分优化器：保持原接口不变
        get_params = lambda model, cond: list(model.parameters()) if cond else []

        # AE 优化器 (5e-4)：不改
        ae_params = (
            get_params(self.gene_encoder, True)
            + get_params(self.gene_decoder, True)
        )
        self.opt_ae = torch.optim.AdamW(ae_params, lr=5e-4, weight_decay=0.01)
        self.scheduler_ae = torch.optim.lr_scheduler.StepLR(
            self.opt_ae, step_size=50, gamma=0.5
        )

        # ============================================================
        # 新增：根据 generator_type 选择 Stage-II generator 参数
        # ============================================================
        if self.generator_type == "diffusion":
            generator_params = get_params(self.diffusion, True)

        elif self.generator_type == "deterministic":
            generator_params = get_params(self.deterministic_predictor, True)

        elif self.generator_type == "gaussian":
            generator_params = get_params(self.gaussian_predictor, True)

        # 保持变量名 opt_diff / scheduler_diff 不变，这样 trainer.py 不用改
        diff_params = (
            generator_params
            + get_params(self.drug_proj, True)
            + get_params(self.fm_proj, True)
        )
        self.opt_diff = torch.optim.AdamW(
            diff_params,
            lr=1e-4,
            weight_decay=1e-3
        )
        self.scheduler_diff = torch.optim.lr_scheduler.StepLR(
            self.opt_diff, step_size=50, gamma=0.5
        )

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        self.hparams = {
            "lat_dim": 128,
            "batch_size": 128,
            "dropout": 0.2,
            "encoder_width": 256,
            "encoder_depth": 4,
            "ED_dropout": 0.1,
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
            drugs_idx, dosages = _move_inputs(
                drugs_idx,
                dosages,
                device=self.device
            )
            latent_drugs = self.drug_embeddings(drugs_idx)
        else:
            # Zero-Shot 时直接传入预计算分子特征
            latent_drugs, dosages = _move_inputs(
                drugs_pre,
                dosages,
                device=self.device
            )

        # 处理剂量维度
        if len(dosages.size()) == 0:
            dosages = dosages.unsqueeze(0)

        dosages = dosages.view(-1, 1)
        dosages = torch.log1p(dosages)

        # chemical embedding + log-dose
        drug_dose_concat = torch.cat([latent_drugs, dosages], dim=1)
        final_drug_emb = self.drug_proj(drug_dose_concat)
        return final_drug_emb

    def get_semantic_condition(self, cell_embeddings, drugs_idx, dosages, drugs_pre):
        # 1. 获取 128 维药物特征
        drug_emb = self.compute_drug_embeddings_(drugs_idx, dosages, drugs_pre)

        # 2. 获取 128 维对照细胞特征
        if cell_embeddings.shape[1] == self.FM_ndim:
            z_control = self.fm_proj(cell_embeddings)
            z_control = F.normalize(z_control, p=2, dim=1)
        else:
            with torch.no_grad():
                z_control = self.gene_encoder(cell_embeddings)

        # 3. 256-d semantic condition
        z_sem = torch.cat([z_control, drug_emb], dim=1)
        return z_sem

    def configure_dmso(self, datasets):

        dmso_conditions = set()

        for dataset in datasets.values():

            if not hasattr(dataset, "drugs_names"):
                continue

            for i, name in enumerate(dataset.drugs_names):

                if str(name).strip().lower() == "dmso":
                    idx = int(dataset.drugs_idx[i])
                    dose = float(dataset.dosages[i])

                    dmso_conditions.add((idx, dose))

        if len(dmso_conditions) != 1:
            raise ValueError(
                f"Expected one unique DMSO condition, "
                f"found: {dmso_conditions}"
            )

        self.dmso_index, self.dmso_dose = dmso_conditions.pop()

        print(
            f"[DMSO] index={self.dmso_index}, "
            f"dose={self.dmso_dose}"
        )

    def _get_dmso_semantic_condition(self, cell_embeddings, dosages):

        if self.dmso_index is None or self.dmso_dose is None:
            raise RuntimeError("DMSO condition has not been configured.")

        DMSO_INDEX = self.dmso_index
        DMSO_DOSE = self.dmso_dose

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

        z_sem_uncond = self.get_semantic_condition(
            cell_embeddings,
            solvent_idx,
            solvent_dosages,
            None,
        )
        return z_sem_uncond

    @torch.no_grad()
    def predict(
            self,
            genes,
            cell_embeddings,
            drugs_idx=None,
            dosages=None,
            covariates=None,
            drugs_pre=None,
            guidance_scale=1.5,
    ):
        """
        三个模型共用相同的：
            scFM condition
            drug/dose condition
            Stage-I gene decoder

        只切换 latent generator。
        """

        genes, cell_embeddings, drugs_idx, dosages = _move_inputs(
            genes,
            cell_embeddings,
            drugs_idx,
            dosages,
            device=self.device
        )

        # 真实 target-drug semantic condition
        z_sem_cond = self.get_semantic_condition(
            cell_embeddings,
            drugs_idx,
            dosages,
            drugs_pre
        )

        z_sem_uncond = None

        # 为了 controlled comparison，三个模型都允许使用同一 DMSO reference。
        # 如果你想做“纯 generator”消融，可以让三个模型统一 guidance_scale=1.0。
        if guidance_scale > 1.0:
            z_sem_uncond = self._get_dmso_semantic_condition(
                cell_embeddings,
                dosages
            )

        # ============================================================
        # A. 原始 DiffCRISP
        # ============================================================
        if self.generator_type == "diffusion":
            img = self.diffusion.generate_ddim(
                z_sem_cond=z_sem_cond,
                z_sem_uncond=z_sem_uncond,
                guidance_scale=guidance_scale,
                ddim_steps=20,
            )

        # ============================================================
        # B. Deterministic latent predictor
        # ============================================================
        elif self.generator_type == "deterministic":
            z_cond = self.deterministic_predictor(z_sem_cond)

            if guidance_scale > 1.0 and z_sem_uncond is not None:
                z_uncond = self.deterministic_predictor(z_sem_uncond)
                # DMSO-referenced latent guidance
                img = z_uncond + guidance_scale * (z_cond - z_uncond)
            else:
                img = z_cond

        # ============================================================
        # C. Conditional Gaussian stochastic predictor
        # ============================================================
        elif self.generator_type == "gaussian":
            mu_cond = self.gaussian_predictor(z_sem_cond)

            img = self.gaussian_predictor.sample(
                mu_cond
            )

        else:
            raise RuntimeError(f"Unknown generator_type: {self.generator_type}")

        # 三种模型共用同一个 decoder
        gene_recon = self.gene_decoder(img)
        gene_recon = torch.nn.functional.relu(gene_recon)
        return gene_recon

    def iter_update(
            self,
            epoch,
            genes,
            cell_embeddings,
            drugs_idx=None,
            dosages=None,
            degs=None,
            **kwargs
    ):
        genes, cell_embeddings, drugs_idx, dosages = _move_inputs(
            genes,
            cell_embeddings,
            drugs_idx,
            dosages,
            device=self.device
        )

        # 定义阶段分界线
        AE_WARMUP = 40

        # ============================================================
        # Stage I：完全保持你的原训练逻辑
        # ============================================================
        if epoch < AE_WARMUP:
            self.gene_encoder.train()
            self.gene_decoder.train()

            z0 = self.gene_encoder(genes)
            genes_recon = self.gene_decoder(z0)

            afloss = self.loss_afmse(y=genes, pred=genes_recon, degs=degs)
            mseloss = self.loss_mse(genes_recon, genes)
            l1loss = self.l1_loss(genes_recon, genes)
            # mmdloss = MMDloss(genes, genes_recon)

            loss_recon = 0.5 * l1loss + 0.5 * mseloss + 0.2 * afloss
            # 保持你 final experiment：MMD 不参与 loss
            loss_ae = loss_recon

            self.opt_ae.zero_grad()
            loss_ae.backward()
            self.opt_ae.step()

            loss_dict = {
                "loss": loss_ae.item(),
                "loss_af": afloss.item(),
                "loss_mse": mseloss.item(),
                "loss_l1": l1loss.item(),
                # "mmdloss": mmdloss.item()
            }

        # ============================================================
        # Stage II：三个模型共享 z0_target 和 z_sem，只替换 generator
        # ============================================================
        else:
            self.gene_encoder.eval()
            self.gene_decoder.eval()

            with torch.no_grad():
                z0_target = self.gene_encoder(genes)

            z_sem = self.get_semantic_condition(
                cell_embeddings,
                drugs_idx,
                dosages,
                None
            )

            # --------------------------------------------------------
            # A. 原始 DiffCRISP
            # --------------------------------------------------------
            if self.generator_type == "diffusion":
                x0_recon, noise, pred_noise, t = self.diffusion(
                    z0_target,
                    z_sem
                )
                loss_diff = self.loss_mse(pred_noise, noise)
                total_diff_loss = loss_diff

                loss_dict = {
                    "loss": total_diff_loss.item(),
                    "loss_reconstruction": 0.0,
                    "loss_diff": loss_diff.item(),
                }

            # --------------------------------------------------------
            # B. Deterministic latent predictor
            # --------------------------------------------------------
            elif self.generator_type == "deterministic":
                z0_pred = self.deterministic_predictor(z_sem)
                loss_latent = self.loss_mse(z0_pred, z0_target)
                total_diff_loss = loss_latent

                # 保留 loss_diff key，trainer.py 的 NaN 检测不需要改
                loss_dict = {
                    "loss": total_diff_loss.item(),
                    "loss_reconstruction": 0.0,
                    "loss_diff": loss_latent.item(),
                    "loss_latent_mse": loss_latent.item(),
                }

            # --------------------------------------------------------
            # C. Conditional Gaussian stochastic predictor
            # --------------------------------------------------------
            elif self.generator_type == "gaussian":
                mu = self.gaussian_predictor(z_sem)

                loss_gaussian = self.gaussian_predictor.mse_loss(
                    target=z0_target,
                    mu=mu
                )

                total_diff_loss = loss_gaussian

                loss_dict = {
                    "loss": total_diff_loss.item(),
                    "loss_reconstruction": 0.0,
                    "loss_diff": loss_gaussian.item(),
                    "loss_gaussian_mse": loss_gaussian.item(),
                }

            else:
                raise RuntimeError(
                    f"Unknown generator_type: {self.generator_type}"
                )

            # 三种 Stage-II 模型共用相同 optimizer 接口
            self.opt_diff.zero_grad()
            total_diff_loss.backward()
            self.opt_diff.step()

        return loss_dict

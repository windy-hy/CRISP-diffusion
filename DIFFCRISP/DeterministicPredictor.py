import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicResidualBlock(nn.Module):
    """
    Deterministic baseline block.

    It keeps the same basic conditioning idea as DiffCRISP's AdaLN MLP blocks:
    z_sem is injected into every residual block. The only thing removed is the
    diffusion/noising process and timestep.
    """

    def __init__(self, hidden_dim: int, z_sem_dim: int, dropout_prob: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.cond_proj = nn.Linear(z_sem_dim, hidden_dim * 2)

        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(z_sem).chunk(2, dim=-1)
        h = self.fc1(x)
        h = self.layer_norm(h)
        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class DeterministicLatentPredictor(nn.Module):
    """
    Controlled deterministic baseline for DiffCRISP.

    Input:
        z_sem: [batch, z_sem_dim], e.g. 256 = 128 cell + 128 drug/dose

    Output:
        z_pred: [batch, latent_dim], e.g. 128

    It directly predicts one treated latent state:
        z_sem -> z_pred

    No diffusion and no stochastic sampling are used.
    """

    def __init__(
        self,
        z_sem_dim: int = 256,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 5,
        dropout_prob: float = 0.15,
    ):
        super().__init__()
        self.z_sem_dim = z_sem_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(z_sem_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DeterministicResidualBlock(
                hidden_dim=hidden_dim,
                z_sem_dim=z_sem_dim,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sem: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(z_sem)
        for block in self.blocks:
            h = block(h, z_sem)
        return self.output_layer(h)

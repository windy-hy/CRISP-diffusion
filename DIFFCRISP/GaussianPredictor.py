import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianResidualBlock(nn.Module):
    """
    Conditional residual block for the Gaussian stochastic baseline.

    z_sem is injected into every block through AdaLN-style modulation,
    matching DiffCRISP's conditioning philosophy while removing diffusion.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_sem_dim: int,
        dropout_prob: float = 0.15
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.cond_proj = nn.Linear(
            z_sem_dim,
            hidden_dim * 2
        )

        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        z_sem: torch.Tensor
    ) -> torch.Tensor:

        gamma, beta = self.cond_proj(
            z_sem
        ).chunk(2, dim=-1)

        h = self.fc1(x)

        h = self.layer_norm(h)

        h = h * (1.0 + gamma) + beta

        h = F.silu(h)

        h = self.dropout(h)

        h = self.fc2(h)

        return x + h


class GaussianLatentPredictor(nn.Module):
    """
    Fixed-variance stochastic Gaussian baseline for DiffCRISP.

    The model predicts only the conditional mean:

        mu = mu(z_sem)

    and samples:

        z_sample = mu + sigma * epsilon

        epsilon ~ N(0, I)

    where sigma is fixed rather than learned.

    This provides a controlled stochastic baseline:
        deterministic predictor:
            z = mu

        Gaussian predictor:
            z = mu + sigma * epsilon

        diffusion:
            learns the full conditional latent distribution
    """

    def __init__(
        self,
        z_sem_dim: int = 256,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 5,
        dropout_prob: float = 0.15,
        fixed_sigma: float = 0.05,
    ):
        super().__init__()

        self.z_sem_dim = z_sem_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Fixed Gaussian standard deviation
        self.fixed_sigma = float(fixed_sigma)

        # Input projection
        self.input_layer = nn.Linear(
            z_sem_dim,
            hidden_dim
        )

        # Residual AdaLN-style blocks
        self.blocks = nn.ModuleList([
            GaussianResidualBlock(
                hidden_dim=hidden_dim,
                z_sem_dim=z_sem_dim,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)
        ])

        # Only predict conditional mean
        self.mu_head = nn.Linear(
            hidden_dim,
            latent_dim
        )

    def forward(
        self,
        z_sem: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict conditional latent mean.

        Parameters
        ----------
        z_sem:
            Semantic condition, shape:
            [batch_size, z_sem_dim]

        Returns
        -------
        mu:
            Predicted treated latent mean,
            shape:
            [batch_size, latent_dim]
        """

        h = self.input_layer(z_sem)

        for block in self.blocks:
            h = block(
                h,
                z_sem
            )

        mu = self.mu_head(h)

        return mu

    def sample(
        self,
        mu: torch.Tensor,
        eps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample from a fixed-variance isotropic Gaussian:

            z = mu + sigma * epsilon

            epsilon ~ N(0, I)
        """

        if eps is None:
            eps = torch.randn_like(mu)

        return (
            mu
            + self.fixed_sigma * eps
        )

    @staticmethod
    def mse_loss(
        target: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        """
        Train the conditional mean using latent-space MSE.

        With fixed Gaussian variance,
        maximizing Gaussian likelihood is equivalent
        to minimizing MSE up to a constant scaling factor.
        """

        return F.mse_loss(
            mu,
            target
        )


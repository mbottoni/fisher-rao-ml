from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fisher_rao_ml.losses import diagonal_gaussian_fisher_rao_squared


class SmallMnistVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 400) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mean(h), self.logvar(h)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


def vae_loss(
    reconstruction_logits: Tensor,
    x: Tensor,
    mean: Tensor,
    logvar: Tensor,
    regularizer: str,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction = F.binary_cross_entropy_with_logits(
        reconstruction_logits,
        x,
        reduction="sum",
    ) / x.shape[0]

    if regularizer == "kl":
        kl_per_sample = -0.5 * torch.sum(
            1.0 + logvar - mean.square() - logvar.exp(),
            dim=-1,
        )
        regularization = kl_per_sample.mean()
    elif regularizer == "fisher_rao":
        regularization = diagonal_gaussian_fisher_rao_squared(mean, logvar).mean()
    else:
        raise ValueError(f"Unknown regularizer: {regularizer}")

    total = reconstruction + beta * regularization
    return total, {
        "loss": total.detach(),
        "reconstruction": reconstruction.detach(),
        "regularization": regularization.detach(),
    }

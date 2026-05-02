from __future__ import annotations

import torch
from torch import Tensor

from fisher_rao_ml.distribution_losses import distribution_loss


def pairwise_student_t_affinities(embedding: Tensor, eps: float = 1e-12) -> Tensor:
    distances = torch.cdist(embedding, embedding).square()
    weights = 1.0 / (1.0 + distances)
    weights = weights.fill_diagonal_(0.0)
    return weights / weights.sum().clamp_min(eps)


def symmetric_gaussian_affinities(x: Tensor, bandwidth: float = 1.0, eps: float = 1e-12) -> Tensor:
    distances = torch.cdist(x, x).square()
    weights = torch.exp(-distances / (2.0 * bandwidth**2))
    weights = weights.fill_diagonal_(0.0)
    return weights / weights.sum().clamp_min(eps)


def tsne_distribution_loss(p: Tensor, q: Tensor, objective: str, eps: float = 1e-12) -> Tensor:
    p_flat = p.flatten()
    q_flat = q.flatten()
    return distribution_loss(
        p_flat.unsqueeze(0),
        q_flat.unsqueeze(0),
        objective=objective,
        eps=eps,
    )

from __future__ import annotations

import torch
from torch import Tensor

from fisher_rao_ml.losses import categorical_fisher_rao_squared


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

    if objective == "kl":
        return (p_flat * (p_flat.clamp_min(eps).log() - q_flat.clamp_min(eps).log())).sum()
    if objective == "kl_smoothed":
        smoothing = 1e-3
        uniform = torch.full_like(p_flat, 1.0 / p_flat.numel())
        p_smooth = (1.0 - smoothing) * p_flat + smoothing * uniform
        q_smooth = (1.0 - smoothing) * q_flat + smoothing * uniform
        return (
            p_smooth
            * (p_smooth.clamp_min(eps).log() - q_smooth.clamp_min(eps).log())
        ).sum()
    if objective == "kl_capped":
        per_edge = p_flat * (p_flat.clamp_min(eps).log() - q_flat.clamp_min(eps).log())
        return per_edge.clamp_max(0.05).sum()
    if objective == "jensen_shannon":
        m = 0.5 * (p_flat + q_flat)
        kl_pm = p_flat * (p_flat.clamp_min(eps).log() - m.clamp_min(eps).log())
        kl_qm = q_flat * (q_flat.clamp_min(eps).log() - m.clamp_min(eps).log())
        return 0.5 * (kl_pm.sum() + kl_qm.sum())
    if objective == "hellinger":
        return (
            0.5
            * (torch.sqrt(p_flat.clamp_min(eps)) - torch.sqrt(q_flat.clamp_min(eps)))
            .square()
            .sum()
        )
    if objective == "fisher_rao":
        return categorical_fisher_rao_squared(p_flat, q_flat, eps=eps)

    raise ValueError(f"Unknown objective: {objective}")

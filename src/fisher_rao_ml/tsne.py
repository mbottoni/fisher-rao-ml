from __future__ import annotations

import math

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


def perplexity_gaussian_affinities(
    x: Tensor,
    perplexity: float = 30.0,
    eps: float = 1e-12,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Tensor:
    """Per-point bandwidth t-SNE affinities via binary search on Shannon entropy.

    For each point i, finds sigma_i such that H(P_{i|.}) = log(perplexity).
    Returns the symmetric joint P_{ij} = (P_{j|i} + P_{i|j}) / (2n), normalized.
    """
    n = x.shape[0]
    target_h = math.log(perplexity)
    sq_dist = torch.cdist(x, x).square()

    betas = torch.ones(n, device=x.device, dtype=x.dtype)
    beta_lo = torch.zeros(n, device=x.device, dtype=x.dtype)
    beta_hi = torch.full((n,), float("inf"), device=x.device, dtype=x.dtype)

    for _ in range(max_iter):
        neg_d = -sq_dist * betas.unsqueeze(1)
        neg_d = neg_d - neg_d.amax(dim=1, keepdim=True)
        p_cond = neg_d.exp()
        p_cond.fill_diagonal_(0.0)
        row_sum = p_cond.sum(dim=1, keepdim=True).clamp_min(eps)
        p_cond = p_cond / row_sum
        log_p = p_cond.clamp_min(eps).log()
        h = -(p_cond * log_p).sum(dim=1)

        diff = h - target_h
        converged = diff.abs() < tol
        need_update = ~converged

        # H > target → beta too small → increase
        increase = need_update & (diff > 0)
        # H < target → beta too large → decrease
        decrease = need_update & (diff < 0)

        beta_lo = torch.where(increase, betas, beta_lo)
        beta_hi = torch.where(decrease, betas, beta_hi)

        new_betas = betas.clone()
        new_betas = torch.where(increase & beta_hi.isfinite(), (betas + beta_hi) / 2.0, new_betas)
        new_betas = torch.where(increase & ~beta_hi.isfinite(), betas * 2.0, new_betas)
        new_betas = torch.where(decrease & (beta_lo > 0), (beta_lo + betas) / 2.0, new_betas)
        new_betas = torch.where(decrease & (beta_lo == 0), betas / 2.0, new_betas)
        betas = new_betas

        if converged.all():
            break

    neg_d = -sq_dist * betas.unsqueeze(1)
    neg_d = neg_d - neg_d.amax(dim=1, keepdim=True)
    p_cond = neg_d.exp()
    p_cond.fill_diagonal_(0.0)
    row_sum = p_cond.sum(dim=1, keepdim=True).clamp_min(eps)
    p_cond = p_cond / row_sum

    p_sym = (p_cond + p_cond.t()) / (2.0 * n)
    p_sym.fill_diagonal_(0.0)
    return p_sym / p_sym.sum().clamp_min(eps)


def tsne_distribution_loss(p: Tensor, q: Tensor, objective: str, eps: float = 1e-12) -> Tensor:
    p_flat = p.flatten()
    q_flat = q.flatten()
    return distribution_loss(
        p_flat.unsqueeze(0),
        q_flat.unsqueeze(0),
        objective=objective,
        eps=eps,
    )

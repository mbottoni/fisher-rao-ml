from __future__ import annotations

import torch
from torch import Tensor


def _safe_probs(probs: Tensor, eps: float) -> Tensor:
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def categorical_fisher_rao_distance(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Fisher-Rao geodesic distance on the probability simplex.

    For categorical distributions, the square-root map sends the simplex to the
    positive orthant of the unit sphere. With the common information-geometry
    convention, the geodesic distance is:

        d_FR(p, q) = 2 arccos(sum_i sqrt(p_i q_i))

    The returned tensor is batch-shaped if ``p`` and ``q`` are batched.
    """
    p = _safe_probs(p, eps)
    q = _safe_probs(q, eps)
    affinity = torch.sqrt(p * q).sum(dim=-1)
    affinity = affinity.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 2.0 * torch.acos(affinity)


def categorical_fisher_rao_squared(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    distance = categorical_fisher_rao_distance(p, q, eps=eps)
    return distance.square()


def diagonal_gaussian_fisher_rao_distance(
    mean: Tensor,
    logvar: Tensor,
    prior_mean: float | Tensor = 0.0,
    prior_logvar: float | Tensor = 0.0,
    eps: float = 1e-8,
) -> Tensor:
    """Approximate diagonal-Gaussian Fisher-Rao distance to a diagonal prior.

    Each latent dimension is treated as an independent univariate Gaussian
    manifold. The univariate Fisher-Rao geometry has a closed-form hyperbolic
    distance; the diagonal-product distance sums squared per-dimension lengths.
    """
    sigma = torch.exp(0.5 * logvar).clamp_min(eps)
    prior_logvar_t = torch.as_tensor(prior_logvar, device=mean.device, dtype=mean.dtype)
    prior_sigma = torch.exp(0.5 * prior_logvar_t)
    prior_sigma = prior_sigma.clamp_min(eps)
    prior_mean_t = torch.as_tensor(prior_mean, device=mean.device, dtype=mean.dtype)

    numerator = (mean - prior_mean_t).square() + 2.0 * (sigma - prior_sigma).square()
    denominator = (4.0 * sigma * prior_sigma).clamp_min(eps)
    arccosh_arg = 1.0 + numerator / denominator
    per_dim = torch.sqrt(torch.tensor(2.0, device=mean.device, dtype=mean.dtype)) * torch.acosh(
        arccosh_arg.clamp_min(1.0 + eps)
    )
    return torch.sqrt(per_dim.square().sum(dim=-1).clamp_min(eps))


def diagonal_gaussian_fisher_rao_squared(
    mean: Tensor,
    logvar: Tensor,
    prior_mean: float | Tensor = 0.0,
    prior_logvar: float | Tensor = 0.0,
    eps: float = 1e-8,
) -> Tensor:
    distance = diagonal_gaussian_fisher_rao_distance(
        mean,
        logvar,
        prior_mean=prior_mean,
        prior_logvar=prior_logvar,
        eps=eps,
    )
    return distance.square()

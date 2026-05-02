from __future__ import annotations

import torch
from torch import Tensor

from fisher_rao_ml.losses import categorical_fisher_rao_squared

OBJECTIVES = (
    "kl",
    "kl_smoothed",
    "kl_capped",
    "jensen_shannon",
    "hellinger",
    "fisher_rao",
)


def distribution_loss(
    target: Tensor,
    prediction: Tensor,
    objective: str,
    eps: float = 1e-6,
) -> Tensor:
    """Compare batched categorical distributions.

    ``target`` and ``prediction`` are expected to have shape ``(..., classes)`` and to be
    normalized probability vectors. The returned value is averaged over all batch dimensions.
    """
    target = target.clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    prediction = prediction.clamp_min(eps)
    prediction = prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(eps)

    if objective == "kl":
        loss = target * (target.log() - prediction.log())
        return loss.sum(dim=-1).mean()
    if objective == "kl_smoothed":
        smoothing = 1e-3
        uniform = torch.full_like(target, 1.0 / target.shape[-1])
        target_smooth = (1.0 - smoothing) * target + smoothing * uniform
        prediction_smooth = (1.0 - smoothing) * prediction + smoothing * uniform
        loss = target_smooth * (target_smooth.log() - prediction_smooth.log())
        return loss.sum(dim=-1).mean()
    if objective == "kl_capped":
        per_class = target * (target.log() - prediction.log())
        return per_class.clamp_max(0.05).sum(dim=-1).mean()
    if objective == "jensen_shannon":
        midpoint = 0.5 * (target + prediction)
        kl_target = target * (target.log() - midpoint.clamp_min(eps).log())
        kl_prediction = prediction * (prediction.log() - midpoint.clamp_min(eps).log())
        return 0.5 * (kl_target.sum(dim=-1) + kl_prediction.sum(dim=-1)).mean()
    if objective == "hellinger":
        return (
            0.5
            * (torch.sqrt(target) - torch.sqrt(prediction))
            .square()
            .sum(dim=-1)
            .mean()
        )
    if objective == "fisher_rao":
        return categorical_fisher_rao_squared(target, prediction, eps=eps).mean()
    raise ValueError(f"Unknown objective: {objective}")


def distribution_loss_from_logits(
    target: Tensor,
    logits: Tensor,
    objective: str,
    eps: float = 1e-6,
) -> Tensor:
    return distribution_loss(
        target,
        torch.softmax(logits, dim=-1),
        objective=objective,
        eps=eps,
    )

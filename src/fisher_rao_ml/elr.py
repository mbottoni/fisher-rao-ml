"""Early-Learning Regularization (Liu et al., NeurIPS 2020).

loss = CE(logits, labels) + lam * log(1 - <p_i, t_i>)

t_i is a per-sample temporally-ensembled target updated as
t_i <- beta * t_i + (1 - beta) * p_i each time sample i is seen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELRLoss(nn.Module):
    def __init__(
        self,
        n_samples: int,
        n_classes: int,
        beta: float = 0.7,
        lam: float = 3.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.register_buffer("targets", torch.zeros(n_samples, n_classes))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        with torch.no_grad():
            t = self.targets[indices]
            t = self.beta * t + (1.0 - self.beta) * probs.detach()
            self.targets[indices] = t

        ce = F.cross_entropy(logits, labels)
        inner = (probs * t).sum(dim=-1)
        reg = torch.log(torch.clamp(1.0 - inner, min=self.eps)).mean()
        return ce + self.lam * reg

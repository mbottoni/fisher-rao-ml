import torch

from fisher_rao_ml.distribution_losses import OBJECTIVES, distribution_loss_from_logits
from fisher_rao_ml.losses import (
    categorical_fisher_rao_squared,
    diagonal_gaussian_fisher_rao_squared,
)


def test_categorical_fisher_rao_has_gradients() -> None:
    logits = torch.randn(4, requires_grad=True)
    q = torch.softmax(logits, dim=-1)
    p = torch.tensor([0.55, 0.20, 0.15, 0.10])

    loss = categorical_fisher_rao_squared(p, q)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_diagonal_gaussian_fisher_rao_has_gradients() -> None:
    mean = torch.randn(8, 3, requires_grad=True)
    logvar = torch.randn(8, 3, requires_grad=True) * 0.1
    logvar.retain_grad()

    loss = diagonal_gaussian_fisher_rao_squared(mean, logvar).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert mean.grad is not None
    assert logvar.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(logvar.grad).all()


def test_distribution_objectives_have_gradients() -> None:
    target = torch.tensor(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.60, 0.20, 0.10],
        ]
    )

    for objective in OBJECTIVES:
        logits = torch.randn(2, 4, requires_grad=True)
        loss = distribution_loss_from_logits(target, logits, objective)
        loss.backward()

        assert torch.isfinite(loss), objective
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), objective

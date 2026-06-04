import torch
import torch.nn.functional as F

from fisher_rao_ml.elr import ELRLoss


def test_elr_has_finite_gradients() -> None:
    loss_fn = ELRLoss(n_samples=4, n_classes=3, beta=0.7, lam=3.0)
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    indices = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(logits, labels, indices)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_elr_buffer_updates_with_momentum() -> None:
    beta = 0.7
    loss_fn = ELRLoss(n_samples=2, n_classes=3, beta=beta, lam=3.0)
    logits = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    indices = torch.tensor([0, 1])

    probs = F.softmax(logits, dim=-1)
    expected_first = (1.0 - beta) * probs

    loss_fn(logits, labels, indices)
    assert torch.allclose(loss_fn.targets, expected_first, atol=1e-6)

    expected_second = beta * expected_first + (1.0 - beta) * probs
    loss_fn(logits, labels, indices)
    assert torch.allclose(loss_fn.targets, expected_second, atol=1e-6)


def test_elr_lambda_zero_is_cross_entropy() -> None:
    loss_fn = ELRLoss(n_samples=4, n_classes=3, beta=0.7, lam=0.0)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    indices = torch.tensor([0, 1, 2, 3])

    loss = loss_fn(logits, labels, indices)
    ce = F.cross_entropy(logits, labels)

    assert torch.allclose(loss, ce, atol=1e-6)

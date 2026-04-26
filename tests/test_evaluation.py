import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fisher_rao_ml.evaluation import (
    evaluate_embedding,
    evaluate_vae_loader,
    latent_knn_accuracy,
    neighborhood_recall,
)
from fisher_rao_ml.vae import SmallMnistVAE


def test_embedding_metrics_are_bounded() -> None:
    rng = np.random.default_rng(0)
    original = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.2, size=(20, 3)),
            rng.normal(loc=2.0, scale=0.2, size=(20, 3)),
        ]
    )
    embedded = original[:, :2]
    labels = np.array([0] * 20 + [1] * 20)

    metrics = evaluate_embedding(original, embedded, labels, n_neighbors=5, seed=0)

    assert 0.0 <= metrics["eval_trustworthiness"] <= 1.0
    assert 0.0 <= metrics["eval_neighborhood_recall"] <= 1.0
    assert 0.0 <= metrics["eval_knn_accuracy"] <= 1.0
    assert metrics["eval_silhouette"] > 0.0


def test_neighborhood_recall_perfect_for_identical_embeddings() -> None:
    points = np.arange(20, dtype=float).reshape(10, 2)

    assert neighborhood_recall(points, points, n_neighbors=3) == 1.0


def test_latent_knn_accuracy() -> None:
    train_latents = np.array([[0.0], [0.1], [1.0], [1.1]])
    train_labels = np.array([0, 0, 1, 1])
    eval_latents = np.array([[0.05], [1.05]])
    eval_labels = np.array([0, 1])

    assert latent_knn_accuracy(train_latents, train_labels, eval_latents, eval_labels, 1) == 1.0


def test_vae_evaluation_metrics_are_finite() -> None:
    model = SmallMnistVAE(latent_dim=2, hidden_dim=8)
    x = torch.rand(4, 1, 28, 28)
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_vae_loader(
        model,
        loader,
        torch.device("cpu"),
        regularizer="kl",
        beta=1.0,
    )

    assert set(metrics) == {
        "eval_loss",
        "eval_reconstruction",
        "eval_regularization",
        "eval_bce_per_pixel",
        "eval_mean_norm",
        "eval_variance_mean",
        "eval_active_units",
    }
    assert all(np.isfinite(value) for value in metrics.values())

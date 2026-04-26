from __future__ import annotations

import numpy as np
import torch
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fisher_rao_ml.vae import SmallMnistVAE, vae_loss


def neighborhood_recall(
    original: np.ndarray,
    embedded: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Average fraction of original-space neighbors recovered in embedding space."""
    original_neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(original)
    embedded_neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedded)
    original_idx = original_neighbors.kneighbors(return_distance=False)[:, 1:]
    embedded_idx = embedded_neighbors.kneighbors(return_distance=False)[:, 1:]

    recalls = []
    for original_row, embedded_row in zip(original_idx, embedded_idx, strict=True):
        recalls.append(len(set(original_row).intersection(embedded_row)) / n_neighbors)
    return float(np.mean(recalls))


def embedding_classification_accuracy(
    embedded: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 5,
    seed: int = 0,
) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        embedded,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    return float(accuracy_score(y_test, classifier.predict(x_test)))


def evaluate_embedding(
    original: np.ndarray,
    embedded: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Shared t-SNE metrics for comparing KL and Fisher-Rao final embeddings."""
    return {
        "eval_trustworthiness": float(
            trustworthiness(original, embedded, n_neighbors=n_neighbors)
        ),
        "eval_neighborhood_recall": neighborhood_recall(
            original,
            embedded,
            n_neighbors=n_neighbors,
        ),
        "eval_silhouette": float(silhouette_score(embedded, labels)),
        "eval_knn_accuracy": embedding_classification_accuracy(
            embedded,
            labels,
            seed=seed,
        ),
    }


@torch.no_grad()
def collect_latents(
    model: SmallMnistVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means = []
    labels = []
    for x, y in loader:
        mean, _ = model.encode(x.to(device))
        means.append(mean.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(means, axis=0), np.concatenate(labels, axis=0)


def latent_knn_accuracy(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    eval_latents: np.ndarray,
    eval_labels: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_latents, train_labels)
    return float(accuracy_score(eval_labels, classifier.predict(eval_latents)))


@torch.no_grad()
def evaluate_vae_loader(
    model: SmallMnistVAE,
    loader: DataLoader,
    device: torch.device,
    regularizer: str,
    beta: float,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    metric_totals = {
        "eval_loss": 0.0,
        "eval_reconstruction": 0.0,
        "eval_regularization": 0.0,
        "eval_bce_per_pixel": 0.0,
        "eval_mean_norm": 0.0,
        "eval_variance_mean": 0.0,
    }
    latent_means = []

    for x, _ in loader:
        x = x.to(device)
        reconstruction_logits, mean, logvar = model(x)
        _, metrics = vae_loss(
            reconstruction_logits,
            x,
            mean,
            logvar,
            regularizer=regularizer,
            beta=beta,
        )
        batch_size = x.shape[0]
        total_samples += batch_size
        for name in ["loss", "reconstruction", "regularization"]:
            metric_totals[f"eval_{name}"] += float(metrics[name].cpu()) * batch_size

        batch_bce = F.binary_cross_entropy_with_logits(
            reconstruction_logits,
            x,
            reduction="mean",
        )
        metric_totals["eval_bce_per_pixel"] += float(batch_bce.cpu()) * batch_size
        metric_totals["eval_mean_norm"] += float(mean.norm(dim=-1).mean().cpu()) * batch_size
        metric_totals["eval_variance_mean"] += float(logvar.exp().mean().cpu()) * batch_size
        latent_means.append(mean.cpu())

    averaged = {name: value / total_samples for name, value in metric_totals.items()}
    latents = torch.cat(latent_means, dim=0)
    active_units = latents.var(dim=0).gt(0.01).float().sum().item()
    averaged["eval_active_units"] = float(active_units)
    return averaged


def format_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metrics for compact reports without changing MLflow logging precision."""
    return {key: round(value, 6) for key, value in metrics.items()}

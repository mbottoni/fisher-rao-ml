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


def latent_linear_probe_accuracy(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    eval_latents: np.ndarray,
    eval_labels: np.ndarray,
) -> float:
    """Linear classification accuracy in latent-mean space."""
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=500, solver="lbfgs")
    classifier.fit(train_latents, train_labels)
    return float(accuracy_score(eval_labels, classifier.predict(eval_latents)))


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(features) <= len(np.unique(labels)):
        return float("nan")
    try:
        return float(silhouette_score(features, labels))
    except ValueError:
        return float("nan")


def median_pairwise_squared_distance(x: np.ndarray, max_samples: int = 512) -> float:
    if len(x) > max_samples:
        rng = np.random.default_rng(0)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    diffs = x[:, None, :] - x[None, :, :]
    distances = np.sum(diffs * diffs, axis=-1)
    upper = distances[np.triu_indices(len(x), k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return max(median, 1e-6)


def rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
    max_samples: int = 512,
) -> float:
    """Biased RBF-kernel MMD estimate for compact generative-quality diagnostics."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) > max_samples:
        rng = np.random.default_rng(1)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        rng = np.random.default_rng(2)
        y = y[rng.choice(len(y), size=max_samples, replace=False)]
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if gamma is None:
        gamma = 1.0 / median_pairwise_squared_distance(np.vstack([x, y]), max_samples=max_samples)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diffs = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diffs * diffs, axis=-1))

    value = kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
    return float(max(value, 0.0))


def nearest_neighbor_distance(
    x: np.ndarray,
    reference: np.ndarray,
    max_reference: int = 2048,
) -> float:
    if len(x) == 0 or len(reference) == 0:
        return float("nan")
    if len(reference) > max_reference:
        rng = np.random.default_rng(3)
        reference = reference[rng.choice(len(reference), size=max_reference, replace=False)]
    neighbors = NearestNeighbors(n_neighbors=1).fit(reference)
    distances, _ = neighbors.kneighbors(x)
    return float(distances.mean())


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
        "eval_mse": 0.0,
        "eval_mean_norm": 0.0,
        "eval_variance_mean": 0.0,
        "eval_posterior_entropy_proxy": 0.0,
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
        reconstruction = torch.sigmoid(reconstruction_logits)
        batch_mse = F.mse_loss(reconstruction, x, reduction="mean")
        metric_totals["eval_bce_per_pixel"] += float(batch_bce.cpu()) * batch_size
        metric_totals["eval_mse"] += float(batch_mse.cpu()) * batch_size
        metric_totals["eval_mean_norm"] += float(mean.norm(dim=-1).mean().cpu()) * batch_size
        variance = logvar.exp()
        metric_totals["eval_variance_mean"] += float(variance.mean().cpu()) * batch_size
        metric_totals["eval_posterior_entropy_proxy"] += float(logvar.mean().cpu()) * batch_size
        latent_means.append(mean.cpu())

    averaged = {name: value / total_samples for name, value in metric_totals.items()}
    latents = torch.cat(latent_means, dim=0)
    active_units = latents.var(dim=0).gt(0.01).float().sum().item()
    averaged["eval_active_units"] = float(active_units)
    return averaged


@torch.no_grad()
def evaluate_vae_reconstruction_corruption(
    model: SmallMnistVAE,
    loader: DataLoader,
    device: torch.device,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    bce_total = 0.0
    mse_total = 0.0
    for x_clean, _ in loader:
        x_clean = x_clean.to(device)
        x_input = x_clean
        if noise_std > 0:
            x_input = x_input + noise_std * torch.randn_like(x_input)
        if dropout_prob > 0:
            keep = torch.rand_like(x_input).gt(dropout_prob).to(x_input.dtype)
            x_input = x_input * keep
        x_input = x_input.clamp(0.0, 1.0)

        reconstruction_logits, _, _ = model(x_input)
        reconstruction = torch.sigmoid(reconstruction_logits)
        batch_size = x_clean.shape[0]
        total_samples += batch_size
        bce = F.binary_cross_entropy_with_logits(
            reconstruction_logits,
            x_clean,
            reduction="mean",
        )
        bce_total += float(bce.cpu()) * batch_size
        mse_total += float(F.mse_loss(reconstruction, x_clean, reduction="mean").cpu()) * batch_size
    prefix = f"eval_noise_{noise_std:g}_dropout_{dropout_prob:g}"
    return {
        f"{prefix}_bce_per_pixel": bce_total / total_samples,
        f"{prefix}_mse": mse_total / total_samples,
    }


@torch.no_grad()
def collect_vae_arrays(
    model: SmallMnistVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    images = []
    reconstructions = []
    means = []
    logvars = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        reconstruction_logits, mean, logvar = model(x)
        images.append(x.cpu().numpy().reshape(x.shape[0], -1))
        reconstruction = torch.sigmoid(reconstruction_logits)
        reconstructions.append(reconstruction.cpu().numpy().reshape(x.shape[0], -1))
        means.append(mean.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        labels.append(y.numpy())
    return (
        np.concatenate(images, axis=0),
        np.concatenate(reconstructions, axis=0),
        np.concatenate(means, axis=0),
        np.concatenate(logvars, axis=0),
        np.concatenate(labels, axis=0),
    )


@torch.no_grad()
def evaluate_vae_generation(
    model: SmallMnistVAE,
    reference_images: np.ndarray,
    device: torch.device,
    n_samples: int,
    latent_dim: int,
) -> dict[str, float]:
    model.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    generated = torch.sigmoid(model.decode(z)).cpu().numpy().reshape(n_samples, -1)
    sample_variance = float(np.var(generated, axis=0).mean())
    return {
        "eval_sample_pixel_variance": sample_variance,
        "eval_sample_test_mmd": rbf_mmd(generated, reference_images),
        "eval_sample_test_nn_distance": nearest_neighbor_distance(generated, reference_images),
    }


def evaluate_latent_geometry(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    eval_latents: np.ndarray,
    eval_labels: np.ndarray,
) -> dict[str, float]:
    prior = np.random.default_rng(4).normal(size=eval_latents.shape)
    return {
        "eval_latent_knn_accuracy": latent_knn_accuracy(
            train_latents,
            train_labels,
            eval_latents,
            eval_labels,
        ),
        "eval_latent_linear_accuracy": latent_linear_probe_accuracy(
            train_latents,
            train_labels,
            eval_latents,
            eval_labels,
        ),
        "eval_latent_silhouette": safe_silhouette(eval_latents, eval_labels),
        "eval_aggregated_posterior_mmd": rbf_mmd(eval_latents, prior),
    }


def format_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metrics for compact reports without changing MLflow logging precision."""
    return {key: round(value, 6) for key, value in metrics.items()}

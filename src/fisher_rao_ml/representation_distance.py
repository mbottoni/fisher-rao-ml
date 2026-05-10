"""Fisher-Rao Representation Distance (FR-RD).

A proper metric for comparing neural network models via their output
distributions on a reference dataset.

Definition:
    FR-RD(theta, phi; X) = (1/n) sum_x d_FR(P_theta(x), P_phi(x))

where P_theta(x) is the softmax output distribution of model theta on input x.

Key properties (inherited from d_FR on the simplex):
  - Symmetric: FR-RD(theta, phi) = FR-RD(phi, theta)
  - Triangle inequality: FR-RD(theta, psi) <= FR-RD(theta, phi) + FR-RD(phi, psi)
  - Bounded: 0 <= FR-RD <= pi
  - Zero iff models agree on every input in X
"""

from __future__ import annotations

import torch
from torch import Tensor

from fisher_rao_ml.losses import categorical_fisher_rao_distance


def fr_representation_distance(
    probs_a: Tensor,
    probs_b: Tensor,
    eps: float = 1e-8,
) -> float:
    """FR-RD between two models given their output probability matrices.

    Args:
        probs_a: (n, C) softmax outputs of model A on a reference dataset.
        probs_b: (n, C) softmax outputs of model B on the same reference dataset.
        eps: numerical stability floor.

    Returns:
        Scalar mean FR distance across the n reference points.
    """
    distances = categorical_fisher_rao_distance(probs_a, probs_b, eps=eps)
    return float(distances.mean().item())


def pairwise_fr_rd(probs_list: list[Tensor], eps: float = 1e-8) -> Tensor:
    """Compute the full pairwise FR-RD matrix for a collection of models.

    Args:
        probs_list: list of (n, C) tensors, one per model.
        eps: numerical stability floor.

    Returns:
        (M, M) symmetric distance matrix where M = len(probs_list).
    """
    m = len(probs_list)
    matrix = torch.zeros(m, m)
    for i in range(m):
        for j in range(i + 1, m):
            d = fr_representation_distance(probs_list[i], probs_list[j], eps=eps)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def cka_linear(features_a: Tensor, features_b: Tensor) -> float:
    """Linear CKA between two (n, d) feature matrices (Kornblith et al. 2019).

    Used as a baseline metric for comparison with FR-RD.
    """
    a = features_a - features_a.mean(dim=0, keepdim=True)
    b = features_b - features_b.mean(dim=0, keepdim=True)
    hsic_ab = (a @ a.T * (b @ b.T)).sum()
    hsic_aa = (a @ a.T).square().sum().sqrt()
    hsic_bb = (b @ b.T).square().sum().sqrt()
    denom = (hsic_aa * hsic_bb).clamp_min(1e-12)
    return float((hsic_ab / denom).item())


def pairwise_cka(features_list: list[Tensor]) -> Tensor:
    """Full pairwise linear CKA matrix."""
    m = len(features_list)
    matrix = torch.zeros(m, m)
    for i in range(m):
        for j in range(i, m):
            v = cka_linear(features_list[i], features_list[j])
            matrix[i, j] = v
            matrix[j, i] = v
    return matrix


def fr_ood_score(probs_id: Tensor, probs_query: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-sample FR distance from an in-distribution reference set centroid.

    The centroid is the mean probability vector over in-distribution samples.
    Returns a (len(probs_query),) OOD score; higher = more OOD.

    Note: this global centroid approach fails for confident classifiers because the
    centroid of many near-one-hot vectors approximates uniform, making ID predictions
    (far from uniform) score higher than uncertain OOD predictions. Use
    fr_ood_score_class_conditional for reliable OOD detection with confident models.
    """
    centroid = probs_id.mean(dim=0, keepdim=True).expand(probs_query.shape[0], -1)
    return categorical_fisher_rao_distance(centroid, probs_query, eps=eps)


def fr_ood_score_class_conditional(
    probs_id: Tensor,
    labels_id: Tensor,
    probs_query: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Per-sample FR OOD score using class-conditional centroids.

    For each class c, computes the centroid of in-distribution softmax outputs
    where the predicted class is c. Scores a query sample as the minimum FR
    distance to any class centroid.

    This avoids the global-centroid failure mode: confident ID samples are near
    their class centroid → low score; OOD samples (uncertain, diffuse predictions)
    are far from all centroids → high score.

    Args:
        probs_id: (n, C) in-distribution softmax outputs.
        labels_id: (n,) integer ground-truth or predicted class labels for ID samples.
        probs_query: (m, C) query softmax outputs.
        eps: numerical stability floor.

    Returns:
        (m,) OOD scores; higher = more OOD.
    """
    n_classes = probs_id.shape[1]
    centroids = []
    for c in range(n_classes):
        mask = labels_id == c
        if mask.sum() == 0:
            # Fallback: global centroid for unseen class
            centroids.append(probs_id.mean(dim=0))
        else:
            centroids.append(probs_id[mask].mean(dim=0))
    centroids_t = torch.stack(centroids, dim=0)  # (C, C)

    # For each query, compute FR distance to each centroid, take min
    # probs_query: (m, C) → expand to (m, C, C)
    m = probs_query.shape[0]
    # d[i, c] = FR(centroid_c, probs_query[i])
    scores = torch.zeros(m, device=probs_query.device)
    for c in range(n_classes):
        centroid_c = centroids_t[c].unsqueeze(0).expand(m, -1)
        d_c = categorical_fisher_rao_distance(centroid_c, probs_query, eps=eps)
        if c == 0:
            scores = d_c
        else:
            scores = torch.minimum(scores, d_c)
    return scores

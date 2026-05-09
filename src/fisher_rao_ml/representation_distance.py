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
    """
    centroid = probs_id.mean(dim=0, keepdim=True).expand(probs_query.shape[0], -1)
    return categorical_fisher_rao_distance(centroid, probs_query, eps=eps)

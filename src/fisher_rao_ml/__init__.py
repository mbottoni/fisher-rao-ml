"""Fisher-Rao distance experiments for probabilistic ML objectives."""

from fisher_rao_ml.losses import (
    categorical_fisher_rao_distance,
    categorical_fisher_rao_squared,
    diagonal_gaussian_fisher_rao_distance,
    diagonal_gaussian_fisher_rao_squared,
)

__all__ = [
    "categorical_fisher_rao_distance",
    "categorical_fisher_rao_squared",
    "diagonal_gaussian_fisher_rao_distance",
    "diagonal_gaussian_fisher_rao_squared",
]

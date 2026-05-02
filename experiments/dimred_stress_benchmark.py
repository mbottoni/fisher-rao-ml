"""Hypothesis-driven dimensionality-reduction stress tests for KL vs Fisher-Rao.

The main paper benchmark asks whether Fisher-Rao wins on broad t-SNE quality metrics. These
stress tests instead target regimes where KL's unbounded, asymmetric pressure might be a
liability: corrupted high-probability edges, influential outliers, continuous manifolds, and
false-positive bridges between nearby manifolds.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits, make_blobs, make_s_curve, make_swiss_roll
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fisher_rao_ml.device import get_device
from fisher_rao_ml.evaluation import evaluate_embedding, neighborhood_recall
from fisher_rao_ml.tsne import (
    pairwise_student_t_affinities,
    symmetric_gaussian_affinities,
    tsne_distribution_loss,
)

OBJECTIVES = ("kl", "fisher_rao")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KL vs Fisher-Rao dimensionality-reduction stress tests."
    )
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument("--experiments", nargs="+", default=["all"])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument(
        "--stress-datasets",
        nargs="+",
        default=["blobs", "digits"],
        choices=["blobs", "digits"],
        help="Datasets used by the noisy_affinity false-neighbor stress test.",
    )
    parser.add_argument(
        "--false-edge-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2],
        help="Approximate bad-edge mass injected into the affinity distribution.",
    )
    parser.add_argument(
        "--corruption-types",
        nargs="+",
        default=["uniform", "hub", "block", "boundary"],
        choices=["uniform", "hub", "block", "boundary"],
        help="Structured false-affinity mechanisms for the noisy_affinity stress test.",
    )
    parser.add_argument("--save-qualitative-dataset", default="digits", choices=["blobs", "digits"])
    parser.add_argument(
        "--save-qualitative-corruption-type",
        default="hub",
        choices=["uniform", "hub", "block", "boundary"],
    )
    parser.add_argument(
        "--save-qualitative-level",
        type=float,
        default=None,
        help="False-edge level saved for the qualitative bad-edge figure. Defaults to max level.",
    )
    parser.add_argument(
        "--outlier-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.05, 0.1],
    )
    parser.add_argument(
        "--manifold-noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1],
    )
    parser.add_argument(
        "--bridge-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2],
        help="Approximate cross-manifold bridge mass injected into P.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standardize(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x).astype(np.float32)


def median_distance(x: np.ndarray) -> float:
    diffs = x[:, None, :] - x[None, :, :]
    distances = np.sqrt(np.maximum((diffs * diffs).sum(axis=-1), 0.0))
    upper = distances[np.triu_indices(len(x), k=1)]
    return float(np.median(upper))


def gaussian_affinity_numpy(x: np.ndarray, bandwidth: float) -> np.ndarray:
    tensor = torch.tensor(x)
    return symmetric_gaussian_affinities(tensor, bandwidth=bandwidth).numpy()


def train_embedding_from_affinities(
    p_train: np.ndarray,
    objective: str,
    steps: int,
    seed: int,
    device: torch.device,
    log_every: int,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    torch.manual_seed(seed)
    p = torch.tensor(p_train, dtype=torch.float32, device=device)
    embedding = torch.nn.Parameter(torch.randn(p.shape[0], 2, device=device) * 1e-3)
    optimizer = torch.optim.Adam([embedding], lr=5e-2)
    history: list[tuple[int, float]] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        q = pairwise_student_t_affinities(embedding)
        loss = tsne_distribution_loss(p, q, objective=objective)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            history.append((step, float(loss.detach().cpu())))

    return embedding.detach().cpu().numpy(), history


def nearest_neighbor_sets(x: np.ndarray, n_neighbors: int) -> list[set[int]]:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    return [set(row.tolist()) for row in indices]


def local_label_purity(embedded: np.ndarray, labels: np.ndarray, n_neighbors: int) -> float:
    neighbor_sets = nearest_neighbor_sets(embedded, n_neighbors)
    purities = []
    for i, row in enumerate(neighbor_sets):
        if labels[i] < 0:
            continue
        neighbor_labels = labels[list(row)]
        neighbor_labels = neighbor_labels[neighbor_labels >= 0]
        if neighbor_labels.size:
            purities.append(float(np.mean(neighbor_labels == labels[i])))
    return float(np.mean(purities)) if purities else float("nan")


def corrupted_edge_preservation(
    embedded: np.ndarray,
    corrupted_pairs: set[tuple[int, int]],
    n_neighbors: int,
) -> float:
    if not corrupted_pairs:
        return 0.0
    neighbor_sets = nearest_neighbor_sets(embedded, n_neighbors)
    preserved = 0
    for i, j in corrupted_pairs:
        if j in neighbor_sets[i] or i in neighbor_sets[j]:
            preserved += 1
    return preserved / len(corrupted_pairs)


def corrupt_edge_q_mass(embedded: np.ndarray, corrupted_pairs: set[tuple[int, int]]) -> float:
    if not corrupted_pairs:
        return 0.0
    q = pairwise_student_t_affinities(torch.tensor(embedded, dtype=torch.float32)).numpy()
    return float(sum(q[i, j] + q[j, i] for i, j in corrupted_pairs))


def inject_cross_label_edges(
    p_clean: np.ndarray,
    labels: np.ndarray,
    level: float,
    seed: int,
    edges_per_point: int = 4,
    corruption_type: str = "uniform",
    x_clean: np.ndarray | None = None,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    if level <= 0:
        return p_clean.copy(), set()
    n = len(labels)
    target_edges = max(1, int(edges_per_point * n * level))
    candidates = select_corrupted_pairs(
        x_clean=x_clean,
        labels=labels,
        target_edges=target_edges,
        seed=seed,
        corruption_type=corruption_type,
    )

    p = p_clean.copy()
    mass_per_direction = level / max(2 * len(candidates), 1)
    for i, j in candidates:
        p[i, j] += mass_per_direction
        p[j, i] += mass_per_direction
    p /= p.sum()
    return p.astype(np.float32), set(candidates)


def select_corrupted_pairs(
    x_clean: np.ndarray | None,
    labels: np.ndarray,
    target_edges: int,
    seed: int,
    corruption_type: str,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    pairs: list[tuple[int, int]] = []

    def add_pair(i: int, j: int) -> None:
        if labels[i] == labels[j]:
            return
        pair = (int(min(i, j)), int(max(i, j)))
        if pair not in pairs:
            pairs.append(pair)

    if corruption_type == "uniform":
        attempts = 0
        while len(pairs) < target_edges and attempts < target_edges * 50:
            i, j = rng.choice(n, size=2, replace=False)
            attempts += 1
            add_pair(int(i), int(j))
        return pairs

    if corruption_type == "hub":
        hub = int(rng.integers(0, n))
        opposite = np.flatnonzero(labels != labels[hub])
        rng.shuffle(opposite)
        for j in opposite:
            add_pair(hub, int(j))
            if len(pairs) >= target_edges:
                break
        return pairs

    if corruption_type == "block":
        classes = np.unique(labels)
        class_a, class_b = rng.choice(classes, size=2, replace=False)
        a_idx = np.flatnonzero(labels == class_a)
        b_idx = np.flatnonzero(labels == class_b)
        while len(pairs) < target_edges:
            add_pair(int(rng.choice(a_idx)), int(rng.choice(b_idx)))
            if len(pairs) >= len(a_idx) * len(b_idx):
                break
        return pairs

    if corruption_type == "boundary":
        if x_clean is None:
            raise ValueError("boundary corruption requires x_clean")
        candidate_scores: list[tuple[float, int, int]] = []
        for i in range(n):
            opposite = np.flatnonzero(labels != labels[i])
            distances = np.linalg.norm(x_clean[opposite] - x_clean[i], axis=1)
            nearest = opposite[np.argsort(distances)[:5]]
            for j in nearest:
                pair = (int(min(i, j)), int(max(i, j)))
                candidate_scores.append((float(np.linalg.norm(x_clean[i] - x_clean[j])), *pair))
        for _distance, i, j in sorted(set(candidate_scores)):
            add_pair(i, j)
            if len(pairs) >= target_edges:
                break
        return pairs

    raise ValueError(f"Unknown corruption_type: {corruption_type}")


def make_cluster_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x, labels = make_blobs(
        n_samples=n_samples,
        n_features=8,
        centers=5,
        cluster_std=1.4,
        random_state=seed,
    )
    return standardize(x), labels.astype(np.int64)


def make_noisy_affinity_dataset(
    name: str,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "blobs":
        return make_cluster_dataset(n_samples, seed)
    if name == "digits":
        bundle = load_digits()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(bundle.data), size=n_samples, replace=False)
        x = bundle.data[idx]
        labels = bundle.target[idx]
        return standardize(x), labels.astype(np.int64)
    raise ValueError(f"Unknown noisy-affinity dataset: {name}")


def embedding_quality_metrics(
    x_clean: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
    seed: int,
) -> dict[str, float]:
    metrics = evaluate_embedding(x_clean, embedding, labels, n_neighbors=n_neighbors, seed=seed)
    metrics["eval_local_label_purity"] = local_label_purity(embedding, labels, n_neighbors)
    return metrics


def run_noisy_affinity_test(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    qualitative_level = (
        max(args.false_edge_levels)
        if args.save_qualitative_level is None
        else args.save_qualitative_level
    )

    print("\n[stress:noisy-affinity] Injecting cross-label high-affinity edges")
    for dataset in args.stress_datasets:
        x_clean, labels = make_noisy_affinity_dataset(dataset, args.samples, seed=13)
        bandwidth = max(median_distance(x_clean) / math.sqrt(2.0), 1e-3)
        p_clean = gaussian_affinity_numpy(x_clean, bandwidth)
        print(f"[stress:noisy-affinity] dataset={dataset} shape={x_clean.shape}")
        for corruption_type in args.corruption_types:
            for level in args.false_edge_levels:
                for seed in args.seeds:
                    p_train, corrupted_pairs = inject_cross_label_edges(
                        p_clean,
                        labels,
                        level=level,
                        seed=9000 + seed,
                        corruption_type=corruption_type,
                        x_clean=x_clean,
                    )
                    should_save_edges = (
                        dataset == args.save_qualitative_dataset
                        and corruption_type == args.save_qualitative_corruption_type
                        and seed == args.seeds[0]
                        and abs(level - qualitative_level) < 1e-12
                    )
                    if should_save_edges:
                        for edge_idx, (i, j) in enumerate(sorted(corrupted_pairs)):
                            edge_rows.append(
                                {
                                    "dataset": dataset,
                                    "corruption_type": corruption_type,
                                    "stress_level": level,
                                    "seed": seed,
                                    "edge_index": edge_idx,
                                    "source": i,
                                    "target": j,
                                }
                            )
                    for objective in OBJECTIVES:
                        embedding, history = train_embedding_from_affinities(
                            p_train,
                            objective,
                            args.steps,
                            seed,
                            device,
                            args.log_every,
                        )
                        metrics = embedding_quality_metrics(
                            x_clean,
                            embedding,
                            labels,
                            args.neighbors,
                            seed,
                        )
                        metrics["eval_corrupted_edge_preservation"] = corrupted_edge_preservation(
                            embedding,
                            corrupted_pairs,
                            args.neighbors,
                        )
                        metrics["eval_corrupted_edge_q_mass"] = corrupt_edge_q_mass(
                            embedding,
                            corrupted_pairs,
                        )
                        rows.append(
                            {
                                "experiment": "noisy_affinity",
                                "dataset": dataset,
                                "corruption_type": corruption_type,
                                "stress_level": level,
                                "seed": seed,
                                "objective": objective,
                                "final_loss": history[-1][1],
                                **metrics,
                            }
                        )
                        should_save_embedding = (
                            dataset == args.save_qualitative_dataset
                            and corruption_type == args.save_qualitative_corruption_type
                            and seed == args.seeds[0]
                            and (level == 0.0 or abs(level - qualitative_level) < 1e-12)
                        )
                        if should_save_embedding:
                            target_state = "clean" if level == 0.0 else "corrupted"
                            for i in range(embedding.shape[0]):
                                embedding_rows.append(
                                    {
                                        "dataset": dataset,
                                        "corruption_type": corruption_type,
                                        "stress_level": level,
                                        "target_state": target_state,
                                        "seed": seed,
                                        "objective": objective,
                                        "index": i,
                                        "x": float(embedding[i, 0]),
                                        "y": float(embedding[i, 1]),
                                        "label": int(labels[i]),
                                    }
                                )
                        print(
                            f"[stress:noisy-affinity] dataset={dataset} type={corruption_type} "
                            f"level={level:.3f} seed={seed} objective={objective} "
                            f"trust={metrics['eval_trustworthiness']:.4f} "
                            f"bad_edge={metrics['eval_corrupted_edge_preservation']:.4f}"
                        )
    return rows, embedding_rows, edge_rows


def aligned_mean_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference - reference.mean(axis=0, keepdims=True)
    cand = candidate - candidate.mean(axis=0, keepdims=True)
    ref_scale = np.linalg.norm(ref)
    cand_scale = np.linalg.norm(cand)
    if ref_scale <= 1e-12 or cand_scale <= 1e-12:
        return float("nan")
    ref = ref / ref_scale
    cand = cand / cand_scale
    u, _, vt = np.linalg.svd(cand.T @ ref, full_matrices=False)
    aligned = cand @ (u @ vt)
    return float(np.linalg.norm(ref - aligned, axis=1).mean())


def add_bridge_outliers(
    x: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction <= 0:
        return x.copy(), labels.copy()
    rng = np.random.default_rng(seed)
    n_outliers = max(1, int(round(len(x) * fraction)))
    class_ids = np.unique(labels)
    centroids = np.stack([x[labels == class_id].mean(axis=0) for class_id in class_ids])
    outliers = []
    for _ in range(n_outliers):
        a, b = rng.choice(len(centroids), size=2, replace=False)
        mix = rng.uniform(0.35, 0.65)
        point = mix * centroids[a] + (1.0 - mix) * centroids[b]
        point += rng.normal(scale=0.15, size=x.shape[1])
        outliers.append(point)
    x_augmented = np.vstack([x, np.asarray(outliers, dtype=np.float32)])
    labels_augmented = np.concatenate([labels, np.full(n_outliers, -1, dtype=np.int64)])
    return x_augmented.astype(np.float32), labels_augmented


def run_outlier_influence_test(
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_clean, labels = make_cluster_dataset(args.samples, seed=29)
    bandwidth = max(median_distance(x_clean) / math.sqrt(2.0), 1e-3)
    p_clean = gaussian_affinity_numpy(x_clean, bandwidth)

    print("\n[stress:outlier] Measuring normal-point drift after bridge outliers")
    baseline_embeddings: dict[tuple[str, int], np.ndarray] = {}
    for seed in args.seeds:
        for objective in OBJECTIVES:
            baseline, _ = train_embedding_from_affinities(
                p_clean,
                objective,
                args.steps,
                seed,
                device,
                args.log_every,
            )
            baseline_embeddings[(objective, seed)] = baseline

    for fraction in args.outlier_fractions:
        for seed in args.seeds:
            x_augmented, labels_augmented = add_bridge_outliers(
                x_clean,
                labels,
                fraction=fraction,
                seed=12000 + seed,
            )
            p_augmented = gaussian_affinity_numpy(x_augmented, bandwidth)
            for objective in OBJECTIVES:
                embedding, history = train_embedding_from_affinities(
                    p_augmented,
                    objective,
                    args.steps,
                    seed,
                    device,
                    args.log_every,
                )
                normal_embedding = embedding[: len(x_clean)]
                metrics = embedding_quality_metrics(
                    x_clean,
                    normal_embedding,
                    labels,
                    args.neighbors,
                    seed,
                )
                metrics["eval_outlier_influence"] = aligned_mean_distance(
                    baseline_embeddings[(objective, seed)],
                    normal_embedding,
                )
                rows.append(
                    {
                        "experiment": "outlier_influence",
                        "dataset": "blobs_bridge_outliers",
                        "stress_level": fraction,
                        "seed": seed,
                        "objective": objective,
                        "n_outliers": len(x_augmented) - len(x_clean),
                        "final_loss": history[-1][1],
                        **metrics,
                    }
                )
                print(
                    f"[stress:outlier] fraction={fraction:.3f} seed={seed} "
                    f"objective={objective} influence={metrics['eval_outlier_influence']:.5f}"
                )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ar = rankdata(a)
    br = rankdata(b)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.linalg.norm(ar) * np.linalg.norm(br)
    return float((ar @ br) / denom) if denom > 1e-12 else float("nan")


def sampled_pairwise_metrics(
    embedded: np.ndarray,
    latent: np.ndarray,
    seed: int,
    max_pairs: int = 8000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(embedded)
    total_pairs = n * (n - 1) // 2
    pair_count = min(max_pairs, total_pairs)
    first = rng.integers(0, n, size=pair_count)
    second = rng.integers(0, n, size=pair_count)
    keep = first != second
    first = first[keep]
    second = second[keep]
    geo = np.linalg.norm(latent[first] - latent[second], axis=1)
    emb = np.linalg.norm(embedded[first] - embedded[second], axis=1)
    geo_z = (geo - geo.mean()) / max(float(geo.std()), 1e-12)
    emb_z = (emb - emb.mean()) / max(float(emb.std()), 1e-12)
    return {
        "eval_geodesic_spearman": spearman_corr(geo, emb),
        "eval_pairwise_distortion": float(np.mean(np.abs(geo_z - emb_z))),
    }


def make_manifold_dataset(
    name: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "swiss_roll":
        x, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
        latent = np.column_stack([t, x[:, 1]])
    elif name == "s_curve":
        x, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
        latent = np.column_stack([t, x[:, 1]])
    else:
        raise ValueError(f"Unknown manifold dataset: {name}")
    labels = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8])).astype(np.int64)
    return standardize(x), latent.astype(np.float32), labels


def run_global_geometry_test(
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print("\n[stress:global] Measuring continuous-manifold geometry preservation")
    for dataset in ("swiss_roll", "s_curve"):
        for noise in args.manifold_noise_levels:
            x, latent, labels = make_manifold_dataset(dataset, args.samples, noise, seed=43)
            bandwidth = max(median_distance(x) / math.sqrt(2.0), 1e-3)
            p_train = gaussian_affinity_numpy(x, bandwidth)
            for seed in args.seeds:
                for objective in OBJECTIVES:
                    embedding, history = train_embedding_from_affinities(
                        p_train,
                        objective,
                        args.steps,
                        seed,
                        device,
                        args.log_every,
                    )
                    metrics = {
                        "eval_trustworthiness": float(
                            trustworthiness(x, embedding, n_neighbors=args.neighbors)
                        ),
                        "eval_continuity": neighborhood_recall(
                            embedding,
                            x,
                            n_neighbors=args.neighbors,
                        ),
                        "eval_neighborhood_recall_k5": neighborhood_recall(x, embedding, 5),
                        "eval_neighborhood_recall_k10": neighborhood_recall(x, embedding, 10),
                        "eval_neighborhood_recall_k20": neighborhood_recall(x, embedding, 20),
                        **sampled_pairwise_metrics(embedding, latent, seed=seed),
                    }
                    rows.append(
                        {
                            "experiment": "global_geometry",
                            "dataset": dataset,
                            "stress_level": noise,
                            "seed": seed,
                            "objective": objective,
                            "final_loss": history[-1][1],
                            **metrics,
                        }
                    )
                    print(
                        f"[stress:global] dataset={dataset} noise={noise:.3f} seed={seed} "
                        f"objective={objective} spear={metrics['eval_geodesic_spearman']:.4f}"
                    )
    return rows


def make_parallel_curves(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    t = np.linspace(-2.5, 2.5, half)
    t += rng.normal(scale=0.03, size=half)
    curve_a = np.column_stack([t, np.sin(2.0 * t), np.full(half, -0.18)])
    curve_b = np.column_stack([t, np.sin(2.0 * t) + 0.25, np.full(half, 0.18)])
    x = np.vstack([curve_a, curve_b])
    x += rng.normal(scale=0.03, size=x.shape)
    labels = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64)])
    latent = np.concatenate([t, t])[:, None].astype(np.float32)
    return standardize(x), latent, labels


def inject_parallel_bridges(
    p_clean: np.ndarray,
    latent: np.ndarray,
    labels: np.ndarray,
    level: float,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    if level <= 0:
        return p_clean.copy(), set()
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(labels)):
        opposite = np.flatnonzero(labels != labels[i])
        distances = np.abs(latent[opposite, 0] - latent[i, 0])
        for j in opposite[np.argsort(distances)[:3]]:
            pairs.append((min(i, int(j)), max(i, int(j)), float(distances[np.argmin(distances)])))
    unique_pairs = sorted({(i, j) for i, j, _ in pairs})
    p = p_clean.copy()
    mass_per_direction = level / max(2 * len(unique_pairs), 1)
    for i, j in unique_pairs:
        p[i, j] += mass_per_direction
        p[j, i] += mass_per_direction
    p /= p.sum()
    return p.astype(np.float32), set(unique_pairs)


def between_manifold_leakage(embedded: np.ndarray, labels: np.ndarray, n_neighbors: int) -> float:
    neighbor_sets = nearest_neighbor_sets(embedded, n_neighbors)
    leakage = []
    for i, row in enumerate(neighbor_sets):
        leakage.append(float(np.mean(labels[list(row)] != labels[i])))
    return float(np.mean(leakage))


def false_neighbor_rate(
    x_clean: np.ndarray,
    embedded: np.ndarray,
    n_neighbors: int,
) -> float:
    clean_sets = nearest_neighbor_sets(x_clean, n_neighbors)
    embedded_sets = nearest_neighbor_sets(embedded, n_neighbors)
    rates = []
    for clean_row, embedded_row in zip(clean_sets, embedded_sets, strict=True):
        rates.append(1.0 - len(clean_row.intersection(embedded_row)) / n_neighbors)
    return float(np.mean(rates))


def boundary_label_purity(
    x_clean: np.ndarray,
    embedded: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
) -> float:
    opposite_distances = []
    for i in range(len(labels)):
        opposite = x_clean[labels != labels[i]]
        opposite_distances.append(float(np.linalg.norm(opposite - x_clean[i], axis=1).min()))
    threshold = float(np.quantile(opposite_distances, 0.25))
    boundary_mask = np.asarray(opposite_distances) <= threshold
    neighbor_sets = nearest_neighbor_sets(embedded, n_neighbors)
    purities = []
    for i in np.flatnonzero(boundary_mask):
        purities.append(float(np.mean(labels[list(neighbor_sets[i])] == labels[i])))
    return float(np.mean(purities)) if purities else float("nan")


def run_symmetric_mismatch_test(
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_clean, latent, labels = make_parallel_curves(args.samples, seed=61)
    bandwidth = max(median_distance(x_clean) / math.sqrt(2.0), 1e-3)
    p_clean = gaussian_affinity_numpy(x_clean, bandwidth)

    print("\n[stress:symmetric] Injecting cross-manifold false-positive bridges")
    for level in args.bridge_levels:
        p_train, bridge_pairs = inject_parallel_bridges(p_clean, latent, labels, level=level)
        for seed in args.seeds:
            for objective in OBJECTIVES:
                embedding, history = train_embedding_from_affinities(
                    p_train,
                    objective,
                    args.steps,
                    seed,
                    device,
                    args.log_every,
                )
                metrics = {
                    "eval_trustworthiness": float(
                        trustworthiness(x_clean, embedding, n_neighbors=args.neighbors)
                    ),
                    "eval_neighborhood_recall": neighborhood_recall(
                        x_clean,
                        embedding,
                        n_neighbors=args.neighbors,
                    ),
                    "eval_between_manifold_leakage": between_manifold_leakage(
                        embedding,
                        labels,
                        args.neighbors,
                    ),
                    "eval_false_neighbor_rate": false_neighbor_rate(
                        x_clean,
                        embedding,
                        args.neighbors,
                    ),
                    "eval_boundary_label_purity": boundary_label_purity(
                        x_clean,
                        embedding,
                        labels,
                        args.neighbors,
                    ),
                    "eval_bridge_edge_preservation": corrupted_edge_preservation(
                        embedding,
                        bridge_pairs,
                        args.neighbors,
                    ),
                }
                rows.append(
                    {
                        "experiment": "symmetric_mismatch",
                        "dataset": "parallel_curves",
                        "stress_level": level,
                        "seed": seed,
                        "objective": objective,
                        "final_loss": history[-1][1],
                        **metrics,
                    }
                )
                print(
                    f"[stress:symmetric] level={level:.3f} seed={seed} "
                    f"objective={objective} leakage={metrics['eval_between_manifold_leakage']:.4f}"
                )
    return rows


def selected_experiments(names: Iterable[str]) -> set[str]:
    selected = set(names)
    all_names = {"noisy_affinity", "outlier_influence", "global_geometry", "symmetric_mismatch"}
    if "all" in selected:
        return all_names
    unknown = selected - all_names
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    return selected


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    output_dir = Path(args.output_dir)
    device = get_device()
    selected = selected_experiments(args.experiments)
    print(f"[stress] device={device} output_dir={output_dir} experiments={sorted(selected)}")

    rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    if "noisy_affinity" in selected:
        noisy_rows, noisy_embedding_rows, noisy_edge_rows = run_noisy_affinity_test(args, device)
        rows.extend(noisy_rows)
        embedding_rows.extend(noisy_embedding_rows)
        edge_rows.extend(noisy_edge_rows)
    if "outlier_influence" in selected:
        rows.extend(run_outlier_influence_test(args, device))
    if "global_geometry" in selected:
        rows.extend(run_global_geometry_test(args, device))
    if "symmetric_mismatch" in selected:
        rows.extend(run_symmetric_mismatch_test(args, device))

    write_rows(output_dir / "dimred_stress_full.csv", rows)
    write_rows(output_dir / "dimred_stress_embeddings.csv", embedding_rows)
    write_rows(output_dir / "dimred_stress_edges.csv", edge_rows)
    print(f"[stress] Wrote {len(rows)} rows to {output_dir / 'dimred_stress_full.csv'}")
    if embedding_rows:
        print(
            f"[stress] Wrote {len(embedding_rows)} rows to "
            f"{output_dir / 'dimred_stress_embeddings.csv'}"
        )
    if edge_rows:
        print(f"[stress] Wrote {len(edge_rows)} rows to {output_dir / 'dimred_stress_edges.csv'}")


if __name__ == "__main__":
    main()

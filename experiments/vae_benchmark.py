"""Multi-seed VAE benchmark for KL vs Fisher-Rao regularization.

The script is intentionally separate from the t-SNE paper benchmark because VAE
experiments have a different grid: datasets, beta sweeps, train/eval splits,
and reconstruction-corruption probes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fisher_rao_ml.device import get_device
from fisher_rao_ml.evaluation import (
    collect_vae_arrays,
    evaluate_latent_geometry,
    evaluate_vae_generation,
    evaluate_vae_loader,
    evaluate_vae_reconstruction_corruption,
    format_metrics,
)
from fisher_rao_ml.vae import SmallMnistVAE, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-seed VAE KL vs Fisher-Rao study.")
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mnist", "fashion_mnist", "kmnist"],
        choices=["mnist", "fashion_mnist", "kmnist"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
    )
    parser.add_argument("--kl-betas", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument(
        "--fr-betas",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0, 3.0],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--dropout-levels", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument(
        "--save-latents-for",
        default="mnist",
        choices=["mnist", "fashion_mnist", "kmnist"],
        help="Dataset for representative latent scatter CSV output.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_factory(name: str) -> type[Dataset]:
    if name == "mnist":
        return datasets.MNIST
    if name == "fashion_mnist":
        return datasets.FashionMNIST
    if name == "kmnist":
        return datasets.KMNIST
    raise ValueError(f"Unknown dataset: {name}")


def seeded_subset(dataset: Dataset, n_samples: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    size = min(n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=size, replace=False)
    return Subset(dataset, indices.tolist())


def make_loaders(
    dataset_name: str,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cls = dataset_factory(dataset_name)
    transform = transforms.ToTensor()
    train_full = dataset_cls(root="data", train=True, download=True, transform=transform)
    eval_full = dataset_cls(root="data", train=False, download=True, transform=transform)
    train_subset = seeded_subset(train_full, train_samples, seed)
    eval_subset = seeded_subset(eval_full, eval_samples, seed + 17)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, train_eval_loader, eval_loader


def iter_configs(args: argparse.Namespace) -> list[tuple[str, float]]:
    configs = [("kl", beta) for beta in args.kl_betas]
    configs.extend(("fisher_rao", beta) for beta in args.fr_betas)
    return configs


def train_one(
    dataset_name: str,
    regularizer: str,
    beta: float,
    seed: int,
    train_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[SmallMnistVAE, list[dict[str, float | int | str]], dict[str, float | int]]:
    torch.manual_seed(seed)
    model = SmallMnistVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dynamics: list[dict[str, float | int | str]] = []
    global_step = 0
    stable = True
    max_grad_norm = 0.0
    completed_epochs = 0

    description = f"{dataset_name} {regularizer} beta={beta:g} seed={seed}"
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"[vae] {description} epoch {epoch + 1}", leave=False)
        for x, _ in progress:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction_logits, mean, logvar = model(x)
            loss, metrics = vae_loss(
                reconstruction_logits,
                x,
                mean,
                logvar,
                regularizer=regularizer,
                beta=beta,
            )
            if not torch.isfinite(loss):
                stable = False
                print(f"[vae] non-finite loss dataset={dataset_name} reg={regularizer} beta={beta}")
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            max_grad_norm = max(max_grad_norm, float(grad_norm.detach().cpu()))
            optimizer.step()

            if global_step % args.log_every == 0:
                compact = format_metrics(
                    {name: float(value.cpu()) for name, value in metrics.items()}
                )
                progress.set_postfix(compact)
                dynamics.append(
                    {
                        "task": "vae",
                        "dataset": dataset_name,
                        "regularizer": regularizer,
                        "beta": beta,
                        "seed": seed,
                        "epoch": epoch,
                        "step": global_step,
                        "grad_norm": max_grad_norm,
                        **compact,
                    }
                )
            global_step += 1
        if not stable:
            break
        completed_epochs = epoch + 1

    stability = {
        "eval_stable": 1.0 if stable else 0.0,
        "eval_completed_epochs": completed_epochs,
        "eval_max_grad_norm": max_grad_norm,
    }
    return model, dynamics, stability


def save_latent_rows(
    rows: list[dict[str, float | int | str]],
    dataset_name: str,
    regularizer: str,
    beta: float,
    seed: int,
    eval_latents: np.ndarray,
    eval_labels: np.ndarray,
    max_points: int = 500,
) -> None:
    from sklearn.decomposition import PCA

    if len(eval_latents) == 0:
        return
    projected = PCA(n_components=2, random_state=0).fit_transform(eval_latents)
    for index in range(min(max_points, len(projected))):
        rows.append(
            {
                "dataset": dataset_name,
                "regularizer": regularizer,
                "beta": beta,
                "seed": seed,
                "index": index,
                "x": float(projected[index, 0]),
                "y": float(projected[index, 1]),
                "label": int(eval_labels[index]),
            }
        )


def run_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    metric_rows: list[dict[str, float | int | str]] = []
    dynamics_rows: list[dict[str, float | int | str]] = []
    latent_rows: list[dict[str, float | int | str]] = []
    configs = iter_configs(args)

    print("\n[vae] Multi-seed, multi-dataset VAE benchmark")
    print(f"[vae] datasets={args.datasets} seeds={args.seeds}")
    print(f"[vae] configs={configs}")

    for dataset_name in args.datasets:
        for seed in args.seeds:
            train_loader, train_eval_loader, eval_loader = make_loaders(
                dataset_name,
                args.train_samples,
                args.eval_samples,
                args.batch_size,
                seed,
            )
            for regularizer, beta in configs:
                model, dynamics, stability = train_one(
                    dataset_name,
                    regularizer,
                    beta,
                    seed,
                    train_loader,
                    device,
                    args,
                )
                dynamics_rows.extend(dynamics)
                metrics = evaluate_vae_loader(model, eval_loader, device, regularizer, beta)
                train_images, _, train_latents, _, train_labels = collect_vae_arrays(
                    model,
                    train_eval_loader,
                    device,
                )
                (
                    eval_images,
                    eval_reconstructions,
                    eval_latents,
                    _,
                    eval_labels,
                ) = collect_vae_arrays(model, eval_loader, device)
                metrics.update(
                    evaluate_latent_geometry(train_latents, train_labels, eval_latents, eval_labels)
                )
                metrics.update(
                    evaluate_vae_generation(
                        model,
                        eval_images,
                        device,
                        n_samples=min(args.sample_count, len(eval_images)),
                        latent_dim=args.latent_dim,
                    )
                )
                metrics["eval_reconstruction_train_nn_distance"] = math.nan
                if len(train_images) > 0 and len(eval_images) > 0:
                    from fisher_rao_ml.evaluation import nearest_neighbor_distance

                    metrics["eval_reconstruction_train_nn_distance"] = nearest_neighbor_distance(
                        eval_reconstructions,
                        train_images,
                    )
                for noise_std in args.noise_levels:
                    metrics.update(
                        evaluate_vae_reconstruction_corruption(
                            model,
                            eval_loader,
                            device,
                            noise_std=noise_std,
                        )
                    )
                for dropout_prob in args.dropout_levels:
                    metrics.update(
                        evaluate_vae_reconstruction_corruption(
                            model,
                            eval_loader,
                            device,
                            dropout_prob=dropout_prob,
                        )
                    )
                metrics.update(stability)
                metric_rows.append(
                    {
                        "task": "vae",
                        "dataset": dataset_name,
                        "regularizer": regularizer,
                        "beta": beta,
                        "seed": seed,
                        **metrics,
                    }
                )
                if (
                    dataset_name == args.save_latents_for
                    and seed == args.seeds[0]
                    and beta in {1.0, 0.3}
                ):
                    save_latent_rows(
                        latent_rows,
                        dataset_name,
                        regularizer,
                        beta,
                        seed,
                        eval_latents,
                        eval_labels,
                    )
                compact = format_metrics(
                    {
                        "bce": metrics["eval_bce_per_pixel"],
                        "knn": metrics["eval_latent_knn_accuracy"],
                        "mmd": metrics["eval_aggregated_posterior_mmd"],
                        "active": metrics["eval_active_units"],
                    }
                )
                print(
                    f"[vae] dataset={dataset_name} seed={seed} reg={regularizer} "
                    f"beta={beta:g} {compact}"
                )

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "vae_full_metrics.csv", metric_rows)
    write_rows(output_dir / "vae_training_dynamics.csv", dynamics_rows)
    write_rows(output_dir / "vae_latent_embeddings.csv", latent_rows)


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    device = get_device()
    print(f"[vae] device={device} output_dir={args.output_dir}")
    run_benchmark(args, device)
    print("[vae] Benchmark complete")


if __name__ == "__main__":
    main()

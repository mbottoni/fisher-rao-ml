from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fisher_rao_ml.device import get_device
from fisher_rao_ml.evaluation import (
    collect_latents,
    evaluate_embedding,
    evaluate_vae_loader,
    format_metrics,
    latent_knn_accuracy,
)
from fisher_rao_ml.tsne import (
    pairwise_student_t_affinities,
    symmetric_gaussian_affinities,
    tsne_distribution_loss,
)
from fisher_rao_ml.vae import SmallMnistVAE, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact paper benchmarks.")
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--tsne-steps", type=int, default=150)
    parser.add_argument("--tsne-samples", type=int, default=160)
    parser.add_argument("--vae-epochs", type=int, default=1)
    parser.add_argument("--vae-train-samples", type=int, default=2048)
    parser.add_argument("--vae-eval-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def has_nonfinite_metrics(metrics: dict[str, float]) -> bool:
    return any(not math.isfinite(value) for value in metrics.values())


def train_tsne_embedding(
    x_for_affinities: torch.Tensor,
    objective: str,
    steps: int,
    seed: int,
    log_every: int,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    torch.manual_seed(seed)
    embedding = torch.nn.Parameter(
        torch.randn(x_for_affinities.shape[0], 2, device=x_for_affinities.device) * 1e-3
    )
    optimizer = torch.optim.Adam([embedding], lr=5e-2)
    p = symmetric_gaussian_affinities(x_for_affinities, bandwidth=5.0)
    history = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        q = pairwise_student_t_affinities(embedding)
        loss = tsne_distribution_loss(p, q, objective=objective)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            loss_value = float(loss.detach().cpu())
            history.append((step, loss_value))
            print(f"[paper:t-SNE] objective={objective} step={step:04d} loss={loss_value:.6f}")

    return embedding.detach().cpu().numpy(), history


def run_tsne_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    rng = np.random.default_rng(args.seed)
    x_clean, labels = make_blobs(
        n_samples=args.tsne_samples,
        n_features=8,
        centers=5,
        cluster_std=1.6,
        random_state=args.seed,
    )
    x_scale = float(np.std(x_clean))
    noise_levels = [0.0, 0.5, 1.0]
    objectives = ["kl", "fisher_rao"]
    metric_rows: list[dict[str, float | int | str]] = []
    dynamics_rows: list[dict[str, float | int | str]] = []

    print("\n[paper:t-SNE] Benchmarking representation quality and feature-noise robustness")
    for noise in noise_levels:
        perturbation = rng.normal(size=x_clean.shape) * x_scale * noise
        x_noisy = x_clean + perturbation
        x_tensor = torch.tensor(x_noisy, dtype=torch.float32, device=device)
        for objective in objectives:
            embedding, history = train_tsne_embedding(
                x_tensor,
                objective=objective,
                steps=args.tsne_steps,
                seed=args.seed,
                log_every=args.log_every,
            )
            metrics = evaluate_embedding(x_clean, embedding, labels, n_neighbors=10, seed=args.seed)
            metric_rows.append(
                {
                    "task": "tsne",
                    "objective": objective,
                    "noise_std_fraction": noise,
                    **metrics,
                }
            )
            for step, loss_value in history:
                dynamics_rows.append(
                    {
                        "task": "tsne",
                        "objective": objective,
                        "noise_std_fraction": noise,
                        "step": step,
                        "loss": loss_value,
                    }
                )
            print(f"[paper:t-SNE] objective={objective} noise={noise} {format_metrics(metrics)}")

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "tsne_robustness_metrics.csv", metric_rows)
    write_rows(output_dir / "tsne_training_dynamics.csv", dynamics_rows)


@torch.no_grad()
def evaluate_noisy_vae_reconstruction(
    model: SmallMnistVAE,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    bce_total = 0.0
    mse_total = 0.0
    for x_clean, _ in loader:
        x_clean = x_clean.to(device)
        x_input = (x_clean + noise_std * torch.randn_like(x_clean)).clamp(0.0, 1.0)
        reconstruction_logits, _, _ = model(x_input)
        reconstruction = torch.sigmoid(reconstruction_logits)
        batch_size = x_clean.shape[0]
        total_samples += batch_size
        bce = F.binary_cross_entropy_with_logits(reconstruction_logits, x_clean, reduction="mean")
        mse = F.mse_loss(reconstruction, x_clean, reduction="mean")
        bce_total += float(bce.cpu()) * batch_size
        mse_total += float(mse.cpu()) * batch_size
    return {
        "eval_noisy_bce_per_pixel": bce_total / total_samples,
        "eval_noisy_mse": mse_total / total_samples,
    }


def run_vae_training(
    regularizer: str,
    beta: float,
    train_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[SmallMnistVAE, list[dict[str, float | int | str]]]:
    torch.manual_seed(args.seed)
    model = SmallMnistVAE(latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows: list[dict[str, float | int | str]] = []
    global_step = 0
    stable = True

    print(f"\n[paper:VAE] Training regularizer={regularizer} beta={beta}")
    for epoch in range(args.vae_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{regularizer} beta={beta} epoch {epoch + 1}")
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
                print(
                    "[paper:VAE] non-finite loss detected; "
                    f"regularizer={regularizer} beta={beta} step={global_step}"
                )
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            if global_step % args.log_every == 0:
                compact = format_metrics(
                    {name: float(value.cpu()) for name, value in metrics.items()}
                )
                progress.set_postfix(compact)
                print(
                    "[paper:VAE] "
                    f"regularizer={regularizer} beta={beta} step={global_step:04d} "
                    f"loss={compact['loss']:.6f} "
                    f"reconstruction={compact['reconstruction']:.6f} "
                    f"regularization={compact['regularization']:.6f}"
                )
                rows.append(
                    {
                        "task": "vae",
                        "regularizer": regularizer,
                        "beta": beta,
                        "step": global_step,
                        **compact,
                    }
                )
            global_step += 1
        if not stable:
            break

    return model, rows


def run_vae_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    eval_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_dataset = Subset(train_dataset, range(min(args.vae_train_samples, len(train_dataset))))
    eval_dataset = Subset(eval_dataset, range(min(args.vae_eval_samples, len(eval_dataset))))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    configs = [
        ("kl", 1.0),
        ("fisher_rao", 1.0),
        ("fisher_rao", 0.3),
    ]
    metric_rows: list[dict[str, float | int | str]] = []
    dynamics_rows: list[dict[str, float | int | str]] = []

    print("\n[paper:VAE] Benchmarking reconstruction, latent geometry, and input-noise robustness")
    for regularizer, beta in configs:
        model, rows = run_vae_training(regularizer, beta, train_loader, device, args)
        dynamics_rows.extend(rows)
        eval_metrics = evaluate_vae_loader(model, eval_loader, device, regularizer, beta)
        train_latents, train_labels = collect_latents(model, train_eval_loader, device)
        eval_latents, eval_labels = collect_latents(model, eval_loader, device)
        if np.isfinite(train_latents).all() and np.isfinite(eval_latents).all():
            eval_metrics["eval_latent_knn_accuracy"] = latent_knn_accuracy(
                train_latents,
                train_labels,
                eval_latents,
                eval_labels,
            )
        else:
            eval_metrics["eval_latent_knn_accuracy"] = float("nan")
        eval_metrics.update(evaluate_noisy_vae_reconstruction(model, eval_loader, device, 0.25))
        eval_metrics["eval_stable"] = 0.0 if has_nonfinite_metrics(eval_metrics) else 1.0
        metric_rows.append(
            {
                "task": "vae",
                "regularizer": regularizer,
                "beta": beta,
                **eval_metrics,
            }
        )
        print(
            f"[paper:VAE] regularizer={regularizer} beta={beta} "
            f"{format_metrics(eval_metrics)}"
        )

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "vae_final_metrics.csv", metric_rows)
    write_rows(output_dir / "vae_training_dynamics.csv", dynamics_rows)


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[paper] device={device} output_dir={output_dir}")
    run_tsne_benchmark(args, device)
    run_vae_benchmark(args, device)
    print("[paper] Benchmark complete")


if __name__ == "__main__":
    main()

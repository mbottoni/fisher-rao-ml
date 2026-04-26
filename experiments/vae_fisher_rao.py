from __future__ import annotations

import argparse

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fisher_rao_ml.device import get_device
from fisher_rao_ml.evaluation import (
    collect_latents,
    evaluate_vae_loader,
    format_metrics,
    latent_knn_accuracy,
)
from fisher_rao_ml.tracking import configure_mlflow
from fisher_rao_ml.vae import SmallMnistVAE, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MNIST VAE with KL or Fisher-Rao regularization."
    )
    parser.add_argument("--regularizer", choices=["kl", "fisher_rao"], default="fisher_rao")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=10_000)
    parser.add_argument("--max-eval-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tracking-dir", default="mlruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    configure_mlflow("fisher-rao-vae", args.tracking_dir)

    device = get_device()
    torch.manual_seed(args.seed)

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    eval_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    if args.max_train_samples > 0:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_samples, len(train_dataset))),
        )
    if args.max_eval_samples > 0:
        eval_dataset = Subset(
            eval_dataset,
            range(min(args.max_eval_samples, len(eval_dataset))),
        )

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SmallMnistVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("\n[VAE] Starting experiment")
    print(f"  regularizer: {args.regularizer}")
    print(f"  device: {device}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  train_samples: {len(train_dataset)}")
    print(f"  eval_samples: {len(eval_dataset)}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  beta: {args.beta}")
    print(f"  lr: {args.lr}")
    print(f"  parameters: {parameter_count:,}")

    with mlflow.start_run(run_name=f"vae-{args.regularizer}"):
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))
        mlflow.log_param("parameters", parameter_count)
        run_id = mlflow.active_run().info.run_id
        print(f"  mlflow_run_id: {run_id}")

        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for x, _ in progress:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction_logits, mean, logvar = model(x)
                loss, metrics = vae_loss(
                    reconstruction_logits,
                    x,
                    mean,
                    logvar,
                    regularizer=args.regularizer,
                    beta=args.beta,
                )
                loss.backward()
                optimizer.step()

                if global_step % args.log_every == 0:
                    for name, value in metrics.items():
                        mlflow.log_metric(name, float(value.cpu()), step=global_step)
                    compact_metrics = format_metrics(
                        {name: float(value.cpu()) for name, value in metrics.items()}
                    )
                    progress.set_postfix(compact_metrics)
                    print(
                        "[VAE] "
                        f"step={global_step:04d} "
                        f"loss={compact_metrics['loss']:.6f} "
                        f"reconstruction={compact_metrics['reconstruction']:.6f} "
                        f"regularization={compact_metrics['regularization']:.6f}"
                    )
                global_step += 1

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value.cpu()), step=global_step - 1)
        final_train_metrics = format_metrics(
            {name: float(value.cpu()) for name, value in metrics.items()}
        )
        print("[VAE] Final training batch metrics")
        for name, value in final_train_metrics.items():
            print(f"  {name}: {value}")

        print("[VAE] Evaluating held-out reconstruction and latent geometry")
        eval_metrics = evaluate_vae_loader(
            model,
            eval_loader,
            device,
            regularizer=args.regularizer,
            beta=args.beta,
        )
        train_latents, train_labels = collect_latents(model, train_eval_loader, device)
        eval_latents, eval_labels = collect_latents(model, eval_loader, device)
        eval_metrics["eval_latent_knn_accuracy"] = latent_knn_accuracy(
            train_latents,
            train_labels,
            eval_latents,
            eval_labels,
        )
        for name, value in eval_metrics.items():
            mlflow.log_metric(name, value, step=global_step - 1)
        print("[VAE] Final evaluation metrics")
        for name, value in format_metrics(eval_metrics).items():
            print(f"  {name}: {value}")

        print("[VAE] Logging model artifact to MLflow")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("[VAE] Finished\n")


if __name__ == "__main__":
    main()

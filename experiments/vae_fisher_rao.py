from __future__ import annotations

import argparse

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from fisher_rao_ml.device import get_device
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tracking-dir", default="mlruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow("fisher-rao-vae", args.tracking_dir)

    device = get_device()
    torch.manual_seed(args.seed)

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    if args.max_train_samples > 0:
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SmallMnistVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name=f"vae-{args.regularizer}"):
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))

        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            for x, _ in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
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

                if global_step % 25 == 0:
                    for name, value in metrics.items():
                        mlflow.log_metric(name, float(value.cpu()), step=global_step)
                global_step += 1

        mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()

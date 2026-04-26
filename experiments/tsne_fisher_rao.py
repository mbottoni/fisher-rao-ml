from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.datasets import make_blobs

from fisher_rao_ml.device import get_device
from fisher_rao_ml.evaluation import evaluate_embedding, format_metrics
from fisher_rao_ml.tracking import configure_mlflow
from fisher_rao_ml.tsne import (
    pairwise_student_t_affinities,
    symmetric_gaussian_affinities,
    tsne_distribution_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KL and Fisher-Rao t-SNE objectives.")
    parser.add_argument("--objective", choices=["kl", "fisher_rao"], default="fisher_rao")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--bandwidth", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-neighbors", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tracking-dir", default="mlruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    configure_mlflow("fisher-rao-tsne", args.tracking_dir)

    device = get_device()
    torch.manual_seed(args.seed)
    x_np, labels_np = make_blobs(
        n_samples=args.n_samples,
        n_features=8,
        centers=5,
        cluster_std=1.6,
        random_state=args.seed,
    )
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    p = symmetric_gaussian_affinities(x, bandwidth=args.bandwidth)

    print("\n[t-SNE] Starting experiment")
    print(f"  objective: {args.objective}")
    print(f"  device: {device}")
    print(f"  samples: {args.n_samples}")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  bandwidth: {args.bandwidth}")
    print(f"  eval_neighbors: {args.eval_neighbors}")

    embedding = torch.randn(args.n_samples, 2, device=device, requires_grad=True) * 1e-3
    embedding = torch.nn.Parameter(embedding)
    optimizer = torch.optim.Adam([embedding], lr=args.lr)

    with mlflow.start_run(run_name=f"tsne-{args.objective}"):
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))
        run_id = mlflow.active_run().info.run_id
        print(f"  mlflow_run_id: {run_id}")

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            q = pairwise_student_t_affinities(embedding)
            loss = tsne_distribution_loss(p, q, objective=args.objective)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                loss_value = float(loss.detach().cpu())
                mlflow.log_metric("loss", loss_value, step=step)
                print(f"[t-SNE] step={step:04d} loss={loss_value:.6f}")

        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)
        fig_path = output_dir / f"tsne_{args.objective}.png"

        emb = embedding.detach().cpu().numpy()
        eval_metrics = evaluate_embedding(
            x_np,
            emb,
            labels_np,
            n_neighbors=args.eval_neighbors,
            seed=args.seed,
        )
        for name, value in eval_metrics.items():
            mlflow.log_metric(name, value, step=args.steps - 1)
        print("[t-SNE] Final evaluation metrics")
        for name, value in format_metrics(eval_metrics).items():
            print(f"  {name}: {value}")

        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=labels_np, s=18, cmap="tab10")
        plt.title(f"t-SNE objective: {args.objective}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        mlflow.log_artifact(str(fig_path))
        print(f"[t-SNE] Saved embedding plot: {fig_path}")
        print("[t-SNE] Finished\n")


if __name__ == "__main__":
    main()

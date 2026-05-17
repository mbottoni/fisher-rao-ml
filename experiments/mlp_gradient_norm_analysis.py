"""MLP gradient norm analysis: noisy/clean ratio for MLP on Digits (sym_40).

Mirrors gradient_norm_analysis.py but uses the MLP + Digits setup
to compare whether FR's gradient bounding also occurs in the MLP regime
(where FR hurts accuracy). If FR's ratio ≈ 1 on MLPs too, the Ghosh
condition — not gradient dynamics — explains the MLP failure.

Outputs:
  reports/results/mlp_gradient_norm_full.csv
  (columns: objective, seed, epoch, sample_type, mean_grad_norm, loss)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

OBJECTIVES = ("kl", "fisher_rao", "gce", "mae", "hellinger", "sce")
N_CLASSES = 10
NOISE_RATE = 0.40
RESULTS = Path("reports/results")
OUT = RESULTS / "mlp_gradient_norm_full.csv"
FIELDNAMES = ["objective", "seed", "epoch", "sample_type", "mean_grad_norm", "loss"]
PROBE_EPOCHS = list(range(0, 100, 5)) + [99]


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_grad_norm(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      objective: str, device: torch.device) -> tuple[float, float]:
    model.train()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = distribution_loss_from_logits(y, logits, objective=objective)
    model.zero_grad()
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    model.zero_grad()
    return float(total_norm ** 0.5), float(loss.item())


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


def run_experiment(objective: str, seed: int, device: torch.device,
                   x_tr: torch.Tensor, y_tr_oh: torch.Tensor,
                   clean_idx: np.ndarray, noisy_idx: np.ndarray,
                   writer: csv.DictWriter) -> None:
    torch.manual_seed(seed)
    model = MLP(x_tr.shape[1], N_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    rng = np.random.default_rng(seed + 10000)
    idx_all = np.arange(len(x_tr))

    probe_clean = x_tr[clean_idx[:64]], y_tr_oh[clean_idx[:64]]
    probe_noisy = x_tr[noisy_idx[:64]], y_tr_oh[noisy_idx[:64]]

    for epoch in range(100):
        rng.shuffle(idx_all)
        for b in range(0, len(x_tr), 128):
            batch_idx = idx_all[b:b + 128]
            xb = x_tr[batch_idx].to(device)
            yb = y_tr_oh[batch_idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=objective)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        if epoch in PROBE_EPOCHS:
            clean_norm, clean_loss = compute_grad_norm(model, *probe_clean, objective, device)
            noisy_norm, noisy_loss = compute_grad_norm(model, *probe_noisy, objective, device)
            writer.writerow({"objective": objective, "seed": seed, "epoch": epoch,
                             "sample_type": "clean", "mean_grad_norm": clean_norm,
                             "loss": clean_loss})
            writer.writerow({"objective": objective, "seed": seed, "epoch": epoch,
                             "sample_type": "noisy", "mean_grad_norm": noisy_norm,
                             "loss": noisy_loss})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = get_device()
    RESULTS.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_digits()
    x_all = data.data.astype(np.float32)
    y_all = data.target.astype(np.int64)

    done: set[tuple[str, int]] = set()
    if OUT.exists() and not args.force:
        with OUT.open() as f:
            for row in csv.DictReader(f):
                done.add((row["objective"], int(row["seed"])))

    write_header = not OUT.exists() or args.force
    with OUT.open("w" if args.force else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            x_tr_raw, x_te_raw, y_tr, y_te = train_test_split(
                x_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
            )
            scaler = StandardScaler()
            x_tr_raw = scaler.fit_transform(x_tr_raw).astype(np.float32)

            noisy = y_tr.copy()
            n_noisy = int(NOISE_RATE * len(y_tr))
            idx = rng.choice(len(y_tr), size=n_noisy, replace=False)
            for i in idx:
                choices = [c for c in range(N_CLASSES) if c != int(noisy[i])]
                noisy[i] = int(rng.choice(choices))

            corrupted_mask = noisy != y_tr
            noisy_idx = np.where(corrupted_mask)[0]
            clean_idx = np.where(~corrupted_mask)[0]

            x_tr_t = torch.tensor(x_tr_raw)
            y_tr_oh = make_one_hot(noisy, N_CLASSES)

            for obj in OBJECTIVES:
                if (obj, seed) in done:
                    print(f"  skip {obj}/seed{seed}")
                    continue
                print(f"Running {obj}/seed{seed}...", flush=True)
                run_experiment(obj, seed, device, x_tr_t, y_tr_oh,
                               clean_idx, noisy_idx, writer)
                f.flush()
                done.add((obj, seed))
                print(f"  done {obj}/seed{seed}")


if __name__ == "__main__":
    main()

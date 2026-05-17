"""Track softmax probability on wrong labels during training.

Directly measures p_k(corrupted_label) across epochs to verify the danger zone
prediction: FR keeps p_k above 0.032 longer than Hellinger (universally soft),
while KL pushes p_k toward 1 (memorisation). The danger zone (p_k > 0.032) is
where FR gradient > KL gradient; once p_k < 0.032, FR enters saturation (FR < KL).

Protocol:
- CIFAR-10 10k subset, sym_40 and asym_40 regimes, 3 seeds
- Every 5 epochs, compute mean p_k(wrong_label) on held-out noisy batch
- Compare KL, FR, Hellinger, GCE, MAE at key epochs
- DANGER_ZONE_THRESHOLD = 0.032 is shown as reference line in output

Outputs:
  reports/results/wrong_label_prob_full.csv
  (columns: objective, seed, noise_regime, epoch, mean_wrong_prob)
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

OBJECTIVES = ("kl", "fisher_rao", "hellinger", "gce", "mae")
N_CLASSES = 10
DANGER_ZONE_THRESHOLD = 0.032
PROBE_EPOCHS = list(range(0, 60, 5)) + [59]
RESULTS = Path("reports/results")
OUT = RESULTS / "wrong_label_prob_full.csv"
FIELDNAMES = [
    "objective", "seed", "noise_regime", "epoch",
    "mean_wrong_prob", "frac_below_threshold",
]


def load_cifar10_subset(n_train: int, n_test: int, seed: int) -> tuple:
    try:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise RuntimeError("torchvision required") from e

    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
    classes = np.unique(labels)
    n_per_class = max(1, n_train // len(classes))
    idxs = []
    for c in classes:
        c_idxs = np.where(labels == c)[0]
        chosen = rng.choice(c_idxs, size=min(n_per_class, len(c_idxs)), replace=False)
        idxs.extend(chosen.tolist())
    idxs = np.array(idxs[:n_train])
    rng.shuffle(idxs)
    x_tr = np.stack([train_ds[int(i)][0].numpy() for i in idxs]).astype(np.float32)
    y_tr = labels[idxs].astype(np.int64)

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    x_tr = (x_tr - mean) / std
    return x_tr, y_tr


def inject_symmetric_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple:
    noisy = y.copy()
    n_noisy = int(rate * len(y))
    idx = rng.choice(len(y), size=n_noisy, replace=False)
    for i in idx:
        choices = [c for c in range(N_CLASSES) if c != int(noisy[i])]
        noisy[i] = int(rng.choice(choices))
    is_noisy = np.zeros(len(y), dtype=bool)
    is_noisy[idx] = True
    return noisy, is_noisy


def inject_asymmetric_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple:
    noisy = y.copy()
    is_noisy = np.zeros(len(y), dtype=bool)
    shift = {0: 2, 1: 9, 2: 0, 3: 5, 4: 3, 5: 4, 6: 1, 7: 6, 8: 7, 9: 8}
    for i, label in enumerate(y):
        if rng.random() < rate:
            noisy[i] = shift[int(label)]
            if noisy[i] != label:
                is_noisy[i] = True
    return noisy, is_noisy


class ConvNet(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


@torch.no_grad()
def compute_wrong_prob(
    model: nn.Module,
    x_probe: torch.Tensor,
    y_noisy_probe: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean p_k(wrong_label), fraction with p_k < DANGER_ZONE_THRESHOLD)."""
    model.eval()
    logits = model(x_probe.to(device))
    probs = torch.softmax(logits, dim=-1).cpu()
    wrong_probs = probs[torch.arange(len(y_noisy_probe)), y_noisy_probe]
    mean_wp = wrong_probs.mean().item()
    frac_below = (wrong_probs < DANGER_ZONE_THRESHOLD).float().mean().item()
    return mean_wp, frac_below


def run_one(
    objective: str,
    seed: int,
    noise_regime: str,
    device: torch.device,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    writer: csv.DictWriter,
) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 42)

    if "asym" in noise_regime:
        rate = float(noise_regime.split("_")[1]) / 100
        y_noisy, is_noisy = inject_asymmetric_noise(y_tr, rate, rng)
    else:
        rate = float(noise_regime.split("_")[1]) / 100 if noise_regime != "clean" else 0.0
        if rate > 0:
            y_noisy, is_noisy = inject_symmetric_noise(y_tr, rate, rng)
        else:
            y_noisy = y_tr.copy()
            is_noisy = np.zeros(len(y_tr), dtype=bool)

    noisy_idx = np.where(is_noisy)[0]
    if len(noisy_idx) < 64:
        print(f"  warning: only {len(noisy_idx)} noisy samples")
        return

    x_tr_t = torch.tensor(x_tr)
    y_noisy_oh = make_one_hot(y_noisy, N_CLASSES)

    probe_x = x_tr_t[noisy_idx[:64]]
    probe_y_wrong = torch.tensor(y_noisy[noisy_idx[:64]], dtype=torch.long)

    model = ConvNet(N_CLASSES).to(device)
    lr = 0.05
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=55, eta_min=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[5])

    idx_all = np.arange(len(x_tr))
    aug_rng = np.random.default_rng(seed + 99)

    for epoch in range(60):
        model.train()
        aug_rng.shuffle(idx_all)
        for start in range(0, len(x_tr), 128):
            batch_idx = idx_all[start:start + 128]
            xb = x_tr_t[batch_idx].to(device)
            yb = y_noisy_oh[batch_idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=objective)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        if epoch in PROBE_EPOCHS:
            mean_wp, frac_below = compute_wrong_prob(model, probe_x, probe_y_wrong, device)
            writer.writerow({
                "objective": objective, "seed": seed, "noise_regime": noise_regime,
                "epoch": epoch, "mean_wrong_prob": mean_wp, "frac_below_threshold": frac_below,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--regimes", nargs="+", default=["sym_40", "asym_40"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = get_device()
    RESULTS.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, int, str]] = set()
    if OUT.exists() and not args.force:
        with OUT.open() as f:
            for row in csv.DictReader(f):
                done.add((row["objective"], int(row["seed"]), row["noise_regime"]))

    write_header = not OUT.exists() or args.force
    with OUT.open("w" if args.force else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for noise_regime in args.regimes:
            for seed in range(args.seeds):
                x_tr, y_tr = load_cifar10_subset(10000, 2000, seed)
                for obj in OBJECTIVES:
                    if (obj, seed, noise_regime) in done:
                        print(f"  skip {obj}/{seed}/{noise_regime}")
                        continue
                    print(f"[wrong-prob] {noise_regime} {obj}/seed{seed}...", flush=True)
                    run_one(obj, seed, noise_regime, device, x_tr, y_tr, writer)
                    f.flush()
                    done.add((obj, seed, noise_regime))
                    print(f"  done {obj}/seed{seed}/{noise_regime}")


if __name__ == "__main__":
    main()

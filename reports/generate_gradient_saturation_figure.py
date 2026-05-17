"""Generate the gradient saturation figure for fr_noisy_labels.tex (Figure 2).

Plots |∂ℓ/∂z_k| for KL, FR, Hellinger, GCE(q=0.7) as a function of p_k,
the probability assigned to the corrupted label k. Shades the danger zone
(p_k ∈ (0.032, 1)) where FR gradient exceeds KL's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# Threshold where FR gradient = KL gradient: solve 4*arccos(√p)·√p·√(1-p) = 1-p
DANGER_ZONE_THRESHOLD = 0.032


def grad_kl(p: np.ndarray) -> np.ndarray:
    return 1 - p


def grad_fr(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 4 * np.arccos(np.sqrt(p)) * np.sqrt(p) * np.sqrt(1 - p)


def grad_hellinger(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 0.5 * np.sqrt(p) * (1 - p)


def grad_gce(p: np.ndarray, q: float = 0.7) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return p**q * (1 - p)


def grad_mae(p: np.ndarray) -> np.ndarray:
    return 2 * p * (1 - p)


def main() -> None:
    p = np.linspace(1e-4, 1 - 1e-4, 2000)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    kl = grad_kl(p)
    fr = grad_fr(p)
    h = grad_hellinger(p)
    gce = grad_gce(p, q=0.7)
    mae = grad_mae(p)

    ax.plot(p, kl, color="tab:blue", linewidth=2, label="KL (CE)")
    ax.plot(p, fr, color="tab:red", linewidth=2, linestyle="--", label="Fisher–Rao")
    ax.plot(p, h, color="tab:green", linewidth=2, linestyle="-.", label="Hellinger")
    ax.plot(p, gce, color="tab:orange", linewidth=2, linestyle=":", label="GCE ($q$=0.7)")
    ax.plot(p, mae, color="tab:purple", linewidth=1.5, linestyle=(0, (3, 1, 1, 1)), label="MAE")

    # Shade danger zone where FR > KL
    in_zone = fr > kl
    ax.fill_between(p, 0, 1.05, where=in_zone, alpha=0.10, color="tab:red",
                    label=f"Danger zone (FR > KL, $p_k > {DANGER_ZONE_THRESHOLD}$)")

    # Mark the crossover point
    ax.axvline(DANGER_ZONE_THRESHOLD, color="tab:red", linewidth=0.8, linestyle=":")
    ax.text(DANGER_ZONE_THRESHOLD + 0.01, 0.95, f"$p_k = {DANGER_ZONE_THRESHOLD}$",
            color="tab:red", fontsize=8, va="top")

    # Mark Hellinger peak
    p_hell_peak = 1 / 3
    ax.annotate(
        f"Hellinger max ≈ {grad_hellinger(np.array([p_hell_peak]))[0]:.2f}",
        xy=(p_hell_peak, grad_hellinger(np.array([p_hell_peak]))[0]),
        xytext=(0.45, 0.23),
        fontsize=8, color="tab:green",
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 0.8},
    )

    # Mark FR peak
    p_fr_peak = p[np.argmax(fr)]
    ax.annotate(
        f"FR max ≈ {max(fr):.2f}",
        xy=(p_fr_peak, max(fr)),
        xytext=(0.35, 1.75),
        fontsize=8, color="tab:red",
        arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 0.8},
    )

    ax.set_xlabel("$p_k$ — probability on corrupted label $k$", fontsize=11)
    ax.set_ylabel("$|{\\partial\\ell}/{\\partial z_k}|$ — gradient magnitude", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Gradient magnitude on corrupted label vs. confidence $p_k$\n"
        "(shaded: danger zone where FR amplifies corrupted gradients above KL)",
        fontsize=10,
    )

    fig.tight_layout()
    out = FIGURES / "gradient_saturation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()

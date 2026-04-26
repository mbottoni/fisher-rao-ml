# Fisher-Rao ML

Starter experiments for replacing KL divergence with Fisher-Rao distances in:

- t-SNE-style probability matching;
- VAE latent regularization.

The first technical goal is gradient feasibility. The losses in `src/fisher_rao_ml/losses.py`
are written as differentiable PyTorch operations, so gradients are obtained through autograd.

## Setup

```bash
uv sync --extra dev
```

PyTorch device selection prefers Apple Silicon MPS when available, then CUDA, then CPU.

## Experiments

Run t-SNE objective comparisons:

```bash
uv run --project . python experiments/tsne_fisher_rao.py --objective fisher_rao
uv run --project . python experiments/tsne_fisher_rao.py --objective kl
```

The t-SNE script prints the device, run configuration, MLflow run id, loss every
`--log-every` steps, final embedding metrics, and the saved plot path.

Run VAE regularizer comparisons:

```bash
uv run --project . python experiments/vae_fisher_rao.py --regularizer fisher_rao
uv run --project . python experiments/vae_fisher_rao.py --regularizer kl
```

The VAE script prints the device, dataset sizes, parameter count, MLflow run id,
periodic loss/reconstruction/regularization values, final training-batch metrics,
held-out evaluation metrics, and model artifact logging status. Use `--log-every`
to control console and MLflow training metric frequency.

Open MLflow:

```bash
uv run --project . mlflow ui --backend-store-uri ./mlruns
```

Open the marimo playground:

```bash
uv run --project . marimo edit notebooks/fisher_rao_playground.py
```

## Research Questions

This codebase is designed to test whether Fisher-Rao geometry is a useful replacement for KL
in distribution-matching objectives.

For t-SNE, the baseline minimizes:

```text
KL(P || Q)
```

The Fisher-Rao variant minimizes:

```text
d_FR(P, Q)^2 = 4 arccos(sum_i sqrt(P_i Q_i))^2
```

For VAEs, the baseline regularizes the approximate posterior with:

```text
KL(q_phi(z | x) || N(0, I))
```

The Fisher-Rao variant uses a diagonal-product approximation based on the closed-form
univariate Gaussian Fisher-Rao distance to the standard normal.

## Early Evaluation Plan

Track these metrics in MLflow:

- objective curves and gradient stability;
- final t-SNE embedding plots;
- final t-SNE trustworthiness, neighborhood recall, silhouette, and kNN accuracy;
- VAE reconstruction loss versus latent regularization;
- final VAE reconstruction, active latent units, latent variance, and latent kNN accuracy;
- sensitivity to `beta`, learning rate, and latent dimension.

Possible advantages to investigate:

- symmetric distribution comparison for t-SNE;
- information-geometric invariance;
- smoother behavior near the standard-normal VAE prior;
- better latent geometry or robustness when KL is too aggressive.

This is intentionally a research scaffold, not a finished claim. The next step is to run paired
KL/Fisher-Rao sweeps and compare outcomes rather than assume Fisher-Rao is better.

# Fisher-Rao ML

A rigorous empirical study of the Fisher-Rao geodesic distance as a replacement or complement
to KL divergence in t-SNE-style affinity matching and variational autoencoders. The repository
contains:

- a differentiable PyTorch implementation of categorical and diagonal-Gaussian
  Fisher-Rao distances (`src/fisher_rao_ml/losses.py`);
- t-SNE and VAE training scripts that toggle between KL and Fisher-Rao objectives;
- a multi-seed, multi-dataset t-SNE robustness benchmark
  (`experiments/paper_benchmark.py`);
- a multi-seed, multi-dataset VAE beta-sweep benchmark
  (`experiments/vae_benchmark.py`);
- hypothesis-driven dimensionality-reduction stress tests for cases where Fisher-Rao may help
  (`experiments/dimred_stress_benchmark.py`);
- paired-comparison statistical aggregation pipelines
  (`experiments/aggregate_results.py`, `experiments/aggregate_vae_results.py`,
  `experiments/aggregate_dimred_stress.py`);
- an arXiv-style LaTeX report (`reports/fisher_rao_vs_kl_arxiv.tex`).

The t-SNE result is negative-leaning: KL has a small but consistent advantage in
silhouette-based cluster separation. The VAE workflow asks a different question: whether
Fisher-Rao's intrinsic diagonal-Gaussian geometry changes reconstruction, latent usefulness,
posterior matching, generation, and corruption robustness after each objective receives its own
beta tuning.

## Setup

```bash
make install
```

PyTorch device selection prefers Apple Silicon MPS when available, then CUDA, then CPU.

## End-to-end paper rebuild

```bash
make paper-all
```

This runs the t-SNE benchmark, the VAE benchmark, aggregation, figure generation, and the
LaTeX report build. The default VAE grid is much larger than the t-SNE grid; use the smoke
command below while iterating.

Individual stages:

```bash
make paper-benchmark       # paired t-SNE runs
make vae-benchmark         # multi-dataset, multi-seed VAE beta sweep
make dimred-stress         # targeted t-SNE stress tests for Fisher-Rao-favorable regimes
make paper-aggregate       # t-SNE mean/std + paired Wilcoxon + Cliff's delta
make vae-aggregate         # VAE best-beta selection + paired tests
make dimred-stress-aggregate # stress-test mean/std + paired tests
make report-figures        # aggregate followed by figure regeneration
make report-pdf            # report-figures followed by pdflatex/bibtex
```

Quick VAE smoke run:

```bash
uv run --project . python experiments/vae_benchmark.py \
  --datasets mnist \
  --seeds 101 \
  --kl-betas 1.0 \
  --fr-betas 0.3 1.0 \
  --epochs 1 \
  --train-samples 256 \
  --eval-samples 128
uv run --project . python experiments/aggregate_vae_results.py
uv run --project . python reports/generate_figures.py
```

The VAE benchmark has a lightweight CSV cache. Existing
`(dataset, seed, regularizer, beta)` cells in `reports/results/vae_full_metrics.csv` are
skipped automatically; pass `--force` to recompute them.

MLflow is used by the single-objective scripts (`experiments/tsne_fisher_rao.py` and
`experiments/vae_fisher_rao.py`). The paper-scale grid benchmarks write CSV artifacts directly
so they can be aggregated and versioned reproducibly.

## Fisher-Rao-favorable stress tests

The headline benchmark asks whether Fisher-Rao improves broad t-SNE-style metrics. The stress
benchmark asks a narrower question: when does KL's unbounded asymmetric pressure become a
liability?

```bash
make dimred-stress
make dimred-stress-aggregate
```

Implemented stress families:

- `noisy_affinity`: inject false high-affinity cross-label edges into `P_train`, then evaluate
  against clean neighborhoods. The default run includes uniform, hub, block, and boundary
  false-edge mechanisms.
- `outlier_influence`: add bridge outliers and measure normal-point embedding drift after
  Procrustes alignment.
- `global_geometry`: use Swiss-roll and S-curve manifolds with continuum-preservation metrics.
- `symmetric_mismatch`: inject false-positive bridges between nearby parallel manifolds and
  measure leakage / false-neighbor rates.

For a quick smoke run:

```bash
uv run --project . python experiments/dimred_stress_benchmark.py \
  --experiments noisy_affinity \
  --samples 80 \
  --steps 5 \
  --seeds 101 \
  --corruption-types uniform \
  --false-edge-levels 0.1
uv run --project . python experiments/aggregate_dimred_stress.py
uv run --project . python reports/generate_figures.py
```

## Single-objective experiments (with MLflow tracking)

```bash
uv run --project . python experiments/tsne_fisher_rao.py --objective fisher_rao
uv run --project . python experiments/tsne_fisher_rao.py --objective kl
uv run --project . python experiments/vae_fisher_rao.py --regularizer fisher_rao
uv run --project . python experiments/vae_fisher_rao.py --regularizer kl
```

Open the MLflow UI:

```bash
make mlflow-ui
```

Open the marimo playground:

```bash
make marimo
```

## Tests and lint

```bash
make test
make lint
```

## Layout

```
src/fisher_rao_ml/
  losses.py        # differentiable Fisher-Rao distances
  tsne.py          # affinity kernels and KL/FR loss switch
  vae.py           # small MNIST VAE with KL/FR regularizer toggle
  evaluation.py    # objective-independent embedding and VAE metrics
  device.py        # MPS/CUDA/CPU selection
  tracking.py      # MLflow setup
experiments/
  tsne_fisher_rao.py       # individual t-SNE runs (MLflow logged)
  vae_fisher_rao.py        # individual VAE runs (MLflow logged)
  paper_benchmark.py       # multi-seed t-SNE robustness sweep
  vae_benchmark.py         # multi-seed VAE beta-sweep study
  dimred_stress_benchmark.py # targeted dimensionality-reduction stress tests
  aggregate_results.py     # t-SNE paired Wilcoxon + Cliff's delta + mean/std
  aggregate_vae_results.py # VAE best-beta selection + paired tests
  aggregate_dimred_stress.py # stress-test paired Wilcoxon + Cliff's delta + mean/std
reports/
  generate_figures.py      # figures consumed by the LaTeX report
  fisher_rao_vs_kl_arxiv.tex
  references.bib
  results/                 # CSV outputs of paper_benchmark + aggregator
  figures/                 # PDF figures consumed by the report
tests/                     # pytest suite for losses + evaluation
notebooks/                 # marimo playground
```

## Citation

If you use the benchmark or the statistical-evaluation pipeline, please cite the report in
`reports/fisher_rao_vs_kl_arxiv.tex`.

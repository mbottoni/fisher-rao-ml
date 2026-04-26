# Fisher-Rao ML

A rigorous empirical study of the Fisher-Rao geodesic distance as a replacement for KL
divergence in t-SNE-style affinity matching. The repository contains:

- a differentiable PyTorch implementation of categorical and diagonal-Gaussian
  Fisher-Rao distances (`src/fisher_rao_ml/losses.py`);
- t-SNE and VAE training scripts that toggle between KL and Fisher-Rao objectives;
- a multi-seed, multi-dataset, multi-noise robustness benchmark
  (`experiments/paper_benchmark.py`);
- a paired-comparison statistical aggregation pipeline
  (`experiments/aggregate_results.py`);
- an arXiv-style LaTeX report (`reports/fisher_rao_vs_kl_arxiv.tex`).

The headline empirical finding is that, across $3$ datasets, $5$ noise levels, and
$10$ seeds, KL has a small but consistent and statistically significant advantage in
silhouette-based cluster separation, and Fisher-Rao does not significantly outperform
KL on any cell. See the paper for full details.

## Setup

```bash
make install
```

PyTorch device selection prefers Apple Silicon MPS when available, then CUDA, then CPU.

## End-to-end paper rebuild

```bash
make paper-all
```

This runs the multi-seed benchmark, aggregates the results, regenerates figures, and
recompiles the LaTeX report. End-to-end runtime is approximately $2$--$3$ minutes on Apple
Silicon MPS plus a few seconds for the LaTeX build.

Individual stages:

```bash
make paper-benchmark       # 150 paired t-SNE runs + 3-config VAE preliminary
make paper-aggregate       # mean/std + paired Wilcoxon + Cliff's delta
make report-figures        # paper-aggregate followed by figure regeneration
make report-pdf            # report-figures followed by pdflatex/bibtex
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
  paper_benchmark.py       # full multi-seed multi-dataset paper sweep
  aggregate_results.py     # paired Wilcoxon + Cliff's delta + mean/std
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

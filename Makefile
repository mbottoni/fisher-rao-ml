.PHONY: install test lint tsne-fr tsne-kl tsne-compare vae-fr vae-kl vae-compare paper-benchmark vae-benchmark vae-aggregate paper-aggregate dimred-stress dimred-stress-aggregate ml-stress ml-stress-aggregate report-figures report-pdf paper-all mlflow-ui marimo

install:
	uv sync --project . --extra dev

test:
	uv run --project . --extra dev pytest tests

lint:
	uv run --project . --extra dev ruff check .

tsne-fr:
	uv run --project . python experiments/tsne_fisher_rao.py --objective fisher_rao

tsne-kl:
	uv run --project . python experiments/tsne_fisher_rao.py --objective kl

tsne-compare:
	uv run --project . python experiments/tsne_fisher_rao.py --objective kl
	uv run --project . python experiments/tsne_fisher_rao.py --objective fisher_rao

vae-fr:
	uv run --project . python experiments/vae_fisher_rao.py --regularizer fisher_rao

vae-kl:
	uv run --project . python experiments/vae_fisher_rao.py --regularizer kl

vae-compare:
	uv run --project . python experiments/vae_fisher_rao.py --regularizer kl
	uv run --project . python experiments/vae_fisher_rao.py --regularizer fisher_rao

paper-benchmark:
	uv run --project . python experiments/paper_benchmark.py --skip-vae

vae-benchmark:
	uv run --project . python experiments/vae_benchmark.py

paper-aggregate:
	uv run --project . python experiments/aggregate_results.py

dimred-stress:
	uv run --project . python experiments/dimred_stress_benchmark.py

dimred-stress-aggregate:
	uv run --project . python experiments/aggregate_dimred_stress.py

ml-stress:
	uv run --project . python experiments/soft_label_benchmark.py
	uv run --project . python experiments/distillation_benchmark.py

ml-stress-aggregate:
	uv run --project . python experiments/aggregate_ml_stress.py

vae-aggregate:
	uv run --project . python experiments/aggregate_vae_results.py

report-figures: paper-aggregate vae-aggregate dimred-stress-aggregate ml-stress-aggregate
	uv run --project . python reports/generate_figures.py

report-pdf: report-figures
	cd reports && pdflatex fisher_rao_vs_kl_arxiv.tex && bibtex fisher_rao_vs_kl_arxiv && pdflatex fisher_rao_vs_kl_arxiv.tex && pdflatex fisher_rao_vs_kl_arxiv.tex

paper-all: paper-benchmark vae-benchmark paper-aggregate vae-aggregate report-pdf

mlflow-ui:
	uv run --project . mlflow ui --backend-store-uri ./mlruns

marimo:
	uv run --project . marimo edit notebooks/fisher_rao_playground.py

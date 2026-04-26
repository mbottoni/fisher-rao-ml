.PHONY: install test lint tsne-fr tsne-kl tsne-compare vae-fr vae-kl vae-compare mlflow-ui marimo

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

mlflow-ui:
	uv run --project . mlflow ui --backend-store-uri ./mlruns

marimo:
	uv run --project . marimo edit notebooks/fisher_rao_playground.py

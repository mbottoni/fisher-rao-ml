from __future__ import annotations

from pathlib import Path

import mlflow


def configure_mlflow(experiment_name: str, tracking_dir: str = "mlruns") -> None:
    tracking_path = Path(tracking_dir).resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment(experiment_name)

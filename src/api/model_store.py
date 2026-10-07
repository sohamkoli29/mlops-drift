"""Load the champion model from the MLflow registry."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, MODEL_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    version: str
    run_id: str | None = None
    loaded_at: str = field(default_factory=_now)
    metrics: dict = field(default_factory=dict)


def load_champion(alias: str = CHAMPION_ALIAS) -> LoadedModel:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    run_metrics = client.get_run(mv.run_id).data.metrics
    metrics = {k: v for k, v in run_metrics.items() if k.startswith("holdout_")}
    return LoadedModel(model=model, version=str(mv.version), run_id=mv.run_id, metrics=metrics)

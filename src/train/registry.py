"""Thin helpers around MLflow tracking and the model registry (aliases, not stages)."""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME


def configure() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient()


def alias_version(client: MlflowClient, alias: str):
    """The ModelVersion an alias points to, or None if the alias/model doesn't exist."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def register_version(client: MlflowClient, run_id: str, alias: str | None = None):
    """Register the run's logged model as a new version; optionally point an alias at it."""
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    if alias:
        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    return mv

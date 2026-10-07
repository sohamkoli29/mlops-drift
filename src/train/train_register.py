"""Train the baseline, log everything to MLflow, register v1 and set alias 'champion'."""

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import CHAMPION_ALIAS, MODEL_NAME, REPORTS_DIR
from src.data.loader import load_holdout, load_reference
from src.drift.profile import build_profile
from src.train.pipeline import build_pipeline, get_candidates
from src.train.registry import alias_version, configure, register_version


def evaluate(pipe, X: pd.DataFrame, y: pd.Series, prefix: str = "holdout_") -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }
    return {f"{prefix}{k}": round(float(v), 6) for k, v in metrics.items()}


def _model_params(pipe) -> dict:
    params = pipe.named_steps["model"].get_params()
    return {f"model__{k}": str(v) for k, v in params.items()}


def train_and_log(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    run_name: str,
    tags: dict | None = None,
) -> tuple[str, dict]:
    """Fit a pipeline, log params/metrics/model/reference profile. Returns (run_id, metrics)."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags or {})

        pipe = build_pipeline(get_candidates()[model_name])
        pipe.fit(X_train, y_train)

        mlflow.log_params(
            {"model_type": model_name, "n_train": len(X_train), **_model_params(pipe)}
        )
        metrics = evaluate(pipe, X_eval, y_eval)
        mlflow.log_metrics(metrics)

        # reference profile for the drift detectors (built from this model's training data)
        mlflow.log_dict(build_profile(X_train), "reference_profile.json")

        sample = X_train.head(100)
        signature = infer_signature(sample, pipe.predict(sample))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )
        return run.info.run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="logreg | random_forest | gradient_boosting")
    parser.add_argument("--force", action="store_true", help="re-register even if champion exists")
    args = parser.parse_args()

    client = configure()
    existing = alias_version(client, CHAMPION_ALIAS)
    if existing and not args.force:
        print(
            f"Alias '{CHAMPION_ALIAS}' already points to v{existing.version}. "
            "Use --force to override."
        )
        return

    if args.model:
        model_name = args.model
    else:
        with open(REPORTS_DIR / "baseline_metrics.json") as f:
            model_name = json.load(f)["best_model"]
    print(f"Training: {model_name}")

    X_ref, y_ref = load_reference()
    X_hold, y_hold = load_holdout()

    run_id, metrics = train_and_log(
        model_name,
        X_ref,
        y_ref,
        X_hold,
        y_hold,
        run_name=f"baseline-{model_name}",
        tags={"stage": "baseline", "train_data": "reference"},
    )
    mv = register_version(client, run_id, alias=CHAMPION_ALIAS)

    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")
    print(f"Registered {MODEL_NAME} v{mv.version} with alias '{CHAMPION_ALIAS}'")


if __name__ == "__main__":
    main()

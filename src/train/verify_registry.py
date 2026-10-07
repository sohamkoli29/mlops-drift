"""Check that models:/<name>@champion loads and reproduces the logged holdout F1."""

import mlflow
import mlflow.artifacts
import mlflow.sklearn
from sklearn.metrics import f1_score

from src.config import CHAMPION_ALIAS, MODEL_NAME
from src.data.loader import load_holdout
from src.train.registry import alias_version, configure


def main() -> None:
    client = configure()
    mv = alias_version(client, CHAMPION_ALIAS)
    assert mv is not None, f"No '{CHAMPION_ALIAS}' alias found. Run train_register first."
    print(f"{MODEL_NAME} v{mv.version} | aliases={mv.aliases} | run_id={mv.run_id}")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    X, y = load_holdout()
    f1 = f1_score(y, model.predict(X))
    logged = client.get_run(mv.run_id).data.metrics["holdout_f1"]
    print(f"Loaded model holdout F1 = {f1:.4f} | logged = {logged:.4f}")
    assert abs(f1 - logged) < 1e-4, "Loaded model does not match the logged run"

    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="reference_profile.json"
    )
    print(f"Reference profile available at: {path}")
    print("OK: registry round trip works")


if __name__ == "__main__":
    main()

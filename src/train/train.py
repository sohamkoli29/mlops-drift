"""Compare candidate models by CV F1 on reference, report final metrics on holdout."""

import json
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import REPORTS_DIR, ROOT, SEED
from src.data.loader import load_holdout, load_reference
from src.train.pipeline import build_pipeline, get_candidates

MODEL_DIR = ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X_ref, y_ref = load_reference()
    X_hold, y_hold = load_holdout()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for name, model in get_candidates().items():
        start = time.time()
        res = cross_validate(
            build_pipeline(model),
            X_ref,
            y_ref,
            cv=cv,
            scoring=["f1", "accuracy"],
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_f1": res["test_f1"].mean(),
                "cv_f1_std": res["test_f1"].std(),
                "cv_accuracy": res["test_accuracy"].mean(),
                "seconds": time.time() - start,
            }
        )
        print(f"{name:20s} cv_f1={rows[-1]['cv_f1']:.4f}  ({rows[-1]['seconds']:.0f}s)")

    table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    print("\n", table.round(4).to_string(index=False))

    best_name = table.iloc[0]["model"]
    print(f"\nBest by CV F1: {best_name}")

    pipe = build_pipeline(get_candidates()[best_name])
    pipe.fit(X_ref, y_ref)

    pred = pipe.predict(X_hold)
    proba = pipe.predict_proba(X_hold)[:, 1]
    holdout = {
        "accuracy": accuracy_score(y_hold, pred),
        "precision": precision_score(y_hold, pred),
        "recall": recall_score(y_hold, pred),
        "f1": f1_score(y_hold, pred),
        "roc_auc": roc_auc_score(y_hold, proba),
    }
    holdout = {k: round(float(v), 4) for k, v in holdout.items()}
    print("Holdout metrics:", holdout)

    joblib.dump(pipe, MODEL_DIR / "baseline.joblib")
    with open(REPORTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "cv_comparison": table.round(4).to_dict(orient="records"),
                "holdout": holdout,
            },
            f,
            indent=2,
        )
    print(f"Saved model to {MODEL_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()

"""Detector comparison: scenarios x detectors x seeds, written to CSV and logged to MLflow.

Usage: python -m experiments.run_experiments [--seeds 5] [--first-seed 200] [--no-mlflow]
"""

import argparse
from pathlib import Path

import pandas as pd

from experiments.metrics import by_scenario, mean_std, per_run, pooled_metrics
from sim.simulator import load_scenario, reference_std, stream
from src.config import MLFLOW_TRACKING_URI, ROOT
from src.data.loader import load_production, load_reference
from src.drift.analysis import analyze_window, explanation_hit
from src.drift.detectors import load_thresholds

SCENARIOS = ["none", "sudden", "gradual"]
OUT_DIR = ROOT / "experiments" / "results"
EXPERIMENT = "drift-experiments"


def run_all(seeds, X_ref, X_pool, y_pool, std) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, expl = [], []
    for seed in seeds:
        for name in SCENARIOS:
            for w in stream(load_scenario(name), X_pool, y_pool, std, seed=seed):
                a = analyze_window(X_ref, w.X, window=w.index)
                base = {
                    "scenario": name,
                    "seed": seed,
                    "window": w.index,
                    "drift_active": w.drift_active,
                    "intensity": w.intensity,
                }
                for det, d in a["detectors"].items():
                    alert = d["severity"] != "none"
                    rows.append({**base, "detector": det, "alert": alert, "n": d["n_flagged"]})
                rows.append(
                    {**base, "detector": "vote_2of3", "alert": a["n_agreeing"] >= 2, "n": 0}
                )
                hit = explanation_hit(a, w.shifted_features)
                if hit:
                    expl.append(
                        {
                            **base,
                            "top_features": ",".join(t["feature"] for t in a["top_features"]),
                            "precision_at_3": hit["precision_at_k"],
                            "recall_at_3": hit["recall_at_k"],
                        }
                    )
        print(f"seed {seed} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(expl)


def log_to_mlflow(runs, pooled, thresholds, out_dir: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    delays = runs.pivot_table(index=["detector", "seed"], columns="scenario", values="delay")
    for _, r in pooled.iterrows():
        metrics = {k: r[k] for k in ("precision", "recall", "f1", "mcc", "false_alarm_rate")}
        for scenario in ("sudden", "gradual"):
            key = (r.detector, r.seed)
            if scenario in delays.columns and key in delays.index:
                metrics[f"delay_{scenario}"] = delays.loc[key, scenario]
        with mlflow.start_run(run_name=f"{r.detector}-seed{int(r.seed)}"):
            mlflow.log_params({"detector": r.detector, "seed": int(r.seed)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
    with mlflow.start_run(run_name="summary"):
        mlflow.log_dict(thresholds, "thresholds.json")
        for csv in sorted(out_dir.glob("*.csv")):
            mlflow.log_artifact(str(csv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    thresholds = load_thresholds()
    print("Thresholds in use:", thresholds, "\n")

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    df, expl = run_all(seeds, X_ref, X_pool, y_pool, reference_std(X_ref))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = per_run(df)
    pooled = pooled_metrics(runs)
    scen = by_scenario(runs)
    df.to_csv(OUT_DIR / "windows.csv", index=False)
    runs.to_csv(OUT_DIR / "per_run.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled_by_seed.csv", index=False)
    scen.to_csv(OUT_DIR / "by_scenario.csv", index=False)
    expl.to_csv(OUT_DIR / "explainability_windows.csv", index=False)

    pd.set_option("display.width", 150)
    print("\nPooled over all scenarios (mean ± std over seeds):\n")
    print(mean_std(pooled).to_string())
    print("\nBy scenario (mean over seeds):\n")
    print(scen.round(3).to_string(index=False))
    if len(expl):
        e = expl.groupby("scenario")[["precision_at_3", "recall_at_3"]].mean().round(3)
        print("\nExplainability: top-3 features vs the features the simulator shifted:\n")
        print(e.to_string())

    if not args.no_mlflow:
        try:
            log_to_mlflow(runs, pooled, thresholds, OUT_DIR)
            print(f"\nLogged to MLflow experiment '{EXPERIMENT}' at {MLFLOW_TRACKING_URI}")
        except Exception as exc:
            print(f"\nMLflow logging skipped ({type(exc).__name__}: {exc}). CSVs are saved.")


if __name__ == "__main__":
    main()

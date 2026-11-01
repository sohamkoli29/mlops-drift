"""Offline recovery experiment: F1 before drift, under drift and after retraining.

Usage: python -m experiments.recovery [--seeds 5] [--first-seed 200]
       [--recent-weight 5] [--reference-rows 5000] [--min-rows 4000]
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sim.simulator import WINDOW_SIZE, load_scenario, reference_std, stream
from src.config import ROOT, SEED
from src.data.loader import load_production, load_reference
from src.drift.analysis import analyze_window
from src.retrain.policy import PolicyConfig, PolicyState, decide, on_retrain_done
from src.retrain.validate import ValidationConfig, compare, f1_at_half
from src.train.pipeline import build_pipeline, get_candidates

OUT_DIR = ROOT / "experiments" / "results"


def get_champion(X_ref, y_ref):
    path = ROOT / "models" / "baseline.joblib"
    if path.exists():
        return joblib.load(path)
    return build_pipeline(get_candidates()["gradient_boosting"]).fit(X_ref, y_ref)


def train_challenger(X_recent, y_recent, X_ref, y_ref, args):
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        X_recent, y_recent, test_size=0.3, stratify=y_recent, random_state=SEED
    )
    X_fit, y_fit, weights = X_tr, y_tr, None
    if args.reference_rows > 0:
        idx = X_ref.sample(min(args.reference_rows, len(X_ref)), random_state=SEED).index
        X_fit = pd.concat([X_ref.loc[idx], X_tr], ignore_index=True)
        y_fit = pd.concat([y_ref.loc[idx], y_tr], ignore_index=True)
        weights = np.concatenate([np.ones(len(idx)), np.full(len(X_tr), args.recent_weight)])
    pipe = build_pipeline(get_candidates()["gradient_boosting"])
    pipe.fit(X_fit, y_fit, model__sample_weight=weights)
    return pipe, X_tr, X_ev, y_ev


def run_seed(seed, champion, X_ref, y_ref, X_pool, y_pool, std, args):
    scenario = load_scenario("sudden_concept")
    onset = scenario["onset_window"]
    pcfg, vcfg = PolicyConfig(), ValidationConfig()
    state = PolicyState()
    model, reference = champion, X_ref
    buffers: dict[int, tuple] = {}
    rows, events = [], []

    for w in stream(scenario, X_pool, y_pool, std, seed=seed):
        rows.append(
            {
                "seed": seed,
                "window": w.index,
                "f1_live": f1_at_half(model, w.X, w.y),
                "f1_never_retrain": f1_at_half(champion, w.X, w.y),
                "model_version": 1 + sum(e["promoted"] for e in events),
            }
        )
        buffers[w.index] = (w.X, w.y)

        a = analyze_window(reference, w.X, window=w.index)
        a["first_id"] = w.index * WINDOW_SIZE + 1
        decide(a, state, pcfg)
        if not state.pending:
            continue
        start = (state.drift_first_id - 1) // WINDOW_SIZE
        idx = [i for i in sorted(buffers) if i >= start]
        if len(idx) * WINDOW_SIZE < args.min_rows:
            continue

        X_recent = pd.concat([buffers[i][0] for i in idx], ignore_index=True)
        y_recent = pd.concat([buffers[i][1] for i in idx], ignore_index=True)
        challenger, X_tr, X_ev, y_ev = train_challenger(X_recent, y_recent, X_ref, y_ref, args)
        res = compare(model, challenger, X_ev, y_ev, vcfg)
        events.append({"seed": seed, "window": w.index, "promoted": res.promote, **res.to_dict()})
        if res.promote:
            model, reference = challenger, X_tr
        on_retrain_done(state, pcfg)
    return pd.DataFrame(rows), events, onset


def summarise(rows: pd.DataFrame, events: list, onset: int) -> dict:
    seed = int(rows.seed.iloc[0])
    first_retrain = events[0]["window"] if events else None
    before = rows[rows.window < onset]
    under_end = first_retrain if first_retrain is not None else rows.window.max()
    under = rows[(rows.window >= onset) & (rows.window <= under_end)]
    after = rows[rows.window > first_retrain] if first_retrain is not None else rows.iloc[0:0]
    return {
        "seed": seed,
        "f1_before": before.f1_live.mean(),
        "f1_under_drift": under.f1_live.mean(),
        "f1_after_with_retraining": after.f1_live.mean() if len(after) else np.nan,
        "f1_after_never_retrain": after.f1_never_retrain.mean() if len(after) else np.nan,
        "retrain_attempts": len(events),
        "promotions": sum(e["promoted"] for e in events),
        "first_retrain_window": first_retrain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=200)
    parser.add_argument("--recent-weight", type=float, default=5.0)
    parser.add_argument("--reference-rows", type=int, default=5000)
    parser.add_argument("--min-rows", type=int, default=4000)
    args = parser.parse_args()

    X_ref, y_ref = load_reference()
    X_pool, y_pool = load_production()
    std = reference_std(X_ref)
    champion = get_champion(X_ref, y_ref)

    all_rows, all_events, summaries = [], [], []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        rows, events, onset = run_seed(seed, champion, X_ref, y_ref, X_pool, y_pool, std, args)
        all_rows.append(rows)
        all_events += events
        summaries.append(summarise(rows, events, onset))
        print(f"seed {seed}: {len(events)} retrain attempt(s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary["gain_from_retraining"] = (
        summary.f1_after_with_retraining - summary.f1_after_never_retrain
    )
    pd.concat(all_rows).to_csv(OUT_DIR / "recovery_windows.csv", index=False)
    pd.DataFrame(all_events).to_csv(OUT_DIR / "recovery_events.csv", index=False)
    summary.to_csv(OUT_DIR / "recovery_summary.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nPer seed:\n")
    print(summary.round(3).to_string(index=False))
    print("\nMean over seeds:\n")
    print(summary.drop(columns="seed").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()

"""Run all detectors over every scenario and save per-window results.

Usage: python -m sim.run_offline [--seed 42]
"""

import argparse

import pandas as pd

from sim.simulator import load_scenario, reference_std, stream
from src.config import REPORTS_DIR
from src.data.loader import load_production, load_reference
from src.drift.detectors import get_detectors

SCENARIOS = ["none", "sudden", "gradual"]
OUT_DIR = REPORTS_DIR / "offline"


def run_scenario(name: str, X_ref, X_pool, y_pool, ref_std, seed: int) -> pd.DataFrame:
    scenario = load_scenario(name)
    detectors = get_detectors()
    rows = []
    for w in stream(scenario, X_pool, y_pool, ref_std, seed=seed):
        for det_name, det in detectors.items():
            res = det.score(X_ref, w.X)
            flagged = [f for f, r in res.items() if r.flagged]
            top = max(res.items(), key=lambda kv: kv[1].score)
            rows.append(
                {
                    "scenario": name,
                    "seed": seed,
                    "window": w.index,
                    "drift_active": w.drift_active,
                    "intensity": w.intensity,
                    "detector": det_name,
                    "n_flagged": len(flagged),
                    "n_features": len(res),
                    "max_score": round(top[1].score, 4),
                    "max_feature": top[0],
                    "flagged_features": ",".join(sorted(flagged)),
                    "alert": len(flagged) > 0,
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        drift = g[g.drift_active]
        calm = g[~g.drift_active]
        first = drift[drift.alert].window.min() if drift.alert.any() else None
        onset = drift.window.min() if len(drift) else None
        return pd.Series(
            {
                "false_alarm_rate": round(calm.alert.mean(), 3) if len(calm) else None,
                "recall": round(drift.alert.mean(), 3) if len(drift) else None,
                "delay_windows": (
                    (first - onset) if first is not None and onset is not None else None
                ),
            }
        )

    return df.groupby(["scenario", "detector"]).apply(stats, include_groups=False).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    ref_std = reference_std(X_ref)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [run_scenario(s, X_ref, X_pool, y_pool, ref_std, args.seed) for s in SCENARIOS]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_DIR / f"windows_seed{args.seed}.csv", index=False)

    pd.set_option("display.width", 140)
    print(summarise(df).to_string(index=False))
    print(f"\nSaved {OUT_DIR / f'windows_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()

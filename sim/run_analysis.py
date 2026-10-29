"""Analyse every window of each scenario: severity, top-k features, JSON (and optional HTML).

Usage: python -m sim.run_analysis [--seed 42] [--html]
"""

import argparse
import json

import pandas as pd

from sim.simulator import load_scenario, reference_std, stream
from src.config import REPORTS_DIR
from src.data.loader import load_production, load_reference
from src.drift.analysis import analyze_window, explanation_hit
from src.drift.report import save_html_report

SCENARIOS = ["none", "sudden", "gradual"]
OUT_DIR = REPORTS_DIR / "monitoring"
SYMBOL = {"none": ".", "medium": "m", "high": "H"}


def run_scenario(name, X_ref, X_pool, y_pool, ref_std, seed, html) -> pd.DataFrame:
    scenario = load_scenario(name)
    out_dir = OUT_DIR / name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = list(stream(scenario, X_pool, y_pool, ref_std, seed=seed))
    html_windows = {0, len(windows) - 1}
    if scenario["onset_window"] is not None:
        html_windows.add(scenario["onset_window"])

    rows = []
    for w in windows:
        a = analyze_window(X_ref, w.X, window=w.index)
        hit = explanation_hit(a, w.shifted_features)
        a.update(
            scenario=name,
            drift_active=w.drift_active,
            intensity=w.intensity,
            true_features=w.shifted_features,
            explanation_match=hit,
        )
        with open(out_dir / f"window_{w.index:02d}.json", "w") as f:
            json.dump(a, f, indent=2)
        if html and w.index in html_windows:
            save_html_report(X_ref, w.X, out_dir / f"window_{w.index:02d}.html")

        rows.append(
            {
                "scenario": name,
                "seed": seed,
                "window": w.index,
                "drift_active": w.drift_active,
                "intensity": w.intensity,
                "severity": a["severity"],
                "n_agreeing": a["n_agreeing"],
                "psi_flagged": a["detectors"]["psi"]["n_flagged"],
                "max_psi": a["max_psi"],
                "top_features": ",".join(t["feature"] for t in a["top_features"]),
                "precision_at_3": hit["precision_at_k"] if hit else None,
                "recall_at_3": hit["recall_at_k"] if hit else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--html", action="store_true", help="also write Evidently HTML reports")
    args = parser.parse_args()

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    ref_std = reference_std(X_ref)

    frames = [
        run_scenario(s, X_ref, X_pool, y_pool, ref_std, args.seed, args.html) for s in SCENARIOS
    ]
    df = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"summary_seed{args.seed}.csv", index=False)

    print("Severity per window (. none, m medium, H high):\n")
    for name, g in df.groupby("scenario", sort=False):
        strip = "".join(SYMBOL[s] for s in g.severity)
        drift = g[g.drift_active]
        line = f"{name:8s} {strip}"
        if len(drift):
            line += (
                f"   top-3 precision {drift.precision_at_3.mean():.2f}"
                f"  recall {drift.recall_at_3.mean():.2f}"
            )
        print(line)
    print(f"\nSaved JSON per window under {OUT_DIR}")


if __name__ == "__main__":
    main()

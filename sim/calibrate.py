"""Calibrate detector thresholds on no-drift windows and check them on unseen seeds.

Usage: python -m sim.calibrate [--seeds 10] [--write]
"""

import argparse
import json

import numpy as np
import pandas as pd

from sim.simulator import load_scenario, reference_std, stream
from src.config import REPORTS_DIR
from src.data.loader import load_production, load_reference
from src.drift.detectors import DEFAULT_THRESHOLDS, THRESHOLD_FILE
from src.drift.ks import KSDetector
from src.drift.psi import PSIDetector
from src.drift.wasserstein import WassersteinDetector

TARGET_FALSE_ALARM = 0.05
KS_ALPHAS = [0.05, 0.02, 0.01, 0.005, 0.001]
VALIDATION_SEEDS = range(100, 105)
DETECTORS = ["psi", "ks", "wasserstein"]


def collect(seeds, X_ref, X_pool, y_pool, ref_std) -> pd.DataFrame:
    """Raw per-feature scores on no-drift windows (thresholds are applied later)."""
    scenario = load_scenario("none")
    detectors = (PSIDetector(), KSDetector(), WassersteinDetector())
    rows = []
    for seed in seeds:
        for w in stream(scenario, X_pool, y_pool, ref_std, seed=seed):
            for det in detectors:
                for feature, r in det.score(X_ref, w.X).items():
                    rows.append(
                        {
                            "seed": seed,
                            "window": w.index,
                            "detector": det.name,
                            "feature": feature,
                            "score": r.score,
                            "p_value": r.p_value,
                        }
                    )
    return pd.DataFrame(rows)


def window_alerts(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Per window: did each detector flag at least one feature, and did 2 of 3 agree?"""
    d = df.copy()
    per_feature = th["wasserstein"]["per_feature"]
    limit = d.feature.map(lambda f: per_feature.get(f, th["wasserstein"]["default"]))
    d["flagged"] = np.select(
        [d.detector == "psi", d.detector == "ks", d.detector == "wasserstein"],
        [d.score >= th["psi"]["threshold"], d.p_value < th["ks"]["alpha"], d.score > limit],
        default=False,
    )
    per = d.groupby(["seed", "window", "detector"]).flagged.any().unstack("detector")
    per["agree_2of3"] = per[DETECTORS].sum(axis=1) >= 2
    return per


def calibrate(df: pd.DataFrame) -> tuple[dict, dict]:
    w = df[df.detector == "wasserstein"]
    per_feature = {
        f: round(max(0.05, 1.2 * float(g.score.quantile(0.99))), 3) for f, g in w.groupby("feature")
    }
    ks = df[df.detector == "ks"]
    ks_rates = {
        a: float(ks.assign(hit=ks.p_value < a).groupby(["seed", "window"]).hit.any().mean())
        for a in KS_ALPHAS
    }
    alpha = next((a for a in KS_ALPHAS if ks_rates[a] <= TARGET_FALSE_ALARM), KS_ALPHAS[-1])
    thresholds = {
        "psi": {"threshold": DEFAULT_THRESHOLDS["psi"]["threshold"]},
        "ks": {"alpha": alpha},
        "wasserstein": {"default": 0.10, "per_feature": per_feature},
    }
    return thresholds, ks_rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--write", action="store_true", help="write src/drift/thresholds.json")
    args = parser.parse_args()

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    std = reference_std(X_ref)

    fit = collect(range(args.seeds), X_ref, X_pool, y_pool, std)
    check = collect(VALIDATION_SEEDS, X_ref, X_pool, y_pool, std)
    calibrated, ks_rates = calibrate(fit)

    table = pd.DataFrame(
        {
            "default thresholds (fit seeds)": window_alerts(fit, DEFAULT_THRESHOLDS).mean(),
            "calibrated (fit seeds)": window_alerts(fit, calibrated).mean(),
            "calibrated (unseen seeds)": window_alerts(check, calibrated).mean(),
        }
    ).T[[*DETECTORS, "agree_2of3"]]

    pd.set_option("display.width", 140)
    print("Window-level false-alarm rate on no-drift windows (lower is better):\n")
    print(table.round(3).to_string())
    print(
        "\nKS: false-alarm rate per alpha (fit seeds):",
        {a: round(r, 3) for a, r in ks_rates.items()},
    )
    print("Chosen KS alpha:", calibrated["ks"]["alpha"])
    print("Wasserstein per-feature thresholds:", calibrated["wasserstein"]["per_feature"])
    psi = fit[fit.detector == "psi"].groupby("feature").score.quantile(0.99).round(3)
    print("\nPSI 99th percentile per feature (threshold stays 0.10):")
    print(psi.sort_values(ascending=False).head(5).to_string())

    n_windows = len(fit.groupby(["seed", "window"]))
    out = {
        "thresholds": calibrated,
        "calibrated_on": {"seeds": args.seeds, "windows": n_windows},
        "false_alarm_rates": table.round(4).to_dict(orient="index"),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "calibration.json").write_text(json.dumps(out, indent=2))
    if args.write:
        THRESHOLD_FILE.write_text(
            json.dumps({**calibrated, "calibrated_on": out["calibrated_on"]}, indent=2)
        )
        print(f"\nWrote {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()

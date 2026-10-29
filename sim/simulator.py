"""Replay the production pool in fixed windows, injecting drift from a scenario YAML."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import NUMERIC_FEATURES, ROOT, SEED

SCENARIO_DIR = ROOT / "sim" / "scenarios"
WINDOW_SIZE = 1000
INTEGER_FEATURES = set(NUMERIC_FEATURES)  # all numeric features are integers in this dataset


@dataclass
class Window:
    index: int
    X: pd.DataFrame
    y: pd.Series
    drift_active: bool
    intensity: float
    shifted_features: list[str] = field(default_factory=list)


def load_scenario(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def intensity_at(scenario: dict, window: int) -> float:
    """0 before onset; ramps linearly to 1 over ramp_windows; 1 afterwards."""
    onset = scenario["onset_window"]
    if onset is None or window < onset:
        return 0.0
    ramp = scenario.get("ramp_windows", 0)
    if ramp <= 0:
        return 1.0
    return min(1.0, (window - onset + 1) / ramp)


def apply_effects(
    X: pd.DataFrame, effects: list[dict], intensity: float, ref_std: dict, rng
) -> pd.DataFrame:
    X = X.copy()
    for eff in effects:
        col = eff["feature"]
        if eff["type"] == "shift_std":
            shifted = X[col].astype(float) + eff["value"] * intensity * ref_std[col]
            X[col] = np.clip(np.round(shifted), 0, None).astype(X[col].dtype)
        elif eff["type"] == "scale_std":
            mean = X[col].astype(float).mean()
            scaled = mean + (X[col].astype(float) - mean) * (1 + (eff["value"] - 1) * intensity)
            X[col] = np.clip(np.round(scaled), 0, None).astype(X[col].dtype)
        elif eff["type"] == "swap_category":
            n = int(round(len(X) * eff["fraction"] * intensity))
            if n > 0:
                idx = rng.choice(X.index.to_numpy(), size=n, replace=False)
                X.loc[idx, col] = eff["to"]
        else:
            raise ValueError(f"unknown effect type: {eff['type']}")
    return X


def stream(
    scenario: dict,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    ref_std: dict,
    seed: int = SEED,
    window_size: int = WINDOW_SIZE,
):
    """Yield Window objects. Each seed gives a different shuffle and different noise."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_pool))
    n_possible = len(X_pool) // window_size
    n_windows = min(scenario["n_windows"], n_possible)
    effects = scenario.get("effects", [])
    shifted = [e["feature"] for e in effects]

    for i in range(n_windows):
        rows = order[i * window_size : (i + 1) * window_size]
        X = X_pool.iloc[rows].reset_index(drop=True)
        y = y_pool.iloc[rows].reset_index(drop=True)
        intensity = intensity_at(scenario, i)
        if intensity > 0:
            X = apply_effects(X, effects, intensity, ref_std, rng)
        yield Window(
            index=i,
            X=X,
            y=y,
            drift_active=intensity > 0,
            intensity=round(intensity, 4),
            shifted_features=shifted if intensity > 0 else [],
        )


def reference_std(X_ref: pd.DataFrame) -> dict:
    return {c: float(X_ref[c].astype(float).std(ddof=1)) for c in NUMERIC_FEATURES}

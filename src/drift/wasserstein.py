"""Wasserstein-1 distance divided by the reference standard deviation (numeric only)."""

import pandas as pd
from scipy.stats import wasserstein_distance

from src.config import NUMERIC_FEATURES
from src.drift.base import FeatureScore, check_inputs, numeric_values

DEFAULT_THRESHOLD = 0.10  # placeholder, calibrated on the no-drift control on Day 11


class WassersteinDetector:
    name = "wasserstein"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def score(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, FeatureScore]:
        check_inputs(reference, current, NUMERIC_FEATURES)
        out: dict[str, FeatureScore] = {}
        for col in NUMERIC_FEATURES:
            ref = numeric_values(reference, col)
            cur = numeric_values(current, col)
            if len(ref) < 2 or len(cur) == 0:
                continue
            std = float(ref.std(ddof=1))
            w = float(wasserstein_distance(ref, cur)) / (std if std > 0 else 1.0)
            out[col] = FeatureScore(w, w > self.threshold)
        return out

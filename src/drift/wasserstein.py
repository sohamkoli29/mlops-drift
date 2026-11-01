"""Wasserstein-1 distance divided by the reference standard deviation (numeric only)."""

import pandas as pd
from scipy.stats import wasserstein_distance

from src.config import NUMERIC_FEATURES
from src.drift.base import FeatureScore, check_inputs, numeric_values

DEFAULT_THRESHOLD = 0.10  # fallback; per-feature values come from calibration (Day 11)


class WassersteinDetector:
    name = "wasserstein"

    def __init__(
        self, threshold: float = DEFAULT_THRESHOLD, per_feature: dict | None = None
    ) -> None:
        self.threshold = threshold
        self.per_feature = per_feature or {}

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
            limit = self.per_feature.get(col, self.threshold)
            out[col] = FeatureScore(w, w > limit)
        return out

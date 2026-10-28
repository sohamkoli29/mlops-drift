"""Two-sample Kolmogorov-Smirnov test (numeric features only)."""

import pandas as pd
from scipy.stats import ks_2samp

from src.config import NUMERIC_FEATURES
from src.drift.base import FeatureScore, check_inputs, numeric_values


class KSDetector:
    name = "ks"

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def score(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, FeatureScore]:
        check_inputs(reference, current, NUMERIC_FEATURES)
        out: dict[str, FeatureScore] = {}
        for col in NUMERIC_FEATURES:
            ref = numeric_values(reference, col)
            cur = numeric_values(current, col)
            if len(ref) == 0 or len(cur) == 0:
                continue
            res = ks_2samp(ref, cur)
            out[col] = FeatureScore(
                float(res.statistic), bool(res.pvalue < self.alpha), float(res.pvalue)
            )
        return out

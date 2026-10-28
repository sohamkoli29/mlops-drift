"""Population Stability Index: sum_i (q_i - p_i) * ln(q_i / p_i)."""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.drift.base import FeatureScore, check_inputs, numeric_values
from src.drift.profile import MISSING, N_BINS, OTHER, category_shares, numeric_bin_shares

EPS = 1e-4
PSI_MEDIUM = 0.10
PSI_HIGH = 0.25


def psi_value(ref_shares, cur_shares) -> float:
    p = np.clip(np.asarray(ref_shares, dtype=float), EPS, None)
    q = np.clip(np.asarray(cur_shares, dtype=float), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_level(score: float) -> str:
    """none < 0.10 <= medium <= 0.25 < high."""
    if score > PSI_HIGH:
        return "high"
    if score >= PSI_MEDIUM:
        return "medium"
    return "none"


class PSIDetector:
    name = "psi"

    def __init__(self, threshold: float = PSI_MEDIUM, n_bins: int = N_BINS) -> None:
        self.threshold = threshold
        self.n_bins = n_bins

    def score(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, FeatureScore]:
        check_inputs(reference, current, NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        out: dict[str, FeatureScore] = {}

        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for col in NUMERIC_FEATURES:
            ref = numeric_values(reference, col)
            cur = numeric_values(current, col)
            if len(ref) == 0 or len(cur) == 0:
                continue
            edges = np.unique(np.quantile(ref, quantiles))
            s = psi_value(numeric_bin_shares(ref, edges), numeric_bin_shares(cur, edges))
            out[col] = FeatureScore(s, s >= self.threshold)

        for col in CATEGORICAL_FEATURES:
            filled = reference[col].astype(object).where(reference[col].notna(), MISSING)
            categories = sorted(filled.unique())
            ref_sh = category_shares(reference[col], categories)
            cur_sh = category_shares(current[col], categories)
            keys = [*categories, OTHER]
            s = psi_value([ref_sh[k] for k in keys], [cur_sh[k] for k in keys])
            out[col] = FeatureScore(s, s >= self.threshold)
        return out

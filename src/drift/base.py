"""Common interface for all drift detectors."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureScore:
    """Result for one feature: the raw score, whether it crossed the threshold."""

    score: float
    flagged: bool
    p_value: float | None = None  # only KS fills this


class Detector(Protocol):
    name: str

    def score(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, FeatureScore]: ...


def check_inputs(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> None:
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must both be non-empty")
    missing = [f for f in features if f not in reference.columns or f not in current.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def numeric_values(df: pd.DataFrame, col: str) -> np.ndarray:
    """Non-null values of a numeric column as a float array."""
    return df[col].dropna().astype(float).to_numpy()

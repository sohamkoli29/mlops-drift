"""Load the parquet splits as (X, y) with consistent missing-value handling."""

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    HOLDOUT_FILE,
    PRODUCTION_FILE,
    REFERENCE_FILE,
    TARGET,
)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn None in categorical columns into np.nan so SimpleImputer sees it."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def _load(path) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_frame(pd.read_parquet(path))
    return df[FEATURES], df[TARGET]


def load_reference() -> tuple[pd.DataFrame, pd.Series]:
    return _load(REFERENCE_FILE)


def load_production() -> tuple[pd.DataFrame, pd.Series]:
    return _load(PRODUCTION_FILE)


def load_holdout() -> tuple[pd.DataFrame, pd.Series]:
    return _load(HOLDOUT_FILE)

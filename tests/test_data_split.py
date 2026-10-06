import pandas as pd
import pytest

from src.config import HOLDOUT_FILE, PRODUCTION_FILE, REFERENCE_FILE, TARGET

pytestmark = pytest.mark.skipif(not REFERENCE_FILE.exists(), reason="run split_data.py first")


def test_split_sizes_and_columns():
    ref = pd.read_parquet(REFERENCE_FILE)
    prod = pd.read_parquet(PRODUCTION_FILE)
    hold = pd.read_parquet(HOLDOUT_FILE)
    assert list(ref.columns) == list(prod.columns) == list(hold.columns)
    assert TARGET in ref.columns
    assert len(prod) >= 10_000  # enough windows of 1,000
    assert abs(ref[TARGET].mean() - prod[TARGET].mean()) < 0.02  # no built-in drift

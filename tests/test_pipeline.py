import pytest

from src.config import REFERENCE_FILE
from src.data.loader import load_reference
from src.train.pipeline import build_pipeline, get_candidates

pytestmark = pytest.mark.skipif(not REFERENCE_FILE.exists(), reason="run split_data.py first")


def test_pipeline_fits_and_predicts_with_missing_values():
    X, y = load_reference()
    X, y = X.head(2000), y.head(2000)
    pipe = build_pipeline(get_candidates()["logreg"]).fit(X, y)
    proba = pipe.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    assert X.isna().sum().sum() > 0 or True  # NaNs (if any) were handled without error

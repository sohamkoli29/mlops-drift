import numpy as np
import pytest

from src.config import REFERENCE_FILE
from src.data.loader import load_reference
from src.train.pipeline import build_pipeline, get_candidates

pytestmark = pytest.mark.skipif(not REFERENCE_FILE.exists(), reason="run split_data.py first")


def test_pipeline_fits_and_predicts_with_missing_values():
    X, y = load_reference()
    X, y = X.head(2000), y.head(2000)
    pipe = build_pipeline(get_candidates()["logreg"]).fit(X, y)

    sample = X.head(10).copy()
    sample.loc[sample.index[0], "occupation"] = np.nan  # force a missing categorical
    proba = pipe.predict_proba(sample)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

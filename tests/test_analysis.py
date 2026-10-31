import json

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.drift.analysis import analyze_window, explanation_hit


def make_frame(n, seed, age_shift=0.0, cat_p=0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(0, 1, n) for col in NUMERIC_FEATURES}
    data["age"] = data["age"] + age_shift
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["a", "b"], n, p=[cat_p, 1 - cat_p])
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def ref():
    return make_frame(20_000, seed=0)


def test_clean_window_has_no_severity_and_no_top_features(ref):
    a = analyze_window(ref, make_frame(1000, seed=1), window=3)
    assert a["severity"] == "none"
    assert a["top_features"] == []
    assert a["window"] == 3
    assert a["summary"] == "No significant drift"


def test_single_feature_drift_is_explained(ref):
    a = analyze_window(ref, make_frame(1000, seed=2, age_shift=2.0))
    assert a["severity"] == "high"
    assert a["n_agreeing"] == 3
    top = a["top_features"][0]
    assert top["feature"] == "age"
    assert top["flagged_by"] == ["ks", "psi", "wasserstein"]
    assert top["change"]["mean_shift_in_ref_std"] == pytest.approx(2.0, abs=0.3)
    json.dumps(a)  # must be JSON-serialisable


def test_two_shifted_features_are_the_top_two(ref):
    cur = make_frame(1000, seed=3, age_shift=2.0)
    cur["hours_per_week"] = cur["hours_per_week"] + 1.5
    a = analyze_window(ref, cur, k=2)
    names = [t["feature"] for t in a["top_features"]]
    assert set(names) == {"age", "hours_per_week"}
    hit = explanation_hit(a, ["age", "hours_per_week"], k=2)
    assert hit == {"precision_at_k": 1.0, "recall_at_k": 1.0}


def test_categorical_change_names_the_category(ref):
    a = analyze_window(ref, make_frame(1000, seed=4, cat_p=0.9), k=13)
    sex = next(t for t in a["top_features"] if t["feature"] == "sex")
    assert sex["flagged_by"] == ["psi"]
    assert sex["change"]["type"] == "categorical"
    assert abs(sex["change"]["cur_share"] - sex["change"]["ref_share"]) > 0.3


def test_explanation_hit_without_ground_truth_is_none(ref):
    a = analyze_window(ref, make_frame(1000, seed=5))
    assert explanation_hit(a, []) is None


def test_feature_scores_cover_every_detector_and_feature(ref):
    a = analyze_window(ref, make_frame(1000, seed=6))
    assert set(a["feature_scores"]) == {"psi", "ks", "wasserstein"}
    assert len(a["feature_scores"]["psi"]) == 13
    assert len(a["feature_scores"]["ks"]) == 5

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REFERENCE_FILE
from src.drift.detectors import get_detectors, score_all
from src.drift.ks import KSDetector
from src.drift.psi import PSIDetector, psi_level
from src.drift.wasserstein import WassersteinDetector


def make_frame(n, seed, age_shift=0.0, cat_p=0.5) -> pd.DataFrame:
    """Standard-normal numerics; 'age' optionally shifted; binary categoricals with P(a)=cat_p."""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(0, 1, n) for col in NUMERIC_FEATURES}
    data["age"] = data["age"] + age_shift
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["a", "b"], n, p=[cat_p, 1 - cat_p])
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def ref():
    return make_frame(20_000, seed=0)


# ---------- interface ----------


def test_registry_has_three_detectors():
    assert set(get_detectors()) == {"psi", "ks", "wasserstein"}


def test_feature_coverage(ref):
    cur = make_frame(1000, seed=1)
    assert set(PSIDetector().score(ref, cur)) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(KSDetector().score(ref, cur)) == set(NUMERIC_FEATURES)
    assert set(WassersteinDetector().score(ref, cur)) == set(NUMERIC_FEATURES)


def test_score_all_table_shape(ref):
    table = score_all(ref, make_frame(1000, seed=1))
    assert list(table.columns) == ["detector", "feature", "score", "flagged", "p_value"]
    assert len(table) == 13 + 5 + 5


def test_empty_current_raises(ref):
    with pytest.raises(ValueError):
        PSIDetector().score(ref, ref.head(0))


def test_nan_in_numeric_current_is_ignored(ref):
    cur = make_frame(1000, seed=1)
    cur.loc[:49, "age"] = np.nan
    assert np.isfinite(PSIDetector().score(ref, cur)["age"].score)


# ---------- PSI ----------


def test_psi_same_distribution_is_stable(ref):
    res = PSIDetector().score(ref, make_frame(1000, seed=1))
    assert all(r.score < 0.05 for r in res.values())
    assert not any(r.flagged for r in res.values())


def test_psi_medium_shift(ref):
    res = PSIDetector().score(ref, make_frame(1000, seed=2, age_shift=0.5))
    assert res["age"].flagged
    assert psi_level(res["age"].score) in {"medium", "high"}
    assert not res["hours_per_week"].flagged  # untouched features stay quiet


def test_psi_large_shift_is_high(ref):
    res = PSIDetector().score(ref, make_frame(1000, seed=3, age_shift=2.0))
    assert res["age"].score > 0.25
    assert psi_level(res["age"].score) == "high"


def test_psi_categorical_share_shift(ref):
    res = PSIDetector().score(ref, make_frame(1000, seed=4, cat_p=0.9))
    assert res["sex"].score > 0.25  # expected about 0.88 for 50/50 -> 90/10


def test_psi_unseen_category_lands_in_other(ref):
    cur = make_frame(1000, seed=5)
    cur.loc[:299, "race"] = "z"
    assert PSIDetector().score(ref, cur)["race"].flagged


# ---------- KS ----------


def test_ks_detects_shift_and_returns_p_value(ref):
    res = KSDetector().score(ref, make_frame(1000, seed=6, age_shift=0.5))
    assert res["age"].flagged
    assert res["age"].p_value < 0.01
    assert 0.0 < res["age"].score <= 1.0


def test_ks_false_alarm_rate_matches_alpha(ref):
    flags = [
        KSDetector().score(ref, make_frame(1000, seed=100 + i))["age"].flagged for i in range(40)
    ]
    assert np.mean(flags) < 0.20  # alpha = 0.05, so about 2 of 40 expected


# ---------- Wasserstein ----------


def test_wasserstein_known_answer_for_pure_shift(ref):
    res = WassersteinDetector().score(ref, make_frame(1000, seed=7, age_shift=2.0))
    assert res["age"].score == pytest.approx(2.0, abs=0.2)  # std(ref) = 1, shift = 2
    assert res["age"].flagged


def test_wasserstein_no_drift_noise_level(ref):
    scores = [
        WassersteinDetector().score(ref, make_frame(1000, seed=200 + i))["age"].score
        for i in range(20)
    ]
    # sampling noise with n=1000 is about 0.04, well under the 0.10 placeholder threshold
    assert np.mean(scores) < 0.08
    assert max(scores) < 0.10


# ---------- real data smoke test ----------


@pytest.mark.skipif(not REFERENCE_FILE.exists(), reason="run split_data.py first")
def test_real_data_window_from_same_distribution_is_quiet():
    from src.data.loader import load_production, load_reference

    X_ref, _ = load_reference()
    X_prod, _ = load_production()
    res = PSIDetector().score(X_ref, X_prod.iloc[:1000])
    assert len(res) == 13
    assert all(res[c].score < 0.10 for c in NUMERIC_FEATURES)

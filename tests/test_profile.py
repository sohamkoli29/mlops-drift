import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES
from src.drift.profile import OTHER, build_profile, category_shares, numeric_bin_shares


def _frame(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(17, 90, n),
            "education_num": rng.integers(1, 17, n),
            "capital_gain": np.where(rng.random(n) < 0.9, 0, rng.integers(1, 99999, n)),
            "capital_loss": np.where(rng.random(n) < 0.95, 0, rng.integers(1, 4000, n)),
            "hours_per_week": rng.integers(1, 99, n),
        }
    )
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.choice(["a", "b", "c"], n)
    df.loc[:49, "occupation"] = np.nan
    return df


def test_shares_sum_to_one():
    profile = build_profile(_frame())
    for p in profile["numeric"].values():
        assert abs(sum(p["shares"]) - 1) < 1e-9
    for p in profile["categorical"].values():
        assert abs(sum(p["shares"].values()) - 1) < 1e-9


def test_zero_heavy_feature_has_few_unique_sorted_edges():
    edges = build_profile(_frame())["numeric"]["capital_gain"]["edges"]
    assert edges == sorted(set(edges))
    assert len(edges) < 9


def test_unseen_category_goes_to_other():
    shares = category_shares(pd.Series(["a", "z"]), ["a", "b"])
    assert shares["a"] == 0.5
    assert shares[OTHER] == 0.5


def test_missing_is_its_own_category():
    profile = build_profile(_frame())
    assert profile["categorical"]["occupation"]["shares"]["__MISSING__"] > 0


def test_same_distribution_gives_similar_shares():
    ref = build_profile(_frame(seed=0))["numeric"]["age"]
    other = numeric_bin_shares(_frame(seed=1)["age"], ref["edges"])
    assert np.max(np.abs(other - np.array(ref["shares"]))) < 0.05

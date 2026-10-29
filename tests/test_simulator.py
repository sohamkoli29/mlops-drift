import pytest

from sim.simulator import intensity_at, load_scenario, reference_std, stream
from src.config import PRODUCTION_FILE, REFERENCE_FILE

needs_data = pytest.mark.skipif(
    not (REFERENCE_FILE.exists() and PRODUCTION_FILE.exists()), reason="run split_data.py first"
)


def test_intensity_none_and_sudden_and_gradual():
    none = {"onset_window": None, "ramp_windows": 0}
    sudden = {"onset_window": 8, "ramp_windows": 0}
    gradual = {"onset_window": 8, "ramp_windows": 6}
    assert all(intensity_at(none, i) == 0 for i in range(18))
    assert intensity_at(sudden, 7) == 0 and intensity_at(sudden, 8) == 1.0
    assert intensity_at(gradual, 7) == 0
    assert intensity_at(gradual, 8) == pytest.approx(1 / 6)
    assert intensity_at(gradual, 13) == 1.0 and intensity_at(gradual, 17) == 1.0


@needs_data
def test_stream_ground_truth_and_effects():
    from src.data.loader import load_production, load_reference

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    std = reference_std(X_ref)

    sudden = list(stream(load_scenario("sudden"), X_pool, y_pool, std, seed=1))
    assert [w.drift_active for w in sudden] == [i >= 8 for i in range(len(sudden))]
    assert all(len(w.X) == 1000 and len(w.y) == 1000 for w in sudden)
    assert sudden[0].shifted_features == []
    assert set(sudden[10].shifted_features) == {"age", "hours_per_week", "marital_status"}
    assert sudden[10].X["age"].mean() > sudden[0].X["age"].mean() + 5  # 0.8 std ~ 11 years
    assert (sudden[10].X["marital_status"] == "Divorced").mean() > 0.30


@needs_data
def test_none_scenario_has_no_drift_and_seeds_differ():
    from src.data.loader import load_production, load_reference

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    std = reference_std(X_ref)
    a = list(stream(load_scenario("none"), X_pool, y_pool, std, seed=1))
    b = list(stream(load_scenario("none"), X_pool, y_pool, std, seed=2))
    assert not any(w.drift_active for w in a)
    assert not a[0].X.equals(b[0].X)

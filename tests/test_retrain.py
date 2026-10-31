import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.monitor.job import get_reference
from src.retrain.flow import RetrainConfig, post_promotion_check, run_check
from src.retrain.policy import (
    PolicyConfig,
    PolicyState,
    decide,
    load_state,
    on_retrain_done,
    save_state,
)
from src.retrain.reference import restore_reference, set_reference
from src.retrain.rollback import promote, rollback_alias
from src.retrain.validate import ValidationConfig, compare
from tests.test_analysis import make_frame

CFG = PolicyConfig(min_agreeing=2, medium_persistence=3, cooldown_windows=2)


def win(i, severity, agreeing=3):
    return {"window": i, "severity": severity, "n_agreeing": agreeing, "first_id": i * 1000 + 1}


# ---------- policy ----------


def test_no_drift_never_triggers():
    state = PolicyState()
    assert not any(decide(win(i, "none"), state, CFG).trigger for i in range(5))


def test_high_with_agreement_triggers_once():
    state = PolicyState()
    assert not decide(win(0, "none"), state, CFG).trigger
    d = decide(win(1, "high"), state, CFG)
    assert d.trigger and state.pending
    assert state.drift_first_id == 1001
    assert not decide(win(2, "high"), state, CFG).trigger  # already pending


def test_medium_needs_three_consecutive_windows():
    state = PolicyState()
    assert not decide(win(0, "medium"), state, CFG).trigger
    assert not decide(win(1, "medium"), state, CFG).trigger
    assert decide(win(2, "medium"), state, CFG).trigger
    assert state.drift_first_id == 1


def test_interrupted_medium_streak_resets():
    state = PolicyState()
    for i, sev in enumerate(["medium", "medium", "none", "medium", "medium"]):
        assert not decide(win(i, sev), state, CFG).trigger


def test_single_detector_is_not_enough():
    state = PolicyState()
    d = decide(win(0, "high", agreeing=1), state, CFG)
    assert not d.trigger and "agree" in d.reason
    assert decide(win(1, "high", agreeing=2), state, CFG).trigger
    assert state.drift_first_id == 1  # streak began at the first drifted window


def test_cooldown_blocks_then_releases():
    state = PolicyState()
    assert decide(win(0, "high"), state, CFG).trigger
    on_retrain_done(state, CFG)
    assert not state.pending and state.cooldown_left == 2
    assert decide(win(1, "high"), state, CFG).reason.startswith("cooldown")
    assert decide(win(2, "high"), state, CFG).reason.startswith("cooldown")
    assert decide(win(3, "high"), state, CFG).trigger


def test_policy_state_roundtrip(tmp_path):
    state = PolicyState(last_window=4, pending=True, drift_first_id=9)
    save_state(tmp_path / "s.json", state)
    assert load_state(tmp_path / "s.json") == state
    assert load_state(tmp_path / "missing.json") == PolicyState()


# ---------- validation ----------


class Rule:
    def __init__(self, fn, delay=0.0):
        self.fn, self.delay = fn, delay

    def predict_proba(self, X):
        if self.delay:
            time.sleep(self.delay)
        p = self.fn(X).astype(float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.integers(0, 2, 200)})
    return X, pd.Series(X["x"].to_numpy())


def test_better_challenger_is_promoted(data):
    X, y = data
    res = compare(Rule(lambda d: d["x"] * 0), Rule(lambda d: d["x"]), X, y)
    assert res.promote and res.f1_gain > 0.5


def test_equal_challenger_is_rejected(data):
    X, y = data
    res = compare(Rule(lambda d: d["x"]), Rule(lambda d: d["x"]), X, y)
    assert not res.promote and "below the required" in res.reason


def test_gain_below_margin_is_rejected(data):
    X, y = data
    res = compare(Rule(lambda d: d["x"] * 0), Rule(lambda d: d["x"]), X, y, ValidationConfig(2.0))
    assert not res.promote


def test_slow_challenger_is_rejected_even_if_better(data):
    X, y = data
    res = compare(Rule(lambda d: d["x"] * 0), Rule(lambda d: d["x"], delay=0.1), X, y)
    assert res.f1_gain > 0.5 and not res.promote and "latency" in res.reason


# ---------- reference swap ----------


def test_set_and_restore_reference(tmp_path):
    set_reference(make_frame(100, 1), tmp_path)
    assert not (tmp_path / "reference_previous.parquet").exists()
    set_reference(make_frame(200, 2), tmp_path)
    assert (tmp_path / "reference_previous.parquet").exists()
    restore_reference(tmp_path)
    assert len(pd.read_parquet(tmp_path / "reference_current.parquet")) == 100
    restore_reference(tmp_path)
    assert not (tmp_path / "reference_current.parquet").exists()


def test_monitor_get_reference_follows_the_file(tmp_path):
    base = make_frame(50, 0)
    assert get_reference(tmp_path, base) is base
    set_reference(make_frame(300, 1), tmp_path)
    assert len(get_reference(tmp_path, base)) == 300
    set_reference(make_frame(500, 2), tmp_path)
    assert len(get_reference(tmp_path, base)) == 500
    restore_reference(tmp_path)
    assert len(get_reference(tmp_path, base)) == 300
    restore_reference(tmp_path)
    assert get_reference(tmp_path, base) is base


# ---------- registry aliases ----------


class FakeClient:
    def __init__(self, aliases, versions=("1", "2")):
        self.aliases, self.versions, self.tags = dict(aliases), set(versions), {}

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException(f"alias {alias} not found")
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        assert isinstance(version, str)
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException("missing")
        del self.aliases[alias]

    def get_model_version(self, name, version):
        if version not in self.versions:
            raise MlflowException("no such version")
        return SimpleNamespace(version=version)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


def test_promote_moves_aliases():
    c = FakeClient({"champion": "1", "challenger": "2"})
    assert promote(c, "2") == "1"
    assert c.aliases == {"champion": "2", "previous": "1"}


def test_rollback_restores_previous_version():
    c = FakeClient({"champion": "2", "previous": "1"})
    assert rollback_alias(c) == {"from": "2", "to": "1", "changed": True}
    assert c.aliases == {"champion": "1"}
    assert c.tags[("2", "status")] == "rolled_back"


def test_rollback_edge_cases():
    with pytest.raises(RuntimeError):
        rollback_alias(FakeClient({"champion": "1"}))
    with pytest.raises(MlflowException):
        rollback_alias(FakeClient({"champion": "1"}), to_version="9")
    assert rollback_alias(FakeClient({"champion": "1"}), to_version=1)["changed"] is False


# ---------- check cycle ----------


def write_windows(monitor_dir, severities):
    d = monitor_dir / "windows"
    d.mkdir(parents=True, exist_ok=True)
    for i, sev in enumerate(severities):
        (d / f"window_{i:05d}.json").write_text(json.dumps(win(i, sev)))


def test_check_waits_for_data_then_retrains_once(tmp_path):
    write_windows(tmp_path, ["none", "none", "high"])
    available = {"n": 1000}
    calls = []

    def fetch(after_id=0, limit=6000):
        return pd.DataFrame({"id": range(available["n"])})

    def retrain(rows, trigger):
        calls.append(trigger)
        return {"promoted": True}

    rcfg = RetrainConfig(min_rows=2000)
    first = run_check(tmp_path, CFG, rcfg, fetch, retrain)
    assert "waiting" in first and not calls
    assert load_state(tmp_path / "policy_state.json").pending

    available["n"] = 2500
    second = run_check(tmp_path, CFG, rcfg, fetch, retrain)
    assert second["retrain"] == {"promoted": True} and len(calls) == 1
    assert calls[0]["drift_first_id"] == 2001
    state = load_state(tmp_path / "policy_state.json")
    assert not state.pending and state.cooldown_left == 2

    assert "retrain" not in run_check(tmp_path, CFG, rcfg, fetch, retrain)  # nothing new


def test_check_resets_when_monitor_numbering_restarts(tmp_path):
    save_state(tmp_path / "policy_state.json", PolicyState(last_window=9, cooldown_left=3))
    write_windows(tmp_path, ["none", "none"])
    report = run_check(tmp_path, CFG, RetrainConfig(), lambda **_: pd.DataFrame(), lambda *_: {})
    assert report["new_windows"] == 2
    assert load_state(tmp_path / "policy_state.json").last_window == 1


# ---------- post-promotion verification ----------


def promotion_record(monitor_dir):
    rec = {"status": "monitoring", "version": "2", "champion_f1": 0.5}
    (monitor_dir / "promotion.json").write_text(json.dumps(rec))


def live_rows(n, correct):
    label = np.tile([0, 1], n // 2)
    pred = label if correct else np.zeros(n, dtype=int)
    return pd.DataFrame({"model_version": "2", "label": label, "prediction": pred})


def test_post_check_waits_for_enough_rows(tmp_path):
    promotion_record(tmp_path)
    out = post_promotion_check(tmp_path, lambda **_: live_rows(400, True), lambda reason: None)
    assert out["status"] == "waiting"


def test_post_check_confirms_a_good_model(tmp_path):
    promotion_record(tmp_path)
    out = post_promotion_check(tmp_path, lambda **_: live_rows(1200, True), lambda reason: None)
    assert out["status"] == "confirmed"
    assert json.loads((tmp_path / "promotion.json").read_text())["status"] == "confirmed"


def test_post_check_rolls_back_a_worse_model(tmp_path):
    promotion_record(tmp_path)
    reasons = []
    out = post_promotion_check(
        tmp_path, lambda **_: live_rows(1200, False), lambda reason: reasons.append(reason)
    )
    assert out["status"] == "rolled_back" and len(reasons) == 1
    assert post_promotion_check(tmp_path, lambda **_: live_rows(1200, True), reasons.append) is None

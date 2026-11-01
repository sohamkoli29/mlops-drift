"""Drift injected -> alert -> policy trigger -> promotion -> API serves v2 -> drift cleared."""

import json

import pytest
from fastapi.testclient import TestClient

from sim.simulator import load_scenario, reference_std, stream
from src.api.main import create_app
from src.api.model_store import LoadedModel
from src.api.request_logger import RequestLogger
from src.config import FEATURES, MODEL_NAME, PRODUCTION_FILE, REFERENCE_FILE
from src.data.loader import clean_frame, load_production, load_reference
from src.monitor.job import get_reference, process_available
from src.retrain.flow import RetrainConfig, run_check
from src.retrain.policy import PolicyConfig, load_state
from src.retrain.reference import set_reference
from src.retrain.rollback import promote
from tests.test_api import EXAMPLE, FakeModel
from tests.test_retrain import FakeClient

pytestmark = pytest.mark.skipif(
    not (REFERENCE_FILE.exists() and PRODUCTION_FILE.exists()), reason="run split_data.py first"
)


def send_window(api: TestClient, w) -> None:
    records = json.loads(w.X.to_json(orient="records"))
    body = api.post("/predict/batch", json={"records": records}).json()
    first = body["first_log_id"]
    ids = list(range(first, first + len(records)))
    r = api.post("/labels", json={"ids": ids, "labels": [int(v) for v in w.y]})
    assert r.status_code == 200


def test_drift_to_promotion_to_recovery(tmp_path):
    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    scenario = load_scenario("sudden_concept")  # drift starts at window 8
    windows = list(stream(scenario, X_pool, y_pool, reference_std(X_ref), seed=3))

    registry = FakeClient({"champion": "1", "challenger": "2"})

    def loader():
        version = registry.get_model_version_by_alias(MODEL_NAME, "champion").version
        return LoadedModel(model=FakeModel(), version=str(version))

    logger = RequestLogger(tmp_path / "api.db")
    mon = tmp_path / "monitor"

    with TestClient(create_app(model_loader=loader, request_logger=logger)) as api:
        assert api.get("/health").json()["model_version"] == "1"

        # 1. traffic: 8 clean windows, then 4 drifted ones, with delayed labels
        for w in windows[:12]:
            send_window(api, w)

        # 2. the monitor sees the onset exactly at window 8
        analyses = process_available(logger, X_ref, mon, window_size=1000)
        severities = [a["severity"] for a in analyses]
        assert [a["window"] for a in analyses] == list(range(12))
        assert "high" not in severities[:8]
        assert severities[8:] == ["high"] * 4
        assert {"age", "hours_per_week"} <= {t["feature"] for t in analyses[8]["top_features"]}

        # 3. policy triggers once; the retrain step promotes v2 and reloads the API
        calls = []

        def fetch(after_id=0, limit=6000):
            return logger.read_labeled(after_id=after_id, limit=limit)

        def retrain(rows, trigger):
            calls.append(trigger)
            assert len(rows) >= 3000
            assert promote(registry, "2") == "1"
            reload = api.post("/reload").json()
            assert reload == {"previous_version": "1", "current_version": "2", "changed": True}
            set_reference(clean_frame(rows[FEATURES]), mon)
            return {"promoted": True}

        report = run_check(mon, PolicyConfig(), RetrainConfig(min_rows=3000), fetch, retrain)
        assert report["retrain"] == {"promoted": True}
        assert len(calls) == 1
        assert calls[0]["window"] == 8 and calls[0]["drift_first_id"] == 8001
        assert load_state(mon / "policy_state.json").cooldown_left == 3

        # 4. after the swap the monitor compares against the new reference: drift is absorbed
        send_window(api, windows[12])
        new_reference = get_reference(mon, X_ref)
        assert len(new_reference) >= 3000
        (after,) = process_available(logger, new_reference, mon, window_size=1000)
        assert after["window"] == 12
        assert after["severity"] != "high"
        psi_flagged = after["detectors"]["psi"]["flagged_features"]
        assert "age" not in psi_flagged and "hours_per_week" not in psi_flagged

        # 5. the API now serves the promoted version
        assert api.post("/predict", json=EXAMPLE).json()["model_version"] == "2"

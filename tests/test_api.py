import itertools

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model_store import LoadedModel
from src.api.request_logger import RequestLogger
from src.api.schemas import PredictRequest

EXAMPLE = PredictRequest.model_config["json_schema_extra"]["example"]


class FakeModel:
    """age >= 40 -> probability 0.9, else 0.1."""

    def predict_proba(self, X):
        p = np.where(X["age"].to_numpy() >= 40, 0.9, 0.1)
        return np.column_stack([1 - p, p])


def make_loader(versions=("1", "2")):
    counter = itertools.count()

    def loader():
        i = min(next(counter), len(versions) - 1)
        return LoadedModel(model=FakeModel(), version=versions[i])

    return loader


def failing_loader():
    raise RuntimeError("mlflow down")


@pytest.fixture
def client(tmp_path):
    logger = RequestLogger(tmp_path / "pred.db")
    app = create_app(model_loader=make_loader(), request_logger=logger)
    with TestClient(app) as c:
        c.logger = logger
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "1"}


def test_predict_and_logging(client):
    r = client.post("/predict", json={**EXAMPLE, "age": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["probability"] == pytest.approx(0.9)
    assert body["model_version"] == "1"

    assert client.logger.count() == 1
    row = client.logger.read().iloc[0]
    assert row["age"] == 50
    assert row["model_version"] == "1"
    assert row["prediction"] == 1


def test_null_categorical_is_accepted_and_logged_as_null(client):
    r = client.post("/predict", json={**EXAMPLE, "occupation": None})
    assert r.status_code == 200
    assert client.logger.read()["occupation"].isna().all()


def test_all_null_categoricals_stay_object_dtype():
    nulls = {k: None for k in ["workclass", "occupation", "native_country"]}
    df = PredictRequest(**{**EXAMPLE, **nulls}).to_frame()
    assert df["workclass"].dtype == object
    assert df["workclass"].isna().all()


def test_invalid_input_rejected_and_not_logged(client):
    assert client.post("/predict", json={**EXAMPLE, "age": -5}).status_code == 422
    assert client.logger.count() == 0


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "label": 1}).status_code == 422


def test_batch(client):
    records = [{**EXAMPLE, "age": a} for a in (25, 45, 60)]
    r = client.post("/predict/batch", json={"records": records})
    assert r.status_code == 200
    assert r.json()["predictions"] == [0, 1, 1]
    assert client.logger.count() == 3


def test_reload_swaps_model(client):
    assert client.post("/predict", json=EXAMPLE).json()["model_version"] == "1"
    r = client.post("/reload")
    assert r.json() == {"previous_version": "1", "current_version": "2", "changed": True}
    assert client.post("/predict", json=EXAMPLE).json()["model_version"] == "2"
    assert client.get("/model-info").json()["version"] == "2"


def test_failed_reload_keeps_current_model(tmp_path):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("registry unreachable")
        return LoadedModel(model=FakeModel(), version="1")

    app = create_app(model_loader=flaky, request_logger=RequestLogger(tmp_path / "p.db"))
    with TestClient(app) as c:
        assert c.post("/reload").status_code == 502
        assert c.get("/health").json()["model_version"] == "1"
        assert c.post("/predict", json=EXAMPLE).status_code == 200


def test_no_model_returns_503(tmp_path):
    app = create_app(model_loader=failing_loader, request_logger=RequestLogger(tmp_path / "p.db"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=EXAMPLE).status_code == 503

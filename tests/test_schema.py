import pytest
from pydantic import ValidationError

from src.api.schemas import PredictRequest
from src.config import FEATURES


def test_schema_fields_match_features():
    assert list(PredictRequest.model_fields) == FEATURES


def test_example_roundtrip_to_frame():
    example = PredictRequest.model_config["json_schema_extra"]["example"]
    df = PredictRequest(**example).to_frame()
    assert list(df.columns) == FEATURES
    assert len(df) == 1


def test_invalid_age_rejected():
    example = dict(PredictRequest.model_config["json_schema_extra"]["example"])
    example["age"] = -5
    with pytest.raises(ValidationError):
        PredictRequest(**example)


def test_null_categorical_allowed():
    example = dict(PredictRequest.model_config["json_schema_extra"]["example"])
    example["occupation"] = None
    assert PredictRequest(**example).to_frame()["occupation"].isna().all()

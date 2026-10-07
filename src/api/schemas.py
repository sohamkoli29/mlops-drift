"""Request/response schemas for the inference API."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import FEATURES


class PredictRequest(BaseModel):
    """One census record. Categorical fields may be null (imputed in the pipeline)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 39,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native_country": "United-States",
            }
        },
    )

    # numeric
    age: int = Field(ge=0, le=120)
    education_num: int = Field(ge=1, le=16)
    capital_gain: int = Field(ge=0)
    capital_loss: int = Field(ge=0)
    hours_per_week: int = Field(ge=0, le=168)

    # categorical
    workclass: str | None = None
    education: str | None = None
    marital_status: str | None = None
    occupation: str | None = None
    relationship: str | None = None
    race: str | None = None
    sex: str | None = None
    native_country: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Single-row DataFrame in the exact column order the pipeline expects."""
        row = {k: (np.nan if v is None else v) for k, v in self.model_dump().items()}
        return pd.DataFrame([row])[FEATURES]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int  # 1 = income > 50K
    probability: float
    model_version: str

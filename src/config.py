"""Central config: paths, seed, target and feature definitions."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
HOLDOUT_DIR = DATA_DIR / "holdout"
REPORTS_DIR = ROOT / "reports"

RAW_FILE = RAW_DIR / "adult.csv"
REFERENCE_FILE = REFERENCE_DIR / "reference.parquet"
PRODUCTION_FILE = PRODUCTION_DIR / "production_pool.parquet"
HOLDOUT_FILE = HOLDOUT_DIR / "holdout.parquet"

SEED = 42

# Split sizes: holdout 20%, remaining 80% divided equally
HOLDOUT_FRACTION = 0.20
REFERENCE_FRACTION_OF_REST = 0.50  # -> 40% reference, 40% production pool

TARGET = "income"  # 1 if income > 50K else 0

NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Dropped: fnlwgt is a census sampling weight, not a real predictive feature
DROP_COLUMNS = ["fnlwgt"]

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "income-drift"
MODEL_NAME = "income-classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

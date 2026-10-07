"""scikit-learn Pipeline: impute -> scale / one-hot -> classifier."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def get_candidates() -> dict:
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }

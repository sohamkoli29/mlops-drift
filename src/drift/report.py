"""Evidently HTML drift report for one window (optional dependency)."""

from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, SEED


def save_html_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
    max_reference_rows: int = 5000,
) -> Path:
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is not installed. Run: pip install -r requirements-reports.txt"
        ) from exc

    ref = reference[FEATURES]
    if len(ref) > max_reference_rows:
        ref = ref.sample(max_reference_rows, random_state=SEED)

    mapping = ColumnMapping(
        numerical_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=current[FEATURES], column_mapping=mapping)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(path))
    return path

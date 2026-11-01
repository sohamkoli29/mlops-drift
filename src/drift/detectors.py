"""Detector registry, calibrated thresholds and a helper that runs all detectors on a window."""

import json
import os
from pathlib import Path

import pandas as pd

from src.drift.base import Detector
from src.drift.ks import KSDetector
from src.drift.psi import PSI_MEDIUM, PSIDetector
from src.drift.wasserstein import DEFAULT_THRESHOLD, WassersteinDetector

THRESHOLD_FILE = Path(__file__).with_name("thresholds.json")

DEFAULT_THRESHOLDS = {
    "psi": {"threshold": PSI_MEDIUM},
    "ks": {"alpha": 0.05},
    "wasserstein": {"default": DEFAULT_THRESHOLD, "per_feature": {}},
}


def load_thresholds(path=None) -> dict:
    """Calibrated thresholds merged over the defaults.

    DRIFT_THRESHOLDS_FILE overrides the path. A missing file means defaults.
    """
    path = Path(path or os.getenv("DRIFT_THRESHOLDS_FILE", THRESHOLD_FILE))
    merged = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    if path.is_file():
        data = json.loads(path.read_text())
        for key in merged:
            merged[key].update(data.get(key, {}))
    return merged


def get_detectors(thresholds: dict | None = None) -> dict[str, Detector]:
    t = thresholds or load_thresholds()
    w = t["wasserstein"]
    detectors = (
        PSIDetector(t["psi"]["threshold"]),
        KSDetector(t["ks"]["alpha"]),
        WassersteinDetector(w["default"], w["per_feature"]),
    )
    return {d.name: d for d in detectors}


def score_all(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    detectors: dict[str, Detector] | None = None,
) -> pd.DataFrame:
    """One row per (detector, feature): score, flagged, p_value."""
    rows = []
    for name, det in (detectors or get_detectors()).items():
        for feature, res in det.score(reference, current).items():
            rows.append(
                {
                    "detector": name,
                    "feature": feature,
                    "score": res.score,
                    "flagged": res.flagged,
                    "p_value": res.p_value,
                }
            )
    return pd.DataFrame(rows)

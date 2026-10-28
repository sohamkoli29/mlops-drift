"""Detector registry and a helper that runs all of them on one window."""

import pandas as pd

from src.drift.base import Detector
from src.drift.ks import KSDetector
from src.drift.psi import PSIDetector
from src.drift.wasserstein import WassersteinDetector


def get_detectors() -> dict[str, Detector]:
    return {d.name: d for d in (PSIDetector(), KSDetector(), WassersteinDetector())}


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

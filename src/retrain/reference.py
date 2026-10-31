"""Swap the reference the monitor compares against (stored next to the monitor state)."""

import os
from pathlib import Path

import pandas as pd

from src.config import FEATURES, MONITOR_DIR

CURRENT_NAME = "reference_current.parquet"
PREVIOUS_NAME = "reference_previous.parquet"


def set_reference(df: pd.DataFrame, monitor_dir=MONITOR_DIR) -> Path:
    """Make df the monitor's reference; keep the one it replaces so a rollback can restore it."""
    d = Path(monitor_dir)
    d.mkdir(parents=True, exist_ok=True)
    current, previous, tmp = d / CURRENT_NAME, d / PREVIOUS_NAME, d / "reference_new.tmp"
    df[FEATURES].reset_index(drop=True).to_parquet(tmp, index=False)
    if current.exists():
        os.replace(current, previous)
    else:
        previous.unlink(missing_ok=True)  # the base reference was active; restoring = deleting
    os.replace(tmp, current)
    return current


def restore_reference(monitor_dir=MONITOR_DIR) -> None:
    d = Path(monitor_dir)
    current, previous = d / CURRENT_NAME, d / PREVIOUS_NAME
    if previous.exists():
        os.replace(previous, current)
    else:
        current.unlink(missing_ok=True)

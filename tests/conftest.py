import pytest


@pytest.fixture(autouse=True)
def default_thresholds(monkeypatch):
    """Tests use the built-in defaults, never the calibrated file."""
    monkeypatch.setenv("DRIFT_THRESHOLDS_FILE", "does-not-exist.json")

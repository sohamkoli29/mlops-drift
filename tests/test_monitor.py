import numpy as np
import pandas as pd
from prometheus_client import REGISTRY

from src.api.request_logger import RequestLogger
from src.config import CATEGORICAL_FEATURES
from src.monitor.job import process_available, read_state, reset_to_now, write_state

WINDOW = 500


def make_frame(n, seed, age_shift=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(17, 90, n) + age_shift,
            "education_num": rng.integers(1, 17, n),
            "capital_gain": np.where(rng.random(n) < 0.9, 0, rng.integers(1, 99999, n)),
            "capital_loss": np.where(rng.random(n) < 0.95, 0, rng.integers(1, 4000, n)),
            "hours_per_week": rng.integers(1, 99, n),
        }
    )
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.choice(["a", "b", "c"], n)
    return df


def log_rows(logger: RequestLogger, df: pd.DataFrame) -> None:
    rows = df.to_dict(orient="records")
    logger.log_batch(rows, [0] * len(rows), [0.1] * len(rows), "1", 1.0)


def test_no_full_window_means_no_analysis(tmp_path):
    logger, ref = RequestLogger(tmp_path / "p.db"), make_frame(5000, 0)
    log_rows(logger, make_frame(WINDOW - 1, 1))
    assert process_available(logger, ref, tmp_path, WINDOW) == []
    assert read_state(tmp_path) == {"last_id": 0, "next_window": 0}


def test_windows_are_consecutive_and_state_persists(tmp_path):
    logger, ref = RequestLogger(tmp_path / "p.db"), make_frame(5000, 0)
    log_rows(logger, make_frame(1200, 1))
    out = process_available(logger, ref, tmp_path, WINDOW)
    assert [a["window"] for a in out] == [0, 1]
    assert [a["last_id"] for a in out] == [500, 1000]
    assert read_state(tmp_path)["last_id"] == 1000

    log_rows(logger, make_frame(300, 2))  # 200 pending + 300 new = one more full window
    out2 = process_available(logger, ref, tmp_path, WINDOW)
    assert [a["window"] for a in out2] == [2]
    assert out2[0]["first_id"] == 1001
    assert (tmp_path / "windows" / "window_00002.json").exists()


def test_drift_is_detected_and_published(tmp_path):
    logger, ref = RequestLogger(tmp_path / "p.db"), make_frame(5000, 0)
    log_rows(logger, make_frame(500, 1))
    log_rows(logger, make_frame(500, 2, age_shift=40))
    clean, drifted = process_available(logger, ref, tmp_path, WINDOW)
    assert clean["severity"] != "high"
    assert drifted["severity"] == "high"
    assert drifted["top_features"][0]["feature"] == "age"
    assert REGISTRY.get_sample_value("drift_window_severity") == 2
    psi_age = {"detector": "psi", "feature": "age"}
    assert REGISTRY.get_sample_value("drift_score", psi_age) > 0.25
    assert REGISTRY.get_sample_value("drift_feature_flagged", psi_age) == 1


def test_reset_to_now_skips_old_rows(tmp_path):
    logger, ref = RequestLogger(tmp_path / "p.db"), make_frame(5000, 0)
    log_rows(logger, make_frame(1000, 1))
    reset_to_now(logger, tmp_path)
    assert process_available(logger, ref, tmp_path, WINDOW) == []
    log_rows(logger, make_frame(500, 2))
    assert len(process_available(logger, ref, tmp_path, WINDOW)) == 1


def test_database_wipe_restarts_from_row_one(tmp_path):
    logger, ref = RequestLogger(tmp_path / "p.db"), make_frame(5000, 0)
    write_state(tmp_path, {"last_id": 99999, "next_window": 7})
    log_rows(logger, make_frame(500, 1))
    out = process_available(logger, ref, tmp_path, WINDOW)
    assert len(out) == 1 and out[0]["window"] == 0

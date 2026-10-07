"""Append-only SQLite log of every prediction (the production window source)."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

META_COLUMNS = ["ts", "model_version", "prediction", "probability", "latency_ms"]


def _create_sql() -> str:
    feature_cols = [f"{c} INTEGER" for c in NUMERIC_FEATURES]
    feature_cols += [f"{c} TEXT" for c in CATEGORICAL_FEATURES]
    return (
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, model_version TEXT NOT NULL, "
        "prediction INTEGER NOT NULL, probability REAL NOT NULL, "
        "latency_ms REAL NOT NULL, label INTEGER, " + ", ".join(feature_cols) + ")"
    )


class RequestLogger:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_create_sql())

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def log_batch(
        self,
        features: list[dict],
        predictions: list[int],
        probabilities: list[float],
        model_version: str,
        latency_ms: float,
    ) -> None:
        """Insert one row per record. latency_ms is the call total split across records."""
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        per_record = latency_ms / max(len(features), 1)
        columns = [*META_COLUMNS, *FEATURES]
        rows = [
            (ts, model_version, int(pred), float(proba), per_record)
            + tuple(feat.get(c) for c in FEATURES)
            for feat, pred, proba in zip(features, predictions, probabilities, strict=True)
        ]
        sql = (
            f"INSERT INTO predictions ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})"
        )
        with closing(self._connect()) as conn, conn:
            conn.executemany(sql, rows)

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def read(self, after_id: int = 0, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM predictions WHERE id > ? ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with closing(self._connect()) as conn:
            return pd.read_sql_query(sql, conn, params=(after_id,))

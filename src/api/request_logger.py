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
    ) -> int:
        """Insert one row per record. Returns the id of the first inserted row.

        latency_ms is the call total split across records.
        """
        if not features:
            return 0
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        per_record = latency_ms / len(features)
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
            # still inside the write transaction, so these ids are ours and contiguous
            last_id = conn.execute("SELECT MAX(id) FROM predictions").fetchone()[0]
        return last_id - len(rows) + 1

    def update_labels(self, ids: list[int], labels: list[int]) -> int:
        """Attach ground-truth labels to logged rows. Returns the number of rows updated."""
        data = [(int(v), int(i)) for i, v in zip(ids, labels, strict=True)]
        with closing(self._connect()) as conn, conn:
            cur = conn.executemany("UPDATE predictions SET label = ? WHERE id = ?", data)
            return cur.rowcount

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def max_id(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COALESCE(MAX(id), 0) FROM predictions").fetchone()[0]

    def read(self, after_id: int = 0, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM predictions WHERE id > ? ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with closing(self._connect()) as conn:
            return pd.read_sql_query(sql, conn, params=(after_id,))

    def read_labeled(self, after_id: int = 0, limit: int = 6000) -> pd.DataFrame:
        """The most recent `limit` labeled rows with id > after_id, oldest first."""
        sql = (
            "SELECT * FROM (SELECT * FROM predictions WHERE label IS NOT NULL AND id > ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id"
        )
        with closing(self._connect()) as conn:
            return pd.read_sql_query(sql, conn, params=(int(after_id), int(limit)))

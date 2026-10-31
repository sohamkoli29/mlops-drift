"""Monitoring job: slice the prediction log into windows, analyse each, publish metrics.

Usage:
  python -m src.monitor.job                  # loop forever, serve /metrics on :8001
  python -m src.monitor.job --once           # process what is available, then exit
  python -m src.monitor.job --reset-to-now   # ignore everything logged so far
  python -m src.monitor.job --reset          # forget state, start from row 1
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from prometheus_client import start_http_server

from src.api.request_logger import RequestLogger
from src.config import (
    FEATURES,
    MONITOR_DIR,
    MONITOR_INTERVAL,
    MONITOR_PORT,
    PREDICTION_DB,
    WINDOW_SIZE,
)
from src.data.loader import clean_frame, load_reference
from src.drift.analysis import analyze_window
from src.monitor import metrics_exporter as mx

log = logging.getLogger("monitor")

FRESH_STATE = {"last_id": 0, "next_window": 0}


def _state_file(out_dir) -> Path:
    return Path(out_dir) / "state.json"


def read_state(out_dir) -> dict:
    p = _state_file(out_dir)
    return json.loads(p.read_text()) if p.exists() else dict(FRESH_STATE)


def write_state(out_dir, state: dict) -> None:
    p = _state_file(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, p)


def reset_to_now(logger: RequestLogger, out_dir) -> None:
    write_state(out_dir, {"last_id": logger.max_id(), "next_window": 0})


def process_available(
    logger: RequestLogger,
    reference,
    out_dir,
    window_size: int = WINDOW_SIZE,
    max_windows: int | None = None,
) -> list[dict]:
    """Analyse every full window not yet processed. Returns the analyses (oldest first)."""
    out_dir = Path(out_dir)
    state = read_state(out_dir)
    if logger.max_id() < state["last_id"]:
        log.warning("Log is behind saved state (database was reset); restarting from row 1")
        state = dict(FRESH_STATE)

    analyses: list[dict] = []
    while max_windows is None or len(analyses) < max_windows:
        df = logger.read(after_id=state["last_id"], limit=window_size)
        if len(df) < window_size:
            break
        current = clean_frame(df[FEATURES])
        with mx.CHECK_SECONDS.time():
            a = analyze_window(reference, current, window=state["next_window"])
        a.update(
            first_id=int(df["id"].iloc[0]),
            last_id=int(df["id"].iloc[-1]),
            ts_start=str(df["ts"].iloc[0]),
            ts_end=str(df["ts"].iloc[-1]),
            model_versions=sorted(df["model_version"].unique().tolist()),
            positive_rate=round(float(df["prediction"].mean()), 4),
            mean_probability=round(float(df["probability"].mean()), 4),
        )
        win_dir = out_dir / "windows"
        win_dir.mkdir(parents=True, exist_ok=True)
        (win_dir / f"window_{a['window']:05d}.json").write_text(json.dumps(a, indent=2))

        mx.publish(a)
        state = {"last_id": a["last_id"], "next_window": state["next_window"] + 1}
        write_state(out_dir, state)
        analyses.append(a)
        log.info("window %d: %s", a["window"], a["summary"])

    mx.PENDING_ROWS.set(max(0, logger.max_id() - state["last_id"]))
    mx.LAST_RUN.set(time.time())
    return analyses


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(PREDICTION_DB))
    parser.add_argument("--out", default=str(MONITOR_DIR))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--interval", type=float, default=MONITOR_INTERVAL)
    parser.add_argument("--port", type=int, default=MONITOR_PORT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--reset-to-now", action="store_true")
    args = parser.parse_args()

    logger = RequestLogger(args.db)
    out_dir = Path(args.out)

    if args.reset:
        _state_file(out_dir).unlink(missing_ok=True)
        print("State cleared.")
        return
    if args.reset_to_now:
        reset_to_now(logger, out_dir)
        print(f"State set to current end of log (id {logger.max_id()}).")
        return

    reference, _ = load_reference()
    if args.once:
        done = process_available(logger, reference, out_dir, args.window_size)
        print(f"Processed {len(done)} window(s).")
        return

    start_http_server(args.port)
    log.info("Serving metrics on :%d, reading %s", args.port, args.db)
    while True:
        try:
            process_available(logger, reference, out_dir, args.window_size)
        except Exception:
            log.exception("Monitoring check failed; will retry")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

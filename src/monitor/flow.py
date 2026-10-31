"""Prefect flow around the monitoring job.

Usage:
  python -m src.monitor.flow --db logs\\docker_snapshot\\predictions.db --out reports\\live_flow
  python -m src.monitor.flow --serve --interval 60 --db ... --out ...
"""

import argparse
from pathlib import Path

from src.api.request_logger import RequestLogger
from src.config import PREDICTION_DB, WINDOW_SIZE
from src.data.loader import load_reference
from src.monitor.job import process_available

try:
    from prefect import flow

    HAVE_PREFECT = True
except ImportError:  # run as plain Python when Prefect isn't installed
    HAVE_PREFECT = False

    def flow(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)


@flow(name="drift-monitoring", retries=2, retry_delay_seconds=5, log_prints=True)
def monitoring_flow(
    db: str | None = None,
    out: str = "reports/live_flow",
    window_size: int = WINDOW_SIZE,
    max_windows: int | None = None,
) -> dict:
    logger = RequestLogger(db or PREDICTION_DB)
    reference, _ = load_reference()
    analyses = process_available(logger, reference, Path(out), window_size, max_windows)
    summary = {
        "windows_processed": len(analyses),
        "severities": [a["severity"] for a in analyses],
        "latest": analyses[-1]["summary"] if analyses else "no full window yet",
    }
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None)
    parser.add_argument("--out", default="reports/live_flow")
    parser.add_argument("--serve", action="store_true", help="run on a schedule (needs Prefect)")
    parser.add_argument("--interval", type=int, default=60, help="seconds between runs")
    args = parser.parse_args()

    if args.serve:
        if not HAVE_PREFECT:
            raise SystemExit("Prefect not installed: pip install -r requirements-flows.txt")
        monitoring_flow.serve(
            name="drift-monitoring",
            interval=args.interval,
            parameters={"db": args.db, "out": args.out},
        )
    else:
        monitoring_flow(db=args.db, out=args.out)


if __name__ == "__main__":
    main()

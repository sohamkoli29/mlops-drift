"""Clear live monitoring and retraining state for a fresh demo run.

Afterwards run: docker compose exec monitor python -m src.monitor.job --reset-to-now
"""

import shutil

from src.config import MONITOR_DIR

FILES = [
    "state.json",
    "policy_state.json",
    "promotion.json",
    "retrain_history.jsonl",
    "reference_current.parquet",
    "reference_previous.parquet",
]


def main() -> None:
    shutil.rmtree(MONITOR_DIR / "windows", ignore_errors=True)
    for name in FILES:
        (MONITOR_DIR / name).unlink(missing_ok=True)
    print(f"Cleared live state in {MONITOR_DIR}")


if __name__ == "__main__":
    main()

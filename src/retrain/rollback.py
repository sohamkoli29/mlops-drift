"""Promote / roll back registry aliases, plus a CLI to roll back the live system.

Usage:
  python -m src.retrain.rollback                     # champion <- 'previous' alias, reload API
  python -m src.retrain.rollback --to-version 1      # pin the champion to a specific version
"""

import argparse
import json
from pathlib import Path

import httpx
from mlflow.exceptions import MlflowException

from src.config import (
    API_URL,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODEL_NAME,
    MONITOR_DIR,
    PREVIOUS_ALIAS,
)
from src.retrain.reference import restore_reference
from src.train.registry import alias_version, configure

PROMOTION_FILE = "promotion.json"


def promote(client, new_version) -> str | None:
    """champion -> new_version; the old champion becomes 'previous'. Returns the old version."""
    old = alias_version(client, CHAMPION_ALIAS)
    old_version = str(old.version) if old else None
    if old_version is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS_ALIAS, old_version)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, str(new_version))
    try:
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    except MlflowException:
        pass
    return old_version


def rollback_alias(client, to_version=None) -> dict:
    """Point 'champion' at the previous (or a given) version."""
    current = alias_version(client, CHAMPION_ALIAS)
    current_v = str(current.version) if current else None
    if to_version is not None:
        target = str(to_version)
        client.get_model_version(MODEL_NAME, target)  # raises if the version does not exist
    else:
        prev = alias_version(client, PREVIOUS_ALIAS)
        if prev is None:
            raise RuntimeError("No 'previous' alias, so there is nothing to roll back to")
        target = str(prev.version)

    if current_v == target:
        return {"from": current_v, "to": target, "changed": False}

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, target)
    if current_v is not None:
        client.set_model_version_tag(MODEL_NAME, current_v, "status", "rolled_back")
    try:
        client.delete_registered_model_alias(MODEL_NAME, PREVIOUS_ALIAS)
    except MlflowException:
        pass
    return {"from": current_v, "to": target, "changed": True}


def mark_promotion(monitor_dir, status: str, reason: str) -> None:
    path = Path(monitor_dir) / PROMOTION_FILE
    if path.exists():
        rec = json.loads(path.read_text())
        rec.update(status=status, rollback_reason=reason)
        path.write_text(json.dumps(rec, indent=2))


def perform_rollback(
    api_url: str = API_URL, monitor_dir=MONITOR_DIR, to_version=None, reason: str = "manual"
) -> dict:
    client = configure()
    result = rollback_alias(client, to_version)
    if result["changed"]:
        restore_reference(monitor_dir)
        try:
            r = httpx.post(f"{api_url}/reload", timeout=60)
            r.raise_for_status()
            result["reload"] = r.json()
        except httpx.HTTPError as exc:
            result["reload_error"] = str(exc)
        mark_promotion(monitor_dir, "rolled_back", reason)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-version", default=None)
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--api-url", default=API_URL)
    args = parser.parse_args()
    out = perform_rollback(args.api_url, MONITOR_DIR, args.to_version, args.reason)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

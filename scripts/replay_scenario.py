"""Send simulator windows (and their true labels) to the running API.

Usage: python -m scripts.replay_scenario --scenario sudden_concept --sleep 10 [--seed 42]
"""

import argparse
import json
import time

import httpx
from sklearn.metrics import f1_score

from sim.simulator import load_scenario, reference_std, stream
from src.config import API_URL
from src.data.loader import load_production, load_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sudden")
    parser.add_argument("--windows", type=int, default=0, help="stop after N windows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.0, help="seconds between windows")
    parser.add_argument("--no-labels", action="store_true", help="do not send ground truth")
    args = parser.parse_args()

    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    scenario = load_scenario(args.scenario)

    with httpx.Client(base_url=API_URL, timeout=120) as c:
        for w in stream(scenario, X_pool, y_pool, reference_std(X_ref), seed=args.seed):
            if args.windows and w.index >= args.windows:
                break
            records = json.loads(w.X.to_json(orient="records"))
            r = c.post("/predict/batch", json={"records": records})
            r.raise_for_status()
            body = r.json()
            preds = body["predictions"]

            if not args.no_labels and body.get("first_log_id"):
                first = body["first_log_id"]
                ids = list(range(first, first + len(preds)))
                lr = c.post("/labels", json={"ids": ids, "labels": [int(v) for v in w.y]})
                lr.raise_for_status()

            f1 = f1_score(w.y, preds, zero_division=0)
            print(
                f"window {w.index:2d} | drift={w.drift_active!s:5} | model v{body['model_version']}"
                f" | positive rate {sum(preds) / len(preds):.3f} | live F1 {f1:.3f}"
            )
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

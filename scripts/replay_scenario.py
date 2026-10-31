"""Send simulator windows to the running API.

Usage: python -m scripts.replay_scenario --scenario sudden --windows 14 [--seed 42] [--sleep 3]
"""

import argparse
import json
import time

import httpx

from sim.simulator import load_scenario, reference_std, stream
from src.config import API_URL
from src.data.loader import load_production, load_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="sudden")
    parser.add_argument("--windows", type=int, default=0, help="stop after N windows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.0, help="seconds between windows")
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
            preds = r.json()["predictions"]
            print(
                f"window {w.index:2d} sent | drift_active={w.drift_active!s:5} "
                f"intensity={w.intensity:.2f} | positive rate {sum(preds) / len(preds):.3f}"
            )
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

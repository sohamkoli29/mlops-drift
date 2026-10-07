"""Smoke-test the running API. Usage: python scripts/call_api.py [--n 200]"""

import argparse
import json

import httpx

from src.api.schemas import PredictRequest
from src.config import API_URL
from src.data.loader import load_production


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="also replay N production rows")
    args = parser.parse_args()

    with httpx.Client(base_url=API_URL, timeout=30) as c:
        print("health    :", c.get("/health").json())
        info = c.get("/model-info").json()
        print("model-info:", info["version"], info["algorithm"], info["holdout_metrics"])

        example = PredictRequest.model_config["json_schema_extra"]["example"]
        print("predict   :", c.post("/predict", json=example).json())

        if args.n:
            X, _ = load_production()
            records = json.loads(X.head(args.n).to_json(orient="records"))
            r = c.post("/predict/batch", json={"records": records})
            r.raise_for_status()
            preds = r.json()["predictions"]
            print(f"batch     : {len(preds)} rows, positive rate {sum(preds) / len(preds):.3f}")


if __name__ == "__main__":
    main()

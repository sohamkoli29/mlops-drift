"""Write 500 real production rows for the load test."""

import json
from pathlib import Path

from src.data.loader import load_production

OUT = Path(__file__).resolve().parents[1] / "load" / "samples.json"


def main() -> None:
    X, _ = load_production()
    records = json.loads(X.head(500).to_json(orient="records"))
    OUT.write_text(json.dumps(records))
    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()

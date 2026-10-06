"""Split raw data into reference (train), production pool and holdout.

40% reference  -> training set AND drift reference
40% production -> replayed by the simulator in windows (drift injected later)
20% holdout    -> champion vs challenger validation
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DROP_COLUMNS,
    HOLDOUT_DIR,
    HOLDOUT_FILE,
    HOLDOUT_FRACTION,
    PRODUCTION_DIR,
    PRODUCTION_FILE,
    RAW_FILE,
    REFERENCE_DIR,
    REFERENCE_FILE,
    REFERENCE_FRACTION_OF_REST,
    SEED,
    TARGET,
)


def main() -> None:
    df = pd.read_csv(RAW_FILE).drop(columns=DROP_COLUMNS)

    rest, holdout = train_test_split(
        df, test_size=HOLDOUT_FRACTION, stratify=df[TARGET], random_state=SEED
    )
    reference, production = train_test_split(
        rest,
        train_size=REFERENCE_FRACTION_OF_REST,
        stratify=rest[TARGET],
        random_state=SEED,
    )

    for d in (REFERENCE_DIR, PRODUCTION_DIR, HOLDOUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    reference.reset_index(drop=True).to_parquet(REFERENCE_FILE, index=False)
    production.reset_index(drop=True).to_parquet(PRODUCTION_FILE, index=False)
    holdout.reset_index(drop=True).to_parquet(HOLDOUT_FILE, index=False)

    summary = {
        name: {
            "rows": int(len(part)),
            "positive_rate": round(float(part[TARGET].mean()), 4),
        }
        for name, part in [
            ("reference", reference),
            ("production_pool", production),
            ("holdout", holdout),
        ]
    }
    summary["windows_of_1000_in_production"] = len(production) // 1000

    with open(HOLDOUT_DIR.parent / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

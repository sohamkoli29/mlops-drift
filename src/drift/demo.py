"""Run all detectors on a clean window and on an artificially drifted one."""

import pandas as pd

from src.data.loader import load_production, load_reference
from src.drift.detectors import score_all


def main() -> None:
    pd.set_option("display.width", 120)
    X_ref, _ = load_reference()
    X_prod, _ = load_production()

    clean = X_prod.iloc[:1000].copy()
    drifted = clean.copy()
    drifted["age"] = drifted["age"] + 10
    drifted["hours_per_week"] = drifted["hours_per_week"] + 8
    drifted.loc[drifted.index[:400], "marital_status"] = "Divorced"

    for name, window in [("CLEAN window", clean), ("DRIFTED window", drifted)]:
        table = score_all(X_ref, window)
        print(f"\n=== {name} ===")
        wide = table.pivot(index="feature", columns="detector", values="score").round(3)
        flags = table.pivot(index="feature", columns="detector", values="flagged")
        wide["flagged_by"] = [
            ",".join(d for d in ("psi", "ks", "wasserstein") if flags.loc[f].get(d) is True)
            for f in wide.index
        ]
        print(wide.to_string())


if __name__ == "__main__":
    main()

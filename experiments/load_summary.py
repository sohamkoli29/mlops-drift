"""Compare Locust runs. Usage: python -m experiments.load_summary reports/load/with_monitor ..."""

import sys

import pandas as pd

COLS = ["Request Count", "Failure Count", "Requests/s", "50%", "95%", "99%"]


def main(prefixes: list[str]) -> None:
    rows = []
    for prefix in prefixes:
        stats = pd.read_csv(f"{prefix}_stats.csv").set_index("Name")
        for name in ("/predict", "/predict/batch", "Aggregated"):
            if name in stats.index:
                rows.append(
                    {"run": prefix.split("/")[-1], "endpoint": name, **stats.loc[name, COLS]}
                )
    out = pd.DataFrame(rows).round(2)
    out = out.rename(columns={"50%": "p50 ms", "95%": "p95 ms", "99%": "p99 ms"})
    print(out.to_string(index=False))
    out.to_csv("experiments/results/load.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])

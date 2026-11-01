"""Time each detector and the full window analysis on 1,000-row windows."""

import time
from itertools import islice

import pandas as pd

from sim.simulator import load_scenario, reference_std, stream
from src.config import ROOT
from src.data.loader import load_production, load_reference
from src.drift.analysis import analyze_window
from src.drift.detectors import get_detectors

N_WINDOWS = 10


def main() -> None:
    X_ref, _ = load_reference()
    X_pool, y_pool = load_production()
    scenario = load_scenario("none")
    windows = list(islice(stream(scenario, X_pool, y_pool, reference_std(X_ref)), N_WINDOWS))

    rows = []
    for name, det in get_detectors().items():
        det.score(X_ref, windows[0].X)  # warm-up
        t = []
        for w in windows:
            start = time.perf_counter()
            det.score(X_ref, w.X)
            t.append((time.perf_counter() - start) * 1000)
        rows.append({"step": name, "ms_per_window": sum(t) / len(t), "max_ms": max(t)})

    t = []
    for w in windows:
        start = time.perf_counter()
        analyze_window(X_ref, w.X, window=w.index)
        t.append((time.perf_counter() - start) * 1000)
    rows.append(
        {
            "step": "full analysis (3 detectors + severity + top-k)",
            "ms_per_window": sum(t) / len(t),
            "max_ms": max(t),
        }
    )

    out = pd.DataFrame(rows).round(1)
    out.to_csv(ROOT / "experiments" / "results" / "overhead.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nOne analysis per 1,000 predictions: about {out.ms_per_window.iloc[-1]:.0f} ms")


if __name__ == "__main__":
    main()

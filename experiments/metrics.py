"""Window-level detection metrics: confusion counts, rates, delay, aggregation."""

import numpy as np
import pandas as pd

NAN = float("nan")


def confusion(truth, alert) -> dict:
    t, a = np.asarray(truth, dtype=bool), np.asarray(alert, dtype=bool)
    return {
        "tp": int((t & a).sum()),
        "fp": int((~t & a).sum()),
        "fn": int((t & ~a).sum()),
        "tn": int((~t & ~a).sum()),
    }


def _div(n: float, d: float) -> float:
    return n / d if d else NAN


def rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0,
        "false_alarm_rate": _div(fp, fp + tn),
    }


def detection_delay(g: pd.DataFrame) -> float:
    """Windows between the true onset and the first alert at or after it (NaN = missed)."""
    drift = g[g.drift_active]
    if drift.empty:
        return NAN
    onset = drift.window.min()
    hits = g[(g.window >= onset) & g.alert]
    return float(hits.window.min() - onset) if len(hits) else NAN


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, detector, seed). df needs scenario, detector, seed, window,
    drift_active and alert."""
    rows = []
    for (scenario, detector, seed), g in df.groupby(["scenario", "detector", "seed"], sort=False):
        c = confusion(g.drift_active, g.alert)
        rows.append(
            {
                "scenario": scenario,
                "detector": detector,
                "seed": seed,
                **c,
                **rates(**c),
                "delay": detection_delay(g),
            }
        )
    return pd.DataFrame(rows)


def pooled_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Counts summed over scenarios, per (detector, seed), then precision, recall, F1, MCC."""
    summed = runs.groupby(["detector", "seed"], sort=False)[["tp", "fp", "fn", "tn"]].sum()
    rows = []
    for (detector, seed), r in summed.iterrows():
        counts = {k: int(r[k]) for k in ("tp", "fp", "fn", "tn")}
        rows.append({"detector": detector, "seed": seed, **counts, **rates(**counts)})
    return pd.DataFrame(rows)


def by_scenario(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per (scenario, detector)."""
    rows = []
    for (scenario, detector), g in runs.groupby(["scenario", "detector"], sort=False):
        has_drift = bool(((g.tp + g.fn) > 0).any())
        rows.append(
            {
                "scenario": scenario,
                "detector": detector,
                "false_alarm_rate": g.false_alarm_rate.mean(),
                "recall": g.recall.mean() if has_drift else NAN,
                "delay_mean": g.delay.mean(),
                "delay_std": g.delay.std(),
                "missed_runs": int(g.delay.isna().sum()) if has_drift else 0,
                "runs": len(g),
            }
        )
    return pd.DataFrame(rows)


def mean_std(pooled: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "mcc", "false_alarm_rate"]
    g = pooled.groupby("detector", sort=False)[cols]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)

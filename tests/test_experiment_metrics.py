import math

import pandas as pd

from experiments.metrics import (
    by_scenario,
    confusion,
    detection_delay,
    per_run,
    pooled_metrics,
    rates,
)


def windows(scenario, drift, alerts, detector="psi", seed=0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": scenario,
            "detector": detector,
            "seed": seed,
            "window": range(len(drift)),
            "drift_active": drift,
            "alert": alerts,
        }
    )


def test_confusion_and_rates():
    c = confusion([0, 0, 1, 1], [0, 1, 1, 0])
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    r = rates(**c)
    assert r["precision"] == r["recall"] == r["f1"] == r["false_alarm_rate"] == 0.5
    assert r["mcc"] == 0.0


def test_perfect_detector_and_undefined_rates():
    assert rates(5, 0, 0, 5)["mcc"] == 1.0
    r = rates(0, 0, 0, 10)  # no drift windows, no alerts
    assert math.isnan(r["precision"]) and math.isnan(r["recall"])
    assert r["false_alarm_rate"] == 0.0


def test_delay_ignores_alerts_before_onset_and_marks_misses():
    drift = [False, False, False, True, True, True]
    early = windows("s", drift, [False, True, False, False, False, True])
    assert detection_delay(early) == 2.0  # onset 3, first alert at/after = 5
    assert math.isnan(detection_delay(windows("s", drift, [True] + [False] * 5)))
    assert math.isnan(detection_delay(windows("none", [False] * 4, [True] * 4)))


def test_pooled_sums_counts_over_scenarios_and_by_scenario_means():
    none = windows("none", [False] * 4, [True, False, False, False])
    sudden = windows("sudden", [False, False, True, True], [False, False, True, True])
    runs = per_run(pd.concat([none, sudden]))
    pooled = pooled_metrics(runs).iloc[0]
    assert (pooled.tp, pooled.fp, pooled.fn, pooled.tn) == (2, 1, 0, 5)
    assert pooled.recall == 1.0 and abs(pooled.precision - 2 / 3) < 1e-9

    table = by_scenario(runs).set_index("scenario")
    assert table.loc["none", "false_alarm_rate"] == 0.25
    assert table.loc["sudden", "delay_mean"] == 0.0
    assert table.loc["sudden", "missed_runs"] == 0

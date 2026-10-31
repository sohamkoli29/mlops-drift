"""Prometheus metrics for drift monitoring."""

from prometheus_client import Counter, Gauge, Histogram

SEVERITY_VALUE = {"none": 0, "medium": 1, "high": 2}

DRIFT_SCORE = Gauge(
    "drift_score", "Per-feature drift score, latest window", ["detector", "feature"]
)
DRIFT_FLAGGED = Gauge(
    "drift_feature_flagged", "1 if the detector flagged the feature", ["detector", "feature"]
)
WINDOW_SEVERITY = Gauge("drift_window_severity", "Latest window severity: 0 none, 1 medium, 2 high")
DETECTOR_SEVERITY = Gauge(
    "drift_detector_severity", "Per-detector severity: 0 none, 1 medium, 2 high", ["detector"]
)
SHARE_FLAGGED = Gauge("drift_share_flagged", "Share of features flagged by PSI")
MAX_PSI = Gauge("drift_max_psi", "Highest per-feature PSI in the latest window")
AGREEING = Gauge("drift_detectors_agreeing", "Detectors with severity above none")
LAST_WINDOW = Gauge("drift_last_window_id", "Index of the latest processed window")
PENDING_ROWS = Gauge("drift_pending_rows", "Logged rows not yet part of a full window")
LAST_RUN = Gauge("drift_monitor_last_run_timestamp_seconds", "Unix time of the last check")
WINDOWS_TOTAL = Counter("drift_windows_processed_total", "Windows analysed", ["severity"])
CHECK_SECONDS = Histogram(
    "drift_check_seconds",
    "Time to analyse one window",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)


def publish(a: dict) -> None:
    """Push one analysed window (output of analyze_window) into the gauges."""
    for det, feats in a["feature_scores"].items():
        for feature, r in feats.items():
            DRIFT_SCORE.labels(det, feature).set(r["score"])
            DRIFT_FLAGGED.labels(det, feature).set(1 if r["flagged"] else 0)
    for det, d in a["detectors"].items():
        DETECTOR_SEVERITY.labels(det).set(SEVERITY_VALUE[d["severity"]])
    WINDOW_SEVERITY.set(SEVERITY_VALUE[a["severity"]])
    SHARE_FLAGGED.set(a["share_flagged"])
    MAX_PSI.set(a["max_psi"] or 0.0)
    AGREEING.set(a["n_agreeing"])
    LAST_WINDOW.set(a["window"])
    WINDOWS_TOTAL.labels(a["severity"]).inc()

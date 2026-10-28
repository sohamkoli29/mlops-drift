"""Prometheus metrics for the inference API."""

from collections import Counter as Tally

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
ERRORS = Counter("api_errors_total", "HTTP 5xx responses", ["endpoint", "status"])

PREDICTIONS = Counter(
    "predictions_total", "Predictions by model version and class", ["model_version", "prediction"]
)
PROBABILITY = Histogram(
    "prediction_probability",
    "Predicted probability of income > 50K",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_VERSION = Gauge("api_model_version", "Version number of the loaded model (0 = none)")
RELOADS = Counter("api_model_reloads_total", "Model reload attempts", ["result"])


def observe_request(endpoint: str, method: str, status: int, seconds: float) -> None:
    REQUESTS.labels(endpoint, method, str(status)).inc()
    LATENCY.labels(endpoint).observe(seconds)
    if status >= 500:
        ERRORS.labels(endpoint, str(status)).inc()


def record_predictions(version: str, predictions: list[int], probabilities: list[float]) -> None:
    for label, n in Tally(predictions).items():
        PREDICTIONS.labels(version, str(label)).inc(n)
    for p in probabilities:
        PROBABILITY.observe(p)


def set_model_version(version: str) -> None:
    try:
        MODEL_VERSION.set(float(version))
    except ValueError:
        MODEL_VERSION.set(-1)

"""Inference API: /predict, /predict/batch, /health, /live, /model-info, /reload, /metrics."""

import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import metrics
from src.api.model_store import LoadedModel, load_champion
from src.api.request_logger import RequestLogger
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    records_to_frame,
)
from src.config import FEATURES, MLFLOW_TRACKING_URI, MODEL_NAME, PREDICTION_DB

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

UNTRACKED_PATHS = {"/metrics", "/live"}  # scrapes and healthchecks would drown real traffic


def create_app(
    model_loader: Callable[[], LoadedModel] = load_champion,
    request_logger: RequestLogger | None = None,
) -> FastAPI:
    reload_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.logger = (
            request_logger if request_logger is not None else RequestLogger(PREDICTION_DB)
        )
        app.state.loaded = None
        try:
            app.state.loaded = model_loader()
            metrics.set_model_version(app.state.loaded.version)
            log.info("Loaded champion v%s", app.state.loaded.version)
        except Exception:
            metrics.MODEL_VERSION.set(0)
            log.exception("Could not load champion at startup; /health will return 503")
        yield

    app = FastAPI(title="Income classifier API", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        if request.url.path in UNTRACKED_PATHS:
            return await call_next(request)
        start = time.perf_counter()
        status = 500  # stays 500 if the handler raises
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = route.path if route is not None else "unmatched"
            metrics.observe_request(endpoint, request.method, status, time.perf_counter() - start)

    def require_model() -> LoadedModel:
        loaded = app.state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return loaded

    def safe_log(features, predictions, probabilities, version, latency_ms) -> None:
        # a logging failure must never break a prediction
        try:
            app.state.logger.log_batch(features, predictions, probabilities, version, latency_ms)
        except Exception:
            log.exception("Failed to write prediction log")

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/health")
    def health():
        loaded = app.state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok", "model_version": loaded.version}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model-info")
    def model_info():
        loaded = require_model()
        step = getattr(loaded.model, "named_steps", {}).get("model")
        return {
            "model_name": MODEL_NAME,
            "version": loaded.version,
            "run_id": loaded.run_id,
            "loaded_at": loaded.loaded_at,
            "algorithm": type(step).__name__ if step is not None else None,
            "holdout_metrics": loaded.metrics,
            "features": FEATURES,
            "tracking_uri": MLFLOW_TRACKING_URI,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        loaded = require_model()
        start = time.perf_counter()
        proba = float(loaded.model.predict_proba(req.to_frame())[0, 1])
        pred = int(proba >= 0.5)  # same as predict() for a binary classifier
        latency_ms = (time.perf_counter() - start) * 1000
        safe_log([req.model_dump()], [pred], [proba], loaded.version, latency_ms)
        metrics.record_predictions(loaded.version, [pred], [proba])
        return PredictResponse(
            prediction=pred, probability=round(proba, 6), model_version=loaded.version
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(req: BatchPredictRequest):
        loaded = require_model()
        rows = [r.model_dump() for r in req.records]
        start = time.perf_counter()
        proba = loaded.model.predict_proba(records_to_frame(rows))[:, 1]
        latency_ms = (time.perf_counter() - start) * 1000
        preds = (proba >= 0.5).astype(int).tolist()
        probs = [round(float(p), 6) for p in proba]
        safe_log(rows, preds, probs, loaded.version, latency_ms)
        metrics.record_predictions(loaded.version, preds, probs)
        return BatchPredictResponse(
            predictions=preds, probabilities=probs, model_version=loaded.version
        )

    @app.post("/reload")
    def reload_model():
        """Hot-swap: re-resolve the champion alias. On failure keep the current model."""
        with reload_lock:
            old = app.state.loaded
            try:
                new = model_loader()
            except Exception as exc:
                metrics.RELOADS.labels("failure").inc()
                log.exception("Reload failed")
                raise HTTPException(
                    status_code=502,
                    detail=f"Reload failed, still serving the previous model: {exc}",
                ) from exc
            app.state.loaded = new  # single reference swap, in-flight requests are safe
            metrics.RELOADS.labels("success").inc()
            metrics.set_model_version(new.version)
        return {
            "previous_version": old.version if old else None,
            "current_version": new.version,
            "changed": old is None or old.version != new.version,
        }

    return app


app = create_app()

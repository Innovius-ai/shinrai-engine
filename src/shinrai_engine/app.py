"""The HTTP layer: wire-compatible with the ShinrAI detection service API.

Ported from the internal shinrai-pii-serve FastAPI app; the predictor behind
it is the torch-free NumpyOnnxPredictor (vendored shinrai_pii_runtime), so
this file changes the concurrency model deliberately: ONNX Runtime has no
thread-affinity constraint and releases the GIL during Run, so inference goes
through asyncio.to_thread guarded by a semaphore (default 1 = the same FIFO
the internal torch service had, while /healthz stays responsive during a long
document).

Endpoints:
    GET  /            service info (JSON, curl-friendly)
    GET  /health(z)   200 only when every model is loaded and warmed up
    GET  /metrics     totals + rolling p50/p95 (auth-gated when auth is on)
    GET  /api/models  name/default/window/precision per model
    POST /api/analyze {"text"|"texts", "model", "threshold", "merge_persons"}

The /api/analyze request/response shape is byte-compatible with the internal
service: entities in the frozen legacy shape (startIndex/endIndex are Unicode
code-point offsets, types via api_mapping), envelope {model, results,
timing_ms, version, release_channel}.
"""

from __future__ import annotations

import asyncio
import hmac
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shinrai_pii_runtime import to_legacy_entities

from . import __version__
from .config import Settings
from .metrics import ServeMetrics
from .registry import LoadedModel

GITHUB_URL = "https://github.com/Innovius-ai/shinrai-engine"

AUTH_DISABLED_BANNER = (
    "AUTH DISABLED — /api/* and /metrics are open to anyone who can reach this "
    "port. Deploy cluster-internal only, or set SHINRAI_API_KEY."
)


class AnalyzeRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    model: str | None = None
    threshold: float = 0.7
    merge_persons: bool = True


def create_app(settings: Settings, registry: dict[str, LoadedModel]) -> FastAPI:
    app = FastAPI(title="shinrai-engine", docs_url=None, redoc_url=None)
    metrics = ServeMetrics()
    inference_gate = asyncio.Semaphore(settings.max_concurrent)

    default_name = settings.default_model
    device = "cuda" if any(m.cuda_active for m in registry.values()) else "cpu"
    default_model = registry[default_name]

    async def require_auth(request: Request) -> None:
        if settings.api_key is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), settings.api_key):
            return
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    auth_dep = [Depends(require_auth)]

    @app.get("/")
    def root() -> dict:
        return {
            "service": "shinrai-engine",
            "version": __version__,
            "models": [
                _model_info(m, default_name) for m in registry.values()
            ],
            "auth": "bearer" if settings.api_key else "disabled",
            "endpoints": {
                "health": "/healthz",
                "metrics": "/metrics",
                "models": "/api/models",
                "analyze": "POST /api/analyze",
            },
            "docs": GITHUB_URL,
        }

    @app.get("/health")
    @app.get("/healthz")
    def health() -> dict:
        # The registry is fully mounted (and warmed up) before uvicorn starts
        # listening; a 200 therefore means every model answers /api/analyze.
        return {
            "status": "ok",
            "models": sorted(registry),
            "device": device,
            "precision": default_model.precision,
            "precision_warning": default_model.precision_warning,
            "providers": default_model.providers,
        }

    @app.get("/metrics", dependencies=auth_dep)
    def metrics_endpoint() -> dict:
        return {
            "service": "shinrai-engine",
            "version": __version__,
            "release_channel": settings.release_channel,
            "device": device,
            "models": sorted(registry),
            "precision": default_model.precision,
            **metrics.snapshot(),
        }

    @app.get("/api/models", dependencies=auth_dep)
    def api_models() -> list[dict]:
        return [_model_info(m, default_name) for m in registry.values()]

    @app.post("/api/analyze", dependencies=auth_dep)
    async def api_analyze(request: AnalyzeRequest):
        started = time.time()
        metrics.start_request()
        ok = False
        try:
            if request.texts is not None:
                texts = request.texts
            elif request.text is not None:
                texts = [request.text]
            else:
                return JSONResponse(
                    {"error": "one of 'text' or 'texts' is required"}, status_code=400
                )
            if len(texts) > settings.max_texts:
                return JSONResponse(
                    {"error": f"too many texts (max {settings.max_texts})"}, status_code=413
                )
            if any(len(t) > settings.max_text_chars for t in texts):
                return JSONResponse(
                    {"error": f"text too long (max {settings.max_text_chars} chars)"},
                    status_code=413,
                )
            model_name = request.model or default_name
            model = registry.get(model_name)
            if model is None:
                return JSONResponse(
                    {"error": f"unknown model {model_name!r}", "models": sorted(registry)},
                    status_code=400,
                )

            predictor = model.predictor
            inference_started = time.time()
            async with inference_gate:
                per_text = await asyncio.to_thread(predictor.predict, texts)
            inference_ms = round((time.time() - inference_started) * 1000, 1)

            results = []
            for text, entities in zip(texts, per_text, strict=True):
                legacy = to_legacy_entities(
                    entities,
                    predictor.label_space,
                    threshold=request.threshold,
                    text=text,
                    merge_persons=request.merge_persons,
                )
                n_tokens = len(
                    predictor.tokenizer(text, add_special_tokens=True)["input_ids"]
                )
                window, stride = predictor.window, predictor.stride
                n_windows = (
                    1
                    if n_tokens <= window
                    else 1 + -(-(n_tokens - window) // (window - stride))
                )
                results.append(
                    {
                        "entities": legacy,
                        "stats": {
                            "chars": len(text),
                            "tokens": n_tokens,
                            "windows": n_windows,
                        },
                    }
                )
            ok = True
            return {
                "model": model_name,
                "results": results,
                "timing_ms": {
                    "total": round((time.time() - started) * 1000, 1),
                    "inference": inference_ms,
                },
                "version": __version__,
                "release_channel": settings.release_channel,
            }
        finally:
            metrics.finish_request(
                ok=ok, duration_ms=round((time.time() - started) * 1000, 1)
            )

    return app


def _model_info(model: LoadedModel, default_name: str) -> dict:
    return {
        "name": model.name,
        "default": model.name == default_name,
        "window": model.predictor.window,
        "languages_hint": None,  # routing lives client-side
        "precision": model.precision,
        "precision_warning": model.precision_warning,
    }

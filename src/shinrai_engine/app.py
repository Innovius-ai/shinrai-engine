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
import gc
import hmac
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shinrai_pii_runtime import scrub_invisibles, to_legacy_entities

from . import __version__
from .config import Settings
from .metrics import ServeMetrics
from .registry import LoadedModel, load_model
from .selftest import run_all

GITHUB_URL = "https://github.com/Innovius-ai/shinrai-engine"

AUTH_DISABLED_BANNER = (
    "AUTH DISABLED — /api/* and /metrics are open to anyone who can reach this "
    "port. Deploy cluster-internal only, or set SHINRAI_API_KEY."
)


class AnalyzeRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    model: str | None = None
    threshold: float | None = None
    merge_persons: bool = True
    segment: Literal["auto", "sentence", "none", "whole"] | None = "auto"


def create_app(settings: Settings, registry: dict[str, LoadedModel]) -> FastAPI:
    metrics = ServeMetrics()
    inference_gate = asyncio.Semaphore(settings.max_concurrent)
    default_name = settings.default_model
    configured = dict(settings.models)
    last_used: dict[str, float] = {}
    activation_locks = {name: asyncio.Lock() for name in configured}

    async def activate_model(name: str) -> LoadedModel:
        """Return a ready model, loading an on-demand version exactly once."""
        async with activation_locks[name]:
            model = registry.get(name)
            if model is None:
                source = configured.get(name)
                if source is None:
                    raise KeyError(name)
                started = time.time()
                model = await asyncio.to_thread(load_model, name, source, settings)
                await asyncio.to_thread(run_all, {name: model}, settings.self_test)
                registry[name] = model
                print(
                    f"[engine] activated {name} [{model.precision}] via "
                    f"{model.providers[0]} in {time.time() - started:.1f}s"
                )
            if name in settings.lazy_models:
                last_used[name] = time.monotonic()
            return model

    async def unload_idle_models() -> None:
        interval = min(60.0, max(1.0, settings.model_idle_ttl_seconds / 4))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            expired = [
                name
                for name, used in last_used.items()
                if name in registry
                and now - used >= settings.model_idle_ttl_seconds
            ]
            for name in expired:
                async with inference_gate, activation_locks[name]:
                    used = last_used.get(name, now)
                    if (
                        name not in registry
                        or time.monotonic() - used < settings.model_idle_ttl_seconds
                    ):
                        continue
                    registry.pop(name, None)
                    last_used.pop(name, None)
                    gc.collect()
                    print(
                        f"[engine] unloaded idle on-demand model {name} "
                        f"after {settings.model_idle_ttl_seconds}s"
                    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sweeper = asyncio.create_task(unload_idle_models())
        try:
            yield
        finally:
            sweeper.cancel()
            await asyncio.gather(sweeper, return_exceptions=True)

    app = FastAPI(
        title="shinrai-engine", docs_url=None, redoc_url=None, lifespan=lifespan
    )

    def device() -> str:
        return "cuda" if any(m.cuda_active for m in registry.values()) else "cpu"

    def model_rows() -> list[dict]:
        return [
            _model_info(
                registry.get(name),
                name,
                default_name,
                on_demand=name in settings.lazy_models,
                idle_ttl_seconds=settings.model_idle_ttl_seconds,
            )
            for name in configured
        ]

    async def require_auth(request: Request) -> None:
        if settings.api_key is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        # Bytes, not str: compare_digest raises TypeError on non-ASCII str
        # input, which turned a stray high-byte token (or a non-ASCII
        # configured key) into 500s instead of 401s.
        if scheme.lower() == "bearer" and hmac.compare_digest(
            token.strip().encode("utf-8"), settings.api_key.encode("utf-8")
        ):
            return
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    auth_dep = [Depends(require_auth)]

    @app.get("/")
    def root() -> dict:
        return {
            "service": "shinrai-engine",
            "version": __version__,
            "models": model_rows(),
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
        default_model = registry[default_name]
        return {
            "status": "ok",
            "models": sorted(configured),
            "loaded_models": sorted(registry),
            "device": device(),
            # Back-compat flat fields describe the DEFAULT model; the
            # per-model truth (multi-model deployments, silent CUDA
            # fallback on a second graph) lives in models_detail.
            "precision": default_model.precision,
            "precision_warning": default_model.precision_warning,
            "providers": default_model.providers,
            "self_test": default_model.self_test,
            "models_detail": model_rows(),
        }

    @app.get("/metrics", dependencies=auth_dep)
    def metrics_endpoint() -> dict:
        default_model = registry[default_name]
        return {
            "service": "shinrai-engine",
            "version": __version__,
            "release_channel": settings.release_channel,
            "device": device(),
            "models": sorted(configured),
            "loaded_models": sorted(registry),
            "precision": default_model.precision,
            **metrics.snapshot(),
        }

    @app.get("/api/models", dependencies=auth_dep)
    def api_models() -> list[dict]:
        return model_rows()

    @app.post("/api/models/{model_name}/activate", dependencies=auth_dep)
    async def api_activate_model(model_name: str):
        if model_name not in configured:
            return JSONResponse(
                {"error": f"unknown model {model_name!r}", "models": sorted(configured)},
                status_code=404,
            )
        async with inference_gate:
            try:
                model = await activate_model(model_name)
            except Exception as exc:
                print(f"[engine] activation failed for {model_name}: {type(exc).__name__}")
                return JSONResponse(
                    {"error": "model activation failed", "model": model_name},
                    status_code=503,
                )
        return _model_info(
            model,
            model_name,
            default_name,
            on_demand=model_name in settings.lazy_models,
            idle_ttl_seconds=settings.model_idle_ttl_seconds,
        )

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
            if model_name not in configured:
                return JSONResponse(
                    {"error": f"unknown model {model_name!r}", "models": sorted(configured)},
                    status_code=400,
                )

            # Length-preserving invisible-char scrub BEFORE tokenization (same
            # slot as the reference service): zero-width/bidi/tag characters
            # would otherwise reach the model as opaque tokens and hide the
            # very entities this service exists to find. Each scrubbed char
            # becomes one space, so the returned offsets stay valid for the
            # caller's original text.
            texts = [scrub_invisibles(t) for t in texts]

            inference_started = time.time()
            async with inference_gate:
                try:
                    model = await activate_model(model_name)
                except Exception as exc:
                    print(f"[engine] activation failed for {model_name}: {type(exc).__name__}")
                    return JSONResponse(
                        {"error": "model activation failed", "model": model_name},
                        status_code=503,
                    )
                predictor = model.predictor
                # EVERYTHING that touches the tokenizer runs in this one
                # gated thread: the HF fast tokenizer is not safe under
                # concurrent calls (Rust core: 'Already borrowed'), and any
                # tokenizer work left on the event loop stalls /healthz for
                # the length of a long document.
                per_text, stats = await asyncio.to_thread(
                    _predict_with_stats, predictor, texts, request.segment
                )
                if model_name in settings.lazy_models:
                    last_used[model_name] = time.monotonic()
            inference_ms = round((time.time() - inference_started) * 1000, 1)

            results = []
            for text, entities, text_stats in zip(texts, per_text, stats, strict=True):
                legacy = to_legacy_entities(
                    entities,
                    predictor.label_space,
                    threshold=_resolve_threshold(request.threshold, predictor),
                    text=text,
                    merge_persons=request.merge_persons,
                )
                results.append({"entities": legacy, "stats": text_stats})
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


def _predict_with_stats(
    predictor, texts: list[str], segment: str | None = "auto"
) -> tuple[list, list[dict]]:
    """Inference plus per-text stats, in ONE worker thread.

    Runs under the inference gate on purpose — every tokenizer touch must be
    serialized (see the call site). The window count uses the tokenizer's real
    step: each window carries `window - 2` content tokens (2 specials), so the
    advance per extra window is `window - 2 - stride`, not `window - stride` —
    the naive formula under-reported one window per ~220 on long documents.
    """
    if segment not in (None, "auto", "sentence", "none", "whole"):
        raise ValueError("segment must be auto, sentence, none, whole or null")
    decode_segment = None if segment in (None, "none", "whole") else segment
    per_text = predictor.predict(texts, segment=decode_segment)
    window, stride = predictor.window, predictor.stride
    content, step = window - 2, window - 2 - stride
    stats = []
    for text in texts:
        n_tokens = len(predictor.tokenizer(text, add_special_tokens=True)["input_ids"])
        n_content = max(n_tokens - 2, 0)
        n_windows = 1 if n_content <= content else 1 + -(-(n_content - content) // step)
        stats.append({"chars": len(text), "tokens": n_tokens, "windows": n_windows})
    return per_text, stats


def _resolve_threshold(requested: float | None, predictor) -> float:
    if requested is not None:
        return float(requested)
    decoder = (getattr(predictor, "meta", None) or {}).get("decoder")
    if isinstance(decoder, dict) and isinstance(decoder.get("serve_threshold"), (int, float)):
        return float(decoder["serve_threshold"])
    return 0.7


def _model_info(
    model: LoadedModel | None,
    name: str,
    default_name: str,
    *,
    on_demand: bool = False,
    idle_ttl_seconds: int = 14_400,
) -> dict:
    if model is None:
        return {
            "name": name,
            "default": name == default_name,
            "state": "cold",
            "on_demand": on_demand,
            "idle_ttl_seconds": idle_ttl_seconds if on_demand else None,
            "window": None,
            "languages_hint": None,
            "precision": None,
            "precision_warning": None,
            "providers": [],
            "self_test": "not_run",
        }
    return {
        "name": name,
        "default": name == default_name,
        "state": "loaded",
        "on_demand": on_demand,
        "idle_ttl_seconds": idle_ttl_seconds if on_demand else None,
        "window": model.predictor.window,
        "languages_hint": None,  # routing lives client-side
        "precision": model.precision,
        "precision_warning": model.precision_warning,
        "providers": model.providers,
        "self_test": model.self_test,
    }

"""Bearer auth: opt-in, probes stay open, /api/* and /metrics gated."""

from __future__ import annotations

from fastapi.testclient import TestClient

from shinrai_engine.app import create_app
from shinrai_engine.config import Settings


def make_client(tiny_registry, api_key: str | None) -> TestClient:
    settings, registry = tiny_registry
    secured = Settings(
        models=settings.models,
        model_cache=settings.model_cache,
        api_key=api_key,
    )
    return TestClient(create_app(secured, registry))


def test_auth_disabled_everything_open(tiny_registry):
    client = make_client(tiny_registry, api_key=None)
    assert client.get("/metrics").status_code == 200
    assert client.post("/api/analyze", json={"text": "x"}).status_code == 200


def test_auth_enabled_gates_api_and_metrics(tiny_registry):
    client = make_client(tiny_registry, api_key="sekrit")
    # Probes and the info page stay open (kubelet has no bearer token).
    assert client.get("/healthz").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/").json()["auth"] == "bearer"

    assert client.get("/metrics").status_code == 401
    assert client.get("/api/models").status_code == 401
    assert client.post("/api/analyze", json={"text": "x"}).status_code == 401

    headers = {"Authorization": "Bearer sekrit"}
    assert client.get("/metrics", headers=headers).status_code == 200
    assert client.get("/api/models", headers=headers).status_code == 200
    assert client.post("/api/analyze", json={"text": "x"}, headers=headers).status_code == 200

    wrong = {"Authorization": "Bearer wrong"}
    assert client.post("/api/analyze", json={"text": "x"}, headers=wrong).status_code == 401

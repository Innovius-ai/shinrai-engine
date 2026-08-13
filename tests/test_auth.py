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


def test_non_ascii_token_is_401_not_500(tiny_registry):
    # compare_digest on str raises TypeError for non-ASCII — an
    # unauthenticated 500 vector until the comparison moved to bytes. The
    # header is sent as raw latin-1 wire bytes (httpx refuses non-ASCII str).
    client = make_client(tiny_registry, api_key="sekrit")
    weird = {b"Authorization": "Bearer s\xe9cret".encode("latin-1")}
    assert client.post("/api/analyze", json={"text": "x"}, headers=weird).status_code == 401


def test_non_ascii_configured_key_refused_at_config_time():
    # A non-ASCII key can never match reliably over HTTP (latin-1 server
    # decode vs UTF-8 clients) — refused with a clear message instead of
    # 401ing every request forever.
    import pytest

    from shinrai_engine.config import ConfigError, load_settings

    with pytest.raises(ConfigError, match="ASCII"):
        load_settings({"SHINRAI_API_KEY": "sécret-ümläut"})

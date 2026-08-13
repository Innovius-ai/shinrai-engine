"""API contract: the exact expectations of shinrai-encryption's RemoteBackend
(bert_backend.py _validate_entities) plus the envelope shape."""

from __future__ import annotations

from fastapi.testclient import TestClient

from shinrai_engine.app import create_app

# The keys RemoteBackend hard-requires on every entity.
REQUIRED_ENTITY_KEYS = ("text", "type", "startIndex", "endIndex")
LEGACY_KEYS = REQUIRED_ENTITY_KEYS + ("tier", "attrs", "source", "confidence", "region")


def make_client(tiny_registry) -> TestClient:
    settings, registry = tiny_registry
    return TestClient(create_app(settings, registry))


def test_healthz_and_root_are_ok(tiny_registry):
    client = make_client(tiny_registry)
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["models"] == ["tiny"]
    assert health["precision"] == "fp32"
    assert health["precision_warning"] is None
    assert health["providers"] == ["CPUExecutionProvider"]

    root = client.get("/").json()
    assert root["service"] == "shinrai-engine"
    assert root["auth"] == "disabled"
    assert root["models"][0]["name"] == "tiny"


def test_api_models_shape(tiny_registry):
    client = make_client(tiny_registry)
    models = client.get("/api/models").json()
    assert models == [
        {
            "name": "tiny",
            "default": True,
            "window": 64,
            "languages_hint": None,
            "precision": "fp32",
            "precision_warning": None,
        }
    ]


def test_analyze_single_text_entities_and_sliceback(tiny_registry):
    client = make_client(tiny_registry)
    text = "Anna Miller lives in Berlin near the main station."
    body = client.post("/api/analyze", json={"text": text}).json()

    assert set(body) == {"model", "results", "timing_ms", "version", "release_channel"}
    assert body["model"] == "tiny"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert set(result["stats"]) == {"chars", "tokens", "windows"}
    assert result["stats"]["chars"] == len(text)

    entities = result["entities"]
    assert entities, "the biased tiny model must produce entities"
    starts = [e["startIndex"] for e in entities]
    assert starts == sorted(starts)
    for ent in entities:
        for key in LEGACY_KEYS:
            assert key in ent, f"missing {key} in {ent}"
        assert ent["source"] == "bert"
        # RemoteBackend's slice-back validation — offsets are code points.
        assert text[ent["startIndex"] : ent["endIndex"]] == ent["text"]
        assert ent["confidence"] >= 0.7


def test_analyze_batch_texts(tiny_registry):
    client = make_client(tiny_registry)
    texts = ["Anna Miller lives in Berlin.", "Peter Schmidt works at the bakery."]
    body = client.post("/api/analyze", json={"texts": texts}).json()
    assert len(body["results"]) == 2


def test_analyze_threshold_filters(tiny_registry):
    client = make_client(tiny_registry)
    text = "Anna Miller lives in Berlin."
    strict = client.post("/api/analyze", json={"text": text, "threshold": 0.9999}).json()
    assert strict["results"][0]["entities"] == []


def test_analyze_error_paths(tiny_registry):
    client = make_client(tiny_registry)
    assert client.post("/api/analyze", json={}).status_code == 400
    unknown = client.post("/api/analyze", json={"text": "x", "model": "nope"})
    assert unknown.status_code == 400
    assert unknown.json()["models"] == ["tiny"]


def test_analyze_limits(tiny_bundle, tmp_path):
    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import build_registry

    settings = load_settings(
        {
            "SHINRAI_MODELS": f"tiny={tiny_bundle}",
            "SHINRAI_MODEL_CACHE": str(tmp_path),
            "SHINRAI_MAX_TEXTS": "2",
            "SHINRAI_MAX_TEXT_CHARS": "50",
        }
    )
    registry = build_registry(settings, log=lambda *a: None)
    client = TestClient(create_app(settings, registry))
    assert client.post("/api/analyze", json={"texts": ["a", "b", "c"]}).status_code == 413
    assert client.post("/api/analyze", json={"text": "x" * 51}).status_code == 413
    assert client.post("/api/analyze", json={"text": "short text"}).status_code == 200

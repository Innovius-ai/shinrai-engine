"""API contract: the exact expectations of shinrai-encryption's RemoteBackend
(bert_backend.py _validate_entities) plus the envelope shape."""

from __future__ import annotations

import time

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
            "state": "loaded",
            "on_demand": False,
            "idle_ttl_seconds": None,
            "providers": ["CPUExecutionProvider"],
            "self_test": "not_run",
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


def test_analyze_scrubs_invisible_characters(tiny_registry):
    """Zero-width characters must not reach the model (they hide entities)
    and must not leak back in entity surfaces. Offsets stay valid because the
    scrub is length-preserving."""
    client = make_client(tiny_registry)
    text = "Anna\u200bMiller lives in Berlin."
    body = client.post("/api/analyze", json={"text": text}).json()
    result = body["results"][0]
    assert result["stats"]["chars"] == len(text)
    for ent in result["entities"]:
        assert "\u200b" not in ent["text"]
        assert 0 <= ent["startIndex"] <= ent["endIndex"] <= len(text)


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


def test_retained_model_activates_and_unloads_after_idle(tiny_bundle, tmp_path):
    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import build_registry

    settings = load_settings(
        {
            "SHINRAI_MODELS": f"current={tiny_bundle},retained={tiny_bundle}",
            "SHINRAI_LAZY_MODELS": "retained",
            "SHINRAI_MODEL_IDLE_TTL_SECONDS": "1",
            "SHINRAI_MODEL_CACHE": str(tmp_path),
            "SHINRAI_SELF_TEST": "off",
        }
    )
    registry = build_registry(settings, log=lambda *a: None)
    assert set(registry) == {"current"}
    with TestClient(create_app(settings, registry)) as client:
        before = {row["name"]: row for row in client.get("/api/models").json()}
        assert before["retained"]["state"] == "cold"
        assert before["retained"]["idle_ttl_seconds"] == 1

        activated = client.post("/api/models/retained/activate")
        assert activated.status_code == 200
        assert activated.json()["state"] == "loaded"

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rows = {row["name"]: row for row in client.get("/api/models").json()}
            if rows["retained"]["state"] == "cold":
                break
            time.sleep(0.1)
        assert rows["retained"]["state"] == "cold"


def _spy_predict(monkeypatch, tiny_registry) -> list[dict]:
    """Wrap the tiny predictor's predict and record the kwargs of every call."""
    _, registry = tiny_registry
    predictor = next(iter(registry.values())).predictor
    real = predictor.predict
    calls: list[dict] = []

    def spy(texts, **kwargs):
        # the engine's call carries `segment`; the runtime's own recursive
        # calls (segment / batch chunking) do not — record only the engine's
        if "segment" in kwargs:
            calls.append(kwargs)
        return real(texts, **kwargs)

    monkeypatch.setattr(predictor, "predict", spy)
    return calls


def test_analyze_language_is_forwarded_to_predict(tiny_registry, monkeypatch):
    calls = _spy_predict(monkeypatch, tiny_registry)
    client = make_client(tiny_registry)
    texts = ["Anna Miller lives in Berlin.", "Peter Schmidt works at the bakery."]
    response = client.post("/api/analyze", json={"texts": texts, "language": "pt-BR"})
    assert response.status_code == 200
    # one language per request: the single predict call carries it for every text
    assert len(calls) == 1
    assert calls[0]["lang"] == "pt-BR"


def test_analyze_without_language_passes_no_lang(tiny_registry, monkeypatch):
    calls = _spy_predict(monkeypatch, tiny_registry)
    client = make_client(tiny_registry)
    for body in ({"text": "Anna Miller lives in Berlin."},
                 {"text": "Anna Miller lives in Berlin.", "language": None}):
        assert client.post("/api/analyze", json=body).status_code == 200
    assert len(calls) == 2
    assert all("lang" not in kw for kw in calls)


def test_analyze_language_does_not_change_unstamped_output(tiny_registry):
    """The tiny checkpoint stamps no per-language setting: language is inert."""
    client = make_client(tiny_registry)
    text = "Anna Miller lives in Berlin near the main station."
    base = client.post("/api/analyze", json={"text": text}).json()["results"]
    for language in ("de", "ja", "he", "pt-BR", "zh_Hant", "EN"):
        with_lang = client.post("/api/analyze", json={"text": text, "language": language})
        assert with_lang.status_code == 200
        assert with_lang.json()["results"] == base


def test_analyze_language_validation(tiny_registry, monkeypatch):
    calls = _spy_predict(monkeypatch, tiny_registry)
    client = make_client(tiny_registry)
    for bad in ("", "d", "de;rm -rf", "../etc", "de--at", "-de", "de-", "x" * 36,
                "de-" + "a" * 9, 42, ["de"], "日本語"):
        response = client.post("/api/analyze", json={"text": "Anna Miller", "language": bad})
        assert response.status_code == 422, bad
    assert calls == []


def test_entities_keep_evidence_with_and_without_language(tiny_registry):
    client = make_client(tiny_registry)
    for body in ({"text": "Anna Miller lives in Berlin."},
                 {"text": "Anna Miller lives in Berlin.", "language": "de"}):
        entities = client.post("/api/analyze", json=body).json()["results"][0]["entities"]
        assert entities
        assert all(e["evidence"] in ("argmax", "floor") for e in entities)

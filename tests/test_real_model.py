"""Slow tests against a real release bundle.

Run with:  SHINRAI_TEST_MODEL_DIR=/path/to/shinrai-pii-pathfinder-m-v1.1 pytest -m slow
"""

from __future__ import annotations

import os

import pytest

MODEL_DIR = os.environ.get("SHINRAI_TEST_MODEL_DIR")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL_DIR, reason="SHINRAI_TEST_MODEL_DIR not set"),
]


@pytest.fixture(scope="module")
def real_registry(tmp_path_factory):
    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import build_registry

    settings = load_settings(
        {
            "SHINRAI_MODELS": f"real={MODEL_DIR}",
            "SHINRAI_MODEL_CACHE": str(tmp_path_factory.mktemp("cache")),
            "SHINRAI_SELF_TEST": "off",
        }
    )
    return settings, build_registry(settings)


def test_golden_selftest_strict(real_registry):
    from shinrai_engine.selftest import run_selftest

    _, registry = real_registry
    assert run_selftest(registry["real"]) is True


def test_analyze_german_sentence(real_registry):
    from fastapi.testclient import TestClient

    from shinrai_engine.app import create_app

    settings, registry = real_registry
    client = TestClient(create_app(settings, registry))
    text = "Anna Müller wohnt in Berlin in der Musterstraße 12."
    body = client.post("/api/analyze", json={"text": text}).json()
    entities = body["results"][0]["entities"]
    types = {e["type"] for e in entities}
    # PERSON may arrive merged (PERSON) or split (FIRSTNAME/SURNAME).
    assert types & {"PERSON", "FIRSTNAME", "SURNAME"}
    assert "CITY" in types
    assert "STREET_ADDRESS" in types
    for ent in entities:
        assert text[ent["startIndex"] : ent["endIndex"]] == ent["text"]
    # "Berlin" sits at code point 21 but byte 22 (the ü before it is two
    # bytes) — proves the offsets on the wire are code points, not bytes.
    berlin = next(e for e in entities if e["text"] == "Berlin")
    assert berlin["startIndex"] == 21
    assert text.encode("utf-8").index(b"Berlin") == 22

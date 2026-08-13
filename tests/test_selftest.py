"""Golden self-test behavior on the tiny model."""

from __future__ import annotations

import json

import pytest

from shinrai_engine.selftest import load_golden, run_all, run_selftest


@pytest.fixture()
def golden_in_bundle(tiny_registry):
    """Write a golden file derived from the tiny model's own predictions."""
    _, registry = tiny_registry
    model = registry["tiny"]
    texts = ["Anna Miller lives in Berlin.", "Peter Schmidt works at the bakery."]
    cases = [
        {"text": text, "entities": model.predictor.predict([text])[0]} for text in texts
    ]
    path = model.bundle_dir / "golden-predictions.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    yield model, cases
    path.unlink()


def test_strict_pass_on_own_predictions(golden_in_bundle):
    model, _ = golden_in_bundle
    assert run_selftest(model, log=lambda *a: None) is True


def test_strict_fails_on_drift(golden_in_bundle):
    model, cases = golden_in_bundle
    cases[0]["entities"][0]["confidence"] = 0.123  # far outside 1e-3
    path = model.bundle_dir / "golden-predictions.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    assert run_selftest(model, log=lambda *a: None) is False


def test_strict_mode_raises(golden_in_bundle):
    model, cases = golden_in_bundle
    # The tiny model only ever emits PERSON — a CITY golden can never match.
    cases[0]["entities"][0]["type"] = "CITY"
    path = model.bundle_dir / "golden-predictions.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_all({"tiny": model}, "strict", log=lambda *a: None)


def test_no_golden_is_a_skip(tiny_registry):
    _, registry = tiny_registry
    model = registry["tiny"]
    assert load_golden(model.bundle_dir, "tiny") is None
    assert run_selftest(model, log=lambda *a: None) is True


def test_packaged_golden_resolves_for_v11(tmp_path):
    """The packaged fallback matches the v1.1 bundle-dir/model naming."""
    bundle_dir = tmp_path / "shinrai-pii-pathfinder-m-v1.1"
    bundle_dir.mkdir()
    golden = load_golden(bundle_dir, "v1.1")
    assert golden and golden[0]["entities"], "packaged golden must load for v1.1"

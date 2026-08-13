"""Downloader: file allowlist, sha verification, atomic install, marker."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from shinrai_engine import download
from shinrai_engine.download import (
    HfSource,
    ModelSourceError,
    ensure_model,
    parse_source,
)

ONNX_REL = "quant/model-fp32.onnx"


def test_parse_source():
    assert parse_source("hf://innovius/some-model") == HfSource("innovius/some-model", "main")
    assert parse_source("hf://innovius/some-model@v2") == HfSource("innovius/some-model", "v2")
    assert parse_source("/models/v1.1") == Path("/models/v1.1")
    with pytest.raises(ModelSourceError):
        parse_source("hf://only-org")


def test_dir_source_used_in_place(tiny_bundle, tmp_path):
    result = ensure_model("tiny", str(tiny_bundle), tmp_path, ONNX_REL, log=lambda *a: None)
    assert result == tiny_bundle


def test_dir_source_missing_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ModelSourceError, match="missing"):
        ensure_model("tiny", str(empty), tmp_path / "cache", ONNX_REL, log=lambda *a: None)


@pytest.fixture()
def fake_hub(tiny_bundle, monkeypatch):
    """hf_hub_download that serves files from the tiny bundle and counts calls."""
    calls: list[str] = []

    def fake_download(*, repo_id, filename, revision, local_dir):
        calls.append(filename)
        source = tiny_bundle / filename
        if not source.is_file():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"no {filename}")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def test_hf_download_installs_atomically_with_marker(fake_hub, tmp_path):
    cache = tmp_path / "cache"
    bundle = ensure_model("v1", "hf://innovius/tiny@main", cache, ONNX_REL, log=lambda *a: None)
    assert bundle == cache / "v1"
    assert (bundle / download.MARKER).is_file()
    assert (bundle / ONNX_REL).is_file()
    assert (bundle / "labels-v2.0.yaml").is_file()
    marker = json.loads((bundle / download.MARKER).read_text())
    assert marker["repo"] == "innovius/tiny"
    assert ONNX_REL in marker["files"]
    # The safetensors weights must never be requested.
    assert not any("safetensors" in f for f in fake_hub)
    # No staging leftovers.
    assert not list(cache.glob(".staging-*"))


def test_hf_download_reuses_cache(fake_hub, tmp_path):
    cache = tmp_path / "cache"
    ensure_model("v1", "hf://innovius/tiny", cache, ONNX_REL, log=lambda *a: None)
    first_calls = len(fake_hub)
    ensure_model("v1", "hf://innovius/tiny", cache, ONNX_REL, log=lambda *a: None)
    assert len(fake_hub) == first_calls, "second call must hit the marker cache"


def test_marker_invalidates_on_source_change(fake_hub, tmp_path):
    """Changing the repo or pinning a revision under the same name must
    re-stage — the old behavior served stale weights forever."""
    cache = tmp_path / "cache"
    ensure_model("v1", "hf://innovius/tiny", cache, ONNX_REL, log=lambda *a: None)
    first = len(fake_hub)
    # Different revision, same name: cache must MISS.
    ensure_model("v1", "hf://innovius/tiny@pinned", cache, ONNX_REL, log=lambda *a: None)
    assert len(fake_hub) > first, "revision change must invalidate the marker"
    second = len(fake_hub)
    # Different repo, same name: cache must MISS again.
    ensure_model("v1", "hf://innovius/other-repo", cache, ONNX_REL, log=lambda *a: None)
    assert len(fake_hub) > second, "repo change must invalidate the marker"


def test_hostile_labels_file_is_refused(tiny_bundle, monkeypatch, tmp_path):
    """config.json is DOWNLOADED content; a traversal path in labels_file
    must be refused before anything is written outside staging."""
    import json as json_mod

    hostile = tmp_path / "hostile-src"
    shutil.copytree(tiny_bundle, hostile)
    config_path = hostile / "config.json"
    config = json_mod.loads(config_path.read_text())
    config["shinrai"]["labels_file"] = "../../evil.yaml"
    config_path.write_text(json_mod.dumps(config))

    def fake_download(*, repo_id, filename, revision, local_dir):
        source = hostile / filename
        if not source.is_file():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"no {filename}")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(ModelSourceError, match="path traversal"):
        ensure_model("v1", "hf://innovius/tiny", tmp_path / "cache", ONNX_REL,
                     log=lambda *a: None)


def test_interrupted_download_keeps_partial_for_resume(tiny_bundle, monkeypatch, tmp_path):
    """A network failure mid-download must keep the staging dir (resume);
    only verification failures start clean."""
    calls = {"n": 0}

    def flaky_download(*, repo_id, filename, revision, local_dir):
        calls["n"] += 1
        if calls["n"] > 2:
            raise ConnectionError("link dropped")
        source = tiny_bundle / filename
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky_download)
    cache = tmp_path / "cache"
    with pytest.raises(ConnectionError):
        ensure_model("v1", "hf://innovius/tiny", cache, ONNX_REL, log=lambda *a: None)
    staged = list(cache.glob(".staging-*"))
    assert staged, "partial download must be kept for resume"
    assert (staged[0] / "config.json").is_file()


def test_sha_mismatch_hard_fails(tiny_bundle, monkeypatch, tmp_path):
    corrupted = tmp_path / "corrupted-src"
    shutil.copytree(tiny_bundle, corrupted)
    manifest_path = corrupted / "quant" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    def fake_download(*, repo_id, filename, revision, local_dir):
        source = corrupted / filename
        if not source.is_file():
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"no {filename}")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cache = tmp_path / "cache"
    with pytest.raises(ModelSourceError, match="sha256 mismatch"):
        ensure_model("v1", "hf://innovius/tiny", cache, ONNX_REL, log=lambda *a: None)
    # Failed install leaves no bundle and no staging litter.
    assert not (cache / "v1").exists()
    assert not list(cache.glob(".staging-*"))

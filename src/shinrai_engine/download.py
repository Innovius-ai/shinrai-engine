"""Model bundle acquisition: mounted directory or Hugging Face download.

Downloads ONLY the serving subset (config, labels YAML, calibration,
tokenizer, quant MANIFEST, the one chosen .onnx) — never model.safetensors or
eval artifacts, which saves >1.2 GB per model. The .onnx is sha256-verified
against quant/MANIFEST.json when the manifest lists it (hard fail on
mismatch; a missing manifest entry downgrades to a warning). Installation is
atomic: everything lands in a staging dir that is renamed into place, then a
.shinrai-complete marker records what was installed — a bundle dir without
its marker is re-staged, a mounted plain directory needs no marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

MARKER = ".shinrai-complete"
MANIFEST_RELPATH = "quant/MANIFEST.json"

# Always fetched. tokenizer_config/special_tokens are optional on the Hub but
# AutoTokenizer wants them when present, so they are fetched best-effort.
_REQUIRED_FILES = ("config.json", "tokenizer/tokenizer.json")
_OPTIONAL_FILES = (
    "calibration.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    MANIFEST_RELPATH,
)


class ModelSourceError(RuntimeError):
    """The bundle cannot be acquired; the message says why."""


@dataclass(frozen=True)
class HfSource:
    repo_id: str
    revision: str = "main"


def parse_source(source: str) -> HfSource | Path:
    """hf://org/repo[@revision] -> HfSource, anything else -> directory Path."""
    if source.startswith("hf://"):
        ref = source[len("hf://") :]
        repo_id, _, revision = ref.partition("@")
        if repo_id.count("/") != 1 or not all(repo_id.split("/")):
            raise ModelSourceError(
                f"{source!r}: expected hf://org/repo[@revision]"
            )
        return HfSource(repo_id=repo_id, revision=revision or "main")
    return Path(source)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_against_manifest(bundle_dir: Path, onnx_relpath: str, log=print) -> None:
    """sha256 the .onnx against its quant/MANIFEST.json entry."""
    manifest_path = bundle_dir / MANIFEST_RELPATH
    onnx_path = bundle_dir / onnx_relpath
    if not manifest_path.is_file():
        log(f"[engine] {bundle_dir.name}: no {MANIFEST_RELPATH} — skipping sha256 verification")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        (a for a in manifest.get("artifacts", []) if a.get("file") == Path(onnx_relpath).name),
        None,
    )
    if entry is None or not entry.get("sha256"):
        log(
            f"[engine] {bundle_dir.name}: {Path(onnx_relpath).name} not listed in the "
            "manifest — skipping sha256 verification"
        )
        return
    actual = sha256_file(onnx_path)
    if actual != entry["sha256"]:
        raise ModelSourceError(
            f"{onnx_path}: sha256 mismatch — manifest says {entry['sha256']}, file is "
            f"{actual}. The download is corrupt or the file was tampered with; delete "
            f"{bundle_dir} and retry."
        )
    log(f"[engine] {bundle_dir.name}: sha256 verified ({Path(onnx_relpath).name})")


def _marker_matches(bundle_dir: Path, onnx_relpath: str) -> bool:
    marker = bundle_dir / MARKER
    if not marker.is_file():
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return onnx_relpath in recorded.get("files", []) and (bundle_dir / onnx_relpath).is_file()


def _download_into(staging: Path, source: HfSource, onnx_relpath: str, log=print) -> list[str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    def fetch(relpath: str, required: bool) -> bool:
        try:
            hf_hub_download(
                repo_id=source.repo_id,
                filename=relpath,
                revision=source.revision,
                local_dir=str(staging),
            )
            return True
        except EntryNotFoundError:
            if required:
                raise ModelSourceError(
                    f"hf://{source.repo_id}@{source.revision} has no {relpath!r}"
                ) from None
            return False

    fetched: list[str] = []
    for relpath in _REQUIRED_FILES:
        fetch(relpath, required=True)
        fetched.append(relpath)

    # The labels file name comes from config.json's shinrai block.
    config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
    shinrai_block = config.get("shinrai") or {}
    labels_file = shinrai_block.get("labels_file", "labels-v2.0.yaml")
    fetch(labels_file, required=True)
    fetched.append(labels_file)

    for relpath in _OPTIONAL_FILES:
        if fetch(relpath, required=False):
            fetched.append(relpath)

    log(f"[engine] downloading {onnx_relpath} from hf://{source.repo_id} (large file)...")
    fetch(onnx_relpath, required=True)
    fetched.append(onnx_relpath)
    return fetched


def ensure_model(
    name: str,
    source: str,
    cache_root: Path,
    onnx_relpath: str,
    log=print,
) -> Path:
    """Return a bundle directory containing config/tokenizer/labels + the graph.

    Directory sources are used in place (and validated). hf:// sources are
    downloaded into <cache_root>/<name> once and reused via the marker file.
    """
    parsed = parse_source(source)

    if isinstance(parsed, Path):
        bundle_dir = parsed
        if not bundle_dir.is_dir():
            raise ModelSourceError(f"{name}: model directory {bundle_dir} does not exist")
        missing = [
            rel for rel in ("config.json", "tokenizer/tokenizer.json", onnx_relpath)
            if not (bundle_dir / rel).is_file()
        ]
        if missing:
            raise ModelSourceError(f"{name}: {bundle_dir} is missing {missing}")
        return bundle_dir

    bundle_dir = cache_root / name
    if _marker_matches(bundle_dir, onnx_relpath):
        log(f"[engine] {name}: using cached bundle {bundle_dir}")
        return bundle_dir

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = cache_root / f".staging-{name}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        fetched = _download_into(staging, parsed, onnx_relpath, log=log)
        verify_against_manifest(staging, onnx_relpath, log=log)
        # A previous partial/foreign dir without a matching marker is replaced.
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        (staging / MARKER).write_text(
            json.dumps(
                {
                    "name": name,
                    "repo": parsed.repo_id,
                    "revision": parsed.revision,
                    "files": fetched,
                    "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.rename(staging, bundle_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    log(f"[engine] {name}: installed hf://{parsed.repo_id}@{parsed.revision} -> {bundle_dir}")
    return bundle_dir

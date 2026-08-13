"""Environment-first configuration.

Every knob is an env var (container-native); the CLI in __main__.py can
override the common ones. Defaults serve the public v1.1 model from Hugging
Face at fp32 on CPU with no auth — the combination that is correct out of the
box AND safe to state loudly (see the AUTH DISABLED banner in app.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODELS = "v1.1=hf://innovius/shinrai-pii-pathfinder-m-v1.1"

# Precision -> the exporter's artifact name inside the bundle's quant/ dir.
PRECISION_TO_FILE = {
    "fp32": "quant/model-fp32.onnx",
    "q8": "quant/model-q8.onnx",
    "int4": "quant/model-q4.onnx",
}


class ConfigError(ValueError):
    """A setting is malformed; the message says which and how to fix it."""


def parse_model_specs(raw: str) -> list[tuple[str, str]]:
    """"name=source[,name=source...]"; source is a directory path or
    hf://org/repo[@revision]. First entry is the default model."""
    specs: list[tuple[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, source = chunk.partition("=")
        if not sep or not name.strip() or not source.strip():
            raise ConfigError(
                f"SHINRAI_MODELS entry {chunk!r} is not name=source "
                "(e.g. v1.1=hf://innovius/shinrai-pii-pathfinder-m-v1.1 or v1.1=/models/v1.1)"
            )
        specs.append((name.strip(), source.strip()))
    if not specs:
        raise ConfigError("SHINRAI_MODELS is empty")
    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        raise ConfigError(f"SHINRAI_MODELS has duplicate model names: {names}")
    return specs


@dataclass(frozen=True)
class Settings:
    models: list[tuple[str, str]] = field(default_factory=lambda: parse_model_specs(DEFAULT_MODELS))
    model_cache: Path = Path("/models")
    precision: str = "fp32"
    onnx_file: str | None = None  # explicit bundle-relative path; overrides precision file
    allow_int4: bool = False
    execution_provider: str = "auto"  # auto | cpu | cuda
    threads: int = 0  # 0 = onnxruntime default
    max_concurrent: int = 1
    api_key: str | None = None
    max_text_chars: int = 200_000
    max_texts: int = 64
    host: str = "0.0.0.0"
    port: int = 8080
    self_test: str = "warn"  # off | warn | strict
    release_channel: str = "public"

    @property
    def default_model(self) -> str:
        return self.models[0][0]

    def onnx_relpath(self) -> str:
        """The bundle-relative path of the graph to load."""
        if self.onnx_file:
            return self.onnx_file
        return PRECISION_TO_FILE[self.precision]


def _read_api_key(env: dict[str, str]) -> str | None:
    key = env.get("SHINRAI_API_KEY", "").strip()
    key_file = env.get("SHINRAI_API_KEY_FILE", "").strip()
    if key and key_file:
        raise ConfigError("set SHINRAI_API_KEY or SHINRAI_API_KEY_FILE, not both")
    if key_file:
        try:
            key = Path(key_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"SHINRAI_API_KEY_FILE {key_file!r} is not readable: {exc}") from exc
        if not key:
            raise ConfigError(f"SHINRAI_API_KEY_FILE {key_file!r} is empty")
    return key or None


def _int(env: dict[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} is below the minimum {minimum}")
    return value


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)

    precision = env.get("SHINRAI_PRECISION", "fp32").strip().lower() or "fp32"
    if precision not in PRECISION_TO_FILE:
        raise ConfigError(
            f"SHINRAI_PRECISION={precision!r} — expected one of {sorted(PRECISION_TO_FILE)}"
        )

    execution_provider = env.get("SHINRAI_EXECUTION_PROVIDER", "auto").strip().lower() or "auto"
    if execution_provider not in ("auto", "cpu", "cuda"):
        raise ConfigError(
            f"SHINRAI_EXECUTION_PROVIDER={execution_provider!r} — expected auto, cpu or cuda"
        )

    self_test = env.get("SHINRAI_SELF_TEST", "warn").strip().lower() or "warn"
    if self_test not in ("off", "warn", "strict"):
        raise ConfigError(f"SHINRAI_SELF_TEST={self_test!r} — expected off, warn or strict")

    return Settings(
        models=parse_model_specs(env.get("SHINRAI_MODELS", DEFAULT_MODELS)),
        model_cache=Path(env.get("SHINRAI_MODEL_CACHE", "/models")),
        precision=precision,
        onnx_file=env.get("SHINRAI_ONNX_FILE", "").strip() or None,
        allow_int4=env.get("SHINRAI_ALLOW_INT4", "").strip() == "1",
        execution_provider=execution_provider,
        threads=_int(env, "SHINRAI_THREADS", 0),
        max_concurrent=_int(env, "SHINRAI_MAX_CONCURRENT", 1, minimum=1),
        api_key=_read_api_key(env),
        max_text_chars=_int(env, "SHINRAI_MAX_TEXT_CHARS", 200_000, minimum=1),
        max_texts=_int(env, "SHINRAI_MAX_TEXTS", 64, minimum=1),
        host=env.get("SHINRAI_HOST", "0.0.0.0").strip() or "0.0.0.0",
        port=_int(env, "SHINRAI_PORT", 8080, minimum=1),
        self_test=self_test,
        release_channel=env.get("SHINRAI_RELEASE_CHANNEL", "public").strip() or "public",
    )

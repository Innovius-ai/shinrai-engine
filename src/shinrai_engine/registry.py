"""Model registry: acquire bundles, pick execution providers, load ONNX
sessions, warm up, and attach the precision verdict + warning to each model.

Everything is EAGER: build_registry returns only when every model answered a
warmup predict whose offsets slice back to the input text — the same "a 200 on
/healthz means the whole registry serves" guarantee the internal service
gives, plus a tokenizer-skew tripwire (offsets are the wire contract;
transformers is pinned exactly for this reason)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shinrai_pii_runtime import NumpyOnnxPredictor

from . import precision as prec
from .config import Settings
from .download import ensure_model

WARMUP_TEXT = "Warmup: Lisa Müller wohnt in der Hauptstraße 10 in Berlin."


class ProviderError(RuntimeError):
    """The requested execution provider is not usable; message says why."""


@dataclass
class LoadedModel:
    name: str
    predictor: NumpyOnnxPredictor
    bundle_dir: Path
    onnx_relpath: str
    precision: str
    precision_warning: str | None
    claim_mismatch: bool
    providers: list[str] = field(default_factory=list)
    default: bool = False

    @property
    def cuda_active(self) -> bool:
        return "CUDAExecutionProvider" in self.providers


def select_providers(setting: str, log=print) -> list[str]:
    """Map auto|cpu|cuda to an onnxruntime provider list.

    cuda hard-fails when the CUDA EP is not available (an explicit ask must
    not silently degrade); auto falls back to CPU with a loud line.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if setting == "cpu":
        return ["CPUExecutionProvider"]
    cuda_available = "CUDAExecutionProvider" in available
    if setting == "cuda":
        if not cuda_available:
            raise ProviderError(
                "SHINRAI_EXECUTION_PROVIDER=cuda but onnxruntime reports no "
                f"CUDAExecutionProvider (available: {available}). Install onnxruntime-gpu "
                "on a CUDA 12 host (see the GPU section of the README) or use auto/cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    if cuda_available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    log("[engine] CUDA execution provider not available — running on CPU")
    return ["CPUExecutionProvider"]


def _warmup(name: str, predictor: NumpyOnnxPredictor) -> None:
    per_text = predictor.predict([WARMUP_TEXT])
    if not isinstance(per_text, list) or len(per_text) != 1:
        raise RuntimeError(f"{name}: warmup predict returned {type(per_text)}")
    for ent in per_text[0]:
        start, end = ent["span"]
        if WARMUP_TEXT[start:end] != ent["text"]:
            raise RuntimeError(
                f"{name}: warmup offsets do not slice back to the text "
                f"({ent['span']} -> {WARMUP_TEXT[start:end]!r} != {ent['text']!r}). "
                "This indicates tokenizer offset skew — check the transformers pin."
            )


def load_model(
    name: str,
    source: str,
    settings: Settings,
    *,
    default: bool = False,
    log=print,
) -> LoadedModel:
    onnx_relpath = settings.onnx_relpath()
    bundle_dir = ensure_model(name, source, settings.model_cache, onnx_relpath, log=log)
    onnx_path = bundle_dir / onnx_relpath

    if settings.onnx_file:
        # Custom file: the claim comes from its name, or is unknown (None).
        claimed = prec.claimed_precision(onnx_path)
    else:
        claimed = settings.precision
    actual, mismatch = prec.resolve_precision(onnx_path, claimed)
    if mismatch:
        log(
            f"[engine] {name}: file claims {claimed} but the graph scan says {actual} — "
            "trusting the scan; the file is mislabeled."
        )
    if actual == prec.INT4 and not settings.allow_int4:
        raise ProviderError(f"{name}: {prec.INT4_REFUSAL}")

    providers = select_providers(settings.execution_provider, log=log)
    predictor = NumpyOnnxPredictor(
        onnx_path,
        bundle_dir,
        providers=providers,
        intra_op_threads=settings.threads or None,
    )
    active = list(predictor.session.get_providers())
    _warmup(name, predictor)

    warning = prec.warning_for(actual, cuda_active="CUDAExecutionProvider" in active)
    return LoadedModel(
        name=name,
        predictor=predictor,
        bundle_dir=bundle_dir,
        onnx_relpath=onnx_relpath,
        precision=actual,
        precision_warning=warning,
        claim_mismatch=mismatch,
        providers=active,
        default=default,
    )


def build_registry(settings: Settings, log=print) -> dict[str, LoadedModel]:
    import time

    registry: dict[str, LoadedModel] = {}
    for name, source in settings.models:
        started = time.time()
        model = load_model(
            name, source, settings, default=(name == settings.default_model), log=log
        )
        registry[name] = model
        log(
            f"[engine] mounted {name} [{model.precision}] ({model.bundle_dir}) via "
            f"{model.providers[0]} in {time.time() - started:.1f}s "
            f"(window {model.predictor.window})"
        )
        block = prec.banner(name, model.precision, model.precision_warning)
        if block:
            log(block)
    return registry

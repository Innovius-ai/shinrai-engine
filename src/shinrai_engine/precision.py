"""Precision detection and the honest warnings.

The warnings state exactly what is measured and what is not. The recorded
benchmarks put q8 within ±0.33 macro-F1 of fp32 on every eval suite, and int4
0.75–2.10 macro-F1 below fp32 on spot checks — yet a 2026-08-11 field test on
real-world documents found q8 quality below its benchmark verdict (not yet
quantified; investigation open). fp32 is therefore the only supported
production precision, and nothing here claims a number that was never
measured.

Detection is two-layered: the filename/env states a CLAIM, and a chunked
binary scan of the .onnx protobuf states the FACT (op-type strings are stored
verbatim, so no proto parse of a 1.2 GB file is needed). On mismatch the scan
wins — a mislabeled file must not silence the warning.
"""

from __future__ import annotations

import re
from pathlib import Path

FP32 = "fp32"
Q8 = "q8"
INT4 = "int4"
INT8 = Q8  # alias so the op-needle table below reads clearly

# Ops that only appear in quantized graphs. MatMulNBits is the int4
# weight-only scheme (which also carries an int8 Gather — int4 wins the
# classification because it is the riskier claim).
_INT4_OPS = (b"MatMulNBits",)
_INT8_OPS = (b"DynamicQuantizeLinear", b"MatMulInteger", b"QLinearMatMul")

_FILENAME_CLAIM = re.compile(r"model-(fp32|q8|q4)\.onnx$")

WARNING_Q8 = (
    "PRECISION WARNING: this model runs int8 (q8) weights. Recorded benchmarks put q8 "
    "within ±0.33 macro-F1 of fp32 on every eval suite, but a 2026-08-11 field test on "
    "real-world documents found q8 quality below its benchmark verdict (not yet "
    "quantified). fp32 is the only supported production precision. Use q8 for "
    "evaluation or resource-constrained trials at your own risk."
)
WARNING_INT4 = (
    "PRECISION WARNING: this model runs int4 (MatMulNBits) weights, measured "
    "0.75-2.10 macro-F1 below fp32 on benchmark spot checks. int4 is experimental "
    "and not supported for production use."
)
WARNING_Q8_CUDA = (
    " Note: int8 dynamic-quantized graphs have no CUDA kernels for their quantized "
    "ops; most of this model executes on CPU despite the CUDA provider. Use fp32 on GPU."
)

INT4_REFUSAL = (
    "refusing to load an int4 model without SHINRAI_ALLOW_INT4=1. int4 is measured "
    "0.75-2.10 macro-F1 below fp32 and is not supported for production; set "
    "SHINRAI_ALLOW_INT4=1 only for evaluation."
)


def claimed_precision(path: str | Path) -> str | None:
    """What the filename claims, or None for custom names."""
    match = _FILENAME_CLAIM.search(Path(path).name)
    if not match:
        return None
    return {"fp32": FP32, "q8": Q8, "q4": INT4}[match.group(1)]


def scan_onnx_precision(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Classify the graph by scanning for quantized op names.

    Overlapping chunks so an op name split across a chunk boundary is still
    found. ~2-4 s for a 1.2 GB file; runs once per startup.
    """
    overlap = 64
    needles = [(op, INT4) for op in _INT4_OPS] + [(op, INT8) for op in _INT8_OPS]
    found_int8 = False
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            window = tail + chunk
            for needle, kind in needles:
                if needle in window:
                    if kind == INT4:
                        return INT4
                    found_int8 = True
            tail = window[-overlap:]
    return Q8 if found_int8 else FP32


def resolve_precision(path: str | Path, claimed: str | None) -> tuple[str, bool]:
    """(actual precision, claim_mismatch). The scan wins over the claim."""
    actual = scan_onnx_precision(path)
    mismatch = claimed is not None and claimed != actual
    return actual, mismatch


# Fewer bits = less trusted. Used to pick the fail-safe verdict when the
# exporter's declaration and the graph scan disagree: warnings must survive a
# wrong label in either direction.
_BADNESS = {INT4: 0, Q8: 1, FP32: 2}


def more_conservative(a: str, b: str) -> str:
    return a if _BADNESS.get(a, 2) <= _BADNESS.get(b, 2) else b


def manifest_precision(bundle_dir: str | Path, onnx_relpath: str) -> str | None:
    """The exporter's declared precision for this artifact, from the bundle's
    quant/MANIFEST.json (sha-verified at download time). None when the bundle
    has no manifest or the file is not listed."""
    import json

    manifest_path = Path(bundle_dir) / "quant" / "MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = next(
        (a for a in manifest.get("artifacts", []) if a.get("file") == Path(onnx_relpath).name),
        None,
    )
    if not entry:
        return None
    fmt = str(entry.get("format", "")).lower()
    if "int4" in fmt or "q4" in fmt:
        return INT4
    if "int8" in fmt or "q8" in fmt:
        return Q8
    if "fp32" in fmt or "float32" in fmt:
        return FP32
    return None


def warning_for(precision: str, *, cuda_active: bool = False) -> str | None:
    if precision == Q8:
        return WARNING_Q8 + (WARNING_Q8_CUDA if cuda_active else "")
    if precision == INT4:
        return WARNING_INT4
    return None


def banner(model_name: str, precision: str, warning: str | None) -> str | None:
    """The bordered startup block; None when there is nothing to warn about."""
    if warning is None:
        return None
    bar = "!" * 78
    return f"{bar}\n!! {model_name} [{precision}]\n!! {warning}\n{bar}"

"""Precision claims, the binary op scan, and the int4 gate."""

from __future__ import annotations

import pytest

from shinrai_engine import precision as prec


def test_claimed_precision_from_filenames():
    assert prec.claimed_precision("quant/model-fp32.onnx") == "fp32"
    assert prec.claimed_precision("quant/model-q8.onnx") == "q8"
    assert prec.claimed_precision("quant/model-q4.onnx") == "int4"
    assert prec.claimed_precision("something-custom.onnx") is None


def _write(path, payload: bytes):
    path.write_bytes(payload)
    return path


def test_scan_classifies_by_op_names(tmp_path):
    fp32 = _write(tmp_path / "a.onnx", b"\x08\x01" + b"Gather MatMul" * 100)
    q8 = _write(tmp_path / "b.onnx", b"junk DynamicQuantizeLinear junk")
    q8b = _write(tmp_path / "b2.onnx", b"junk MatMulInteger junk")
    int4 = _write(tmp_path / "c.onnx", b"junk MatMulNBits junk")
    assert prec.scan_onnx_precision(fp32) == "fp32"
    assert prec.scan_onnx_precision(q8) == "q8"
    assert prec.scan_onnx_precision(q8b) == "q8"
    assert prec.scan_onnx_precision(int4) == "int4"


def test_scan_finds_needle_across_chunk_boundary(tmp_path):
    needle = b"MatMulNBits"
    payload = b"x" * (1024 - 5) + needle + b"y" * 64
    path = _write(tmp_path / "split.onnx", payload)
    assert prec.scan_onnx_precision(path, chunk_bytes=1024) == "int4"


def test_resolve_mismatch_scan_wins(tmp_path):
    # File named q8 but the graph carries no quantized op -> fp32 + mismatch.
    path = _write(tmp_path / "model-q8.onnx", b"Gather MatMul only")
    actual, mismatch = prec.resolve_precision(path, "q8")
    assert actual == "fp32"
    assert mismatch is True


def test_warnings_wording():
    assert prec.warning_for("fp32") is None
    q8 = prec.warning_for("q8")
    assert "fp32 is the only supported production precision" in q8
    assert "not yet" in q8  # unquantified field test stays unquantified
    assert "65" not in q8  # no invented numbers
    q8_cuda = prec.warning_for("q8", cuda_active=True)
    assert "no CUDA kernels" in q8_cuda
    int4 = prec.warning_for("int4")
    assert "0.75-2.10" in int4
    assert "not supported for production" in int4


def test_int4_gate_refuses_without_optin(tiny_bundle, tmp_path):
    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import ProviderError, load_model

    # Give the bundle an int4-looking file (scan classifies by content).
    q4 = tiny_bundle / "quant" / "model-q4.onnx"
    q4.write_bytes(b"MatMulNBits fake graph")
    try:
        settings = load_settings(
            {
                "SHINRAI_MODELS": f"tiny={tiny_bundle}",
                "SHINRAI_MODEL_CACHE": str(tmp_path),
                "SHINRAI_PRECISION": "int4",
            }
        )
        with pytest.raises(ProviderError, match="SHINRAI_ALLOW_INT4"):
            load_model("tiny", str(tiny_bundle), settings, log=lambda *a: None)
    finally:
        q4.unlink()


def test_mislabeled_file_loads_with_scan_verdict(tiny_bundle, tmp_path):
    """A file claiming q8 that is really fp32 loads fine and reports fp32."""
    import shutil

    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import load_model

    bundle_copy = tmp_path / "bundle"
    shutil.copytree(tiny_bundle, bundle_copy)
    shutil.copy(bundle_copy / "quant" / "model-fp32.onnx", bundle_copy / "quant" / "model-q8.onnx")
    settings = load_settings(
        {
            "SHINRAI_MODELS": f"tiny={bundle_copy}",
            "SHINRAI_MODEL_CACHE": str(tmp_path / "cache"),
            "SHINRAI_PRECISION": "q8",
        }
    )
    model = load_model("tiny", str(bundle_copy), settings, log=lambda *a: None)
    assert model.precision == "fp32"
    assert model.claim_mismatch is True
    assert model.precision_warning is None  # the fact wins: fp32 has no warning

"""Settings parsing and validation."""

from __future__ import annotations

import pytest

from shinrai_engine.config import ConfigError, load_settings, parse_model_specs


def test_defaults():
    settings = load_settings({})
    assert settings.models == [("v1.1", "hf://innovius/shinrai-pii-pathfinder-m-v1.1")]
    assert settings.default_model == "v1.1"
    assert settings.precision == "fp32"
    assert settings.onnx_relpath() == "quant/model-fp32.onnx"
    assert settings.api_key is None
    assert settings.port == 8080
    assert settings.self_test == "warn"


def test_parse_model_specs():
    assert parse_model_specs("a=/x, b=hf://o/r") == [("a", "/x"), ("b", "hf://o/r")]
    with pytest.raises(ConfigError):
        parse_model_specs("bare-path-without-name")
    with pytest.raises(ConfigError):
        parse_model_specs("")
    with pytest.raises(ConfigError):
        parse_model_specs("a=/x,a=/y")


def test_precision_selects_file():
    assert load_settings({"SHINRAI_PRECISION": "q8"}).onnx_relpath() == "quant/model-q8.onnx"
    assert (
        load_settings({"SHINRAI_PRECISION": "int4", "SHINRAI_ALLOW_INT4": "1"}).onnx_relpath()
        == "quant/model-q4.onnx"
    )
    with pytest.raises(ConfigError, match="SHINRAI_PRECISION"):
        load_settings({"SHINRAI_PRECISION": "fp16"})


def test_onnx_file_overrides_precision():
    settings = load_settings({"SHINRAI_ONNX_FILE": "custom/my.onnx"})
    assert settings.onnx_relpath() == "custom/my.onnx"


def test_api_key_sources(tmp_path):
    assert load_settings({"SHINRAI_API_KEY": "abc"}).api_key == "abc"
    key_file = tmp_path / "key"
    key_file.write_text("  from-file\n")
    assert load_settings({"SHINRAI_API_KEY_FILE": str(key_file)}).api_key == "from-file"
    with pytest.raises(ConfigError, match="not both"):
        load_settings({"SHINRAI_API_KEY": "a", "SHINRAI_API_KEY_FILE": str(key_file)})
    with pytest.raises(ConfigError, match="not readable"):
        load_settings({"SHINRAI_API_KEY_FILE": str(tmp_path / "absent")})


def test_validation_errors():
    with pytest.raises(ConfigError):
        load_settings({"SHINRAI_EXECUTION_PROVIDER": "tpu"})
    with pytest.raises(ConfigError):
        load_settings({"SHINRAI_SELF_TEST": "loud"})
    with pytest.raises(ConfigError):
        load_settings({"SHINRAI_THREADS": "many"})
    with pytest.raises(ConfigError):
        load_settings({"SHINRAI_MAX_CONCURRENT": "0"})


def test_max_concurrent_above_one_refused():
    # The shared tokenizer is not thread-safe under concurrent calls; the
    # knob refuses loudly instead of accepting-and-crashing under load.
    with pytest.raises(ConfigError, match="thread-safe"):
        load_settings({"SHINRAI_MAX_CONCURRENT": "4"})


def test_int4_requires_optin_at_config_time():
    # Refusing here beats downloading 257 MB and then refusing (which on an
    # emptyDir deployment crash-loops against Hugging Face).
    with pytest.raises(ConfigError, match="SHINRAI_ALLOW_INT4"):
        load_settings({"SHINRAI_PRECISION": "int4"})
    settings = load_settings({"SHINRAI_PRECISION": "int4", "SHINRAI_ALLOW_INT4": "1"})
    assert settings.allow_int4 and settings.precision == "int4"

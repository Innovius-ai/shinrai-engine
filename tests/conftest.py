"""Test fixtures: a ~2 MB fully synthetic model bundle.

The tiny ONNX graph reproduces the exact serving IO contract (input_ids /
attention_mask int64 [batch, sequence]; logits_{person,city,street,org}
[b,s,7]; attr_logits_{origin,gender_expression,name_part}) with weights
biased so every content token argmaxes to B-PERSON-COMMON — deterministic,
non-empty entities with valid offsets, no real weights involved. The
tokenizer is a tiny BPE trained on the fixture sentences. labels-v2.0.yaml is
the canonical (publishable) label inventory. quant/MANIFEST.json carries the
REAL sha256 of the generated graph so download verification is exercised.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

SENTENCES = [
    "Anna Miller lives in Berlin near the main station.",
    "Peter Schmidt works at the bakery on Bahnhofstrasse.",
    "Warmup: Lisa Miller wohnt in der Hauptstrasse 10 in Berlin.",
    "The quick brown fox jumps over the lazy dog.",
]

HEAD_OUTPUTS = [
    # (output name, classes, hot index — 1 = B-<HEAD>-<TIER0>, 0 = O)
    ("logits_person", 7, 1),
    ("logits_city", 7, 0),
    ("logits_street", 7, 0),
    ("logits_org", 7, 0),
    ("attr_logits_origin", 42, 0),
    ("attr_logits_gender_expression", 4, 0),
    ("attr_logits_name_part", 3, 0),
]

VOCAB_LIMIT = 4096
HIDDEN = 8


def build_tiny_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    embedding = np.ones((VOCAB_LIMIT, HIDDEN), dtype=np.float32)
    initializers = [numpy_helper.from_array(embedding, "embedding")]
    nodes = [helper.make_node("Gather", ["embedding", "input_ids"], ["hidden"], axis=0)]
    outputs = []
    for name, classes, hot in HEAD_OUTPUTS:
        weight = np.zeros((HIDDEN, classes), dtype=np.float32)
        weight[:, hot] = 1.0
        initializers.append(numpy_helper.from_array(weight, f"W_{name}"))
        nodes.append(helper.make_node("MatMul", ["hidden", f"W_{name}"], [name]))
        outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", "sequence", classes])
        )
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
    ]
    graph = helper.make_graph(nodes, "tiny-shinrai", inputs, outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def build_tiny_tokenizer(tokenizer_dir: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], vocab_size=300
    )
    tokenizer.train_from_iterator(SENTENCES, trainer)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 64,
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_tiny_bundle(root: Path) -> Path:
    """Create a complete bundle under root and return its directory."""
    bundle = root / "tiny-shinrai-test-model"
    onnx_path = bundle / "quant" / "model-fp32.onnx"
    build_tiny_onnx(onnx_path)
    build_tiny_tokenizer(bundle / "tokenizer")
    shutil.copy(FIXTURES / "labels-v2.0.yaml", bundle / "labels-v2.0.yaml")
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test",
                "shinrai": {
                    "format": "shinrai-multihead-v1",
                    "heads": ["PERSON", "CITY", "STREET", "ORG"],
                    "attribute_heads": True,
                    "labels_file": "labels-v2.0.yaml",
                    "max_length": 64,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle / "calibration.json").write_text(
        json.dumps({"PERSON": 1.0, "CITY": 1.0, "STREET": 1.0, "ORG": 1.0}),
        encoding="utf-8",
    )
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    (bundle / "quant" / "MANIFEST.json").write_text(
        json.dumps(
            {
                "report_version": "1.0",
                "artifacts": [
                    {
                        "file": "model-fp32.onnx",
                        "format": "onnx-fp32",
                        "size_bytes": onnx_path.stat().st_size,
                        "sha256": sha,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory) -> Path:
    return build_tiny_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture(scope="session")
def tiny_registry(tiny_bundle: Path, tmp_path_factory):
    """A loaded registry over the tiny bundle (shared — loading is ~1 s)."""
    from shinrai_engine.config import load_settings
    from shinrai_engine.registry import build_registry

    settings = load_settings(
        {
            "SHINRAI_MODELS": f"tiny={tiny_bundle}",
            "SHINRAI_MODEL_CACHE": str(tmp_path_factory.mktemp("cache")),
            "SHINRAI_SELF_TEST": "off",
        }
    )
    return settings, build_registry(settings, log=lambda *a: None)

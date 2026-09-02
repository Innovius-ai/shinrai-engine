"""NumpyOnnxPredictor: torch-free mirror of export.onnx_export.OnnxPredictor.

Exists so shinrai-encryption can run Q8 ONNX inference in-process without a
torch wheel in its image. Standalone rather than a Predictor subclass because
Predictor.predict imports torch; everything decision-bearing is still shared:
constrained IOB2 decoding comes from training.decode.spans_from_labels and the
label space from training.labels (both torch-free by design — WP-09: never
fork the decoder). The numpy softmax/argmax and per-token attr-logit pooling
are arithmetic ports verified against OnnxPredictor by
tests/test_numpy_predictor.py.

Vendoring: scripts/vendor-bert.sh copies this file (plus decode/labels/adapter)
into shinrai-encryption as `shinrai_pii_runtime/` and rewrites the
`shinrai_pii.training.*` imports below to package-relative form — keep them on
single lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .decode import spans_from_labels
from .labels import load_label_space

# Literal copies of training.model.SHINRAI_CONFIG_KEY and
# export.onnx_export.ATTR_OUTPUT_PREFIX — importing either module would pull
# torch into the runtime.
SHINRAI_CONFIG_KEY = "shinrai"
ATTR_OUTPUT_PREFIX = "attr_logits_"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class NumpyOnnxPredictor:
    """Same contract as OnnxPredictor: predict() returns per-text lists of
    {span, text, type, tier, confidence, attrs} entity dicts."""

    def __init__(
        self,
        onnx_path: str | Path,
        checkpoint_dir: str | Path | None = None,
        *,
        providers: list[str] | None = None,
        window: int | None = None,
        intra_op_threads: int | None = None,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = Path(onnx_path)
        checkpoint_dir = Path(
            checkpoint_dir if checkpoint_dir is not None else onnx_path.parent.parent
        )
        raw = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        meta = raw.get(SHINRAI_CONFIG_KEY)
        if not meta:
            raise ValueError(
                f"{checkpoint_dir}: config.json has no '{SHINRAI_CONFIG_KEY}' block "
                "(pass checkpoint_dir explicitly if the .onnx lives elsewhere)"
            )

        self.checkpoint_dir = checkpoint_dir
        self.onnx_path = onnx_path
        self.label_space = load_label_space(
            checkpoint_dir / meta.get("labels_file", "labels-v2.0.yaml")
        )
        self.meta = meta
        self.heads = list(meta["heads"])
        self.window = int(window or meta.get("max_length") or 512)
        self.stride = max(32, self.window // 8)
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir / "tokenizer"))
        calibration_path = checkpoint_dir / "calibration.json"
        self.temperatures = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_path.is_file()
            else {}
        )
        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = int(intra_op_threads)
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, texts: list[str], *, batch_size: int = 16,
                segment: str | None = None) -> list[list[dict]]:
        if segment is not None:
            # sentence-sized pieces (2026-08-22 long-input finding); None is
            # byte-identical to the sliding-window decode below
            from shinrai_pii.segment import predict_auto, predict_segmented

            if segment not in ("sentence", "auto"):
                raise ValueError(f"segment must be 'sentence', 'auto' or None, got {segment!r}")
            runner = predict_auto if segment == "auto" else predict_segmented
            return runner(
                lambda pieces, **kw: self.predict(pieces, **kw), texts, self._merge_windows,
                batch_size=batch_size,
            )
        per_text: list[list[dict]] = [[] for _ in texts]
        encoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.window,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        sample_map = encoding.pop("overflow_to_sample_mapping").tolist()
        offsets = encoding.pop("offset_mapping")
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        n_windows = input_ids.shape[0]

        for start in range(0, n_windows, batch_size):
            window_ids = list(range(start, min(start + batch_size, n_windows)))
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids[window_ids],
                    "attention_mask": attention_mask[window_ids],
                },
            )
            named = dict(zip(self._output_names, outputs, strict=True))

            for local_i, window_i in enumerate(window_ids):
                text = texts[sample_map[window_i]]
                label_ids: dict[str, list[int]] = {}
                confidences: dict[str, list[float]] = {}
                for head in self.heads:
                    logits = named[f"logits_{head.lower()}"][local_i]
                    temp = float(self.temperatures.get(head, 1.0)) or 1.0
                    probs = _softmax(logits / temp)
                    label_ids[head] = probs.argmax(axis=-1).tolist()
                    confidences[head] = probs.max(axis=-1).tolist()
                token_offsets = [tuple(o) for o in offsets[window_i].tolist()]
                entities = spans_from_labels(
                    label_ids, token_offsets, self.label_space, text, confidences
                )
                entities = self._attach_attrs(named, local_i, token_offsets, entities)
                per_text[sample_map[window_i]].extend(entities)

        return [self._merge_windows(entities) for entities in per_text]

    @staticmethod
    def _merge_windows(entities: list[dict]) -> list[dict]:
        """Verbatim port of Predictor._merge_windows: same head + overlapping
        span keeps the higher-confidence prediction."""
        kept: list[dict] = []
        for ent in sorted(entities, key=lambda e: -e.get("confidence", 0.0)):
            clash = any(
                k["type"] == ent["type"]
                and not (ent["span"][1] <= k["span"][0] or ent["span"][0] >= k["span"][1])
                for k in kept
            )
            if not clash:
                kept.append(ent)
        return sorted(kept, key=lambda e: (e["span"][0], e["span"][1]))

    def _attach_attrs(
        self,
        named_outputs: dict[str, np.ndarray],
        batch_index: int,
        token_offsets: list[tuple[int, int]],
        entities: list[dict],
    ) -> list[dict]:
        """Numpy port of OnnxPredictor._attach_attrs: pool per-token attribute
        logits over the entity's tokens, argmax to a class name. Linear
        commutes with mean, so pooling logits equals pooling hidden states."""
        attr_outputs = {
            name: named_outputs[f"{ATTR_OUTPUT_PREFIX}{name}"][batch_index]
            for name in self.label_space.attributes
            if f"{ATTR_OUTPUT_PREFIX}{name}" in named_outputs
        }
        for ent in entities:
            ent_start, ent_end = ent["span"]
            token_ids = [
                idx
                for idx, (s, e) in enumerate(token_offsets)
                if s != e and s < ent_end and e > ent_start
            ]
            attrs: dict[str, str] = {}
            if token_ids:
                lo, hi = token_ids[0], token_ids[-1] + 1
                for name, space in self.label_space.attributes.items():
                    if ent["type"] not in space.applies_to or name not in attr_outputs:
                        continue
                    pooled = attr_outputs[name][lo:hi].mean(axis=0)
                    attrs[name] = space.classes[int(pooled.argmax())]
            ent["attrs"] = attrs
        return entities

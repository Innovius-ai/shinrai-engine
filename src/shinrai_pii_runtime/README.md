# shinrai_pii_runtime — generated, do not edit by hand

Torch-free BERT inference runtime vendored from `shinrai-pii-bert` so
the shinrai-engine HTTP service runs ONNX detection without a checkout
of the training repo, cross-repo pip access, or a torch wheel.

- **Source repo:** shinrai-pii-bert
- **Source commit:** `808d49120e99e2adb8e9dba4d81e20c7c5aaae95`
- **Synced:** 2026-08-13
- **Regenerate:** `scripts/vendor-engine.sh` in shinrai-pii-bert (requires the
  sibling checkout or `SHINRAI_ENGINE_REPO=<path>`)

Contents: `decode.py` (WP-09 constrained IOB2 decoder — canonical, never
fork), `labels.py` (label space + api_mapping), `adapter.py`
(`to_legacy_entities` / `merge_person_spans`, CLI stripped),
`onnx_numpy.py` (`NumpyOnnxPredictor`). Imports are rewritten to
package-relative form by the vendor script; the anti-fork guard
`tests/test_vendor_bert_script.py` (source repo) asserts the vendored
decoder stays line-identical to the canonical one apart from that rewrite.

Model artifacts are NOT vendored — they arrive at runtime via Hugging Face
download or a mounted volume (see the shinrai-engine README).

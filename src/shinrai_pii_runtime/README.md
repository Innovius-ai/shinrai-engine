# shinrai_pii_runtime — generated, do not edit by hand

Torch-free BERT inference runtime vendored from `shinrai-pii-bert` so
the shinrai-engine HTTP service runs ONNX detection without a checkout
of the training repo, cross-repo pip access, or a torch wheel.

- **Source repo:** shinrai-pii-bert
- **Source commit:** `64e09809c7dbe91153b4d895f9a379e7c5639a2c`
- **Synced:** 2026-09-27
- **Regenerate:** `scripts/vendor-engine.sh` in shinrai-pii-bert (requires the
  sibling checkout or `SHINRAI_ENGINE_REPO=<path>`)

Contents: `decode.py` (WP-09 constrained IOB2 decoder — canonical, never
fork), `labels.py` (label space + api_mapping), `adapter.py`
(`to_legacy_entities` / `merge_person_spans`, CLI stripped),
`onnx_numpy.py` (`NumpyOnnxPredictor`), `segment.py` (long-input
sentence/auto decode), `scrub.py`
(`scrub_invisibles`, the length-preserving input scrub). Imports are rewritten to
package-relative form by the vendor script; the anti-fork guard
`tests/test_vendor_bert_script.py` (source repo) asserts the vendored
decoder stays line-identical to the canonical one apart from that rewrite.

Model artifacts are NOT vendored — they arrive at runtime via Hugging Face
download or a mounted volume (see the shinrai-engine README).

## Engine-local deltas (re-apply after every re-vendor until upstreamed)

1. `temporal_negatives.py` is copied verbatim, and the imports of it in
   `decode.py` and `segment.py` are rewritten to package-relative form. The
   upstream vendor script at the source commit does not do this yet.
2. Bounded inference: `NumpyOnnxPredictor.predict` takes `check_cancel`,
   defaults `batch_size` to 4, and splits `texts` into `batch_size` chunks;
   `predict_segmented` calls `predict` in `batch_size` chunks. The cancel hook
   runs before each chunk and each window batch.
3. Public-repo hygiene: a personal name is removed from one `decode.py`
   comment (the D23 decoder-floor note).

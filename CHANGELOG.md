# Changelog

## 0.1.7 — 2026-10-02

- Runtime re-vendored from shinrai-pii-bert
  `4e72aea38f473c9240fe529321c334111e70f2f2`: the decoder key
  `signoff_floor` (F001). A checkpoint that stamps it lowers that head's
  floor on the one or two short lines after a closing formula
  ("Mit freundlichen Grüßen," / "Max Munster"); a span rescued only by it
  carries its own bar. Without the key the decode is unchanged.
- `POST /api/analyze` applies that bar when the request sets no threshold
  or one at or below the checkpoint's stamped `serve_threshold`; a stricter
  request threshold holds for every span.

## 0.1.6 — 2026-10-01

- Runtime re-vendored from shinrai-pii-bert
  `9fdc40c3bbdfd7b37d856eb3e1188b59c8196c48`: the decoder of the ShinrAI
  v1.5.3 release (21 heads including DATE and AGE, guard v11 from the
  checkpoint stamp). A month abbreviation now counts as part of a date only
  with its period, so "12. März 2026. Sie" no longer grows a DATE into the
  next sentence.

## 0.1.5 — 2026-09-27

- `POST /api/analyze` accepts an optional `language` (BCP-47 tag, for
  example `de`, `ja`, `pt-BR`). One language applies to every text in the
  request. The engine passes it to the runtime, which uses it to select the
  per-language decoder settings that a checkpoint stamps (per-head recall
  floors, particle strip, in-word gap fill, word completion, name join).
  The runtime lower-cases the tag, maps `_` to `-`, and falls back from the
  full tag to its primary subtag (`pt-BR` -> `pt-br` -> `pt`). A malformed
  value returns 422. Without `language`, no per-language setting applies.
- Runtime re-vendored from shinrai-pii-bert
  `64e09809c7dbe91153b4d895f9a379e7c5639a2c`. New: the stampable decoder
  settings block (also settable per deployment with
  `SHINRAI_DECODER_SETTINGS`), `temporal_negatives.py` (month tables the
  decoder and the sentence splitter use), and stricter label-schema checks.
  The engine's bounded-inference changes to the runtime (`check_cancel`,
  batch chunking) are re-applied; see `src/shinrai_pii_runtime/README.md`.
- Behaviour change without `language`: the re-vendored sentence splitter no
  longer splits after a 1-2 digit day ordinal (`15. März`) or a month
  abbreviation (`3. Sept. 2025`), and it glues a short first piece (`Dr.`)
  forward. This changes `segment: "sentence"` output and the long-input part
  of `segment: "auto"` (texts above 1,200 characters). Whole-text decode is
  unchanged: `segment: "none"`/`"whole"` and every `auto` text up to 1,200
  characters give identical entities. Measured on 324 mixed texts (en, de, es,
  fr, and a few others) without `language`: v1.3 `auto` 17 texts differ
  (1,715 -> 1,707 entities), `sentence` 103 differ; v1.4 `auto` 19 differ
  (3,323 -> 3,317), `sentence` 124 differ. Every difference traces to changed
  sentence pieces, except two 4th-decimal confidence changes from different
  batch padding.
- The entity output shape is unchanged, including `evidence`.

## 0.1.4 — 2026-09-11

- Added retained-model lifecycle controls: entries named in
  `SHINRAI_LAZY_MODELS` load on first activation and unload after
  `SHINRAI_MODEL_IDLE_TTL_SECONDS` of inactivity (four hours by default).
- Added authenticated, idempotent `POST /api/models/{name}/activate` and
  model state metadata on health/model endpoints.
- Default detection thresholds can now come from the sealed bundle's decoder
  stamp; callers can still provide an explicit threshold.
- Added sentence-aware segmentation support required by the v1.4 decoder and
  refreshed the vendored, model-independent inference runtime.

## 0.1.3 — 2026-08-13

- License files name the full legal entity: Innovius UG (haftungsbeschränkt).
  No code change.

## 0.1.2 — 2026-08-13

Pre-publication review release. License changed to BSD 3-Clause (Innovius UG (haftungsbeschränkt)).

Fixed, found by an 8-angle review before going public:
- Concurrency: all tokenizer work now runs in one gated worker thread — the
  stats block previously re-tokenized on the event loop with the shared HF
  tokenizer (crashes under concurrent requests, stalls /healthz on long
  documents). `SHINRAI_MAX_CONCURRENT` is fixed at 1 until per-slot
  predictors land.
- Restored the length-preserving invisible-character scrub from the
  reference service (zero-width/bidi characters could hide entities).
- Person merging now thresholds components first (upstream fix, vendored):
  a weak family name could previously sink a strong given name entirely.
- Helm: default image tag is v-prefixed (matches published tags — plain
  `helm install` no longer 404s); `gpu.enabled` keeps its `-gpu` suffix on
  pinned tags; wholesale `resources` overrides render correctly; `env`
  cannot silently duplicate first-class keys; values.schema covers the full
  surface and refuses int4 without the opt-in.
- Downloader: install marker now checks repo AND revision (source switches
  re-stage instead of serving stale weights); interrupted downloads keep
  their partial for resume; `labels_file` from downloaded configs is
  traversal-guarded; hf:// repo ids validated properly.
- Auth: non-ASCII bearer tokens get 401 instead of 500; non-ASCII configured
  keys are refused at startup with a clear message.
- Self-test: golden files resolve by exact bundle identity (never name
  substrings); verdicts are recorded and surfaced on /healthz;
  `selftest --url` works inside the shipped image (stdlib HTTP).
- Precision: the sha-verified MANIFEST declaration now participates — on
  disagreement with the graph scan, the more conservative verdict wins.
- stats.windows uses the tokenizer's real step (was under-reporting on long
  documents); int4-without-opt-in is refused at config time.
- CI: lint/helm-lint/tests run on merge-request pipelines; kaniko layer
  cache enabled. Dockerfile.bundle ships its own dockerignore (the root one
  excluded the models it must COPY).

## 0.1.1 — 2026-08-13

- Fix release CI: image tags were built with literal quote characters on tag
  pipelines (kaniko refused the destination). No runtime change.

## 0.1.0 — 2026-08-13

Initial public release.

- HTTP inference service for the ShinrAI PII detection models (ONNX,
  torch-free): `POST /api/analyze`, `GET /api/models`, `/healthz`, `/metrics`.
- Model acquisition: Hugging Face auto-download (serving subset only,
  sha256-verified, atomic install) or mounted bundle.
- Precision handling: fp32 default; int8 loads with an honest warning; int4
  gated behind `SHINRAI_ALLOW_INT4=1`. Graph-scan detection beats filenames.
- Optional bearer-token auth; loud AUTH DISABLED banner otherwise.
- Golden self-test at startup (`off|warn|strict`).
- CPU image (python-slim + onnxruntime 1.28) and CUDA-12 GPU image
  (onnxruntime-gpu 1.26 line), compose profiles, bundle variant.
- Standalone Helm chart; plain `helm install` works on stock microk8s.

# Changelog

## 0.1.2 — 2026-08-13

Pre-publication review release. License changed to BSD 3-Clause (Innovius UG).

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

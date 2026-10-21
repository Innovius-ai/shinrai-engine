# Changelog

## 0.1.6 — 2026-10-21

- `certifi>=2026.10.19`: the CA bundle of the model downloads from Hugging Face
  no longer trusts the emSign roots with compromised intermediate keys
  (GHSA-6q48-6c7w-9gw2). An image built before 2026-10-20 kept the older
  bundle.
- Both images pin their base by digest: CPU `python:3.12.15-slim-trixie`
  (2026-10-21), GPU `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (built
  2024-12-03). Both run `apt-get upgrade`, so the Debian and Ubuntu security
  updates published after the base was built apply at build time. The CUDA
  packages of the GPU base are apt-held and do not change.
- transformers stays 4.57.6. Its open advisories need model loading,
  `save_pretrained` or custom generation code; the engine uses only the fast
  tokenizer of a local bundle. The tokenizer output of 5.20.2 is byte-identical
  for the published bundles; a switch is a later release.

## 0.1.5 — 2026-10-11

- The default model is the highest version in `SHINRAI_MODELS` (`v1.10` >
  `v1.9`), wherever it is listed; it was the first entry. A deployment that
  lists its newest model first sees no change. `SHINRAI_DEFAULT_MODEL` (or
  `--default-model`) pins another listed model. The default model still cannot
  be lazy.
- README: v1.1 and v1.2 are public under Apache-2.0 at
  `innovius/shinrai-pii-m-v1.1` and `innovius/shinrai-pii-m-v1.2`.
- Package version strings agree again (`__version__` lagged at 0.1.3).

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

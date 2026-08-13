# Changelog

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

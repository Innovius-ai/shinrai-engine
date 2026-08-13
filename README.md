# ShinrAI Engine

Self-hostable HTTP inference service for the [ShinrAI PII detection
models](https://huggingface.co/innovius) — run PII detection centrally on
your own hardware, CPU or NVIDIA GPU. BSD-3-Clause.

The engine answers one question — *which spans of this text are personal
data?* — over a small, stable HTTP API. It is the detection backend for the
ShinrAI Connector desktop app and for central ShinrAI deployments, and it
works standalone for anything else that wants span-level PII detection.

> Development happens on Innovius' internal GitLab; this GitHub repository
> is the public mirror. Issues are welcome here. Pull requests are reviewed
> and ported manually.

## Quickstart

```bash
docker run -p 8080:8080 -v shinrai-models:/models \
  ghcr.io/innovius-ai/shinrai-engine:latest
```

First start downloads the fp32 v1.1 model (~1.3 GB) from Hugging Face into
the volume, verifies its sha256, loads it, and self-tests. Then:

```bash
curl -s -X POST http://127.0.0.1:8080/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Anna Miller lives in Berlin, her office is on Bahnhofstrasse 5."}'
```

```json
{
  "model": "v1.1",
  "results": [{
    "entities": [
      {"text": "Anna Miller", "type": "PERSON", "startIndex": 0, "endIndex": 11,
       "tier": "common", "attrs": {"origin": "US", "gender_expression": "fem",
       "name_part": "full"}, "source": "bert", "confidence": 0.99, "region": "US"},
      {"text": "Berlin", "type": "CITY", "startIndex": 21, "endIndex": 27, "...": "..."}
    ],
    "stats": {"chars": 63, "tokens": 18, "windows": 1}
  }],
  "timing_ms": {"total": 31.2, "inference": 28.9},
  "version": "0.1.2", "release_channel": "public"
}
```

Alternatives:

```bash
# docker compose (profiles: default CPU, gpu, mounted)
docker compose -f docker/docker-compose.yml up engine

# build from source
docker build -f docker/Dockerfile.cpu -t shinrai-engine .

# bare venv, no container
pip install -e ".[cpu]"
SHINRAI_MODEL_CACHE=./models python -m shinrai_engine
```

## Models & precision

| Artifact | Size | Status |
|---|---|---|
| fp32 (`quant/model-fp32.onnx`) | 1.23 GB | **The supported production precision.** |
| int8 / q8 (`quant/model-q8.onnx`) | 309 MB | Evaluation only — see the warning below. |
| int4 / q4 (`quant/model-q4.onnx`) | 257 MB | Experimental; refuses to load without `SHINRAI_ALLOW_INT4=1`. |

**Why fp32 only in production.** Measured macro-F1 (strict span match, over
the 4 entity types; full per-type detail ships with the model cards):

| suite | fp32 | int8 | Δ | int4 |
|---|---|---|---|---|
| v1-suite | 99.55 | 99.22 | −0.33 | — |
| v1.2-de | 92.35 | 92.38 | +0.03 | 91.15 |
| v1.2-en | 94.45 | 94.45 | ±0.00 | — |
| v1.2-it | 92.75 | 92.55 | −0.20 | — |
| v1.2-fr | 94.18 | 94.07 | −0.10 | — |
| v1.2-es | 94.05 | 94.20 | +0.15 | — |
| v1.2-pl | 89.40 | 89.10 | −0.30 | 88.65 |
| v1.1-de | 82.22 | 82.72 | +0.50 | — |
| v1.1-en | 84.25 | 84.33 | +0.08 | — |
| v1.1-ja | 60.80 | 60.80 | ±0.00 | 58.70 |

On benchmarks int8 looks nearly free — **but a 2026-08-11 field test on
real-world documents found int8 output quality below its benchmark verdict.
That gap is not yet quantified and is under investigation.** Until it is
resolved, fp32 is the only precision we support for production use; the
engine says so at startup, on `/healthz`, and in the Helm release notes
whenever you select anything else. int4 is additionally measured 0.75–2.10
macro-F1 below fp32 on spot checks and is gated behind an explicit opt-in.

The precision is detected from the actual graph (a binary scan for
quantized ops), not just the filename — a mislabeled file warns.

**Models.** `v1.1` ([public](https://huggingface.co/innovius/shinrai-pii-pathfinder-m-v1.1),
de/en/ja, 512-token window) is the default. `v1.2` (de/en/it/fr/es/pl,
1024-token window, ja stays on v1.1) is currently a private repo — set
`HF_TOKEN` and `SHINRAI_MODELS=v1.2=hf://innovius/shinrai-pii-pathfinder-m-v1.2`
once you have access. Entity heads: PERSON, CITY, STREET, ORG, plus span
attributes (origin, gender expression, name part).

## Configuration

Everything is an env var:

| Variable | Default | Meaning |
|---|---|---|
| `SHINRAI_MODELS` | `v1.1=hf://innovius/shinrai-pii-pathfinder-m-v1.1` | Comma-separated `name=source`; source is a bundle dir or `hf://org/repo[@rev]`. First = default model. |
| `SHINRAI_MODEL_CACHE` | `/models` | Where hf:// bundles are installed. |
| `SHINRAI_PRECISION` | `fp32` | `fp32` \| `q8` \| `int4` — picks `quant/model-*.onnx`. |
| `SHINRAI_ONNX_FILE` | – | Explicit bundle-relative graph path; overrides the precision file. |
| `SHINRAI_ALLOW_INT4` | – | Must be `1` for int4 graphs to load. |
| `SHINRAI_EXECUTION_PROVIDER` | `auto` | `auto` \| `cpu` \| `cuda`. `cuda` hard-fails if unavailable; `auto` falls back loudly. |
| `SHINRAI_THREADS` | ORT default | `intra_op_num_threads`. |
| `SHINRAI_MAX_CONCURRENT` | `1` | Fixed at 1 for now (the tokenizer is not thread-safe under concurrent calls) — scale replicas instead. |
| `SHINRAI_API_KEY` / `SHINRAI_API_KEY_FILE` | – | Enables bearer auth on `/api/*` and `/metrics`. |
| `SHINRAI_MAX_TEXT_CHARS` / `SHINRAI_MAX_TEXTS` | `200000` / `64` | Request limits (413 beyond). |
| `SHINRAI_HOST` / `SHINRAI_PORT` | `0.0.0.0` / `8080` | Bind address. |
| `SHINRAI_SELF_TEST` | `warn` | `off` \| `warn` \| `strict` — golden self-test at startup; verdict on `/healthz`. |
| `SHINRAI_RELEASE_CHANNEL` | `public` | Echoed in `/api/analyze` and `/metrics` responses. |
| `HF_TOKEN` / `HF_HUB_OFFLINE` | – | Private repos / air-gapped operation. |

## API

All offsets are **Unicode code-point indices** into the submitted text
(`text[startIndex:endIndex] == entity.text` in Python). UTF-8 or UTF-16
consumers must convert — a `ü` is one code point but two UTF-8 bytes.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | open | Service info: version, models, precision, auth state. |
| `GET /healthz` (also `/health`) | open | 200 only when every model is loaded and warmed up. Reports `precision`, `precision_warning`, `providers`, the golden `self_test` verdict, and per-model `models_detail`. |
| `GET /metrics` | gated | Totals + rolling p50/p95 (JSON). |
| `GET /api/models` | gated | Per model: `name`, `default`, `window`, `precision`. |
| `POST /api/analyze` | gated | The inference call. |

`POST /api/analyze` request:

```json
{"text": "...",            // or "texts": ["...", "..."] (max 64)
 "model": "v1.1",          // optional; default = first configured model
 "threshold": 0.7,          // confidence gate (calibrated)
 "merge_persons": true}     // join adjacent given+family name spans
```

Entity `type` values: `FIRSTNAME`, `SURNAME`, `PERSON` (by detected name
part), `CITY`, `STREET_ADDRESS`, `COMPANY`. Each entity carries `tier`,
`attrs` (origin / gender_expression / name_part), `confidence`, `region`,
`source: "bert"`.

Auth, when enabled: `Authorization: Bearer <key>`. Without a key configured
the engine prints `AUTH DISABLED` at startup — run it cluster-internal only.

## GPU

Use the `-gpu` image (CUDA 12 runtime + `onnxruntime-gpu` 1.26 line):

```bash
docker run --gpus all -p 8080:8080 -v shinrai-models:/models \
  ghcr.io/innovius-ai/shinrai-engine:latest-gpu
```

Why the 1.26 pin: `onnxruntime-gpu >= 1.27` is built for CUDA 13, which
dropped Pascal support entirely. The 1.26/CUDA-12 line targets compute
capability >= 6.0, so Pascal cards (P40, sm_61) are expected to work but are
not something we have verified on every card — check `/healthz`:
`"providers": ["CUDAExecutionProvider", ...]` means the GPU is actually in
use. Turing or newer can bump `ORT_GPU_VERSION` (build arg) to a CUDA-13
line.

Two honest caveats: int8 graphs have no CUDA kernels for their quantized ops
and mostly run on CPU regardless of provider (GPU serving is effectively
fp32-only), and we publish no GPU latency numbers yet because we have not
measured any.

## Kubernetes / Helm / microk8s

```bash
helm install shinrai-engine ./helm/shinrai-engine \
  --namespace shinrai-engine --create-namespace \
  --set persistence.enabled=true \
  --set apiKey.value="$(openssl rand -hex 24)"
```

Plain `helm install` with defaults works on a stock microk8s (emptyDir +
first-start download). See [helm/shinrai-engine/README.md](helm/shinrai-engine/README.md)
for the knobs and the `microk8s enable hostpath-storage|gpu|ingress` walkthrough.

## Security

- **Stateless.** Texts are processed in memory and never persisted or
  logged; `/metrics` carries counts and latencies only.
- **Auth is opt-in** and off by default (for drop-in compatibility with
  existing internal consumers). Off means: anyone who can reach the port can
  use the API. Keep it cluster-internal or set `SHINRAI_API_KEY`.
- **TLS** is out of scope for the container — terminate at your ingress or
  reverse proxy.
- **Downloads are verified**: sha256 against the model bundle's signed-off
  manifest; staged and atomically installed.

## Integrations

**ShinrAI Connector (desktop app).** Point the connector at your engine:
Engine tab → *Self-hosted engine* → enter `http://<host>:8080` (+ API key),
Test connection, Use this endpoint. The daemon equivalent is
`shinraid --model-endpoint http://<host>:8080` (token via
`SHINRAI_ENDPOINT_TOKEN`). Pattern-based detection stays local; the engine
adds the model heads, and pseudonym mapping never leaves the user's machine.

**ShinrAI central service.** The engine is a drop-in remote BERT backend:
set the service's `bert_detection.mode: remote` with `remote.url:
http://<engine-service>:8080`. Note the service sends no auth header today —
leave engine auth off on that path and keep both cluster-internal.

## Performance (measured, CPU)

Single-request latency (concurrency 1), v1.1, Apple M-series dev machine,
`intra_op_threads: 2` — measurement conditions in the model repo's serving
bench:

| payload | fp32 p50 | int8 p50 |
|---|---|---|
| ~200 chars | 26 ms | 22 ms |
| ~400 tokens | 101 ms | 102 ms |
| ~1–2k tokens | 588 ms | 537 ms |

fp32 over int8 costs 0–18% latency on this CPU class. Memory: ~1.5 GiB RSS
per loaded fp32 model — the fp32 default costs RAM, not speed. Server-class
Xeon numbers and GPU numbers are not yet published (not measured).

## Development

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                    # hermetic: runs against a synthetic 2 MB bundle
ruff check src tests
# against a real bundle:
SHINRAI_TEST_MODEL_DIR=/path/to/shinrai-pii-pathfinder-m-v1.1 pytest -m slow
```

`src/shinrai_pii_runtime/` is **generated** from the ShinrAI model training
repo (single source of truth for decoding — see its README for provenance).
Never edit it by hand; PRs against it will be closed and ported upstream
instead.

## License

BSD 3-Clause — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Redistributions
must retain the copyright notice naming Innovius UG. The models are published
separately under Apache-2.0 at https://huggingface.co/innovius.

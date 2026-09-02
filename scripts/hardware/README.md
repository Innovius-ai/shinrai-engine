# Hardware benchmark kit (vendored)

Copied from `scripts/hardware/` in the ShinrAI training repo (shinrai-pii-bert) —
do not hand-edit here; changes go upstream and get re-copied together with
`src/shinrai_pii_runtime/`.

- `bench-local.py` — direct ONNX Runtime benchmark (payload + raw modes) using the
  vendored torch-free runtime; a correctness guard replays the release smoke predictions
  (`guard-shinrai-pii-m-v1.3.json`) before any timing is recorded.
- `bench-http.py` — the same payloads against a running engine `/api/analyze`.
- `fingerprint.py` — hardware / runtime fingerprint embedded in every record.
- `payloads/` — the two real-text fixtures (paragraph, doctor's letter).

Run it on your own machine (model bundle downloaded once, ~1.2 GB):

```bash
pip install "onnxruntime==1.28.0" "transformers==4.57.6" numpy psutil pyyaml "huggingface_hub[cli]>=0.34"
hf download innovius/shinrai-pii-m-v1.3 --include "quant/model-fp32.onnx" "tokenizer/*" "config.json" "labels-*.yaml" "calibration.json" --local-dir model
python scripts/hardware/bench-local.py --platform-id my-machine --model-dir model --threads all,4,2 --mode both --payloads S,P,L,W --out bench.jsonl
```

The GitHub workflow `.github/workflows/hardware-bench.yml` runs exactly this on the free
hosted runners (x64, Arm, macOS M1). Results feed the hardware map published with the model:
https://huggingface.co/spaces/innovius/shinrai-pii-benchmarks

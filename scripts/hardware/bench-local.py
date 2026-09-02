#!/usr/bin/env python3
"""Direct ONNX Runtime benchmark for the ShinrAI PII model — no HTTP, no torch.

Feeds research/hardware-map/. Two modes:

  payload  end-to-end predict() on real texts: tokenize → sliding windows → ORT →
           IOB2 decode. The number a customer experiences.
  raw      session.run() on synthetic token ids at fixed sequence lengths and
           batch sizes. The number the analytic cost model calibrates on.

One JSON line per cell (research/hardware-map/results/<platform-id>-<date>.jsonl),
hardware fingerprint embedded, never a hostname. A correctness guard replays the
release smoke predictions through the same session first, so a wrong graph or a
broken tokenizer fails the run instead of producing a fast wrong number.

Examples:
  python scripts/hardware/bench-local.py --platform-id mac-mini-m4-pro --threads 1,2,4,8,all
  python scripts/hardware/bench-local.py --platform-id pi4-4gb --threads all --payloads S,P,W
  python scripts/hardware/bench-local.py --platform-id gh-ubuntu-arm --model-dir ./model --out bench.jsonl

Works from this repo (src/shinrai_pii) and from shinrai-engine's vendored copy
(src/shinrai_pii_runtime); payload fixtures resolve from tests/fixtures/parity or
scripts/hardware/payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent  # not resolve(): ConfigMap mounts are symlinks
ROOT = HERE.parents[1]
for candidate in (ROOT / "src", HERE):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from fingerprint import fingerprint  # noqa: E402

try:
    from shinrai_pii.serve.onnx_numpy import NumpyOnnxPredictor  # noqa: E402
except ImportError:  # shinrai-engine ships the same class vendored
    from shinrai_pii_runtime.onnx_numpy import NumpyOnnxPredictor  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover — RSS then comes from resource.getrusage
    psutil = None

SCHEMA = "shinrai-hardware-bench/1"
DEFAULT_MODEL_DIR = ROOT / "models" / "release" / "shinrai-pii-m-v1.3"
PAYLOAD_DIRS = (ROOT / "tests" / "fixtures" / "parity", HERE / "payloads")
GUARD_FILES = ("smoke-predictions.json", HERE / "guard-shinrai-pii-m-v1.3.json")
SEQ_LENS = (64, 128, 256, 512, 1024)
BATCH_SIZES = (1, 8)
DEFAULT_PAYLOADS = ("S", "P", "L", "W", "D")
XL_CHARS = 100 * 1024  # aiportal CHUNK_SIZE_CHARS, see scripts/serving/payloads.py

# Same short message as scripts/serving/payloads.py — keep byte-identical.
S_TEXT = (
    "Hallo Frau Weber, bitte senden Sie die Unterlagen an Lisa Müller, "
    "Hauptstraße 10, 04109 Leipzig. Telefonisch erreichen Sie uns unter "
    "+49 341 555 0192. Viele Grüße, Jonas Keller (EECC GmbH)"
)

PROVIDERS: dict[str, list] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}), "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}


# ---- helpers ----------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(ms: list[float]) -> dict:
    return {
        "iterations": len(ms),
        "p50_ms": round(_pct(ms, 0.50), 2),
        "p95_ms": round(_pct(ms, 0.95), 2),
        "p99_ms": round(_pct(ms, 0.99), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "min_ms": round(min(ms), 2),
        "max_ms": round(max(ms), 2),
    }


def rss_mb() -> float:
    if psutil is not None:
        return psutil.Process().memory_info().rss / 2**20
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 2**20 if sys.platform == "darwin" else usage / 2**10


def sha256_of(path: Path) -> str:
    """Hash the model file once; cache next to it when the directory is writable."""
    cache = path.with_suffix(path.suffix + ".sha256")
    if cache.is_file():
        cached = cache.read_text(encoding="utf-8").split()
        if cached and len(cached[0]) == 64:
            return cached[0]
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    try:
        cache.write_text(f"{digest.hexdigest()}  {path.name}\n", encoding="utf-8")
    except OSError:
        pass
    return digest.hexdigest()


def timed_loop(fn, *, warmup: int, iters: int, max_seconds: float) -> tuple[list[float], float]:
    """Run fn() warmup times (capped at max_seconds/3 but at least 3 calls), then
    time it iters times or until max_seconds elapse. Returns (latencies_ms, peak_rss_mb)."""
    warm_deadline = time.monotonic() + max(max_seconds / 3.0, 1.0)
    for i in range(warmup):
        fn()
        if i >= 2 and time.monotonic() > warm_deadline:
            break
    latencies: list[float] = []
    peak = rss_mb()
    deadline = time.monotonic() + max_seconds
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - started) * 1000.0)
        peak = max(peak, rss_mb())
        if time.monotonic() > deadline:
            break
    return latencies, peak


# ---- payload corpus ----------------------------------------------------------------


def load_fixture_texts() -> dict[str, str]:
    fixtures = next((d for d in PAYLOAD_DIRS if (d / "pii_mail_01.txt").is_file()), None)
    if fixtures is None:
        raise FileNotFoundError(
            "pii_mail_01.txt not found in tests/fixtures/parity or scripts/hardware/payloads"
        )
    paragraph = (fixtures / "pii_mail_01.txt").read_text(encoding="utf-8")
    letter_path = fixtures / "ePA-Beispiel-Arztbrief.md"
    if not letter_path.is_file():
        letter_path = max(fixtures.glob("*.*"), key=lambda p: p.stat().st_size)
    letter = letter_path.read_text(encoding="utf-8")
    seed = paragraph + "\n\n" + letter + "\n\n"
    chunk = (seed * (XL_CHARS // len(seed) + 1))[:XL_CHARS]
    return {"S": S_TEXT, "P": paragraph, "L": letter, "XL": chunk}


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def count_windows(predictor, text: str) -> int:
    encoding = predictor.tokenizer(
        text,
        truncation=True,
        max_length=predictor.window,
        stride=predictor.stride,
        return_overflowing_tokens=True,
    )
    return len(encoding["input_ids"])


def synth_text(tokenizer, base: str, target_tokens: int, max_tokens: int | None = None) -> str:
    """Concatenate sentences of `base` cyclically until the text holds >= target_tokens
    tokens (and <= max_tokens when given). Keeps real prose, so the sliding-window
    decode sees realistic entity density instead of repeated random ids."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", base.strip()) if s]
    parts: list[str] = []
    text = ""
    i = 0
    while count_tokens(tokenizer, text) < target_tokens:
        parts.append(sentences[i % len(sentences)])
        i += 1
        text = " ".join(parts)
    while max_tokens is not None and count_tokens(tokenizer, text) > max_tokens and len(parts) > 1:
        parts.pop()
        text = " ".join(parts)
    return text


def build_corpus(predictor, wanted: list[str]) -> dict[str, str]:
    fixtures = load_fixture_texts()
    corpus: dict[str, str] = {}
    for name in wanted:
        if name in fixtures:
            corpus[name] = fixtures[name]
        elif name == "W":  # one full window, no overlap
            corpus["W"] = synth_text(predictor.tokenizer, fixtures["P"], target_tokens=980, max_tokens=1020)
        elif name == "D":  # ~10k tokens: a long document, ~12 overlapping windows
            corpus["D"] = synth_text(predictor.tokenizer, fixtures["L"], target_tokens=10_000)
        else:
            raise ValueError(f"unknown payload {name!r}; choose from S,P,L,XL,W,D")
    return corpus


# ---- session / predictor -------------------------------------------------------------


def install_session_patch(disable_kleidiai: bool) -> None:
    """NumpyOnnxPredictor builds its own SessionOptions; hook in via the module attribute
    so the KleidiAI switch (mlas.disable_kleidiai, ORT >= 1.25) reaches every session."""
    import onnxruntime as ort

    if not disable_kleidiai:
        return
    base = ort.SessionOptions

    class PatchedSessionOptions(base):  # type: ignore[misc,valid-type]
        def __init__(self):
            super().__init__()
            self.add_session_config_entry("mlas.disable_kleidiai", "1")

    ort.SessionOptions = PatchedSessionOptions  # type: ignore[misc]


def make_predictor(args, threads: int | None):
    providers = PROVIDERS[args.provider]
    started = time.perf_counter()
    predictor = NumpyOnnxPredictor(
        args.onnx,
        args.model_dir,
        providers=providers,
        intra_op_threads=threads,
    )
    load_s = time.perf_counter() - started
    return predictor, load_s


def run_guard(predictor, guard_path: Path | None) -> dict:
    if guard_path is None:
        return {"status": "skipped", "reason": "no guard file"}
    samples = json.loads(guard_path.read_text(encoding="utf-8"))
    predicted = predictor.predict([s["text"] for s in samples])
    mismatches = []
    for sample, got in zip(samples, predicted, strict=True):
        want = {(e["type"], tuple(e["span"])) for e in sample["entities"]}
        have = {(e["type"], tuple(e["span"])) for e in got}
        if want != have:
            mismatches.append({
                "text": sample["text"][:80],
                "missing": sorted(f"{t}@{s}-{e}" for t, (s, e) in want - have),
                "extra": sorted(f"{t}@{s}-{e}" for t, (s, e) in have - want),
            })
    return {
        "status": "pass" if not mismatches else "fail",
        "file": guard_path.name,
        "samples": len(samples),
        "mismatches": mismatches,
    }


# ---- cells -----------------------------------------------------------------------------


def payload_cells(args, predictor, base: dict, corpus: dict[str, str]) -> list[dict]:
    records = []
    for name, text in corpus.items():
        tokens = count_tokens(predictor.tokenizer, text)
        windows = count_windows(predictor, text)
        latencies, peak = timed_loop(
            lambda: predictor.predict([text]),
            warmup=args.warmup,
            iters=args.iters,
            max_seconds=args.max_seconds,
        )
        stats = summarize(latencies)
        record = {
            **base,
            "mode": "payload",
            "payload": name,
            "chars": len(text),
            "tokens": tokens,
            "windows": windows,
            **stats,
            "tokens_per_s": round(tokens / (stats["mean_ms"] / 1000.0), 1),
            "peak_rss_mb": round(peak, 1),
        }
        records.append(record)
        _emit(args, record)
    return records


def raw_cells(args, predictor, base: dict) -> list[dict]:
    session = predictor.session
    vocab_hi = 1000  # plain sub-word ids, same range the ONNX export traced with
    rng = np.random.default_rng(42)
    records = []
    for seq_len in args.seq_lens:
        for batch in args.batch_sizes:
            input_ids = rng.integers(5, vocab_hi, size=(batch, seq_len), dtype=np.int64)
            attention_mask = np.ones((batch, seq_len), dtype=np.int64)
            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            latencies, peak = timed_loop(
                lambda: session.run(None, feed),
                warmup=args.warmup,
                iters=args.iters,
                max_seconds=args.max_seconds,
            )
            stats = summarize(latencies)
            record = {
                **base,
                "mode": "raw",
                "seq_len": seq_len,
                "batch": batch,
                "tokens": seq_len * batch,
                **stats,
                "tokens_per_s": round(seq_len * batch / (stats["mean_ms"] / 1000.0), 1),
                "peak_rss_mb": round(peak, 1),
            }
            records.append(record)
            _emit(args, record)
    return records


def _emit(args, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    if args.out:
        with Path(args.out).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    if not args.quiet:
        cell = record.get("payload") or f"raw{record['seq_len']}x{record['batch']}"
        print(
            f"{record['label']:<28} thr={str(record['threads_requested']):>4} {cell:<10} "
            f"tok={record['tokens']:>6} p50={record['p50_ms']:>9.1f}ms p95={record['p95_ms']:>9.1f}ms "
            f"tok/s={record['tokens_per_s']:>9.1f} rss={record['peak_rss_mb']:>7.0f}MB",
            flush=True,
        )


# ---- main --------------------------------------------------------------------------------


def parse_threads(spec: str) -> list[int | None]:
    values: list[int | None] = []
    for item in spec.split(","):
        item = item.strip().lower()
        if item in ("all", "default", ""):
            values.append(None)
        else:
            values.append(int(item))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform-id", required=True, help="machine class slug, e.g. mac-mini-m4-pro, pi4-4gb")
    parser.add_argument("--label", default=None, help="cell label; default <platform-id>-<precision>-<provider>")
    parser.add_argument("--note", default=None, help="free text about the machine state (cooling, VM, power mode)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx", type=Path, default=None, help="default <model-dir>/quant/model-fp32.onnx")
    parser.add_argument("--precision", default=None, help="fp32|q8|q4; inferred from the ONNX filename")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="cpu")
    parser.add_argument("--threads", default="all", help="comma list of intra-op thread counts, 'all' = ORT default")
    parser.add_argument("--disable-kleidiai", action="store_true", help="ARM: force generic NEON kernels (no SME/KleidiAI)")
    parser.add_argument("--mode", choices=["payload", "raw", "both"], default="both")
    parser.add_argument("--payloads", default=",".join(DEFAULT_PAYLOADS), help="S,P,L,XL,W,D (default: no XL)")
    parser.add_argument("--seq-lens", default=",".join(map(str, SEQ_LENS)))
    parser.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="time cap per cell after warm-up")
    parser.add_argument("--guard", type=Path, default=None, help="smoke-predictions.json to replay; default: auto")
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--skip-sha", action="store_true", help="skip hashing the ONNX file (slow on SD cards)")
    parser.add_argument("--out", default=None, help="JSONL path; default research/hardware-map/results/<platform>-<date>.jsonl")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.onnx = (args.onnx or args.model_dir / "quant" / "model-fp32.onnx").resolve()
    if not args.onnx.is_file():
        print(f"ONNX file not found: {args.onnx} (git lfs pull?)", file=sys.stderr)
        return 2
    precision = args.precision or (re.search(r"model-(fp32|q8|q4|fp16)", args.onnx.name) or [None, "fp32"])[1]
    label = args.label or f"{args.platform_id}-{precision}-{args.provider}" + ("-nokleidi" if args.disable_kleidiai else "")
    args.seq_lens = [int(x) for x in args.seq_lens.split(",") if x]
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    wanted = [p.strip().upper() for p in args.payloads.split(",") if p.strip()]
    if wanted == ["ALL"]:
        wanted = ["S", "P", "L", "XL", "W", "D"]

    if args.out is None:
        results_dir = ROOT / "research" / "hardware-map" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(results_dir / f"{args.platform_id}-{datetime.now(UTC):%Y%m%d}.jsonl")

    guard_path = None
    if not args.no_guard:
        if args.guard is not None:
            guard_path = args.guard
        else:
            for candidate in (args.model_dir / GUARD_FILES[0], Path(GUARD_FILES[1])):
                if Path(candidate).is_file():
                    guard_path = Path(candidate)
                    break

    install_session_patch(args.disable_kleidiai)
    import onnxruntime as ort

    fp = fingerprint(args.platform_id, args.note)
    model_sha = None if args.skip_sha else sha256_of(args.onnx)
    print(f"model {args.model_dir.name} · {args.onnx.name} ({args.onnx.stat().st_size / 2**20:.0f} MB) · "
          f"ORT {ort.__version__} · cpu {fp['cpu'].get('model')} · out {args.out}", flush=True)

    all_records: list[dict] = []
    for threads in parse_threads(args.threads):
        predictor, load_s = make_predictor(args, threads)
        effective = predictor.session.get_providers()
        base = {
            "schema": SCHEMA,
            "label": label,
            "platform_id": args.platform_id,
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": args.model_dir.name,
            "onnx_file": args.onnx.name,
            "model_sha256": model_sha,
            "precision": precision,
            "provider": args.provider,
            "providers_effective": effective,
            "threads_requested": threads if threads is not None else "default",
            "logical_cores": os.cpu_count(),
            "kleidiai_disabled": bool(args.disable_kleidiai),
            "window": predictor.window,
            "stride": predictor.stride,
            "load_s": round(load_s, 2),
            "rss_after_load_mb": round(rss_mb(), 1),
            "fingerprint": fp,
        }
        if threads is None or threads == parse_threads(args.threads)[0]:
            guard = run_guard(predictor, guard_path)
            base["guard"] = {k: v for k, v in guard.items() if k != "mismatches"} | {
                "mismatches": guard.get("mismatches", [])[:3]
            }
            print(f"guard: {guard['status']} ({guard.get('samples', 0)} samples, "
                  f"{len(guard.get('mismatches', []))} mismatches)", flush=True)
            if guard["status"] == "fail":
                print(json.dumps(guard["mismatches"][:3], ensure_ascii=False, indent=1), file=sys.stderr)
                print("FAIL: guard mismatch — this session does not reproduce the release smoke "
                      "predictions; refusing to record timings (use --no-guard to override)", file=sys.stderr)
                return 1
        if args.mode in ("payload", "both"):
            corpus = build_corpus(predictor, wanted)
            all_records += payload_cells(args, predictor, base, corpus)
        if args.mode in ("raw", "both"):
            all_records += raw_cells(args, predictor, base)
        del predictor

    print(f"{len(all_records)} cells → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

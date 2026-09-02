#!/usr/bin/env python3
"""HTTP benchmark against a running shinrai-pii-serve / shinrai-engine `/api/analyze`.

Companion of bench-local.py for machines we cannot run Python on directly (the
production GPU service). Records, per request, the client round-trip AND the
server-side `timing_ms.inference` / `timing_ms.total` and `stats.tokens` that the
serve API returns, so the map can separate model time from hop time.

Same JSONL schema family as bench-local.py (schema shinrai-hardware-bench/1,
mode "http"). The client fingerprint is embedded; describe the server with
--server-platform-id / --server-note (e.g. "tesla-p40", "1x P40 24 GB, CUDA EP").

  python scripts/hardware/bench-http.py \
      --target http://localhost:8080 \
      --model v1.3 --platform-id tesla-p40 --concurrency 1,4,16 --out results.jsonl

Dependencies: httpx (present in the serve image), stdlib otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(os.path.abspath(__file__)).parent  # not resolve(): ConfigMap mounts are symlinks
sys.path.insert(0, str(HERE))
from fingerprint import fingerprint  # noqa: E402

SCHEMA = "shinrai-hardware-bench/1"
PAYLOAD_DIRS = (HERE.parents[1] / "tests" / "fixtures" / "parity", HERE / "payloads")
S_TEXT = (
    "Hallo Frau Weber, bitte senden Sie die Unterlagen an Lisa Müller, "
    "Hauptstraße 10, 04109 Leipzig. Telefonisch erreichen Sie uns unter "
    "+49 341 555 0192. Viele Grüße, Jonas Keller (EECC GmbH)"
)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def summarize(prefix: str, values: list[float]) -> dict:
    if not values:
        return {f"{prefix}p50_ms": None, f"{prefix}p95_ms": None, f"{prefix}p99_ms": None, f"{prefix}mean_ms": None}
    return {
        f"{prefix}p50_ms": _pct(values, 0.50),
        f"{prefix}p95_ms": _pct(values, 0.95),
        f"{prefix}p99_ms": _pct(values, 0.99),
        f"{prefix}mean_ms": round(statistics.fmean(values), 2),
    }


def load_payloads(wanted: list[str]) -> dict[str, str]:
    fixtures = next((d for d in PAYLOAD_DIRS if (d / "pii_mail_01.txt").is_file()), None)
    if fixtures is None:
        raise FileNotFoundError("payload fixtures not found (tests/fixtures/parity or scripts/hardware/payloads)")
    paragraph = (fixtures / "pii_mail_01.txt").read_text(encoding="utf-8")
    letter_path = fixtures / "ePA-Beispiel-Arztbrief.md"
    letter = letter_path.read_text(encoding="utf-8") if letter_path.is_file() else paragraph * 3
    seed = paragraph + "\n\n" + letter + "\n\n"
    corpus = {
        "S": S_TEXT,
        "P": paragraph,
        "L": letter,
        "D": (seed * 12)[: len(seed) * 12],  # ~10k tokens, sliding windows on the server
        "XL": (seed * (100 * 1024 // len(seed) + 1))[: 100 * 1024],
    }
    return {k: corpus[k] for k in wanted}


def run_cell(client_factory, *, path: str, body: dict, concurrency: int, duration_s: float) -> dict:
    client_ms: list[float] = []
    server_inf_ms: list[float] = []
    server_total_ms: list[float] = []
    tokens: list[int] = []
    windows: list[int] = []
    errors: list[str] = []
    lock = threading.Lock()
    stop_at = time.monotonic() + duration_s

    def worker() -> None:
        client = client_factory()
        while time.monotonic() < stop_at:
            started = time.perf_counter()
            try:
                response = client.post(path, json=body)
                elapsed = (time.perf_counter() - started) * 1000.0
            except httpx.HTTPError as exc:
                with lock:
                    errors.append(type(exc).__name__)
                continue
            with lock:
                if response.status_code >= 400:
                    errors.append(f"HTTP {response.status_code}")
                    continue
                client_ms.append(elapsed)
                try:
                    data = response.json()
                except ValueError:
                    continue
                timing = data.get("timing_ms") or {}
                if "inference" in timing:
                    server_inf_ms.append(float(timing["inference"]))
                if "total" in timing:
                    server_total_ms.append(float(timing["total"]))
                stats = data.get("stats") or {}
                if not stats:  # serve API: stats live per text under results[]
                    results = data.get("results") or []
                    stats = (results[0].get("stats") or {}) if results else {}
                if "tokens" in stats:
                    tokens.append(int(stats["tokens"]))
                if "windows" in stats:
                    windows.append(int(stats["windows"]))
        client.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n = len(client_ms)
    mean_tokens = round(statistics.fmean(tokens), 1) if tokens else None
    return {
        "requests": n,
        "errors": len(errors),
        "error_kinds": sorted(set(errors))[:5],
        "throughput_rps": round(n / duration_s, 2),
        "tokens": mean_tokens,
        "windows": round(statistics.fmean(windows), 1) if windows else None,
        "tokens_per_s": round(mean_tokens * n / duration_s, 1) if mean_tokens else None,
        **summarize("client_", client_ms),
        **summarize("server_inference_", server_inf_ms),
        **summarize("server_total_", server_total_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="base URL of the serve / engine service")
    parser.add_argument("--path", default="/api/analyze")
    parser.add_argument("--model", default=None, help="model key sent in the body, e.g. v1.3")
    parser.add_argument("--platform-id", required=True, help="server hardware slug, e.g. tesla-p40")
    parser.add_argument("--server-note", default=None, help="server description for the map")
    parser.add_argument("--label", default=None)
    parser.add_argument("--payloads", default="S,P,L,D")
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds per cell")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--token", default=None, help="bearer token if the service requires one")
    parser.add_argument("--segment", default=None, help="serve decode mode to request: auto|sentence|none (omit = server default)")
    parser.add_argument("--extra-json", default=None, help='extra request fields, e.g. \'{"threshold":0.5}\'')
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    label = args.label or f"{args.platform_id}-http"
    wanted = [p.strip().upper() for p in args.payloads.split(",") if p.strip()]
    corpus = load_payloads(wanted)
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    def client_factory() -> httpx.Client:
        return httpx.Client(base_url=args.target, timeout=args.timeout, headers=headers)

    fp = fingerprint(f"{args.platform_id}-client")
    server_info: dict = {}
    with client_factory() as probe:
        for probe_path in ("/api/models", "/health", "/healthz"):
            try:
                response = probe.get(probe_path)
                if response.status_code < 400:
                    server_info[probe_path] = response.json()
            except (httpx.HTTPError, ValueError):
                continue

    all_records = []
    for name, text in corpus.items():
        body = {"text": text}
        if args.model:
            body["model"] = args.model
        if args.segment:
            body["segment"] = args.segment
        if args.extra_json:
            body.update(json.loads(args.extra_json))
        with client_factory() as warm:
            for _ in range(args.warmup):
                try:
                    warm.post(args.path, json=body)
                except httpx.HTTPError:
                    pass
        for concurrency in concurrencies:
            cell = run_cell(client_factory, path=args.path, body=body, concurrency=concurrency, duration_s=args.duration)
            record = {
                "schema": SCHEMA,
                "label": label,
                "platform_id": args.platform_id,
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "http",
                "target_path": args.path,
                "model": args.model,
                "segment": args.segment,
                "server_note": args.server_note,
                "server_info": server_info,
                "payload": name,
                "chars": len(text),
                "concurrency": concurrency,
                "duration_s": args.duration,
                **cell,
                "fingerprint": fp,
            }
            all_records.append(record)
            line = json.dumps(record, ensure_ascii=False)
            if args.out:
                with Path(args.out).open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            print(
                f"{label:<22} {name:<3} c={concurrency:<3} n={cell['requests']:>5} err={cell['errors']:<3} "
                f"client p50={cell['client_p50_ms']} p95={cell['client_p95_ms']} | "
                f"server inf p50={cell['server_inference_p50_ms']} p95={cell['server_inference_p95_ms']} | "
                f"tok={cell['tokens']} rps={cell['throughput_rps']}",
                flush=True,
            )
    print(f"{len(all_records)} cells → {args.out or 'stdout only'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Service metrics: totals plus a 300 s rolling latency window with p50/p95.

Hand-ported from shinrai-pii-bert src/shinrai_pii/serve/metrics.py (too small
to vendor-script; keep in sync by eye). Lock-guarded — inference runs on a
worker thread and /metrics is served concurrently by uvicorn.
"""

from __future__ import annotations

import threading
import time

WINDOW_SECONDS = 300


class ServeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._requests_total = 0
        self._failures_total = 0
        self._active = 0
        self._latencies: list[tuple[float, float]] = []  # (timestamp, ms)

    def start_request(self) -> None:
        with self._lock:
            self._requests_total += 1
            self._active += 1

    def finish_request(self, *, ok: bool, duration_ms: float) -> None:
        now = time.time()
        with self._lock:
            self._active = max(0, self._active - 1)
            if not ok:
                self._failures_total += 1
            self._latencies.append((now, duration_ms))
            cutoff = now - WINDOW_SECONDS
            while self._latencies and self._latencies[0][0] < cutoff:
                self._latencies.pop(0)

    @staticmethod
    def _percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return round(ordered[index], 1)

    def snapshot(self) -> dict:
        with self._lock:
            values = [ms for _, ms in self._latencies]
            return {
                "uptime_seconds": int(time.time() - self._started),
                "requests_total": self._requests_total,
                "failures_total": self._failures_total,
                "active": self._active,
                "latency_ms": {
                    "p50": self._percentile(values, 0.50),
                    "p95": self._percentile(values, 0.95),
                },
                "window_s": WINDOW_SECONDS,
            }

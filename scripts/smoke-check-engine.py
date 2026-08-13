#!/usr/bin/env python3
"""Black-box smoke check against a running shinrai-engine.

Standalone on purpose (stdlib only) so it runs from any machine against any
deployment without installing the package:

    python3 scripts/smoke-check-engine.py http://127.0.0.1:8080 [--api-key K]

Asserts: /healthz is ok, /api/models answers, and an /api/analyze probe
returns at least one entity whose offsets slice back to the input text —
the same offset contract every consumer depends on.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

PROBE_TEXT = "Anna Miller lives in Berlin, her office is on Bahnhofstrasse 5."


def fetch(url: str, api_key: str | None, payload: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = fetch(f"{base}/healthz", args.api_key)
    if status != 200 or health.get("status") != "ok":
        print(f"FAIL /healthz -> {status}: {health}")
        return 1
    print(f"ok /healthz — models={health['models']} precision={health.get('precision')}")
    if health.get("precision_warning"):
        print(f"   note: {health['precision_warning']}")

    status, models = fetch(f"{base}/api/models", args.api_key)
    if status != 200:
        print(f"FAIL /api/models -> {status}: {models}")
        return 1
    print(f"ok /api/models — {[m['name'] for m in models]}")

    status, body = fetch(f"{base}/api/analyze", args.api_key, {"text": PROBE_TEXT})
    if status != 200:
        print(f"FAIL /api/analyze -> {status}: {body}")
        return 1
    entities = body["results"][0]["entities"]
    if not entities:
        print("FAIL /api/analyze returned no entities for the probe sentence")
        return 1
    for entity in entities:
        surface = PROBE_TEXT[entity["startIndex"] : entity["endIndex"]]
        if surface != entity["text"]:
            print(f"FAIL offsets do not slice back: {entity} -> {surface!r}")
            return 1
    print(f"ok /api/analyze — {len(entities)} entities, offsets verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

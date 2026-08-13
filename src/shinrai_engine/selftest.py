"""Golden self-test: prove the mounted model still answers like the release.

Golden files are per-model lists of {text, entities} produced at release time
(fp32). Sources, in order: <bundle>/golden-predictions.json (present in
mounted release bundles), else a packaged fallback under
shinrai_engine/golden/ matched by bundle-dir basename or model name (the HF
bundles do not carry the golden file).

fp32 compares strictly (span/text/type/tier equality, confidence within 1e-3
— ONNX-vs-torch parity is ~3.5e-05); q8/int4 compare leniently (entities
exist where golden has them, every offset slices back, no strict spans —
quantization legitimately shifts confidences and boundaries).

Also runs black-box against a URL:  python -m shinrai_engine.selftest --url http://host:8080
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from .registry import LoadedModel

CONFIDENCE_TOLERANCE = 1e-3


def load_golden(bundle_dir: Path, model_name: str) -> list[dict] | None:
    candidate = bundle_dir / "golden-predictions.json"
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    package_dir = resources.files("shinrai_engine") / "golden"
    for entry in package_dir.iterdir():
        if not entry.name.endswith(".json"):
            continue
        stem = entry.name[: -len(".json")]
        if stem == bundle_dir.name or model_name in stem:
            return json.loads(entry.read_text(encoding="utf-8"))
    return None


def _strict_failures(golden_case: dict, predicted: list[dict]) -> list[str]:
    failures: list[str] = []
    want = {
        (tuple(e["span"]), e["text"], e["type"], e.get("tier")): e for e in golden_case["entities"]
    }
    got = {(tuple(e["span"]), e["text"], e["type"], e.get("tier")): e for e in predicted}
    for key in want.keys() - got.keys():
        failures.append(f"missing entity {key}")
    for key in got.keys() - want.keys():
        failures.append(f"unexpected entity {key}")
    for key in want.keys() & got.keys():
        delta = abs(float(want[key]["confidence"]) - float(got[key]["confidence"]))
        if delta > CONFIDENCE_TOLERANCE:
            failures.append(f"confidence drift {delta:.4f} on {key}")
    return failures


def _lenient_failures(golden_case: dict, predicted: list[dict]) -> list[str]:
    failures: list[str] = []
    text = golden_case["text"]
    for ent in predicted:
        start, end = ent["span"]
        if text[start:end] != ent["text"]:
            failures.append(f"offsets do not slice back: {ent['span']} != {ent['text']!r}")
    if golden_case["entities"] and not predicted:
        failures.append("golden has entities but the model found none")
    return failures


def run_selftest(model: LoadedModel, log=print) -> bool:
    golden = load_golden(model.bundle_dir, model.name)
    if golden is None:
        log(f"[selftest] {model.name}: no golden file found — skipped")
        return True
    strict = model.precision == "fp32"
    failures: list[str] = []
    for case in golden:
        predicted = model.predictor.predict([case["text"]])[0]
        check = _strict_failures if strict else _lenient_failures
        for failure in check(case, predicted):
            failures.append(f"{case['text'][:40]!r}...: {failure}")
    if failures:
        log(
            f"[selftest] {model.name} [{model.precision}] FAILED "
            f"({'strict' if strict else 'lenient'}, {len(failures)} problems):"
        )
        for failure in failures[:20]:
            log(f"[selftest]   - {failure}")
        return False
    log(
        f"[selftest] {model.name} [{model.precision}] passed "
        f"({'strict' if strict else 'lenient'}, {len(golden)} texts)"
    )
    return True


def run_all(registry: dict[str, LoadedModel], mode: str, log=print) -> None:
    """mode: off | warn | strict. strict raises on any failure."""
    if mode == "off":
        return
    all_ok = all(run_selftest(model, log=log) for model in registry.values())
    if not all_ok and mode == "strict":
        raise SystemExit("[selftest] FAILED and SHINRAI_SELF_TEST=strict — refusing to serve")


def check_url(base_url: str, api_key: str | None = None) -> int:
    """Black-box smoke against a running engine; returns a process exit code."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    health = httpx.get(f"{base_url}/healthz", timeout=10)
    if health.status_code != 200 or health.json().get("status") != "ok":
        print(f"[selftest] {base_url}/healthz -> {health.status_code} {health.text[:200]}")
        return 1
    text = "Warmup: Lisa Müller wohnt in der Hauptstraße 10 in Berlin."
    response = httpx.post(
        f"{base_url}/api/analyze", json={"text": text}, headers=headers, timeout=120
    )
    if response.status_code != 200:
        print(f"[selftest] analyze -> {response.status_code} {response.text[:200]}")
        return 1
    entities = response.json()["results"][0]["entities"]
    for ent in entities:
        if text[ent["startIndex"] : ent["endIndex"]] != ent["text"]:
            print(f"[selftest] offsets do not slice back: {ent}")
            return 1
    print(f"[selftest] {base_url} ok — {len(entities)} entities on the probe sentence")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="check a running engine instead of loading models")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    if args.url:
        return check_url(args.url.rstrip("/"), args.api_key)

    from .config import load_settings
    from .registry import build_registry

    settings = load_settings()
    registry = build_registry(settings)
    ok = all(run_selftest(model) for model in registry.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

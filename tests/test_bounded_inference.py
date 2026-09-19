import asyncio
import threading
from dataclasses import replace

import httpx
import pytest

from shinrai_engine.app import create_app


def test_timed_out_thread_keeps_permit_and_bounds_admission(tiny_registry, monkeypatch):
    settings, registry = tiny_registry
    settings = replace(settings, max_pending=1)
    started, release = threading.Event(), threading.Event()
    calls = []
    predictor = next(iter(registry.values())).predictor
    real = predictor.predict
    def blocked(*args, **kwargs):
        calls.append(1)
        started.set()
        release.wait(3)
        return real(*args, **kwargs)
    monkeypatch.setattr(predictor, "predict", blocked)
    async def run():
        app = create_app(settings, registry)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/api/analyze", json={"text": "Anna Miller", "processing_timeout_s": .05})
            assert started.is_set()
            assert first.status_code == 504
            second = await client.post("/api/analyze", json={"text": "Peter Schmidt"})
            assert second.status_code == 429
            assert len(calls) == 1
            assert (await client.get("/healthz")).status_code == 200
            release.set()
            for _ in range(50):
                await asyncio.sleep(.02)
                response = await client.post("/api/analyze", json={"text": "Anna Miller"})
                if response.status_code != 429:
                    assert response.status_code == 200
                    break
            else:
                pytest.fail("abandoned work did not release admission")
    try:
        asyncio.run(run())
    finally:
        release.set()


def test_runtime_batching_preserves_decode(tiny_registry):
    _, registry = tiny_registry
    predictor = next(iter(registry.values())).predictor
    texts = ["Anna Miller lives in Berlin. " * 8] * 9
    for segment in (None, "sentence", "auto"):
        assert predictor.predict(texts, batch_size=4, segment=segment) == predictor.predict(texts, batch_size=16, segment=segment)


def test_v14_smoke_requires_real_valid_person(tiny_registry, monkeypatch):
    from shinrai_engine.selftest import run_smoke
    _, registry = tiny_registry
    model = next(iter(registry.values()))
    for result, expected in [([[]], False),
                             ([[{'span': [0, 11], 'text': 'Anna Müller', 'type': 'PERSON'}]], True),
                             ([[{'span': [0, 99], 'text': 'Anna Müller', 'type': 'PERSON'}]], False)]:
        monkeypatch.setattr(model.predictor, 'predict', lambda *args, **kw: result)
        assert run_smoke(model, log=lambda _: None) is expected

"""CLI entrypoint: python -m shinrai_engine

Environment-first (every option is an env var, see config.py); the flags here
override the common ones for local runs.
"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shinrai-engine",
        description="Self-hostable HTTP inference service for the ShinrAI PII models.",
    )
    parser.add_argument("--models", help='overrides SHINRAI_MODELS ("name=source,...")')
    parser.add_argument("--precision", help="overrides SHINRAI_PRECISION (fp32|q8|int4)")
    parser.add_argument("--host", help="overrides SHINRAI_HOST")
    parser.add_argument("--port", type=int, help="overrides SHINRAI_PORT")
    parser.add_argument(
        "--execution-provider", help="overrides SHINRAI_EXECUTION_PROVIDER (auto|cpu|cuda)"
    )
    args = parser.parse_args(argv)

    for flag, env_name in (
        (args.models, "SHINRAI_MODELS"),
        (args.precision, "SHINRAI_PRECISION"),
        (args.host, "SHINRAI_HOST"),
        (args.port, "SHINRAI_PORT"),
        (args.execution_provider, "SHINRAI_EXECUTION_PROVIDER"),
    ):
        if flag is not None:
            os.environ[env_name] = str(flag)

    from .app import AUTH_DISABLED_BANNER, create_app
    from .config import ConfigError, load_settings
    from .registry import build_registry
    from .selftest import run_all

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"shinrai-engine: {exc}")
        return 2

    registry = build_registry(settings)
    run_all(registry, settings.self_test)

    if settings.api_key is None:
        print(f"[engine] {AUTH_DISABLED_BANNER}")

    app = create_app(settings, registry)

    import uvicorn

    print(f"shinrai-engine listening on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

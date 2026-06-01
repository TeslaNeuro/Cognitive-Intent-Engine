"""Entry point: `python -m backend.main`.

Starts the orchestrator (which starts the audio + video pipelines), then
either:

    - runs the FastAPI server (default), or
    - runs headless with a rich console table (`--headless`).

Either mode is graceful on Ctrl-C.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

import uvicorn

from .api.server import ConsoleRenderer, create_app
from .pipelines.orchestrator import Orchestrator
from .utils.config import load_config
from .utils.logging import get_logger


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cognitive State & Intent Engine")
    p.add_argument("--config", type=str, default=None, help="Optional override YAML")
    p.add_argument("--headless", action="store_true", help="Run without HTTP server")
    p.add_argument("--no-console", action="store_true", help="Disable rich console output")
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port

    log = get_logger("main", level=cfg.app.log_level)
    log.info(f"Starting {cfg.app.name} v0.1.0")

    orch = Orchestrator(cfg)
    orch.start()

    console: Optional[ConsoleRenderer] = None
    if cfg.logging.console_table and not args.no_console:
        console = ConsoleRenderer()
        console.start()
        orch.subscribe(console.update)

    # Graceful shutdown.
    def shutdown(signum=None, frame=None):  # noqa: ANN001
        log.info("Shutting down...")
        try:
            orch.stop()
        finally:
            if console is not None:
                console.stop()

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    if args.headless:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            shutdown()
        return 0

    app = create_app(cfg, orch)
    try:
        uvicorn.run(
            app,
            host=cfg.server.host,
            port=cfg.server.port,
            log_level=cfg.app.log_level.lower(),
            access_log=False,
        )
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Record a session of FusedFrames to JSONL for offline training.

Reads from the running backend's /ws endpoint and writes one JSON per line
into `logs/sessions/<session_id>/frames.jsonl`. Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import websockets  # provided by uvicorn[standard]


async def record(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "frames.jsonl"
    print(f"recording → {target}")
    async with websockets.connect(url) as ws:
        with target.open("a", encoding="utf-8") as f:
            async for msg in ws:
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue
                f.write(json.dumps(obj) + "\n")
                f.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/ws")
    p.add_argument(
        "--out-root",
        default="logs/sessions",
        help="parent dir for session subdir",
    )
    p.add_argument(
        "--session",
        default=None,
        help="optional session name (default: timestamp)",
    )
    args = p.parse_args()

    session = args.session or time.strftime("session_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / session

    try:
        asyncio.run(record(args.url, out_dir))
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

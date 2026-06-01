"""FastAPI app + WebSocket broadcaster + MJPEG video stream + console table.

Endpoints:
    GET  /              -> {"name": ..., "version": ...}
    GET  /healthz       -> {"ok": true}
    GET  /config        -> sanitized config dump
    GET  /baseline      -> baseline stats
    GET  /events        -> recent events
    GET  /video.mjpg    -> annotated MJPEG stream of the webcam
    GET  /frame.jpg     -> single annotated JPEG frame
    WS   /ws            -> live JSON stream of FusedFrame
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..pipelines.orchestrator import Orchestrator
from ..utils.config import AppConfig
from ..utils.logging import get_logger
from ..utils.schemas import FusedFrame

log = get_logger("api")


# --------------------------------------------------------------------------
# WebSocket broadcaster
# --------------------------------------------------------------------------

class WSBroadcaster:
    def __init__(self) -> None:
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self._task: Optional[asyncio.Task] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._task = loop.create_task(self._pump())

    async def _pump(self) -> None:
        while True:
            try:
                msg = await self._queue.get()
            except asyncio.CancelledError:
                break
            async with self._lock:
                dead: List[WebSocket] = []
                for ws in self._clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    if ws in self._clients:
                        self._clients.remove(ws)

    def publish_threadsafe(self, frame: FusedFrame) -> None:
        if self._loop is None:
            return
        try:
            msg = frame.model_dump_json()
        except Exception:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, msg)
        except RuntimeError:
            pass

    def _enqueue(self, msg: str) -> None:
        try:
            if self._queue.full():
                # Drop oldest to keep latency bounded.
                _ = self._queue.get_nowait()
            self._queue.put_nowait(msg)
        except Exception:
            pass

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)


# --------------------------------------------------------------------------
# Console renderer (rich table)
# --------------------------------------------------------------------------

class ConsoleRenderer:
    """Pretty live console output.

    Implemented as a *passive* subscriber that re-renders a rich Table on
    each frame. Optional — only enabled when `cfg.logging.console_table`.
    """

    def __init__(self) -> None:
        self.console = Console()
        self._live: Optional[Live] = None
        self._last_frame: Optional[FusedFrame] = None

    def start(self) -> None:
        self._live = Live(self._render(), refresh_per_second=8, console=self.console)
        self._live.__enter__()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    def update(self, frame: FusedFrame) -> None:
        self._last_frame = frame
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Table:
        f = self._last_frame
        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style="bold cyan")
        t.add_column()
        if f is None:
            t.add_row("status", "[yellow]starting…[/yellow]")
            return t

        emo_color = {
            "happy": "green", "sad": "blue", "angry": "red",
            "neutral": "white", "frustrated": "magenta",
        }.get(f.emotion, "white")
        t.add_row("emotion", f"[{emo_color}]{f.emotion}[/] ({f.confidence:.0%})")
        t.add_row("cognitive", f"{f.cognitive_state}")
        t.add_row("intent", f.intent)
        t.add_row("attention", f.attention)
        t.add_row("trend", f.trend)
        t.add_row("stress", f"{f.stress:.2f}")
        t.add_row("engagement", f"{f.engagement:.2f}")
        t.add_row("fatigue", f"{f.fatigue:.2f}")
        t.add_row("cog. load", f"{f.cognitive_load:.2f}")
        t.add_row("anomaly", f"{f.anomaly_score:.2f}")
        t.add_row("latency", f"{f.latency_ms:.0f} ms")
        if f.adaptive_action:
            t.add_row("action", f"[bold yellow]{f.adaptive_action}[/]")
        if f.events:
            t.add_row("events", ", ".join(e.type for e in f.events[-3:]))
        if f.explanation:
            t.add_row("why", " · ".join(f.explanation[:3]))
        return t


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

def create_app(cfg: AppConfig, orch: Orchestrator) -> FastAPI:
    broadcaster = WSBroadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        broadcaster.attach_loop(asyncio.get_running_loop())
        # subscribe orchestrator -> broadcaster (called from worker thread)
        orch.subscribe(broadcaster.publish_threadsafe)
        yield

    app = FastAPI(title=cfg.app.name, version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"name": cfg.app.name, "version": "0.1.0"}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "ts": time.time()}

    @app.get("/config")
    async def get_config():
        return cfg.model_dump()

    @app.get("/baseline")
    async def get_baseline():
        return {
            "samples": orch.baseline.samples,
            "ready": orch.baseline.ready,
            "stats": orch.baseline.stats(),
        }

    @app.get("/events")
    async def get_events(seconds: float = 60.0):
        return {"events": [e.model_dump() for e in orch.store.recent_events(seconds)]}

    @app.get("/latest")
    async def latest():
        f = orch.store.latest_frame()
        if f is None:
            return JSONResponse({"frame": None})
        return JSONResponse({"frame": f.model_dump()})

    @app.get("/frame.jpg")
    async def frame_jpg():
        buf = orch.vision.get_latest_frame_jpeg()
        if buf is None:
            return Response(status_code=204)
        return Response(content=buf, media_type="image/jpeg")

    @app.get("/video.mjpg")
    async def video_mjpg(request: Request):
        boundary = "frame"

        async def gen():
            while True:
                if await request.is_disconnected():
                    break
                buf = orch.vision.get_latest_frame_jpeg()
                if buf is not None:
                    yield (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(buf)}\r\n\r\n"
                    ).encode("ascii") + buf + b"\r\n"
                await asyncio.sleep(1.0 / 25.0)

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await broadcaster.connect(ws)
        try:
            # Send the latest cached frame immediately, then idle until disconnect.
            f = orch.store.latest_frame()
            if f is not None:
                await ws.send_text(f.model_dump_json())
            while True:
                # We just keep the socket alive; the broadcaster pushes frames.
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_text("pong")
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await broadcaster.disconnect(ws)

    return app

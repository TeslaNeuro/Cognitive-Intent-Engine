"""Thread-safe rolling state store.

Holds:
    - The most recent AudioFeatures / VisionFeatures snapshots
    - A bounded history of FusedFrames (for trend + anomaly modules)
    - A list of recent events (with cooldown bookkeeping)

This is intentionally simple — a deque per signal — because everything
the rest of the engine needs is "give me the last N seconds".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional

from ..utils.schemas import AudioFeatures, Event, FusedFrame, VisionFeatures


class StateStore:
    def __init__(self, history_seconds: float = 60.0, tick_hz: float = 10.0) -> None:
        self.history_seconds = history_seconds
        self.tick_hz = tick_hz
        maxlen = int(history_seconds * tick_hz) + 8

        self._lock = threading.RLock()
        self._audio: Optional[AudioFeatures] = None
        self._vision: Optional[VisionFeatures] = None
        self._frames: Deque[FusedFrame] = deque(maxlen=maxlen)
        self._events: Deque[Event] = deque(maxlen=200)
        self._last_event_ts: dict[str, float] = {}

    # ---------- writers ----------
    def update_audio(self, feats: AudioFeatures) -> None:
        with self._lock:
            self._audio = feats

    def update_vision(self, feats: VisionFeatures) -> None:
        with self._lock:
            self._vision = feats

    def push_frame(self, frame: FusedFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def push_event(self, event: Event, cooldown_s: float = 0.0) -> bool:
        with self._lock:
            last = self._last_event_ts.get(event.type, 0.0)
            if cooldown_s > 0 and (event.ts - last) < cooldown_s:
                return False
            self._last_event_ts[event.type] = event.ts
            self._events.append(event)
            return True

    # ---------- readers ----------
    def snapshot(self) -> tuple[Optional[AudioFeatures], Optional[VisionFeatures]]:
        with self._lock:
            return self._audio, self._vision

    def frames(self, last_seconds: Optional[float] = None) -> List[FusedFrame]:
        with self._lock:
            if last_seconds is None:
                return list(self._frames)
            cutoff = time.time() - last_seconds
            return [f for f in self._frames if f.ts >= cutoff]

    def latest_frame(self) -> Optional[FusedFrame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def recent_events(self, last_seconds: float = 10.0) -> List[Event]:
        with self._lock:
            cutoff = time.time() - last_seconds
            return [e for e in self._events if e.ts >= cutoff]

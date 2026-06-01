"""Event detection.

We watch the rolling history of fused frames and emit discrete events when
specific patterns appear. Each event type has its own cooldown so we never
spam the dashboard.

Implemented events:
    - frustration_spike   pitch_z + brow_z jointly cross a threshold
    - attention_drop      attention_score drops more than X within a window
    - disengagement       engagement stays below threshold for N seconds
    - fatigue_onset       EAR sustained below threshold for N seconds
    - stress_rising       stress slope is positive and large over the window
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, List

import numpy as np

from ..state.store import StateStore
from ..utils.config import EventsSection
from ..utils.schemas import Event, FusedFrame


class EventDetector:
    def __init__(self, cfg: EventsSection, store: StateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._ear_below_since: float | None = None
        self._engagement_below_since: float | None = None
        self._last_attention: float | None = None
        self._attn_history: Deque[float] = deque(maxlen=20)

    def step(self, frame: FusedFrame, feature_ctx: dict) -> List[Event]:
        out: List[Event] = []
        now = frame.ts

        # --- frustration_spike ---
        pitch_z = float(feature_ctx.get("pitch_z", 0.0))
        brow_z = float(feature_ctx.get("brow_tension_z", 0.0))
        if pitch_z + brow_z > 2 * self.cfg.frustration_z and feature_ctx.get("has_voice", 0) > 0.5:
            ev = Event(
                type="frustration_spike",
                severity=float(min(1.0, (pitch_z + brow_z) / 4.0)),
                ts=now,
                detail=f"pitch_z={pitch_z:.1f} brow_z={brow_z:.1f}",
            )
            if self.store.push_event(ev, cooldown_s=self.cfg.cooldown_s):
                out.append(ev)

        # --- attention_drop ---
        attn = float(feature_ctx.get("attention_score", 0.5))
        self._attn_history.append(attn)
        if len(self._attn_history) >= 6:
            old = float(np.mean(list(self._attn_history)[:3]))
            new = float(np.mean(list(self._attn_history)[-3:]))
            drop = old - new
            if drop > 0.25 and new < 0.4:
                ev = Event(
                    type="attention_drop",
                    severity=float(min(1.0, drop * 2)),
                    ts=now,
                    detail=f"{old:.2f} -> {new:.2f}",
                )
                if self.store.push_event(ev, cooldown_s=self.cfg.cooldown_s):
                    out.append(ev)

        # --- disengagement ---
        if frame.engagement < self.cfg.disengagement_engagement_threshold:
            if self._engagement_below_since is None:
                self._engagement_below_since = now
            elif (now - self._engagement_below_since) >= 5.0:
                ev = Event(
                    type="disengagement",
                    severity=float(1.0 - frame.engagement),
                    ts=now,
                    detail="engagement low for 5s+",
                )
                if self.store.push_event(ev, cooldown_s=self.cfg.cooldown_s):
                    out.append(ev)
        else:
            self._engagement_below_since = None

        # --- fatigue_onset ---
        ear = float(feature_ctx.get("ear", 1.0))
        if ear and ear < self.cfg.fatigue_ear_threshold:
            if self._ear_below_since is None:
                self._ear_below_since = now
            elif (now - self._ear_below_since) >= self.cfg.fatigue_min_seconds:
                ev = Event(
                    type="fatigue_onset",
                    severity=float(min(1.0, (self.cfg.fatigue_ear_threshold - ear) * 5.0)),
                    ts=now,
                    detail=f"EAR {ear:.2f} below {self.cfg.fatigue_ear_threshold:.2f}",
                )
                if self.store.push_event(ev, cooldown_s=self.cfg.cooldown_s * 2):
                    out.append(ev)
        else:
            self._ear_below_since = None

        # --- stress_rising ---
        recent = self.store.frames(last_seconds=6.0)
        if len(recent) >= 6:
            stresses = np.array([f.stress for f in recent], dtype=np.float32)
            slope = _slope(stresses)
            if slope > 0.05 and stresses[-1] > 0.55:
                ev = Event(
                    type="stress_rising",
                    severity=float(min(1.0, slope * 5)),
                    ts=now,
                    detail=f"slope={slope:.2f}",
                )
                if self.store.push_event(ev, cooldown_s=self.cfg.cooldown_s):
                    out.append(ev)

        return out


def _slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    yc = y - y.mean()
    return float(np.dot(x, yc) / (np.dot(x, x) + 1e-6))

"""Pydantic schemas shared by all pipelines.

These dataclasses double as the wire format for the WebSocket and the
internal contract between modules. Keeping them in one place avoids
drift and lets the dashboard generate types from them.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Canonical 5-class emotion taxonomy for the whole system.
EMOTION_LABELS: List[str] = ["happy", "sad", "angry", "neutral", "frustrated"]
COGNITIVE_LABELS: List[str] = ["focused", "confused", "overloaded", "fatigued"]
INTENT_LABELS: List[str] = ["problem-solving", "asking-for-help", "idle", "exploring"]
TREND_LABELS: List[str] = ["improving", "stable", "deteriorating", "disengaging"]


class EmotionDistribution(BaseModel):
    """Probability distribution over the 5 canonical emotions."""

    happy: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    neutral: float = 1.0
    frustrated: float = 0.0

    def top(self) -> tuple[str, float]:
        items = self.model_dump()
        label = max(items, key=items.get)
        return label, items[label]

    def as_array(self):
        return [getattr(self, k) for k in EMOTION_LABELS]


class AudioFeatures(BaseModel):
    """Compact, model-agnostic audio feature snapshot."""

    ts: float = Field(default_factory=time.time)
    rms: float = 0.0
    pitch_hz: float = 0.0
    pitch_voiced: bool = False
    speech_rate: float = 0.0          # voiced-frame ratio over the window
    pause_ratio: float = 1.0          # silence ratio over the window
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    mfcc_mean: List[float] = Field(default_factory=list)
    mfcc_delta_mean: List[float] = Field(default_factory=list)
    has_voice: bool = False


class VisionFeatures(BaseModel):
    """Per-frame visual features (mostly geometric / interpretable)."""

    ts: float = Field(default_factory=time.time)
    face_detected: bool = False
    bbox: Optional[List[float]] = None  # [x, y, w, h] normalized 0–1
    ear: float = 0.0                    # eye aspect ratio (avg both eyes)
    mouth_curvature: float = 0.0        # >0 = smile, <0 = frown
    brow_tension: float = 0.0           # higher = more tension
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    gaze_x: float = 0.0                 # -1..1 normalized
    gaze_y: float = 0.0
    blink_rate_hz: float = 0.0
    attention_score: float = 0.5        # 0..1, derived from gaze + pose stability


class Event(BaseModel):
    type: str
    severity: float = 0.5
    ts: float = Field(default_factory=time.time)
    detail: Optional[str] = None


class FusedFrame(BaseModel):
    """One published tick of the engine — what the dashboard consumes."""

    ts: float = Field(default_factory=time.time)

    # Emotion
    emotion: str = "neutral"
    confidence: float = 0.0
    probs: Dict[str, float] = Field(default_factory=dict)
    source_weights: Dict[str, float] = Field(default_factory=dict)

    # Higher-order states
    cognitive_state: str = "focused"
    intent: str = "idle"
    attention: str = "normal"        # high / normal / low / dropping
    trend: str = "stable"

    # Continuous scores (0..1)
    stress: float = 0.0
    engagement: float = 0.5
    fatigue: float = 0.0
    cognitive_load: float = 0.0

    # Diagnostics
    events: List[Event] = Field(default_factory=list)
    anomaly_score: float = 0.0
    explanation: List[str] = Field(default_factory=list)
    adaptive_action: Optional[str] = None
    calibration: Dict[str, float | int | bool] = Field(default_factory=dict)
    latency_ms: float = 0.0
    has_voice: bool = False
    face_detected: bool = False

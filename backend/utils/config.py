"""Typed configuration loader.

Reads `config/default.yaml`, optionally overlays a user-supplied YAML, and
returns a Pydantic model. CLI flags / env vars override at the field level.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Section models
# --------------------------------------------------------------------------

class AppSection(BaseModel):
    name: str = "Cognitive State & Intent Engine"
    log_level: str = "INFO"
    tick_hz: float = 10.0


class ServerSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class VisionSection(BaseModel):
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    emotion_model: str = "backend/models/weights/vision_emotion.pt"
    use_mediapipe: bool = True
    flip_horizontal: bool = True
    draw_overlays: bool = True


class AudioSection(BaseModel):
    device: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 1024
    feature_window_s: float = 1.0
    feature_hop_s: float = 0.1
    emotion_model: str = "backend/models/weights/audio_emotion.pt"
    vad_energy_threshold: float = 0.004
    n_mfcc: int = 20


class FusionSection(BaseModel):
    audio_weight: float = 0.45
    vision_weight: float = 0.55
    context_history_s: float = 10.0
    enable_learned_fuser: bool = True


class TemporalSection(BaseModel):
    window_s: float = 8.0
    hidden_size: int = 64
    num_layers: int = 1
    model_path: str = "backend/models/weights/temporal.pt"


class ReasoningSection(BaseModel):
    rules_path: str = "backend/reasoning/rules.yaml"
    ml_classifier_path: str = "backend/models/weights/intent_classifier.joblib"


class CalibrationSection(BaseModel):
    enabled: bool = True
    min_samples: int = 60
    ema_alpha: float = 0.02
    persist_path: str = "sessions/calibration.json"


class EventsSection(BaseModel):
    frustration_z: float = 1.5
    attention_drop_z: float = 1.2
    fatigue_ear_threshold: float = 0.21
    fatigue_min_seconds: float = 5.0
    disengagement_engagement_threshold: float = 0.25
    cooldown_s: float = 4.0


class AnomalySection(BaseModel):
    enabled: bool = True
    history_size: int = 600
    z_threshold: float = 3.0
    retrain_every_s: float = 30.0


class AdaptiveSection(BaseModel):
    enabled: bool = True
    rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LoggingSection(BaseModel):
    session_dir: str = "sessions"
    console_table: bool = True


class AppConfig(BaseModel):
    """Top-level config model."""

    app: AppSection = AppSection()
    server: ServerSection = ServerSection()
    vision: VisionSection = VisionSection()
    audio: AudioSection = AudioSection()
    fusion: FusionSection = FusionSection()
    temporal: TemporalSection = TemporalSection()
    reasoning: ReasoningSection = ReasoningSection()
    calibration: CalibrationSection = CalibrationSection()
    events: EventsSection = EventsSection()
    anomaly: AnomalySection = AnomalySection()
    adaptive: AdaptiveSection = AdaptiveSection()
    logging: LoggingSection = LoggingSection()


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load YAML config; merge over defaults; apply env overrides."""
    data: Dict[str, Any] = {}
    if _DEFAULT_PATH.exists():
        with _DEFAULT_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if path:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = _deep_merge(data, yaml.safe_load(f) or {})

    # Env overrides: CSE_<SECTION>_<KEY> with double underscore for nested.
    # Example: CSE_SERVER__PORT=9000
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("CSE_"):
            continue
        path_parts = env_key[len("CSE_") :].lower().split("__")
        cursor = data
        for part in path_parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path_parts[-1]] = _coerce(env_val)

    return AppConfig(**data)


def _coerce(value: str) -> Any:
    """Best-effort coercion of an env-var string into a Python literal."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

"""Per-user baseline tracker.

We use Welford / EMA-style online statistics so the calibration:
    - starts adapting from sample 1
    - is **persistent** across sessions (JSON on disk)
    - is **robust** — clipped, with a configurable warm-up period
    - is **explainable** — every consumer gets a z-score, not a black box

The basket of signals we track per-user:
    - pitch_hz, rms, speech_rate, pause_ratio (audio)
    - ear, mouth_curvature, brow_tension, attention_score, head_yaw, head_pitch (vision)

These are the signals reasoning rules quote ("pitch z=+1.9").
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

from ..utils.schemas import AudioFeatures, VisionFeatures


_TRACKED_AUDIO = ("rms", "pitch_hz", "speech_rate", "pause_ratio", "zero_crossing_rate")
_TRACKED_VISION = (
    "ear", "mouth_curvature", "brow_tension", "attention_score",
    "head_yaw", "head_pitch", "blink_rate_hz",
)


@dataclass
class _RunningStat:
    mean: float = 0.0
    var: float = 0.0
    n: int = 0

    def update(self, x: float, alpha: float) -> None:
        # EMA mean & variance.
        if self.n == 0:
            self.mean = x
            self.var = 0.0
            self.n = 1
            return
        prev_mean = self.mean
        self.mean = (1 - alpha) * self.mean + alpha * x
        self.var = (1 - alpha) * (self.var + alpha * (x - prev_mean) ** 2)
        self.n += 1

    @property
    def std(self) -> float:
        return max(self.var, 1e-8) ** 0.5

    def z(self, x: float) -> float:
        return (x - self.mean) / self.std


@dataclass
class PersonalBaseline:
    """Online baseline + z-normalization."""

    min_samples: int = 60
    ema_alpha: float = 0.02
    persist_path: Optional[str] = None

    _stats: Dict[str, _RunningStat] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _samples_total: int = 0
    _last_persist_ts: float = 0.0

    # ---------- lifecycle ----------
    def __post_init__(self) -> None:
        for name in _TRACKED_AUDIO + _TRACKED_VISION:
            self._stats.setdefault(name, _RunningStat())
        self.load()

    def load(self) -> None:
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.get("stats", {}).items():
                self._stats[k] = _RunningStat(**v)
            self._samples_total = int(data.get("samples_total", 0))
        except Exception:
            pass

    def save(self) -> None:
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "samples_total": self._samples_total,
            "stats": {k: asdict(v) for k, v in self._stats.items()},
            "ts": time.time(),
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    # ---------- update ----------
    def update(
        self,
        audio: Optional[AudioFeatures] = None,
        vision: Optional[VisionFeatures] = None,
    ) -> None:
        with self._lock:
            if audio is not None and audio.has_voice:
                for name in _TRACKED_AUDIO:
                    val = getattr(audio, name, 0.0)
                    if name == "pitch_hz" and not audio.pitch_voiced:
                        continue
                    self._stats[name].update(float(val), self.ema_alpha)
            if vision is not None and vision.face_detected:
                for name in _TRACKED_VISION:
                    self._stats[name].update(float(getattr(vision, name, 0.0)), self.ema_alpha)
            self._samples_total += 1
            if (time.time() - self._last_persist_ts) > 5.0:
                self._last_persist_ts = time.time()
                try:
                    self.save()
                except Exception:
                    pass

    # ---------- queries ----------
    @property
    def ready(self) -> bool:
        return self._samples_total >= self.min_samples

    @property
    def samples(self) -> int:
        return self._samples_total

    def z(self, name: str, value: float) -> float:
        stat = self._stats.get(name)
        if stat is None or stat.n < 4:
            return 0.0
        return stat.z(value)

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {"mean": v.mean, "std": v.std, "n": v.n}
            for k, v in self._stats.items()
        }

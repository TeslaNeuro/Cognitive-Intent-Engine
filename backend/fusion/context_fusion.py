"""Level-3 fusion: context-aware refinement.

Inputs:
    - Current fused emotion (post-decision fusion)
    - Recent history of fused emotions + scalar trajectories
    - Temporal model's trend estimate
    - Behaviour variability (rolling std of stress / attention)

Outputs:
    - Final emotion (possibly smoothed or shifted by context)
    - Calibrated `cognitive_load` and `fatigue` continuous scores
    - Confidence on the final label

This is where "frustrated for the past 8 seconds + variability low" is
allowed to override a single brief "neutral" tick caused by, e.g., a blink.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from ..utils.schemas import EMOTION_LABELS, EmotionDistribution


class ContextFuser:
    def __init__(self, history_seconds: float = 10.0, tick_hz: float = 10.0) -> None:
        self.maxlen = int(history_seconds * tick_hz) + 4
        self._hist: Deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._ear: Deque[float] = deque(maxlen=self.maxlen)
        self._stress: Deque[float] = deque(maxlen=self.maxlen)
        self._attn: Deque[float] = deque(maxlen=self.maxlen)
        self._brow_z: Deque[float] = deque(maxlen=self.maxlen)
        self._pitch_z: Deque[float] = deque(maxlen=self.maxlen)

    def refine(
        self,
        fused: EmotionDistribution,
        ear: float,
        blink_rate_hz: float,
        attention: float,
        brow_z: float,
        pitch_z: float,
        stress: float,
        smoothing: float = 0.6,
    ) -> Tuple[EmotionDistribution, float, float, float]:
        """Return (refined emotion, confidence, cognitive_load, fatigue)."""

        probs = np.array(fused.as_array(), dtype=np.float32)
        self._hist.append(probs)
        self._ear.append(ear)
        self._stress.append(stress)
        self._attn.append(attention)
        self._brow_z.append(brow_z)
        self._pitch_z.append(pitch_z)

        # Temporal smoothing (exponentially-weighted mean over the history).
        if len(self._hist) > 1:
            arr = np.stack(self._hist, axis=0)
            weights = np.linspace(0.3, 1.0, num=arr.shape[0])
            avg = (arr * weights[:, None]).sum(axis=0) / weights.sum()
            probs = smoothing * avg + (1.0 - smoothing) * probs
            probs /= probs.sum() + 1e-12

        # Cognitive load: combine recent brow tension, pause, pitch variability.
        cog_load = _cognitive_load(self._brow_z, self._pitch_z, self._stress)

        # Fatigue: low EAR sustained + high blink rate decay.
        ear_arr = np.array(self._ear, dtype=np.float32) if self._ear else np.zeros(1)
        low_ear_frac = float(np.mean(ear_arr < 0.21))
        fatigue = float(np.clip(0.6 * low_ear_frac + 0.4 * min(1.0, blink_rate_hz / 0.5), 0.0, 1.0))

        # Confidence: peakiness of refined distribution scaled by history support.
        conf = _peakiness(probs) * min(1.0, len(self._hist) / max(1, self.maxlen // 2))

        # Pack back into distribution.
        payload = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        return EmotionDistribution(**payload), float(conf), float(cog_load), float(fatigue)

    def behaviour_variability(self) -> Dict[str, float]:
        def safe_std(d: Deque[float]) -> float:
            return float(np.std(np.array(d, dtype=np.float32))) if len(d) > 2 else 0.0
        return {
            "stress_std": safe_std(self._stress),
            "attention_std": safe_std(self._attn),
            "brow_z_std": safe_std(self._brow_z),
            "pitch_z_std": safe_std(self._pitch_z),
        }


# --------------------------------------------------------------------------
def _cognitive_load(brow: Deque[float], pitch: Deque[float], stress: Deque[float]) -> float:
    """Approximate cognitive load on 0..1.

    Higher load when:
        - brow tension z-score is high & rising
        - pitch z-score variance is high (speech effort)
        - stress is sustained, not just spiky
    """
    b = np.array(brow, dtype=np.float32) if brow else np.zeros(1)
    p = np.array(pitch, dtype=np.float32) if pitch else np.zeros(1)
    s = np.array(stress, dtype=np.float32) if stress else np.zeros(1)
    brow_lvl = float(np.clip(np.mean(np.maximum(0.0, b)) / 2.0, 0.0, 1.0))
    pitch_var = float(np.clip(np.std(p) / 2.0, 0.0, 1.0))
    stress_lvl = float(np.clip(np.mean(s), 0.0, 1.0))
    return float(np.clip(0.45 * brow_lvl + 0.30 * pitch_var + 0.25 * stress_lvl, 0.0, 1.0))


def _peakiness(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0)
    h = -float(np.sum(p * np.log(p)))
    h_max = float(np.log(len(p)))
    return max(0.0, min(1.0, 1.0 - h / h_max))

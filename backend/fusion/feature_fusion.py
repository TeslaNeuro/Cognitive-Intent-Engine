"""Level-1 fusion: build a single normalized feature vector per tick.

The feature vector concatenates:
    1. A handful of raw audio scalars
    2. The same scalars expressed as z-scores against the user's baseline
    3. A handful of vision scalars + their z-scores
    4. The previous-frame predicted emotion probabilities (vision + audio)

Returning a *stable column order* (see FEATURE_NAMES) means the temporal
LSTM, the anomaly detector, and the rule engine can all reference the same
indices unambiguously.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..calibration.baseline import PersonalBaseline
from ..utils.schemas import (
    EMOTION_LABELS,
    AudioFeatures,
    EmotionDistribution,
    VisionFeatures,
)


# --------------------------------------------------------------------------
# Stable column order. Anything added here must be appended at the end.
# --------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # raw audio
    "rms", "pitch_hz", "speech_rate", "pause_ratio",
    "zero_crossing_rate", "spectral_centroid",
    # audio z-scores
    "rms_z", "pitch_z", "speech_rate_z", "pause_ratio_z",
    # raw vision
    "ear", "mouth_curvature", "brow_tension", "attention_score",
    "head_yaw", "head_pitch", "head_roll",
    "gaze_x", "gaze_y", "blink_rate_hz",
    # vision z-scores
    "ear_z", "mouth_z", "brow_tension_z", "attention_z",
    # prior emotion probs (vision)
    *[f"vp_{e}" for e in EMOTION_LABELS],
    # prior emotion probs (audio)
    *[f"ap_{e}" for e in EMOTION_LABELS],
    # presence flags
    "has_voice", "face_detected",
]

FEATURE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
FEATURE_DIM = len(FEATURE_NAMES)


def build_feature_vector(
    audio: Optional[AudioFeatures],
    vision: Optional[VisionFeatures],
    audio_emo: Optional[EmotionDistribution],
    vision_emo: Optional[EmotionDistribution],
    baseline: PersonalBaseline,
) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    a = audio or AudioFeatures()
    v = vision or VisionFeatures()
    ae = audio_emo or EmotionDistribution()
    ve = vision_emo or EmotionDistribution()

    # raw audio
    vec[FEATURE_INDEX["rms"]] = a.rms
    vec[FEATURE_INDEX["pitch_hz"]] = a.pitch_hz
    vec[FEATURE_INDEX["speech_rate"]] = a.speech_rate
    vec[FEATURE_INDEX["pause_ratio"]] = a.pause_ratio
    vec[FEATURE_INDEX["zero_crossing_rate"]] = a.zero_crossing_rate
    vec[FEATURE_INDEX["spectral_centroid"]] = a.spectral_centroid

    # audio z
    vec[FEATURE_INDEX["rms_z"]] = baseline.z("rms", a.rms)
    vec[FEATURE_INDEX["pitch_z"]] = baseline.z("pitch_hz", a.pitch_hz) if a.pitch_voiced else 0.0
    vec[FEATURE_INDEX["speech_rate_z"]] = baseline.z("speech_rate", a.speech_rate)
    vec[FEATURE_INDEX["pause_ratio_z"]] = baseline.z("pause_ratio", a.pause_ratio)

    # raw vision
    vec[FEATURE_INDEX["ear"]] = v.ear
    vec[FEATURE_INDEX["mouth_curvature"]] = v.mouth_curvature
    vec[FEATURE_INDEX["brow_tension"]] = v.brow_tension
    vec[FEATURE_INDEX["attention_score"]] = v.attention_score
    vec[FEATURE_INDEX["head_yaw"]] = v.head_yaw
    vec[FEATURE_INDEX["head_pitch"]] = v.head_pitch
    vec[FEATURE_INDEX["head_roll"]] = v.head_roll
    vec[FEATURE_INDEX["gaze_x"]] = v.gaze_x
    vec[FEATURE_INDEX["gaze_y"]] = v.gaze_y
    vec[FEATURE_INDEX["blink_rate_hz"]] = v.blink_rate_hz

    # vision z
    vec[FEATURE_INDEX["ear_z"]] = baseline.z("ear", v.ear)
    vec[FEATURE_INDEX["mouth_z"]] = baseline.z("mouth_curvature", v.mouth_curvature)
    vec[FEATURE_INDEX["brow_tension_z"]] = baseline.z("brow_tension", v.brow_tension)
    vec[FEATURE_INDEX["attention_z"]] = baseline.z("attention_score", v.attention_score)

    # prior emotion probs
    vis_probs = ve.as_array()
    aud_probs = ae.as_array()
    for i, e in enumerate(EMOTION_LABELS):
        vec[FEATURE_INDEX[f"vp_{e}"]] = vis_probs[i]
        vec[FEATURE_INDEX[f"ap_{e}"]] = aud_probs[i]

    vec[FEATURE_INDEX["has_voice"]] = 1.0 if a.has_voice else 0.0
    vec[FEATURE_INDEX["face_detected"]] = 1.0 if v.face_detected else 0.0
    return vec


def vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    return {n: float(vec[i]) for n, i in FEATURE_INDEX.items()}

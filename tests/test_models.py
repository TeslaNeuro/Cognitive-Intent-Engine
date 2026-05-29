"""Model wrapper tests — verify fallbacks work when no weights are present."""

import numpy as np

from backend.fusion.feature_fusion import FEATURE_DIM, FEATURE_INDEX
from backend.models.audio_emotion import AudioEmotionModel
from backend.models.temporal import TemporalEstimator
from backend.models.vision_emotion import VisionEmotionModel
from backend.utils.schemas import EMOTION_LABELS


def test_vision_emotion_uninitialized_returns_distribution():
    model = VisionEmotionModel(weights_path=None)
    img = np.zeros((48, 48), dtype=np.uint8)
    d = model.predict(img)
    arr = d.as_array()
    assert len(arr) == len(EMOTION_LABELS)
    assert abs(sum(arr) - 1.0) < 1e-4


def test_audio_emotion_uninitialized_returns_distribution():
    model = AudioEmotionModel(weights_path=None)
    log_mel = np.zeros((64, 96), dtype=np.float32)
    d = model.predict(log_mel)
    assert abs(sum(d.as_array()) - 1.0) < 1e-4


def test_temporal_fallback_trend():
    est = TemporalEstimator(input_dim=FEATURE_DIM, weights_path=None)
    window = np.zeros((20, FEATURE_DIM), dtype=np.float32)
    # Inject rising stress proxy
    pitch_z_col = FEATURE_INDEX["pitch_z"]
    window[:, pitch_z_col] = np.linspace(-1.0, 2.0, 20)
    brow_col = FEATURE_INDEX["brow_tension_z"]
    window[:, brow_col] = np.linspace(-1.0, 2.0, 20)
    out = est.predict(window, FEATURE_INDEX)
    assert out["trend"] in {"improving", "stable", "deteriorating", "disengaging"}
    assert 0.0 <= out["stress"] <= 1.0

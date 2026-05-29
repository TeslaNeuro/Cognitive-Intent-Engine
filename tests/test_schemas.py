"""Sanity-check the schema layer (used by every other test)."""

from backend.utils.schemas import (
    EMOTION_LABELS,
    AudioFeatures,
    EmotionDistribution,
    FusedFrame,
    VisionFeatures,
)


def test_emotion_distribution_top():
    d = EmotionDistribution(happy=0.1, sad=0.1, angry=0.1, neutral=0.1, frustrated=0.6)
    label, conf = d.top()
    assert label == "frustrated"
    assert abs(conf - 0.6) < 1e-6


def test_emotion_as_array():
    d = EmotionDistribution(happy=0.2, sad=0.2, angry=0.2, neutral=0.2, frustrated=0.2)
    arr = d.as_array()
    assert len(arr) == len(EMOTION_LABELS)
    assert abs(sum(arr) - 1.0) < 1e-6


def test_fused_frame_defaults():
    f = FusedFrame()
    assert f.emotion == "neutral"
    assert f.events == []
    assert f.attention == "normal"
    assert f.calibration == {}

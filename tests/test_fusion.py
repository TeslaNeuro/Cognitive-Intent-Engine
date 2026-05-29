"""Tests for the three fusion layers + feature vector layout."""

import numpy as np

from backend.calibration.baseline import PersonalBaseline
from backend.fusion.context_fusion import ContextFuser
from backend.fusion.decision_fusion import DecisionFuser
from backend.fusion.feature_fusion import (
    FEATURE_DIM,
    FEATURE_INDEX,
    FEATURE_NAMES,
    build_feature_vector,
    vector_to_dict,
)
from backend.utils.schemas import (
    EMOTION_LABELS,
    AudioFeatures,
    EmotionDistribution,
    VisionFeatures,
)


def test_feature_vector_layout_stable():
    assert FEATURE_DIM == len(FEATURE_NAMES)
    # Unique column names.
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_feature_vector_round_trip():
    baseline = PersonalBaseline(min_samples=1)
    audio = AudioFeatures(rms=0.05, pitch_hz=180, has_voice=True, pitch_voiced=True)
    vision = VisionFeatures(face_detected=True, ear=0.25, attention_score=0.7)
    audio_emo = EmotionDistribution(neutral=1.0)
    vision_emo = EmotionDistribution(happy=0.6, neutral=0.4)

    vec = build_feature_vector(audio, vision, audio_emo, vision_emo, baseline)
    d = vector_to_dict(vec)
    assert d["rms"] == 0.05
    assert d["pitch_hz"] == 180
    assert d["ear"] == 0.25
    assert d["has_voice"] == 1.0
    assert d["face_detected"] == 1.0
    # Vision prior probs should match.
    assert abs(d["vp_happy"] - 0.6) < 1e-6


def test_decision_fusion_collapses_to_present_modality():
    fuser = DecisionFuser(enable_learned=False)
    audio_emo = EmotionDistribution(angry=0.8, neutral=0.2)
    vision_emo = EmotionDistribution(happy=0.7, neutral=0.3)

    # Only voice present.
    fused, w = fuser.fuse(audio_emo, vision_emo, has_voice=True, face_detected=False)
    label, conf = fused.top()
    assert label == "angry"
    assert w["vision"] == 0.0

    # Only face present.
    fused, w = fuser.fuse(audio_emo, vision_emo, has_voice=False, face_detected=True)
    label, conf = fused.top()
    assert label == "happy"
    assert w["audio"] == 0.0


def test_decision_fusion_blends_when_both_present():
    fuser = DecisionFuser(audio_weight=0.5, vision_weight=0.5, enable_learned=False)
    a = EmotionDistribution(angry=0.6, neutral=0.4)
    v = EmotionDistribution(happy=0.6, neutral=0.4)
    fused, _ = fuser.fuse(a, v, has_voice=True, face_detected=True)
    # Geometric mean ⇒ neutral should dominate when peaks disagree.
    label, conf = fused.top()
    assert label in {"angry", "happy", "neutral"}
    # Probabilities sum to 1 (within fp).
    assert abs(sum(fused.as_array()) - 1.0) < 1e-5


def test_context_fuser_smoothing():
    cf = ContextFuser(history_seconds=1.0, tick_hz=10.0)
    # Push a frustrated streak, then one neutral tick — refined should still
    # lean frustrated due to smoothing.
    frustrated = EmotionDistribution(frustrated=0.9, neutral=0.1)
    neutral = EmotionDistribution(neutral=0.9, frustrated=0.1)
    for _ in range(8):
        cf.refine(frustrated, ear=0.3, blink_rate_hz=0.1,
                  attention=0.6, brow_z=1.5, pitch_z=1.2, stress=0.7)
    refined, conf, cog, fat = cf.refine(
        neutral, ear=0.3, blink_rate_hz=0.1,
        attention=0.6, brow_z=1.5, pitch_z=1.2, stress=0.7,
    )
    label, _ = refined.top()
    assert label == "frustrated"
    assert 0.0 <= conf <= 1.0
    assert 0.0 <= cog <= 1.0
    assert 0.0 <= fat <= 1.0

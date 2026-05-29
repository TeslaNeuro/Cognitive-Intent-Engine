"""Tests for the rule engine + explainability."""

from pathlib import Path

from backend.reasoning.rule_engine import RuleEngine
from backend.reasoning.explain import build_explanation
from backend.utils.schemas import EmotionDistribution


RULES_PATH = Path("backend/reasoning/rules.yaml")


def test_rule_engine_loads():
    eng = RuleEngine(RULES_PATH)
    assert len(eng.rules) > 0


def test_frustration_rule_fires():
    eng = RuleEngine(RULES_PATH)
    ctx = {
        "pitch_z": 1.6,
        "brow_tension_z": 1.2,
        "has_voice": 1.0,
        # Plenty of others, defaulted to 0
        "rms_z": 0.0, "speech_rate_z": 0.0, "pause_ratio_z": 0.0,
        "ear_z": 0.0, "mouth_z": 0.0, "attention_z": 0.0,
        "speech_rate": 0.6, "pause_ratio": 0.2, "face_detected": 1.0,
        "attention_score": 0.7, "fatigue": 0.0, "cognitive_load": 0.4,
        "mouth_curvature": 0.0, "stress": 0.6, "engagement": 0.5,
        "gaze_x": 0.0, "emotion": "neutral",
    }
    hits = eng.evaluate(ctx)
    names = [h.name for h in hits]
    assert "frustration_voice_face" in names


def test_apply_emotion_boost_shifts_argmax():
    eng = RuleEngine(RULES_PATH)
    emo = EmotionDistribution(happy=0.4, neutral=0.4, frustrated=0.2)

    class _Hit:
        name = "rule"
        explanation = ["x"]
        conclude = {"emotion_boost": {"frustrated": 0.8}, "cognitive_state": "overloaded"}

    state = eng.apply([_Hit()], emo, defaults={"emotion": "neutral", "explanation": []})
    assert state["emotion"] == "frustrated"
    assert state["cognitive_state"] == "overloaded"


def test_build_explanation_orders_callouts():
    ctx = {
        "pitch_z": 1.9, "brow_tension_z": -1.2, "ear_z": 0.1,
        "mouth_z": 0.0, "attention_z": -0.3,
        "rms_z": 0.0, "speech_rate_z": 0.0, "pause_ratio_z": 0.0,
    }
    lines = build_explanation(ctx, hits=[])
    assert lines and "pitch_z" in lines[0]

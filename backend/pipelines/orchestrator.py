"""The orchestrator stitches every component into one tick loop.

It runs in its own thread at `cfg.app.tick_hz` (default 10 Hz) and on each
tick:

    1. Reads the latest AudioFeatures + VisionFeatures (non-blocking).
    2. Updates the per-user baseline.
    3. Pulls per-modality emotion distributions.
    4. Builds the level-1 fused feature vector.
    5. Runs level-2 decision fusion.
    6. Pushes the feature vector into the temporal model (level-3 context).
    7. Runs the rule engine + ML reasoning head.
    8. Runs event + anomaly detectors.
    9. Runs the adaptive responder.
    10. Publishes a FusedFrame to subscribers (the FastAPI WebSocket).

All compute outside this thread is concurrent — pipelines have their own
threads, the WebSocket is async, and the heaviest scikit-learn refits are
periodic, not per-tick.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional

import numpy as np

from ..adaptive.responder import AdaptiveResponder
from ..anomaly.detector import AnomalyDetector
from ..calibration.baseline import PersonalBaseline
from ..events.detector import EventDetector
from ..fusion.context_fusion import ContextFuser
from ..fusion.decision_fusion import DecisionFuser
from ..fusion.feature_fusion import (
    FEATURE_DIM,
    FEATURE_INDEX,
    FEATURE_NAMES,
    build_feature_vector,
    vector_to_dict,
)
from ..models.temporal import TemporalEstimator
from ..reasoning.explain import build_explanation
from ..reasoning.ml_classifier import IntentStateClassifier
from ..reasoning.rule_engine import RuleEngine
from ..state.store import StateStore
from ..utils.config import AppConfig
from ..utils.logging import get_logger
from ..utils.schemas import (
    COGNITIVE_LABELS,
    EMOTION_LABELS,
    INTENT_LABELS,
    FusedFrame,
)
from .audio_pipeline import AudioPipeline
from .vision_pipeline import VisionPipeline

log = get_logger("orch")


# Callback signature for subscribers (e.g. WebSocket broadcaster).
FrameCallback = Callable[[FusedFrame], None]


class Orchestrator:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.store = StateStore(history_seconds=60.0, tick_hz=cfg.app.tick_hz)

        # Pipelines.
        self.vision = VisionPipeline(cfg, self.store)
        self.audio = AudioPipeline(cfg, self.store)

        # Calibration.
        self.baseline = PersonalBaseline(
            min_samples=cfg.calibration.min_samples,
            ema_alpha=cfg.calibration.ema_alpha,
            persist_path=cfg.calibration.persist_path if cfg.calibration.enabled else None,
        )

        # Fusion.
        self.decision = DecisionFuser(
            audio_weight=cfg.fusion.audio_weight,
            vision_weight=cfg.fusion.vision_weight,
            enable_learned=cfg.fusion.enable_learned_fuser,
        )
        self.context = ContextFuser(
            history_seconds=cfg.fusion.context_history_s,
            tick_hz=cfg.app.tick_hz,
        )

        # Temporal model.
        self.temporal = TemporalEstimator(
            input_dim=FEATURE_DIM,
            weights_path=cfg.temporal.model_path,
        )

        # Reasoning.
        self.rules = RuleEngine(cfg.reasoning.rules_path)
        self.ml = IntentStateClassifier(persist_path=cfg.reasoning.ml_classifier_path)

        # Events + anomaly + adaptive.
        self.events = EventDetector(cfg.events, self.store)
        self.anomaly = AnomalyDetector(
            history_size=cfg.anomaly.history_size,
            z_threshold=cfg.anomaly.z_threshold,
            retrain_every_s=cfg.anomaly.retrain_every_s,
        )
        self.adaptive = AdaptiveResponder(
            rules=cfg.adaptive.rules,
            store=self.store,
            enabled=cfg.adaptive.enabled,
        )

        # Sliding feature window for the temporal model.
        self._win_size = int(cfg.temporal.window_s * cfg.app.tick_hz)
        self._window: Deque[np.ndarray] = deque(maxlen=self._win_size)

        # Subscribers.
        self._callbacks: list[FrameCallback] = []
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        self.vision.start()
        self.audio.start()
        self._thread = threading.Thread(target=self._loop, name="Orchestrator", daemon=True)
        self._thread.start()
        log.info("Orchestrator started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.vision.stop()
        self.audio.stop()
        try:
            self.baseline.save()
        except Exception:
            pass
        log.info("Orchestrator stopped")

    def subscribe(self, cb: FrameCallback) -> None:
        with self._lock:
            self._callbacks.append(cb)

    # ---------- core tick ----------
    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.cfg.app.tick_hz)
        next_t = time.time()
        while not self._stop.is_set():
            t_tick = time.time()
            try:
                frame = self._tick(t_tick)
            except Exception as e:
                log.exception(f"Tick failed: {e}")
                frame = None

            if frame is not None:
                self.store.push_frame(frame)
                with self._lock:
                    cbs = list(self._callbacks)
                for cb in cbs:
                    try:
                        cb(frame)
                    except Exception:
                        pass

            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    def _tick(self, ts: float) -> FusedFrame:
        audio, vision = self.store.snapshot()
        self.baseline.update(audio=audio, vision=vision)

        audio_emo = self.audio.latest_emotion()
        vision_emo = self.vision.latest_emotion()

        # Level-1 fusion: concatenated feature vector + z-scores.
        vec = build_feature_vector(audio, vision, audio_emo, vision_emo, self.baseline)
        self._window.append(vec)
        ctx = vector_to_dict(vec)

        # Level-2 fusion: decision fusion.
        has_voice = bool(audio.has_voice) if audio else False
        face_detected = bool(vision.face_detected) if vision else False
        fused_emo, source_weights = self.decision.fuse(
            audio_emo=audio_emo,
            vision_emo=vision_emo,
            has_voice=has_voice,
            face_detected=face_detected,
        )

        # Level-3: temporal model.
        window = np.stack(self._window, axis=0) if self._window else np.zeros((1, FEATURE_DIM))
        temporal_out = self.temporal.predict(window, FEATURE_INDEX)

        # Level-3: context fusion (smoothing + cognitive_load + fatigue).
        refined_emo, conf, cog_load, fatigue = self.context.refine(
            fused=fused_emo,
            ear=ctx["ear"],
            blink_rate_hz=ctx["blink_rate_hz"],
            attention=ctx["attention_score"],
            brow_z=ctx["brow_tension_z"],
            pitch_z=ctx["pitch_z"],
            stress=temporal_out["stress"],
        )

        # Build a flat reasoning context the rules can dot-into.
        reasoning_ctx = dict(ctx)
        reasoning_ctx.update({
            "stress": temporal_out["stress"],
            "engagement": temporal_out["engagement"],
            "attention": temporal_out["attention"],
            "fatigue": fatigue,
            "cognitive_load": cog_load,
            "trend": temporal_out["trend"],
            "emotion": refined_emo.top()[0],
        })

        # --- rule engine ---
        hits = self.rules.evaluate(reasoning_ctx)
        defaults = {
            "emotion": refined_emo.top()[0],
            "cognitive_state": _default_cog_state(refined_emo, cog_load, fatigue, temporal_out),
            "intent": _default_intent(refined_emo, has_voice, ctx),
            "attention": _attention_label(temporal_out["attention"]),
            "explanation": [],
        }
        applied = self.rules.apply(hits, refined_emo, defaults)

        # --- ML head (self-supervised) ---
        # Record the rule-applied label as the pseudo-target & train.
        self.ml.record(vec, applied["cognitive_state"], applied["intent"])
        ml_cog, ml_int, ml_proba = self.ml.predict(vec)
        cog_state = ml_cog or applied["cognitive_state"]
        intent = ml_int or applied["intent"]
        # When the rule engine has high confidence (boost present), it wins.
        if any("emotion_boost" in h.conclude for h in hits):
            cog_state = applied["cognitive_state"]
            intent = applied["intent"]

        emotion = applied["emotion"]
        probs = applied["emotion_probs"]

        # --- anomaly ---
        anomaly_score = self.anomaly.update_and_score(vec)

        # --- build frame ---
        frame = FusedFrame(
            ts=ts,
            emotion=emotion,
            confidence=float(probs.get(emotion, conf)),
            probs=probs,
            source_weights=source_weights,
            cognitive_state=cog_state,
            intent=intent,
            attention=applied["attention"],
            trend=temporal_out["trend"],
            stress=temporal_out["stress"],
            engagement=temporal_out["engagement"],
            fatigue=fatigue,
            cognitive_load=cog_load,
            anomaly_score=anomaly_score,
            calibration={
                "samples": int(self.baseline.samples),
                "ready": bool(self.baseline.ready),
            },
            has_voice=has_voice,
            face_detected=face_detected,
        )

        # Train decision fuser online when we have a stable consensus.
        if conf > 0.65 and has_voice and face_detected:
            try:
                pseudo = EMOTION_LABELS.index(emotion)
                self.decision.update(
                    audio_probs=np.array(audio_emo.as_array(), dtype=np.float32),
                    vision_probs=np.array(vision_emo.as_array(), dtype=np.float32),
                    pseudo_label=pseudo,
                )
            except Exception:
                pass

        # --- events ---
        events = self.events.step(frame, reasoning_ctx)
        frame.events = events

        # --- adaptive ---
        action = self.adaptive.step(frame)
        frame.adaptive_action = action

        # --- explanation ---
        frame.explanation = build_explanation(
            ctx=reasoning_ctx,
            hits=hits,
            cog_pred=ml_cog,
            intent_pred=ml_int,
            ml_proba=ml_proba,
        )

        frame.latency_ms = float((time.time() - ts) * 1000.0)
        return frame


# --------------------------------------------------------------------------
# Default-state helpers used when no rules fire.
# --------------------------------------------------------------------------

def _default_cog_state(emo, cog_load: float, fatigue: float, temporal_out: dict) -> str:
    label, _ = emo.top()
    if fatigue > 0.6:
        return "fatigued"
    if cog_load > 0.65:
        return "overloaded"
    if label == "frustrated" and cog_load > 0.4:
        return "overloaded"
    if temporal_out["attention"] < 0.4:
        return "fatigued"
    if label in ("happy", "neutral") and temporal_out["engagement"] > 0.5:
        return "focused"
    return "focused"


def _default_intent(emo, has_voice: bool, ctx: dict) -> str:
    if has_voice and ctx.get("speech_rate", 0) > 0.4:
        return "problem-solving"
    if ctx.get("attention_score", 0) > 0.6 and ctx.get("pause_ratio", 1) > 0.5:
        return "exploring"
    if ctx.get("attention_score", 0) < 0.3:
        return "idle"
    return "problem-solving"


def _attention_label(score: float) -> str:
    if score >= 0.7: return "high"
    if score >= 0.45: return "normal"
    if score >= 0.3: return "dropping"
    return "low"

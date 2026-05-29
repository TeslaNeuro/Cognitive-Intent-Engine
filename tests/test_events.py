"""Event + anomaly detector tests."""

import numpy as np

from backend.anomaly.detector import AnomalyDetector
from backend.events.detector import EventDetector
from backend.state.store import StateStore
from backend.utils.config import EventsSection
from backend.utils.schemas import FusedFrame


def test_frustration_spike_emits_event():
    cfg = EventsSection()
    store = StateStore()
    det = EventDetector(cfg, store)
    frame = FusedFrame(stress=0.7, engagement=0.4, fatigue=0.1)
    ctx = {
        "pitch_z": 2.5, "brow_tension_z": 2.0, "has_voice": 1.0,
        "attention_score": 0.6, "ear": 0.3, "pause_ratio": 0.2,
    }
    out = det.step(frame, ctx)
    assert any(e.type == "frustration_spike" for e in out)


def test_anomaly_score_in_range():
    det = AnomalyDetector(history_size=50, z_threshold=3.0, retrain_every_s=1e9)
    for _ in range(60):
        vec = np.random.randn(40).astype(np.float32) * 0.1
        s = det.update_and_score(vec)
        assert 0.0 <= s <= 1.0
    # Inject an outlier
    s = det.update_and_score(np.ones(40, dtype=np.float32) * 10)
    assert 0.0 <= s <= 1.0

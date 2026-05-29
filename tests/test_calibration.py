"""Calibration baseline tests."""

import os
import tempfile

from backend.calibration.baseline import PersonalBaseline
from backend.utils.schemas import AudioFeatures, VisionFeatures


def test_baseline_warmup_and_zscore():
    b = PersonalBaseline(min_samples=5, ema_alpha=0.5)
    for v in [120.0, 130.0, 125.0, 128.0, 122.0, 124.0]:
        b.update(audio=AudioFeatures(rms=0.05, pitch_hz=v, has_voice=True, pitch_voiced=True))
    assert b.ready
    # A pitch of 200 Hz should be well above the recent baseline.
    z = b.z("pitch_hz", 200.0)
    assert z > 2.0


def test_baseline_persistence(tmp_path):
    path = tmp_path / "baseline.json"
    b = PersonalBaseline(min_samples=2, ema_alpha=0.5, persist_path=str(path))
    for _ in range(10):
        b.update(audio=AudioFeatures(rms=0.05, pitch_hz=150, has_voice=True, pitch_voiced=True))
    b.save()
    assert path.exists()

    b2 = PersonalBaseline(min_samples=2, ema_alpha=0.5, persist_path=str(path))
    assert b2.samples >= 10
    z = b2.z("pitch_hz", 250.0)
    assert z > 0

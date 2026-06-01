"""Audio pipeline.

Continuously captures from the microphone with sounddevice and runs a
rolling-buffer feature extractor at `cfg.audio.feature_hop_s` (default 10 Hz).

Features (librosa):
    - RMS energy
    - MFCC mean + delta-MFCC mean (n_mfcc configurable)
    - Pitch (fundamental frequency) via YIN
    - Zero-crossing rate
    - Spectral centroid
    - Speech rate proxy = ratio of voiced frames
    - Pause / silence ratio
    - VAD flag (`has_voice`)

Spectrogram log-mel is built once per inference tick and fed to
`AudioEmotionModel.predict`.

The capture is non-blocking — sounddevice calls into a callback that just
writes to a thread-safe ring buffer; feature extraction runs in its own
worker thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from ..models.audio_emotion import AudioEmotionModel
from ..state.store import StateStore
from ..utils.config import AppConfig
from ..utils.logging import get_logger
from ..utils.schemas import AudioFeatures, EmotionDistribution

log = get_logger("audio")


# Optional imports — degrade gracefully if missing or no devices.
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:  # pragma: no cover
    sd = None  # type: ignore
    _SD_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:  # pragma: no cover
    librosa = None  # type: ignore
    _LIBROSA_AVAILABLE = False


class _RingBuffer:
    """Fixed-size float32 mono ring buffer."""

    def __init__(self, length: int) -> None:
        self.buf = np.zeros(length, dtype=np.float32)
        self.length = length
        self.write = 0
        self.filled = 0
        self._lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        x = x.astype(np.float32, copy=False).reshape(-1)
        with self._lock:
            n = x.size
            if n >= self.length:
                self.buf[:] = x[-self.length :]
                self.write = 0
                self.filled = self.length
                return
            end = self.write + n
            if end <= self.length:
                self.buf[self.write : end] = x
            else:
                first = self.length - self.write
                self.buf[self.write :] = x[:first]
                self.buf[: n - first] = x[first:]
            self.write = (self.write + n) % self.length
            self.filled = min(self.length, self.filled + n)

    def snapshot(self) -> np.ndarray:
        """Return the buffer ordered oldest -> newest."""
        with self._lock:
            if self.filled < self.length:
                return self.buf[: self.filled].copy()
            return np.concatenate([self.buf[self.write :], self.buf[: self.write]])


class AudioPipeline:
    def __init__(self, cfg: AppConfig, store: StateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None

        self._buffer = _RingBuffer(int(cfg.audio.sample_rate * cfg.audio.feature_window_s))
        self._latest_emotion: EmotionDistribution = EmotionDistribution()
        self._ser_model = AudioEmotionModel(weights_path=cfg.audio.emotion_model)
        self._mel_basis: Optional[np.ndarray] = None  # cached

    # ---------- lifecycle ----------
    def start(self) -> None:
        if not _SD_AVAILABLE:
            log.warning("sounddevice not available; audio pipeline will idle.")
        else:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.cfg.audio.sample_rate,
                    channels=self.cfg.audio.channels,
                    blocksize=self.cfg.audio.block_size,
                    dtype="float32",
                    device=self.cfg.audio.device,
                    callback=self._on_audio,
                )
                self._stream.start()
            except Exception as e:
                log.warning(f"Could not open microphone: {e}; audio pipeline will idle.")
                self._stream = None
        self._thread = threading.Thread(target=self._run, name="AudioPipeline", daemon=True)
        self._thread.start()
        log.info("Audio pipeline started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        log.info("Audio pipeline stopped")

    # ---------- public ----------
    def latest_emotion(self) -> EmotionDistribution:
        return self._latest_emotion

    # ---------- audio thread ----------
    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Overruns/underruns happen — non-fatal.
            pass
        if indata.ndim > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata
        self._buffer.push(mono)

    # ---------- worker thread ----------
    def _run(self) -> None:
        hop = max(0.02, self.cfg.audio.feature_hop_s)
        while not self._stop.is_set():
            t0 = time.time()
            samples = self._buffer.snapshot()
            if samples.size > int(0.2 * self.cfg.audio.sample_rate):
                feats, log_mel = self._extract(samples)
                self.store.update_audio(feats)
                if feats.has_voice and log_mel is not None:
                    self._latest_emotion = self._ser_model.predict(log_mel)
                elif not feats.has_voice:
                    # When user is silent, decay to neutral so we don't latch.
                    self._latest_emotion = _decay(self._latest_emotion, decay=0.1)
            else:
                self.store.update_audio(AudioFeatures())
            elapsed = time.time() - t0
            if elapsed < hop:
                time.sleep(hop - elapsed)

    # ---------- feature extraction ----------
    def _extract(self, samples: np.ndarray):
        sr = self.cfg.audio.sample_rate
        n_mfcc = self.cfg.audio.n_mfcc
        rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
        has_voice = rms > self.cfg.audio.vad_energy_threshold

        feats = AudioFeatures(
            ts=time.time(),
            rms=rms,
            has_voice=bool(has_voice),
        )

        if not _LIBROSA_AVAILABLE:
            feats.mfcc_mean = [0.0] * n_mfcc
            feats.mfcc_delta_mean = [0.0] * n_mfcc
            return feats, None

        # Zero-crossing rate & spectral centroid are cheap & informative.
        feats.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(samples)))
        try:
            spec_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
            feats.spectral_centroid = float(np.mean(spec_centroid))
        except Exception:
            feats.spectral_centroid = 0.0

        # MFCC + delta
        try:
            mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
            d_mfcc = librosa.feature.delta(mfcc)
            feats.mfcc_mean = mfcc.mean(axis=1).astype(float).tolist()
            feats.mfcc_delta_mean = d_mfcc.mean(axis=1).astype(float).tolist()
        except Exception:
            feats.mfcc_mean = [0.0] * n_mfcc
            feats.mfcc_delta_mean = [0.0] * n_mfcc

        # Pitch via YIN — only meaningful if we have voice.
        if has_voice:
            try:
                f0 = librosa.yin(samples, fmin=60, fmax=400, sr=sr)
                f0 = f0[np.isfinite(f0)]
                f0 = f0[(f0 > 60) & (f0 < 400)]
                if f0.size > 0:
                    feats.pitch_hz = float(np.median(f0))
                    feats.pitch_voiced = True
            except Exception:
                pass

        # VAD-style voiced-frame ratio (speech rate proxy)
        try:
            frame_rms = librosa.feature.rms(y=samples, frame_length=512, hop_length=256)[0]
            voiced = frame_rms > self.cfg.audio.vad_energy_threshold
            feats.speech_rate = float(np.mean(voiced))
            feats.pause_ratio = float(1.0 - np.mean(voiced))
        except Exception:
            feats.speech_rate = 1.0 if has_voice else 0.0
            feats.pause_ratio = 0.0 if has_voice else 1.0

        # Log-mel for the CNN
        log_mel = None
        try:
            mel = librosa.feature.melspectrogram(
                y=samples, sr=sr, n_mels=AudioEmotionModel.INPUT_MELS,
                n_fft=1024, hop_length=160, fmin=20, fmax=sr // 2,
            )
            log_mel = librosa.power_to_db(mel + 1e-10)
        except Exception:
            log_mel = None

        return feats, log_mel


def _decay(dist: EmotionDistribution, decay: float) -> EmotionDistribution:
    """Drift an emotion distribution toward neutral by `decay` proportion."""
    payload = dist.model_dump()
    neutral = {k: (1.0 if k == "neutral" else 0.0) for k in payload}
    blended = {k: (1 - decay) * payload[k] + decay * neutral[k] for k in payload}
    s = sum(blended.values()) or 1.0
    return EmotionDistribution(**{k: v / s for k, v in blended.items()})

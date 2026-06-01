"""Lightweight log-mel CNN for speech emotion recognition.

Input: 64 mel bands x 96 time frames (≈1 second @ 10 ms hop).
Output: probability distribution over `EMOTION_LABELS`.

The model is intentionally tiny (~120K params) to keep latency low. Weights
can be trained with `training/train_audio_emotion.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.schemas import EMOTION_LABELS, EmotionDistribution


class AudioEmotionCNN(nn.Module):
    def __init__(self, num_classes: int = len(EMOTION_LABELS), n_mels: int = 64) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AudioEmotionModel:
    INPUT_MELS = 64
    INPUT_FRAMES = 96

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = AudioEmotionCNN(n_mels=self.INPUT_MELS).to(self.device).eval()
        self.has_pretrained = False
        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                self.has_pretrained = True
            except Exception:
                self.has_pretrained = False

    @torch.inference_mode()
    def predict(self, log_mel: np.ndarray) -> EmotionDistribution:
        """log_mel: shape (n_mels, time) — variable time is OK, will be cropped/padded."""
        if log_mel is None or log_mel.size == 0:
            return EmotionDistribution()
        x = _fit_mel(log_mel, self.INPUT_MELS, self.INPUT_FRAMES)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device).float()
        logits = self.net(x_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
        payload = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        return EmotionDistribution(**payload)


def _fit_mel(mel: np.ndarray, n_mels: int, frames: int) -> np.ndarray:
    """Crop/pad mel to (n_mels, frames) and normalize."""
    m, t = mel.shape
    # If wrong mel-band count, resample by simple linear interpolation along axis 0
    if m != n_mels:
        idx = np.linspace(0, m - 1, n_mels)
        mel = np.stack([np.interp(idx, np.arange(m), mel[:, j]) for j in range(t)], axis=1)
    # Crop / pad time axis
    if t >= frames:
        mel = mel[:, -frames:]  # use most recent
    else:
        pad = np.full((n_mels, frames - t), mel.min() if mel.size else -80.0, dtype=mel.dtype)
        mel = np.concatenate([pad, mel], axis=1)
    # Per-clip normalization
    mu = mel.mean()
    sigma = mel.std() + 1e-6
    return ((mel - mu) / sigma).astype(np.float32)

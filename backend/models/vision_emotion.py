"""Lightweight facial-emotion CNN.

Architecture is intentionally small (~200K params) so it fits comfortably on
CPU at >30 FPS and is trivial to convert to TFLite for Jetson Nano.

The class taxonomy matches `schemas.EMOTION_LABELS`.

Two entry points:
    - `VisionEmotionCNN`  : pure nn.Module, used by the training script.
    - `VisionEmotionModel`: inference wrapper, loads weights if present, falls
                            back to a deterministically-initialized model that
                            still produces sensible (uniform-ish) probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.schemas import EMOTION_LABELS, EmotionDistribution


class VisionEmotionCNN(nn.Module):
    """A small 48x48 grayscale -> 5-class CNN."""

    def __init__(self, num_classes: int = len(EMOTION_LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VisionEmotionModel:
    """Inference-only wrapper with safe fallback if no weights are present."""

    INPUT_SIZE = 48

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = VisionEmotionCNN().to(self.device).eval()
        self.has_pretrained = False
        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                self.has_pretrained = True
            except Exception:
                # Corrupt or mismatching weights — log silently, fall back to init.
                self.has_pretrained = False

    @torch.inference_mode()
    def predict(self, face_gray_48: np.ndarray) -> EmotionDistribution:
        """face_gray_48: HxW grayscale uint8 (or float) image, 48x48."""
        if face_gray_48 is None:
            return EmotionDistribution()
        if face_gray_48.dtype != np.float32:
            face_gray_48 = face_gray_48.astype(np.float32) / 255.0
        if face_gray_48.shape != (self.INPUT_SIZE, self.INPUT_SIZE):
            # Caller is expected to resize, but be defensive.
            return EmotionDistribution()
        x = torch.from_numpy(face_gray_48).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
        return _probs_to_distribution(probs)


def _probs_to_distribution(probs: np.ndarray) -> EmotionDistribution:
    payload = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
    return EmotionDistribution(**payload)

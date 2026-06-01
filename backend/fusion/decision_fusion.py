"""Level-2 fusion: combine the per-modality emotion distributions.

Two strategies:

    1. Weighted geometric mean with **confidence-aware** weights.
       - If audio has no voice we collapse to vision-only.
       - If vision has no face we collapse to audio-only.
       - When both are present, weights are scaled by per-modality confidence
         (entropy-based) so a peaky distribution wins over a vague one.

    2. Optional online MLP fuser. Takes the concatenated probabilities and
       learns to predict the *self-supervised pseudo-label* (the historical
       agreed-upon emotion). See `events/adaptive.py` for the trainer.

Returns the fused distribution + per-source weights for transparency.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.schemas import EMOTION_LABELS, EmotionDistribution


class _LearnedFuser(nn.Module):
    """Tiny MLP that takes [audio_probs, vision_probs] -> fused logits."""

    def __init__(self, num_classes: int = len(EMOTION_LABELS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecisionFuser:
    def __init__(
        self,
        audio_weight: float = 0.45,
        vision_weight: float = 0.55,
        enable_learned: bool = True,
        lr: float = 1e-2,
    ) -> None:
        self.audio_w = audio_weight
        self.vision_w = vision_weight
        self.enable_learned = enable_learned
        self._lock = threading.Lock()
        self.learned = _LearnedFuser() if enable_learned else None
        if self.learned is not None:
            self._opt = torch.optim.Adam(self.learned.parameters(), lr=lr)
            self.learned.eval()
        self._learned_ready = False

    # ---------- inference ----------
    def fuse(
        self,
        audio_emo: EmotionDistribution,
        vision_emo: EmotionDistribution,
        has_voice: bool,
        face_detected: bool,
    ) -> Tuple[EmotionDistribution, Dict[str, float]]:
        a = np.array(audio_emo.as_array(), dtype=np.float32)
        v = np.array(vision_emo.as_array(), dtype=np.float32)

        if not has_voice and not face_detected:
            uniform = np.full_like(a, 1.0 / len(EMOTION_LABELS))
            return _to_dist(uniform), {"audio": 0.0, "vision": 0.0}

        # Confidence-aware weights.
        wa = self.audio_w * _confidence(a) if has_voice else 0.0
        wv = self.vision_w * _confidence(v) if face_detected else 0.0
        total = wa + wv
        if total <= 1e-6:
            # Fall back to whichever modality is present
            if face_detected and not has_voice:
                wa, wv = 0.0, 1.0
            elif has_voice and not face_detected:
                wa, wv = 1.0, 0.0
            else:
                wa, wv = self.audio_w, self.vision_w
            total = wa + wv
        wa /= total; wv /= total

        # Geometric mean (log-domain) — robust against confident outliers.
        log_a = np.log(np.clip(a, 1e-6, 1.0))
        log_v = np.log(np.clip(v, 1e-6, 1.0))
        fused_log = wa * log_a + wv * log_v
        fused_log -= fused_log.max()
        fused = np.exp(fused_log)
        fused /= fused.sum() + 1e-12

        # If a learned fuser is ready, blend it in.
        if self.learned is not None and self._learned_ready:
            with torch.inference_mode():
                x = torch.from_numpy(np.concatenate([a, v]))
                logits = self.learned(x).numpy()
            learned = _softmax(logits)
            fused = 0.6 * fused + 0.4 * learned
            fused /= fused.sum() + 1e-12

        return _to_dist(fused), {"audio": float(wa), "vision": float(wv)}

    # ---------- online learning ----------
    def update(self, audio_probs: np.ndarray, vision_probs: np.ndarray, pseudo_label: int) -> None:
        """Take one gradient step of the learned fuser toward `pseudo_label`."""
        if self.learned is None:
            return
        with self._lock:
            self.learned.train()
            x = torch.from_numpy(np.concatenate([audio_probs, vision_probs])).float().unsqueeze(0)
            y = torch.tensor([int(pseudo_label)], dtype=torch.long)
            self._opt.zero_grad()
            logits = self.learned(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            self._opt.step()
            self.learned.eval()
            self._learned_ready = True


# --------------------------------------------------------------------------
def _confidence(probs: np.ndarray) -> float:
    """1 - normalized entropy. Peaky -> ~1, uniform -> ~0."""
    p = np.clip(probs, 1e-6, 1.0)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(p))
    return float(1.0 - h / h_max)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _to_dist(arr: np.ndarray) -> EmotionDistribution:
    payload = {label: float(arr[i]) for i, label in enumerate(EMOTION_LABELS)}
    return EmotionDistribution(**payload)

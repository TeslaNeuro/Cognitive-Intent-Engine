"""Temporal trend model.

A small GRU consumes a sliding window of fused-feature vectors and predicts:
    - trend class (improving / stable / deteriorating / disengaging)
    - 3 continuous trajectory scores: stress, engagement, attention

When no trained weights are present we fall back to a deterministic,
rule-based estimator that uses simple first/second-derivative statistics
on the window — this means the engine is *useful from the first run*.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.schemas import TREND_LABELS


class TemporalTrendModel(nn.Module):
    """GRU + multi-head output."""

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.trend_head = nn.Linear(hidden_size, len(TREND_LABELS))
        self.scalar_head = nn.Linear(hidden_size, 3)  # stress, engagement, attention

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        _, h = self.gru(x)
        last = h[-1]                             # (B, H)
        trend_logits = self.trend_head(last)
        scalars = torch.sigmoid(self.scalar_head(last))
        return trend_logits, scalars


class TemporalEstimator:
    """Inference wrapper with a *deterministic fallback*.

    The fallback computes:
      - stress       = z-score of recent pitch+brow signal mapped to 0..1
      - engagement   = inverse of pause_ratio combined with attention
      - trend        = sign of linear fit on stress over the last window
    """

    def __init__(
        self,
        input_dim: int,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.net = TemporalTrendModel(input_dim).to(self.device).eval()
        self.has_pretrained = False
        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                self.has_pretrained = True
            except Exception:
                self.has_pretrained = False

    @torch.inference_mode()
    def predict(
        self,
        window: np.ndarray,
        feature_index: dict,
    ) -> dict:
        """window: (T, F) recent fused feature vectors, oldest first.

        feature_index maps semantic name -> column index, used by the
        deterministic fallback when no NN weights are present.
        """
        if window is None or window.shape[0] < 2:
            return {
                "trend": "stable",
                "stress": 0.0,
                "engagement": 0.5,
                "attention": 0.5,
                "trend_probs": _uniform_trend(),
            }

        if self.has_pretrained and window.shape[1] == self.net.input_dim:
            x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)
            trend_logits, scalars = self.net(x)
            probs = torch.softmax(trend_logits, dim=-1).cpu().numpy().squeeze(0)
            s = scalars.cpu().numpy().squeeze(0)
            return {
                "trend": TREND_LABELS[int(np.argmax(probs))],
                "stress": float(s[0]),
                "engagement": float(s[1]),
                "attention": float(s[2]),
                "trend_probs": {l: float(p) for l, p in zip(TREND_LABELS, probs)},
            }

        return _fallback(window, feature_index)


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------

def _uniform_trend() -> dict:
    p = 1.0 / len(TREND_LABELS)
    return {l: p for l in TREND_LABELS}


def _fallback(window: np.ndarray, idx: dict) -> dict:
    def col(name: str, default: float = 0.0) -> np.ndarray:
        i = idx.get(name)
        if i is None or i >= window.shape[1]:
            return np.full(window.shape[0], default, dtype=np.float32)
        return window[:, i].astype(np.float32)

    # Build a stress proxy.
    pitch = _zclip(col("pitch_z"))
    brow = _zclip(col("brow_tension_z"))
    rms = _zclip(col("rms_z"))
    pause = col("pause_ratio", 0.5)
    attn = col("attention_score", 0.5)

    stress_series = _smooth(0.45 * pitch + 0.45 * brow + 0.10 * rms)
    stress = float(np.clip(_sigmoid(stress_series[-1]), 0.0, 1.0))
    engagement_series = _smooth((1.0 - pause) * 0.6 + attn * 0.4)
    engagement = float(np.clip(engagement_series[-1], 0.0, 1.0))
    attention = float(np.clip(attn[-1], 0.0, 1.0))

    # Trend = slope of stress vs slope of engagement
    slope_stress = _linfit_slope(stress_series)
    slope_engage = _linfit_slope(engagement_series)
    if slope_stress > 0.04 and slope_engage < -0.02:
        trend = "deteriorating"
    elif slope_stress < -0.04 and slope_engage > 0.02:
        trend = "improving"
    elif engagement < 0.25 and slope_engage <= 0.0:
        trend = "disengaging"
    else:
        trend = "stable"

    probs = {l: 0.05 for l in TREND_LABELS}
    probs[trend] = 0.85
    s = sum(probs.values())
    probs = {k: v / s for k, v in probs.items()}

    return {
        "trend": trend,
        "stress": stress,
        "engagement": engagement,
        "attention": attention,
        "trend_probs": probs,
    }


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if x.shape[0] <= k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _zclip(x: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    return np.clip(np.nan_to_num(x), lo, hi)


def _linfit_slope(y: np.ndarray) -> float:
    if y.shape[0] < 2:
        return 0.0
    x = np.arange(y.shape[0], dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    yc = y - y.mean()
    return float(np.dot(x, yc) / (np.dot(x, x) + 1e-6))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

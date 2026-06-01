"""Streaming anomaly detection over the fused feature vector.

We keep two layers:

    - Fast layer: per-feature streaming z-score on a sliding window. Cheap,
      always-on, gives instant reaction to spikes.
    - Slow layer: scikit-learn IsolationForest refit periodically over the
      same window. Catches multivariate weirdness that per-feature z-scores
      miss (e.g., "smiling but very high pitch").

The output is a single anomaly score in 0..1 that the dashboard plots and
the adaptive responder can route on.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    IsolationForest = None  # type: ignore
    _SKLEARN_AVAILABLE = False


class AnomalyDetector:
    def __init__(
        self,
        history_size: int = 600,
        z_threshold: float = 3.0,
        retrain_every_s: float = 30.0,
    ) -> None:
        self.history_size = history_size
        self.z_threshold = z_threshold
        self.retrain_every_s = retrain_every_s

        self._history: Deque[np.ndarray] = deque(maxlen=history_size)
        self._forest: Optional[IsolationForest] = None  # type: ignore
        self._last_refit = 0.0
        self._lock = threading.Lock()

    # ---------- maintenance ----------
    def _maybe_refit(self) -> None:
        if not _SKLEARN_AVAILABLE:
            return
        if len(self._history) < 50:
            return
        now = time.time()
        if (now - self._last_refit) < self.retrain_every_s:
            return
        try:
            X = np.stack(list(self._history))
            self._forest = IsolationForest(
                n_estimators=64,
                contamination=0.05,
                random_state=0,
            )
            self._forest.fit(X)
            self._last_refit = now
        except Exception:
            self._forest = None

    # ---------- public ----------
    def update_and_score(self, vec: np.ndarray) -> float:
        with self._lock:
            self._history.append(vec.astype(np.float32))
            arr = np.stack(list(self._history))

            # Fast layer: per-feature z-score over the window
            mu = arr.mean(axis=0)
            sigma = arr.std(axis=0) + 1e-6
            z = np.abs((vec - mu) / sigma)
            z_score = float(np.clip(np.max(z) / max(1e-6, self.z_threshold * 2), 0.0, 1.0))

            # Slow layer
            self._maybe_refit()
            forest_score = 0.0
            if self._forest is not None:
                try:
                    s = float(-self._forest.score_samples(vec.reshape(1, -1))[0])
                    # IsolationForest score_samples ∈ ~[-0.5, 0.5], invert + scale.
                    forest_score = float(np.clip((s + 0.5), 0.0, 1.0))
                except Exception:
                    forest_score = 0.0

            return float(np.clip(0.6 * z_score + 0.4 * forest_score, 0.0, 1.0))

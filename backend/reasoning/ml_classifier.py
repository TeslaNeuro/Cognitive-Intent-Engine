"""ML-based reasoning: predict cognitive state + intent from the fused vector.

This is intentionally a *very* small scikit-learn ensemble so it can be
trained online from the rule-engine's pseudo-labels (self-supervised), and
remains interpretable via feature importance.

If the model isn't yet trained the predictor returns `None` and the rule
engine alone decides — that's perfectly fine.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
    from sklearn.linear_model import SGDClassifier
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    joblib = None  # type: ignore
    SGDClassifier = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from ..fusion.feature_fusion import FEATURE_NAMES
from ..utils.schemas import COGNITIVE_LABELS, INTENT_LABELS


class IntentStateClassifier:
    """Two heads: cognitive state + intent, both incrementally trained."""

    MIN_TRAIN_SAMPLES = 30

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self._cog: Optional[SGDClassifier] = None
        self._int: Optional[SGDClassifier] = None
        self._buf_x: Deque[np.ndarray] = deque(maxlen=2000)
        self._buf_cog: Deque[str] = deque(maxlen=2000)
        self._buf_int: Deque[str] = deque(maxlen=2000)
        self.load()

    # ---------- persistence ----------
    def load(self) -> None:
        if not (_SKLEARN_AVAILABLE and self.persist_path and self.persist_path.exists()):
            return
        try:
            obj = joblib.load(self.persist_path)
            self._cog = obj.get("cog")
            self._int = obj.get("intent")
        except Exception:
            self._cog = self._int = None

    def save(self) -> None:
        if not (_SKLEARN_AVAILABLE and self.persist_path):
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump({"cog": self._cog, "intent": self._int}, self.persist_path)
        except Exception:
            pass

    # ---------- training ----------
    def record(self, x: np.ndarray, cog_label: str, intent_label: str) -> None:
        if not _SKLEARN_AVAILABLE:
            return
        self._buf_x.append(x.astype(np.float32))
        self._buf_cog.append(cog_label)
        self._buf_int.append(intent_label)
        # Periodically (cheaply) refit.
        if len(self._buf_x) % 25 == 0 and len(self._buf_x) >= self.MIN_TRAIN_SAMPLES:
            self._fit()

    def _fit(self) -> None:
        X = np.stack(list(self._buf_x))
        y_cog = np.array(list(self._buf_cog))
        y_int = np.array(list(self._buf_int))
        try:
            if self._cog is None:
                self._cog = SGDClassifier(
                    loss="log_loss", alpha=1e-4, max_iter=5, tol=None,
                    random_state=0,
                )
            if self._int is None:
                self._int = SGDClassifier(
                    loss="log_loss", alpha=1e-4, max_iter=5, tol=None,
                    random_state=0,
                )
            self._cog.partial_fit(X, y_cog, classes=np.array(COGNITIVE_LABELS))
            self._int.partial_fit(X, y_int, classes=np.array(INTENT_LABELS))
            self.save()
        except Exception:
            # Bad batch; skip.
            pass

    # ---------- inference ----------
    def predict(self, x: np.ndarray) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
        if not _SKLEARN_AVAILABLE or self._cog is None or self._int is None:
            return None, None, {}
        X = x.reshape(1, -1)
        cog = str(self._cog.predict(X)[0])
        intent = str(self._int.predict(X)[0])
        proba: Dict[str, float] = {}
        try:
            probs = self._cog.predict_proba(X)[0]
            for label, p in zip(self._cog.classes_, probs):
                proba[f"cog_{label}"] = float(p)
            probs_i = self._int.predict_proba(X)[0]
            for label, p in zip(self._int.classes_, probs_i):
                proba[f"intent_{label}"] = float(p)
        except Exception:
            pass
        return cog, intent, proba

    def feature_importance(self) -> Dict[str, float]:
        if not _SKLEARN_AVAILABLE or self._cog is None:
            return {}
        try:
            coefs = np.abs(self._cog.coef_).mean(axis=0)
            return {FEATURE_NAMES[i]: float(coefs[i]) for i in range(min(len(coefs), len(FEATURE_NAMES)))}
        except Exception:
            return {}

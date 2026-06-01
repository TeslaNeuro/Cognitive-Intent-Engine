"""Adaptive responder.

Maps the current FusedFrame to a suggested "system action" — the simplest
form of agency this engine has. The dashboard renders the action; in a real
deployment it could drive notifications, surface help, or dim a UI.

Rules are config-driven (see `adaptive` section of `default.yaml`) so the
behaviour can be tuned without code changes.

Each rule supports:
    - when_stress_above
    - when_engagement_below
    - when_state          (cognitive_state equals)
    - when_trend          (trend equals)
    - sustained_s         (must hold for this many seconds before firing)
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from ..state.store import StateStore
from ..utils.schemas import FusedFrame


class AdaptiveResponder:
    def __init__(self, rules: Dict[str, Dict], store: StateStore, enabled: bool = True) -> None:
        self.rules = rules or {}
        self.store = store
        self.enabled = enabled
        self._since: Dict[str, float] = {}
        self._last_fired: Dict[str, float] = {}

    def step(self, frame: FusedFrame) -> Optional[str]:
        if not self.enabled or not self.rules:
            return None

        now = frame.ts
        fired: Optional[str] = None
        cooldown_s = 30.0

        for action, params in self.rules.items():
            sustained = float(params.get("sustained_s", 0.0))
            condition = self._matches(frame, params)
            if condition:
                t0 = self._since.setdefault(action, now)
                if (now - t0) >= sustained and (now - self._last_fired.get(action, 0)) > cooldown_s:
                    self._last_fired[action] = now
                    fired = action
                    break
            else:
                self._since.pop(action, None)
        return fired

    def _matches(self, frame: FusedFrame, params: Dict) -> bool:
        if "when_stress_above" in params and frame.stress < float(params["when_stress_above"]):
            return False
        if "when_engagement_below" in params and frame.engagement > float(params["when_engagement_below"]):
            return False
        if "when_state" in params and frame.cognitive_state != str(params["when_state"]):
            return False
        if "when_trend" in params and frame.trend != str(params["when_trend"]):
            return False
        return True

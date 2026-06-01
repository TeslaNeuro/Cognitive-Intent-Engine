"""Rule-based reasoning engine.

Rules are declarative (YAML) so non-engineers can extend the system without
touching code. Each rule:

    - is AND-combined over its `when` clauses
    - emits a partial `FusedFrame` update via `conclude`
    - contributes human-readable explanation strings

Conflict resolution: later rules can override earlier ones (the YAML order
defines priority). A boost dictionary can additively bump the emotion
distribution before the final argmax — this is how soft cues stack.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import yaml

from ..utils.schemas import EMOTION_LABELS, EmotionDistribution


_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    ">":  operator.gt,
    "<":  operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda a, b: a in b,
}


@dataclass
class RuleHit:
    name: str
    explanation: List[str]
    conclude: Dict[str, Any]


class RuleEngine:
    def __init__(self, rules_path: str | Path) -> None:
        self.rules_path = Path(rules_path)
        self.rules: List[Dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        if not self.rules_path.exists():
            self.rules = []
            return
        with self.rules_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.rules = list(data.get("rules", []))

    def evaluate(self, ctx: Dict[str, Any]) -> List[RuleHit]:
        hits: List[RuleHit] = []
        for rule in self.rules:
            try:
                if self._match(rule.get("when", []), ctx):
                    hits.append(
                        RuleHit(
                            name=rule.get("name", "rule"),
                            explanation=[
                                _fmt(t, ctx) for t in rule.get("explanation", [])
                            ],
                            conclude=dict(rule.get("conclude", {})),
                        )
                    )
            except Exception:
                # A misconfigured rule must never crash inference.
                continue
        return hits

    def apply(
        self,
        hits: List[RuleHit],
        emotion: EmotionDistribution,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply hits to produce a final state dict."""
        state: Dict[str, Any] = dict(defaults)
        explanation: List[str] = list(defaults.get("explanation", []))

        # Apply emotion boosts first (additive on probabilities).
        probs = np.array(emotion.as_array(), dtype=np.float32)
        for hit in hits:
            boost = hit.conclude.get("emotion_boost")
            if boost:
                for label, delta in boost.items():
                    if label in EMOTION_LABELS:
                        probs[EMOTION_LABELS.index(label)] += float(delta)

        # Re-normalize.
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum() + 1e-12

        # Apply rest of the conclusions (later rules win).
        for hit in hits:
            for k, v in hit.conclude.items():
                if k == "emotion_boost":
                    continue
                state[k] = v
            explanation.extend(hit.explanation)

        state["emotion_probs"] = {
            label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)
        }
        state["emotion"] = EMOTION_LABELS[int(np.argmax(probs))]
        state["explanation"] = explanation
        return state

    # ---------- private ----------
    def _match(self, clauses: List[Dict[str, Any]], ctx: Dict[str, Any]) -> bool:
        if not clauses:
            return False
        for clause in clauses:
            feat = clause.get("feature")
            op_name = clause.get("op")
            value = clause.get("value")
            op = _OPS.get(op_name, None)
            if feat is None or op is None or feat not in ctx:
                return False
            if not op(ctx[feat], value):
                return False
        return True


def _fmt(template: str, ctx: Dict[str, Any]) -> str:
    try:
        return template.format(**ctx)
    except Exception:
        return template

"""Assemble the final, ordered explanation for a frame.

The explanation has three parts:

    1. **Why this emotion?** — top contributing features ranked by z-score.
    2. **Which rules fired?** — names + their printed explanation strings.
    3. **What did the ML head say?** — top class & confidence (if trained).

The list is intentionally a flat list of strings so the dashboard can
render it as bullets without further parsing.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..reasoning.rule_engine import RuleHit


_INTERESTING_Z = (
    "pitch_z", "brow_tension_z", "ear_z", "mouth_z", "attention_z",
    "rms_z", "speech_rate_z", "pause_ratio_z",
)


def build_explanation(
    ctx: Dict[str, float],
    hits: Iterable[RuleHit],
    cog_pred: str | None = None,
    intent_pred: str | None = None,
    ml_proba: Dict[str, float] | None = None,
) -> List[str]:
    lines: List[str] = []

    # 1. Z-score callouts: top three by |z| above 0.8.
    callouts: List[Tuple[str, float]] = []
    for name in _INTERESTING_Z:
        v = float(ctx.get(name, 0.0))
        if abs(v) >= 0.8:
            callouts.append((name, v))
    callouts.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for name, v in callouts[:3]:
        sign = "+" if v >= 0 else ""
        lines.append(f"{name} {sign}{v:.1f}σ vs baseline")

    # 2. Rule explanations (de-duplicated, preserving order).
    seen: set[str] = set()
    for hit in hits:
        for s in hit.explanation:
            if s not in seen:
                seen.add(s)
                lines.append(s)

    # 3. ML head summary.
    if cog_pred or intent_pred:
        proba_bits = []
        if ml_proba:
            cog_key = f"cog_{cog_pred}"
            int_key = f"intent_{intent_pred}"
            if cog_key in ml_proba:
                proba_bits.append(f"{cog_pred} {ml_proba[cog_key]:.0%}")
            if int_key in ml_proba:
                proba_bits.append(f"{intent_pred} {ml_proba[int_key]:.0%}")
        if proba_bits:
            lines.append("ML: " + ", ".join(proba_bits))
        else:
            lines.append(f"ML: {cog_pred or '?'} / {intent_pred or '?'}")

    return lines

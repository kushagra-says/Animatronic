"""
emotion.py

EmotionEngine — dimensional PAD state + derived traits for the Cognitive Engine.

PAD (Pleasure-Arousal-Dominance) is the canonical affect space. We layer
a few derived scalars on top because LLMs and humans both reason about
emotion better when it's named.

Decay: pulled toward neutral baseline at a fixed rate per tick.
Perturbation: rule-based mapping from observation signals.

This is intentionally simple and rule-based in v1. The wiring is the
contract — the values can be replaced with a learned model later.

See: Docs/ARCHITECTURE.md
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ==========================================================
# State
# ==========================================================

@dataclass
class EmotionState:
    """Plain data carrier. The engine owns the only instance."""

    # PAD — range [-1, 1]
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Derived scalars — range [0, 1]
    curiosity: float = 0.5
    trust: float = 0.5
    boredom: float = 0.0

    updated_at: float = field(default_factory=time.time)


# ==========================================================
# Perturbation rules
# ==========================================================

@dataclass
class EmotionPerturbation:
    """A delta applied to the state by some signal."""

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    curiosity: float = 0.0
    trust: float = 0.0
    boredom: float = 0.0


class EmotionRules:
    """
    Maps observation signals -> EmotionPerturbation.

    v1 is deterministic. v2 will consider the LLM's own self-report.
    """

    # Observation signals -> perturbation
    NOVELTY_HIGH = EmotionPerturbation(curiosity=0.10, arousal=0.05)
    PERSONAL_HIGH = EmotionPerturbation(pleasure=0.05, trust=0.05)
    CONTRADICTION = EmotionPerturbation(dominance=-0.10, arousal=0.10)
    COMPLIMENT = EmotionPerturbation(pleasure=0.10, trust=0.05)
    THREAT = EmotionPerturbation(pleasure=-0.20, arousal=0.15, dominance=-0.10)
    SILENCE_LONG = EmotionPerturbation(boredom=0.10, arousal=-0.05)

    # Thresholds tuned to the heuristic ObservationEngine's actual range
    # (~0.1–0.7, not 0.0–1.0). The previous values (>0.7 / >0.6) never
    # fired, which is why emotion scores stayed frozen at the baseline.
    NOVELTY_HIGH_THRESHOLD = 0.40
    PERSONAL_HIGH_THRESHOLD = 0.30

    @staticmethod
    def for_observation(
        *,
        novelty: float,
        personal_relevance: float,
        goal_relevance: float,
        is_contradiction: bool,
        text: str,
    ) -> EmotionPerturbation:

        deltas = EmotionPerturbation()

        if novelty > EmotionRules.NOVELTY_HIGH_THRESHOLD:
            _apply(deltas, EmotionRules.NOVELTY_HIGH)

        if personal_relevance > EmotionRules.PERSONAL_HIGH_THRESHOLD:
            _apply(deltas, EmotionRules.PERSONAL_HIGH)

        if is_contradiction:
            _apply(deltas, EmotionRules.CONTRADICTION)

        lowered = text.lower()

        if any(w in lowered for w in ["thanks", "thank you", "appreciate", "great job"]):
            _apply(deltas, EmotionRules.COMPLIMENT)

        if any(w in lowered for w in ["hate", "shut up", "stupid", "idiot", "die"]):
            _apply(deltas, EmotionRules.THREAT)

        return deltas


def _apply(target: EmotionPerturbation, delta: EmotionPerturbation) -> None:

    target.pleasure += delta.pleasure
    target.arousal += delta.arousal
    target.dominance += delta.dominance
    target.curiosity += delta.curiosity
    target.trust += delta.trust
    target.boredom += delta.boredom


# ==========================================================
# Engine
# ==========================================================

# Pull rate per tick. 0.10 = 10% pull toward baseline each maintenance tick.
BASELINE_PULL = 0.10

# Baselines (neutral resting state).
PLEASURE_BASELINE = 0.0
AROUSAL_BASELINE = 0.0
DOMINANCE_BASELINE = 0.0
CURIOSITY_BASELINE = 0.5
TRUST_BASELINE = 0.5
BOREDOM_BASELINE = 0.0


class EmotionEngine:
    """
    Owns the single EmotionState instance and the decay rules.

    Used by:
        - BeliefEngine / ObservationEngine (perturbation)
        - Consciousness tick loop (maintenance)
        - ContextBuilder (read current state)
    """

    def __init__(self, state: Optional[EmotionState] = None):

        self.state = state or EmotionState()

    # ==========================================================
    # Mutations
    # ==========================================================

    def perturb(self, perturbation: EmotionPerturbation) -> None:

        s = self.state

        s.pleasure = _clamp(s.pleasure + perturbation.pleasure, -1.0, 1.0)
        s.arousal = _clamp(s.arousal + perturbation.arousal, -1.0, 1.0)
        s.dominance = _clamp(s.dominance + perturbation.dominance, -1.0, 1.0)
        s.curiosity = _clamp_unit(s.curiosity + perturbation.curiosity)
        s.trust = _clamp_unit(s.trust + perturbation.trust)
        s.boredom = _clamp_unit(s.boredom + perturbation.boredom)

        s.updated_at = time.time()

    # ------------------------------------------------------

    def apply_to_observation(
        self,
        *,
        novelty: float,
        personal_relevance: float,
        goal_relevance: float,
        is_contradiction: bool,
        text: str,
    ) -> EmotionPerturbation:

        deltas = EmotionRules.for_observation(
            novelty=novelty,
            personal_relevance=personal_relevance,
            goal_relevance=goal_relevance,
            is_contradiction=is_contradiction,
            text=text,
        )

        self.perturb(deltas)

        return deltas

    # ------------------------------------------------------

    def decay(self) -> None:

        """Maintenance tick — pull state toward baseline."""

        s = self.state

        s.pleasure = _lerp(s.pleasure, PLEASURE_BASELINE, BASELINE_PULL)
        s.arousal = _lerp(s.arousal, AROUSAL_BASELINE, BASELINE_PULL)
        s.dominance = _lerp(s.dominance, DOMINANCE_BASELINE, BASELINE_PULL)
        s.curiosity = _lerp(s.curiosity, CURIOSITY_BASELINE, BASELINE_PULL)
        s.trust = _lerp(s.trust, TRUST_BASELINE, BASELINE_PULL)
        s.boredom = _lerp(s.boredom, BOREDOM_BASELINE, BASELINE_PULL)

        s.updated_at = time.time()

    # ------------------------------------------------------

    def apply_silence(self, seconds: float) -> None:

        """Special perturbation for long inactivity."""

        if seconds > 300:  # 5 minutes
            self.perturb(EmotionRules.SILENCE_LONG)

    # ==========================================================
    # Reads
    # ==========================================================

    def snapshot(self) -> EmotionState:

        return EmotionState(
            pleasure=self.state.pleasure,
            arousal=self.state.arousal,
            dominance=self.state.dominance,
            curiosity=self.state.curiosity,
            trust=self.state.trust,
            boredom=self.state.boredom,
            updated_at=self.state.updated_at,
        )

    # ------------------------------------------------------

    def summary(self) -> str:

        s = self.state

        return (
            f"pleasure={s.pleasure:+.2f}  "
            f"arousal={s.arousal:+.2f}  "
            f"dominance={s.dominance:+.2f}  "
            f"curiosity={s.curiosity:.2f}  "
            f"trust={s.trust:.2f}  "
            f"boredom={s.boredom:.2f}"
        )

    # ------------------------------------------------------

    def dominant_label(self) -> str:

        """
        Heuristic mapping PAD + scalars to a single descriptive label.
        Used in the CLI and in prompts.

        Order matters. Arousal is checked early so "calm" and "agitated"
        take precedence over the broader valence-only labels.
        """

        s = self.state

        if s.boredom > 0.7:
            return "bored"

        if s.curiosity > 0.7 and s.arousal > 0.2:
            return "curious"

        if s.pleasure > 0.4 and s.arousal > 0.3:
            return "excited"

        if s.pleasure > 0.3:
            return "pleasant"

        # Arousal takes precedence over valence-only labels.
        if s.arousal > 0.3 and s.pleasure < -0.3:
            return "agitated"

        if s.arousal < -0.3:
            return "calm"

        if s.pleasure < -0.3:
            return "down"

        return "neutral"


# ==========================================================
# Helpers
# ==========================================================

def _clamp(value: float, low: float, high: float) -> float:

    return max(low, min(high, value))


def _clamp_unit(value: float) -> float:

    return _clamp(value, 0.0, 1.0)


def _lerp(current: float, target: float, rate: float) -> float:

    return current + (target - current) * rate

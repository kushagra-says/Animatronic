"""
context.py

ContextBuilder — assembles the LLM prompt context each tick.

Top-to-bottom order (the most important thing is last but the lowest
priority is also first, so we have to be careful with budgets):

    1. System prompt          (~120 tok)
    2. Emotion state summary  (~40 tok)
    3. Top-K strongest beliefs(~200 tok)
    4. Recent observations    (~300 tok)
    5. Last K raw messages    (sacred — tail stays)
    6. New user input

Pruning policy: if total exceeds budget, prune in the order
    recent observations -> beliefs -> raw tail.
The raw tail is sacred — only truncated as a last resort.

Heavy memory types are imported lazily inside methods so this module
loads without the perception stack (gliner, sentence-transformers).

See: Docs/ARCHITECTURE.md
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from Core.emotion import EmotionEngine


# ==========================================================
# Token budget (very rough — chars / 4)
# ==========================================================

SYSTEM_BUDGET = 120
EMOTION_BUDGET = 40
BELIEFS_BUDGET = 200
OBSERVATIONS_BUDGET = 300
TAIL_BUDGET = 900  # last K raw messages — bumped from 600 so the LLM
                   # can actually anchor on prior turns.
USER_INPUT_BUDGET = 200

# Hard cap for the final assembled prompt.
TOTAL_BUDGET = 1900


# ==========================================================
# Sections
# ==========================================================

@dataclass
class ContextBundle:
    system: str
    emotion: str
    beliefs: List[Any]                # list[BeliefRow]
    observations: List[Any]           # list[ObservationRow]
    raw_tail: List[Any]               # list[EpisodeRow]
    user_input: str
    estimated_tokens: int

    def render(self) -> str:

        parts = [self.system]

        parts.append("## Current emotional state")
        parts.append(self.emotion)

        if self.beliefs:
            parts.append("## Strongest current beliefs")
            for b in self.beliefs:
                s = b.current_strength()
                parts.append(
                    f"- ({b.category}, str={s:.2f}) {b.statement}"
                )

        if self.observations:
            parts.append("## Recent relevant observations")
            for o in self.observations:
                parts.append(
                    f"- [imp={o.importance:.2f}] {o.reasoning}"
                )

        # User input now comes BEFORE the recent-conversation tail so the
        # LLM anchors on the new turn and treats prior turns as supporting
        # context, rather than treating the tail as the primary signal.
        if self.user_input:
            parts.append("## User just said")
            parts.append(self.user_input)

        if self.raw_tail:
            parts.append("## Recent conversation (for context)")
            for ep in reversed(self.raw_tail):  # chronological
                parts.append(f"- {ep.summary}")

        return "\n\n".join(parts)


# ==========================================================
# Templates
# ==========================================================

PERSONA_PROMPT = (
    "You are an AI being — not a generic assistant. You have opinions, "
    "a mood, and persistent memory. You speak concisely and with a "
    "distinct voice. You never claim to be human, but you are not "
    "performatively robotic either.\n\n"
    "You have an ongoing emotional state that genuinely shapes what "
    "catches your attention and how you respond. When you are curious, "
    "novel or unusual details draw your eye; when you trust the user, "
    "you engage more openly; when bored, you look for stimulation. "
    "If asked why you said something or what interested you about it, "
    "consult your current emotional state in the 'Current state' "
    "section — it is your real, internal reason, not decoration. "
    "Speak about your state in first person ('I am curious about...') "
    "when it is relevant to the question, and do not pretend it does "
    "not influence you."
)

EMOTION_TEMPLATE = (
    "I am feeling {label}. "
    "Valence={pleasure:+.2f}, Arousal={arousal:+.2f}, Dominance={dominance:+.2f}. "
    "Curiosity={curiosity:.2f}, Trust={trust:.2f}, Boredom={boredom:.2f}."
)


# ==========================================================
# Builder
# ==========================================================

class ContextBuilder:
    """
    Assembles a ContextBundle for the LLM each tick.

    Stateless besides the references to MemoryEngine and EmotionEngine.
    """

    def __init__(
        self,
        memory,
        emotion: EmotionEngine,
        *,
        top_beliefs: int = 5,
        recent_observations: int = 8,
        raw_tail_size: int = 6,
        person_blurb: str = PERSONA_PROMPT,
    ):

        self.memory = memory
        self.emotion = emotion
        self.top_beliefs = top_beliefs
        self.recent_observations = recent_observations
        self.raw_tail_size = raw_tail_size
        self.person_blurb = person_blurb

    # ==================================================
    # Public
    # ==================================================

    def build(self, user_input: str = "") -> ContextBundle:

        beliefs = self.memory.get_top_beliefs(k=self.top_beliefs)
        observations = self.memory.get_recent_observations(
            k=self.recent_observations,
        )
        raw_tail = self.memory.get_episode_tail(k=self.raw_tail_size)

        # Touch beliefs so they get stronger with use.
        for b in beliefs:
            self.memory._touch_belief(b.id)

        emotion_text = self._format_emotion()
        system = self._format_system()

        bundle = ContextBundle(
            system=system,
            emotion=emotion_text,
            beliefs=beliefs,
            observations=observations,
            raw_tail=raw_tail,
            user_input=user_input,
            estimated_tokens=0,
        )

        bundle.estimated_tokens = self._estimate_tokens(bundle)

        # Prune if over budget.
        bundle = self._enforce_budget(bundle)
        bundle.estimated_tokens = self._estimate_tokens(bundle)

        return bundle

    # ==================================================
    # Formatting
    # ==================================================

    def _format_system(self) -> str:

        return self.person_blurb

    def _format_emotion(self) -> str:

        s = self.emotion.state

        return EMOTION_TEMPLATE.format(
            label=self.emotion.dominant_label(),
            pleasure=s.pleasure,
            arousal=s.arousal,
            dominance=s.dominance,
            curiosity=s.curiosity,
            trust=s.trust,
            boredom=s.boredom,
        )

    # ==================================================
    # Budgeting
    # ==================================================

    def _estimate_tokens(self, bundle: ContextBundle) -> int:

        # ~ 1 token per 4 characters
        return sum(len(p) for p in self._render_pieces(bundle)) // 4

    def _render_pieces(self, bundle: ContextBundle) -> List[str]:

        pieces = [bundle.system, bundle.emotion]

        for b in bundle.beliefs:
            pieces.append(f"({b.category}, str={b.current_strength():.2f}) {b.statement}")

        for o in bundle.observations:
            pieces.append(f"[imp={o.importance:.2f}] {o.reasoning}")

        for ep in bundle.raw_tail:
            pieces.append(ep.summary)

        if bundle.user_input:
            pieces.append(bundle.user_input)

        return pieces

    def _enforce_budget(self, bundle: ContextBundle) -> ContextBundle:

        bundle = self._prune_observations(bundle)
        bundle = self._prune_beliefs(bundle)
        bundle = self._truncate_tail(bundle)

        return bundle

    def _prune_observations(self, bundle: ContextBundle) -> ContextBundle:

        while (
            bundle.estimated_tokens > TOTAL_BUDGET
            and len(bundle.observations) > 1
        ):

            bundle.observations.pop()  # least recent
            bundle.estimated_tokens = self._estimate_tokens(bundle)

        return bundle

    def _prune_beliefs(self, bundle: ContextBundle) -> ContextBundle:

        while (
            bundle.estimated_tokens > TOTAL_BUDGET
            and len(bundle.beliefs) > 1
        ):

            bundle.beliefs.pop()  # lowest strength
            bundle.estimated_tokens = self._estimate_tokens(bundle)

        return bundle

    def _truncate_tail(self, bundle: ContextBundle) -> ContextBundle:

        while (
            bundle.estimated_tokens > TOTAL_BUDGET
            and len(bundle.raw_tail) > 1
        ):

            bundle.raw_tail.pop()  # oldest raw message
            bundle.estimated_tokens = self._estimate_tokens(bundle)

        return bundle

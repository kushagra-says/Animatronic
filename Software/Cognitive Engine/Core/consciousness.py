"""
consciousness.py

The tick loop — the "moment of consciousness" of the Cognitive Engine.

Two threads:

    1. Event thread (synchronous, called from the CLI / main loop)
       on_input(text) -> runs the full perception -> ... -> output pipeline.

    2. Maintenance thread (synchronous, called periodically by the caller)
       maintenance_tick() -> decay, emotion pull-to-baseline, prune.

The split is deliberate — we don't want one fast async runtime deciding
what the engine is. The caller (CLI, animatronic body) drives the cadence.

See: Docs/ARCHITECTURE.md
"""

import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from Core.memory import MemoryEngine
from Core.memory_store import MemoryStore
from Core.emotion import EmotionEngine
from Core.context import ContextBuilder
from Core.llm import EngineLLM

# Heavy engine imports are deferred so this module loads even when the
# perception stack (gliner, sentence-transformers) is not installed.
# Callers who don't pass in their own engines will get the real ones
# via _default_*().
if TYPE_CHECKING:
    from Core.perception import PerceptionEngine
    from Core.observation_engine import ObservationEngine
    from Core.belief_engine import BeliefEngine


# ==========================================================
# Result type
# ==========================================================

@dataclass
class TickResult:
    perception_id: int
    observation_id: int
    belief_id: Optional[int]
    emotion_summary: str
    response_text: str
    elapsed_sec: float


# ==========================================================
# Consciousness
# ==========================================================

class Consciousness:
    """
    Owns the wiring between every cognitive module.

    Lifecycle: build → on_input() repeatedly → maintenance_tick() periodically.
    """

    # Maintenance interval between reflection passes. Reflection clusters
    # recent observations into higher-order "Pattern" beliefs; doing it
    # every 10th maintenance tick means roughly every 10 minutes at the
    # default 60s cadence.
    REFLECTION_EVERY_N_TICKS = 10

    def __init__(
        self,
        store: MemoryStore,
        *,
        perception: "Optional[PerceptionEngine]" = None,
        observation_engine: "Optional[ObservationEngine]" = None,
        belief_engine: "Optional[BeliefEngine]" = None,
        memory: Optional[MemoryEngine] = None,
        emotion: Optional[EmotionEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[EngineLLM] = None,
    ):

        self.store = store

        self.memory = memory or MemoryEngine(store)
        self.emotion = emotion or EmotionEngine()
        self.context_builder = context_builder or ContextBuilder(
            self.memory, self.emotion
        )
        self.llm = llm or EngineLLM()

        self.perception = perception or _default_perception()
        self.observation_engine = observation_engine or _default_observation()
        # Give the observation engine a handle to memory so it can
        # compute novelty against the recent episode tail instead of
        # returning a flat 0.50 placeholder.
        self.observation_engine._memory_engine = self.memory
        self.belief_engine = belief_engine or _default_belief()

        self._tick_count = 0

    # ==========================================================
    # Event path
    # ==========================================================

    def on_input(self, text: str) -> TickResult:

        start = time.perf_counter()

        # 1. Perceive
        perception = self.perception.perceive(text)

        # 2. Persist perception
        perception_id = self.memory.persist_perception(perception)

        # 3. Observe
        observation = self.observation_engine.observe(perception)
        observation_id = self.memory.persist_observation(
            observation, perception_id=perception_id
        )

        # 4. Update belief
        belief = self.belief_engine.process_observation(observation)
        belief_id = None

        if belief is not None:

            existing = self.memory.find_belief_by_statement(belief.statement)

            if existing is None:
                belief_id = self.memory.persist_belief(
                    statement=belief.statement,
                    category=belief.category,
                    importance=belief.importance,
                    base_strength=0.7,
                    confidence=belief.confidence,
                    observation_id=observation_id,
                    relation="support",
                )
            else:
                belief_id = existing.id
                self.memory.link_evidence(
                    belief_id, observation_id, relation="support"
                )

        # 5. Perturb emotion
        is_contradiction = (
            belief is not None and belief.contradiction_count > 0
        )

        self.emotion.apply_to_observation(
            novelty=observation.novelty,
            personal_relevance=observation.personal_relevance,
            goal_relevance=observation.goal_relevance,
            is_contradiction=is_contradiction,
            text=text,
        )

        # 6. Persist this moment as an episode (user turn)
        self.memory.persist_episode(
            summary=f"user: {text[:160]}",
            importance=observation.importance,
            started_at=time.time(),
        )

        # 7. Build context
        bundle = self.context_builder.build(user_input=text)

        # 8. Generate response
        result = self.llm.generate(
            system=bundle.render(),
            user=text,
        )

        # 9. Persist the assistant's reply as its own episode. Including
        # bot-side turns in the raw_tail lets the LLM reflect on what
        # it just said (and stitch together multi-step reasoning like
        # "I picked that line because..."). Speaker prefix ('assistant:')
        # matches the existing 'user:' convention so the renderer can
        # show both as-is.
        self.memory.persist_episode(
            summary=f"assistant: {result.text[:160]}",
            importance=observation.importance,
            started_at=time.time(),
        )

        elapsed = time.perf_counter() - start

        return TickResult(
            perception_id=perception_id,
            observation_id=observation_id,
            belief_id=belief_id,
            emotion_summary=self.emotion.summary(),
            response_text=result.text,
            elapsed_sec=elapsed,
        )

    # ==========================================================
    # Maintenance path
    # ==========================================================

    def maintenance_tick(self) -> dict:

        pruned = self.memory.apply_decay()
        self.emotion.decay()

        # Reflection: cluster recent observations into higher-order
        # beliefs. Runs every Nth tick to amortise cost.
        self._tick_count += 1
        beliefs_synthesised = 0

        if self._tick_count >= self.REFLECTION_EVERY_N_TICKS:
            beliefs_synthesised = self.memory.reflect()
            self._tick_count = 0

        return {
            "pruned_beliefs": pruned,
            "beliefs_synthesised": beliefs_synthesised,
            "tick_count": self._tick_count,
            "emotion": self.emotion.summary(),
            "stats": self.memory.stats(),
        }


# ==========================================================
# Lazy defaults — keep heavy imports out of this module's import time
# ==========================================================

def _default_perception():
    from Core.perception import PerceptionEngine
    return PerceptionEngine()


def _default_observation():
    from Core.observation_engine import ObservationEngine
    return ObservationEngine()


def _default_belief():
    from Core.belief_engine import BeliefEngine
    return BeliefEngine()


def _default_llm():
    from Core.llm import EngineLLM
    return EngineLLM()

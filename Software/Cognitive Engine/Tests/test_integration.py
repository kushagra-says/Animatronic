"""
Integration smoke test for the Cognitive Engine.

Boots Consciousness against an in-memory SQLite store, mocks the heavy
modules (LLM, PerceptionEngine, ObservationEngine, BeliefEngine), and
verifies that on_input() runs the full pipeline without raising.

Goal: catch wiring bugs between modules that unit tests can't see.

Run:
    python Tests/test_integration.py
"""

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Make Core.* and Utilities.* importable
ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from Core.memory_store import MemoryStore  # noqa: E402
from Core.memory import MemoryEngine        # noqa: E402
from Core.emotion import EmotionEngine     # noqa: E402
from Core.context import ContextBuilder     # noqa: E402
from Core.llm import LLMResult             # noqa: E402
from Core.consciousness import Consciousness, TickResult  # noqa: E402


# ==========================================================
# Fakes — minimal stand-ins for the heavy modules
# ==========================================================

@dataclass
class FakePerception:
    """Mirrors enough of Core.perception.Perception for memory writes."""
    timestamp: "FakeTimestamp"
    raw_text: str
    language: "FakeLanguage"
    semantic_evidence: list = field(default_factory=list)


@dataclass
class FakeLanguage:
    language: str = "en"
    confidence: float = 1.0


class FakeTimestamp:
    def timestamp(self) -> float:
        import time as _t
        return _t.time()


@dataclass
class FakeObservation:
    timestamp: "FakeTimestamp"
    novelty: float = 0.5
    persistence: float = 0.5
    personal_relevance: float = 0.5
    goal_relevance: float = 0.5
    temporal_relevance: float = 0.5
    importance: float = 0.5
    reasoning: str = "fake observation"


@dataclass
class FakeBelief:
    statement: str = "user is talking to me"
    category: str = "Interaction"
    importance: float = 0.6
    confidence: float = 0.7
    contradiction_count: int = 0


class FakePerceptionEngine:
    """Returns a deterministic Perception for any input."""
    def perceive(self, text: str) -> FakePerception:
        return FakePerception(
            timestamp=FakeTimestamp(),
            raw_text=text,
            language=FakeLanguage(),
        )


class FakeObservationEngine:
    """Returns a deterministic Observation."""
    def observe(self, perception: FakePerception) -> FakeObservation:
        return FakeObservation(timestamp=FakeTimestamp())


class FakeBeliefEngine:
    """Returns a fixed belief for the first input, then nothing."""
    call_count: int = 0

    def process_observation(self, observation: FakeObservation) -> Optional[FakeBelief]:
        self.call_count += 1
        if self.call_count == 1:
            return FakeBelief()
        return None


class FakeLLM:
    """Returns a canned response without hitting Ollama."""
    text: str = "(stubbed response)"

    def generate(self, system: str, user: str) -> LLMResult:
        return LLMResult(
            text=self.text,
            prompt_tokens=len(system) // 4,
            response_tokens=len(self.text) // 4,
        )


# ==========================================================
# Tests
# ==========================================================

def test_consciousness_boot_and_first_tick(tmp_path: Path) -> None:
    """Boots the full Consciousness with fakes, runs on_input once."""

    store = MemoryStore(db_path=tmp_path / "integration.db")
    try:
        consciousness = Consciousness(
            store=store,
            perception=FakePerceptionEngine(),
            observation_engine=FakeObservationEngine(),
            belief_engine=FakeBeliefEngine(),
            memory=MemoryEngine(store),
            emotion=EmotionEngine(),
            context_builder=ContextBuilder(
                MemoryEngine(store),
                EmotionEngine(),
            ),
            llm=FakeLLM(),
        )

        result = consciousness.on_input("hello there")

        assert isinstance(result, TickResult), \
            f"Expected TickResult, got {type(result).__name__}"

        assert result.response_text == "(stubbed response)", \
            f"Stub LLM should return its canned text, got: {result.response_text!r}"

        assert result.perception_id > 0, "perception_id should be assigned"
        assert result.observation_id > 0, "observation_id should be assigned"
        assert result.belief_id is not None, \
            "belief_id should be set (FakeBeliefEngine returns a belief on first call)"
        assert result.elapsed_sec >= 0.0, "elapsed_sec should be non-negative"
        assert result.emotion_summary, "emotion_summary should be non-empty"

        # Memory should actually contain something now.
        stats = consciousness.memory.stats()
        assert stats["perceptions"] == 1, f"Expected 1 perception, got {stats['perceptions']}"
        assert stats["observations"] == 1, f"Expected 1 observation, got {stats['observations']}"
        assert stats["beliefs"] >= 1, f"Expected at least 1 belief, got {stats['beliefs']}"
        assert stats["episodes"] == 1, f"Expected 1 episode, got {stats['episodes']}"
        assert stats["active_beliefs"] >= 1

        # Evidence graph should link the belief to the observation.
        row = store.conn.execute(
            """
            SELECT relation FROM belief_evidence
            WHERE belief_id = ? AND observation_id = ?
            """,
            (result.belief_id, result.observation_id),
        ).fetchone()

        assert row is not None, "Belief should be linked to its observation"
        assert row["relation"] == "support"

        # Second on_input should also work (this is where wiring bugs often hide).
        result_2 = consciousness.on_input("how are you?")

        assert isinstance(result_2, TickResult)
        assert result_2.response_text == "(stubbed response)"
        assert result_2.perception_id != result.perception_id, \
            "Each on_input should produce a fresh perception_id"

    finally:
        store.close()

    print("test_consciousness_boot_and_first_tick: PASS")


def test_maintenance_tick_runs() -> None:
    """The maintenance path should not raise even on an empty store."""

    store = MemoryStore(db_path=tmp_path_for_maintenance() / "maintenance.db")
    try:
        consciousness = Consciousness(
            store=store,
            perception=FakePerceptionEngine(),
            observation_engine=FakeObservationEngine(),
            belief_engine=FakeBeliefEngine(),
            memory=MemoryEngine(store),
            emotion=EmotionEngine(),
            context_builder=ContextBuilder(MemoryEngine(store), EmotionEngine()),
            llm=FakeLLM(),
        )

        report = consciousness.maintenance_tick()

        assert "pruned_beliefs" in report
        assert "beliefs_synthesised" in report
        assert "emotion" in report
        assert "stats" in report
        assert report["pruned_beliefs"] == 0, "Nothing to prune on a fresh store"
        # Default cadence is 10 ticks between reflection passes; the first
        # tick should not trigger reflection.
        assert report["beliefs_synthesised"] == 0
        assert report["tick_count"] == 1

    finally:
        store.close()

    print("test_maintenance_tick_runs: PASS")


def test_reflection_triggers_every_n_ticks() -> None:
    """After REFLECTION_EVERY_N_TICKS calls to maintenance_tick(),
    reflect() should have run and synthesised at least one Pattern belief."""

    with tempfile.TemporaryDirectory(prefix="cog-refl-") as tmp:
        store = MemoryStore(db_path=Path(tmp) / "reflection.db")
        try:
            memory = MemoryEngine(store)
            consciousness = Consciousness(
                store=store,
                perception=FakePerceptionEngine(),
                observation_engine=FakeObservationEngine(),
                belief_engine=FakeBeliefEngine(),
                memory=memory,
                emotion=EmotionEngine(),
                context_builder=ContextBuilder(MemoryEngine(store), EmotionEngine()),
                llm=FakeLLM(),
            )

            # Seed three observations that share content words so they
            # cluster together under the Jaccard heuristic.
            for reasoning in [
                "User likes Tame Impala and psychedelic music",
                "User mentioned Tame Impala again today",
                "Tame Impala is a recurring topic in conversation",
            ]:
                memory.store.conn.execute(
                    """
                    INSERT INTO observations
                        (timestamp, perception_id, novelty, persistence,
                         personal_relevance, goal_relevance, temporal_relevance,
                         importance, reasoning)
                    VALUES (?, NULL, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, ?)
                    """,
                    (time.time(), reasoning),
                )

            # Force the reflection threshold to fire on the very next tick.
            consciousness.REFLECTION_EVERY_N_TICKS = 2

            # First tick: counts up, no reflection yet.
            report_1 = consciousness.maintenance_tick()
            assert report_1["beliefs_synthesised"] == 0

            # Second tick: reflection runs.
            report_2 = consciousness.maintenance_tick()
            assert report_2["beliefs_synthesised"] >= 1, \
                f"Expected at least one synthesised belief, got {report_2}"

            # A "Pattern" belief should now exist linking the three
            # observations.
            row = store.conn.execute(
                """
                SELECT id, statement, category FROM beliefs
                WHERE category = 'Pattern' AND is_active = 1
                """
            ).fetchone()

            assert row is not None, "Expected a Pattern belief after reflection"
            assert row["statement"].startswith("Recurring theme:")

            # Evidence links should be in place.
            links = store.conn.execute(
                """
                SELECT COUNT(*) AS c FROM belief_evidence be
                JOIN beliefs b ON be.belief_id = b.id
                WHERE b.id = ? AND be.relation = 'support'
                """,
                (row["id"],),
            ).fetchone()["c"]

            assert links == 3, f"Expected 3 evidence links, got {links}"

        finally:
            store.close()

    print("test_reflection_triggers_every_n_ticks: PASS")


def tmp_path_for_maintenance() -> Path:
    return Path(tempfile.mkdtemp(prefix="cog-maint-"))


def main() -> None:

    print("=" * 60)
    print("CONSCIOUSNESS INTEGRATION TESTS")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="cog-int-") as tmp:
        test_consciousness_boot_and_first_tick(Path(tmp))

    test_maintenance_tick_runs()
    test_reflection_triggers_every_n_ticks()

    print("\nAll Consciousness integration tests passed.")


if __name__ == "__main__":
    main()

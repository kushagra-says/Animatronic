"""
Tests the ContextBuilder assembly and budget enforcement.
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from Core.context import ContextBuilder, TOTAL_BUDGET  # noqa: E402
from Core.emotion import EmotionEngine, EmotionState  # noqa: E402


class FakeMemory:
    """Stand-in for MemoryEngine — returns canned data."""

    def __init__(self, beliefs=None, observations=None, tail=None):

        self.beliefs = beliefs or []
        self.observations = observations or []
        self.tail = tail or []
        self._touched = []

    def get_top_beliefs(self, k=5):

        return self.beliefs[:k]

    def get_recent_observations(self, k=8, **kwargs):

        return self.observations[:k]

    def get_episode_tail(self, k):

        return self.tail[:k]

    def _touch_belief(self, belief_id):

        self._touched.append(belief_id)


def test_basic_assembly() -> None:

    fake = FakeMemory()
    emotion = EmotionEngine(EmotionState(pleasure=0.2, curiosity=0.7))
    builder = ContextBuilder(fake, emotion)

    bundle = builder.build(user_input="hello")

    assert "AI being" in bundle.system
    assert "Curiosity" in bundle.emotion
    assert "hello" in bundle.user_input

    print("test_basic_assembly: PASS")


def test_budget_enforcement() -> None:

    # Pad beliefs with very long statements to blow the budget.
    long_beliefs = []

    for i in range(20):

        class B:
            id = i
            category = "Test"
            statement = "x" * 500
            def current_strength(self):
                return 1.0

        long_beliefs.append(B())

    fake = FakeMemory(beliefs=long_beliefs)
    emotion = EmotionEngine()
    builder = ContextBuilder(fake, emotion, top_beliefs=20)

    bundle = builder.build(user_input="hi")

    assert bundle.estimated_tokens <= TOTAL_BUDGET + 80, \
        f"Bundle over budget: {bundle.estimated_tokens}"

    print(f"test_budget_enforcement: PASS (tokens={bundle.estimated_tokens})")


def test_tail_is_sacred() -> None:

    # Trigger pruning by stuffing observations; the tail must still be there.
    class StubObs:
        id = 0
        importance = 0.9
        reasoning = "x" * 400

    observations = [StubObs() for _ in range(20)]

    class Tail:
        id = 0
        summary = "important recent message"

    fake = FakeMemory(observations=observations, tail=[Tail()])
    emotion = EmotionEngine()
    builder = ContextBuilder(fake, emotion)

    bundle = builder.build(user_input="")

    # Even with pruning, the tail should be the last thing to go.
    assert len(bundle.raw_tail) >= 1, "Tail should be preserved"

    print("test_tail_is_sacred: PASS")


def main():

    print("=" * 60)
    print("CONTEXT BUILDER TESTS")
    print("=" * 60)

    test_basic_assembly()
    test_budget_enforcement()
    test_tail_is_sacred()

    print("\nAll ContextBuilder tests passed.")


if __name__ == "__main__":
    main()

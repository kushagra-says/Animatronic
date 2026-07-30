"""
Tests the EmotionEngine state and perturbation rules.
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from Core.emotion import (  # noqa: E402
    EmotionEngine,
    EmotionState,
    EmotionRules,
    EmotionPerturbation,
)


def test_initial_state_neutral() -> None:

    engine = EmotionEngine()

    assert engine.state.pleasure == 0.0
    assert engine.state.arousal == 0.0
    assert engine.state.dominance == 0.0
    assert engine.state.curiosity == 0.5

    print("test_initial_state_neutral: PASS")


def test_perturbation_clamps() -> None:

    engine = EmotionEngine()

    big = EmotionPerturbation(pleasure=5.0, arousal=-5.0, trust=10.0)
    engine.perturb(big)

    assert engine.state.pleasure == 1.0
    assert engine.state.arousal == -1.0
    assert engine.state.trust == 1.0

    print("test_perturbation_clamps: PASS")


def test_decay_pulls_to_baseline() -> None:

    engine = EmotionEngine(EmotionState(pleasure=1.0, trust=1.0, curiosity=0.0))
    engine.decay()

    # 10% pull toward baseline.
    # pleasure baseline = 0.0 → 1.0 - 0.10 = 0.90
    # trust baseline    = 0.5 → 1.0 - 0.05 = 0.95
    # curiosity baseline = 0.5 → 0.0 + 0.05 = 0.05
    assert 0.85 < engine.state.pleasure < 0.95
    assert 0.90 < engine.state.trust < 1.00
    assert 0.00 < engine.state.curiosity < 0.10

    print("test_decay_pulls_to_baseline: PASS")


def test_threat_rule() -> None:

    engine = EmotionEngine()

    deltas = EmotionRules.for_observation(
        novelty=0.5,
        personal_relevance=0.5,
        goal_relevance=0.5,
        is_contradiction=False,
        text="I hate you, shut up",
    )

    assert deltas.pleasure < 0, "Threat should drop pleasure"
    assert deltas.arousal > 0, "Threat should spike arousal"

    print("test_threat_rule: PASS")


def test_compliment_rule() -> None:

    engine = EmotionEngine()

    deltas = EmotionRules.for_observation(
        novelty=0.5,
        personal_relevance=0.5,
        goal_relevance=0.5,
        is_contradiction=False,
        text="Thanks, appreciate it",
    )

    assert deltas.pleasure > 0
    assert deltas.trust > 0

    print("test_compliment_rule: PASS")


def test_dominant_label() -> None:

    engine = EmotionEngine()

    engine.state.pleasure = 0.5
    engine.state.arousal = 0.5
    assert engine.dominant_label() in ("excited", "pleasant")

    engine.state.pleasure = -0.5
    engine.state.boredom = 0.8
    assert engine.dominant_label() == "bored"

    engine.state.boredom = 0.0
    engine.state.arousal = -0.5
    assert engine.dominant_label() == "calm"

    print("test_dominant_label: PASS")


def main():

    print("=" * 60)
    print("EMOTION ENGINE TESTS")
    print("=" * 60)

    test_initial_state_neutral()
    test_perturbation_clamps()
    test_decay_pulls_to_baseline()
    test_threat_rule()
    test_compliment_rule()
    test_dominant_label()

    print("\nAll EmotionEngine tests passed.")


if __name__ == "__main__":
    main()

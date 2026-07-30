"""
Tests the MemoryEngine + ACT-R decay semantics.
"""

import sys
import time
from pathlib import Path

# Make Core.* and Utilities.* importable
ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from Core.memory import MemoryEngine  # noqa: E402
from Core.memory_store import MemoryStore  # noqa: E402


def make_engine(tmp_path: Path) -> tuple:

    """Returns (engine, store) so the caller can close the store on exit."""

    store = MemoryStore(db_path=tmp_path / "test.db")
    return MemoryEngine(store), store


def test_persist_and_retrieve(tmp_path: Path) -> None:

    engine, store = make_engine(tmp_path)
    try:

        bid = engine.persist_belief(
            statement="User likes Tame Impala",
            category="Preference",
            importance=0.7,
            base_strength=0.7,
        )

        assert bid > 0, "Expected a row id"

        top = engine.get_top_beliefs(k=5)

        assert len(top) == 1, "Should have one belief"
        assert top[0].statement == "User likes Tame Impala"

    finally:
        store.close()

    print("test_persist_and_retrieve: PASS")


def test_act_r_decay(tmp_path: Path) -> None:

    engine, store = make_engine(tmp_path)
    try:

        bid = engine.persist_belief(
            statement="Decay test",
            category="Test",
            importance=0.5,
            base_strength=1.0,
        )

        # Fresh belief at t=0: strength = base * ln(1 + 0) / 1 = 0
        belief = engine.get_belief_by_id(bid)
        assert belief.current_strength() == 0.0, \
            "Fresh belief with 0 accesses should have strength 0"

        # Force accesses = 3
        store.conn.execute(
            "UPDATE beliefs SET accesses = 3 WHERE id = ?", (bid,)
        )

        belief = engine.get_belief_by_id(bid)
        s = belief.current_strength()
        # base=1.0, accesses=3 -> ln(4) / 1 ~ 1.386
        assert 1.3 < s < 1.5, f"Expected ~1.386, got {s}"

    finally:
        store.close()

    print(f"test_act_r_decay: PASS (strength={s:.3f})")


def test_prune_below_threshold(tmp_path: Path) -> None:

    engine, store = make_engine(tmp_path)
    try:

        # A belief with very low base_strength that hasn't been touched
        engine.persist_belief(
            statement="weak belief",
            category="Test",
            importance=0.1,
            base_strength=0.01,
        )

        pruned = engine.apply_decay()

        assert pruned == 1, f"Should have pruned 1 weak belief, pruned {pruned}"

        top = engine.get_top_beliefs(k=5)
        assert len(top) == 0, "Weak belief should be soft-deleted"

    finally:
        store.close()

    print("test_prune_below_threshold: PASS")


def test_top_belief_ranking(tmp_path: Path) -> None:

    engine, store = make_engine(tmp_path)
    try:

        engine.persist_belief(
            statement="weak",
            category="A",
            importance=0.1,
            base_strength=0.3,
        )
        engine.persist_belief(
            statement="strong",
            category="B",
            importance=0.9,
            base_strength=0.9,
        )
        engine.persist_belief(
            statement="medium",
            category="C",
            importance=0.5,
            base_strength=0.6,
        )

        top = engine.get_top_beliefs(k=3)

        assert len(top) == 3
        # Right after creation, accesses is 0, so strength = 0 for all
        # So the ranking by base_strength kicks in
        assert top[0].statement == "strong"
        assert top[-1].statement == "weak"

    finally:
        store.close()

    print("test_top_belief_ranking: PASS")


def main():

    print("=" * 60)
    print("MEMORY ENGINE TESTS")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:

        tmp_path = Path(tmp)
        test_persist_and_retrieve(tmp_path)

    with tempfile.TemporaryDirectory() as tmp:

        tmp_path = Path(tmp)
        test_act_r_decay(tmp_path)

    with tempfile.TemporaryDirectory() as tmp:

        tmp_path = Path(tmp)
        test_prune_below_threshold(tmp_path)

    with tempfile.TemporaryDirectory() as tmp:

        tmp_path = Path(tmp)
        test_top_belief_ranking(tmp_path)

    print("\nAll MemoryEngine tests passed.")


if __name__ == "__main__":
    main()

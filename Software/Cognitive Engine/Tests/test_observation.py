"""
test_observation.py

End-to-end integration test for the Observation Engine.
"""

import time

from Core.perception import PerceptionEngine
from Core.observation_engine import ObservationEngine


TEXT = """
My name is Kushagra.

I study AI & ML at Ramaiah Institute of Technology.

My favorite band is Tame Impala.

I am building an offline animatronic AI.
"""


def print_line():
    print("-" * 60)


def main():

    print("=" * 70)
    print("OBSERVATION ENGINE TEST")
    print("=" * 70)

    # ------------------------------------------------------
    # Initialize Engines
    # ------------------------------------------------------

    perception_engine = PerceptionEngine()
    observation_engine = ObservationEngine()

    # ------------------------------------------------------
    # Perception
    # ------------------------------------------------------

    start = time.perf_counter()

    perception = perception_engine.perceive(TEXT)

    perception_time = time.perf_counter() - start

    # ------------------------------------------------------
    # Observation
    # ------------------------------------------------------

    start = time.perf_counter()

    observation = observation_engine.observe(perception)

    observation_time = time.perf_counter() - start

    total_time = perception_time + observation_time

    # ======================================================
    # PERCEPTION
    # ======================================================

    print("\nPERCEPTION")
    print_line()

    print(f"Language : {perception.language.language}")
    print(f"Confidence : {perception.language.confidence:.3f}")

    print("\nEntities")
    print_line()

    for entity in perception.entities:

        print(
            f"{entity.text:<35}"
            f"{entity.label:<20}"
            f"{entity.confidence:.3f}"
        )

    print("\nEmbedding")
    print_line()

    print(f"Dimension : {perception.semantics.dimension}")

    # ======================================================
    # OBSERVATION
    # ======================================================

    print("\nOBSERVATION")
    print_line()

    print(f"Novelty             : {observation.novelty:.3f}")

    print(f"Persistence         : {observation.persistence:.3f}")

    print(f"Personal Relevance  : {observation.personal_relevance:.3f}")

    print(f"Goal Relevance      : {observation.goal_relevance:.3f}")

    print(f"Temporal Relevance  : {observation.temporal_relevance:.3f}")

    print(f"\nImportance          : {observation.importance:.3f}")

    print("\nReasoning")
    print_line()

    print(observation.reasoning)

    # ======================================================
    # PERFORMANCE
    # ======================================================

    print("\nPERFORMANCE")
    print_line()

    print(f"Perception Time : {perception_time*1000:.2f} ms")

    print(f"Observation Time: {observation_time*1000:.2f} ms")

    print(f"Total Pipeline  : {total_time*1000:.2f} ms")

    print("\nTimestamp")
    print_line()

    print(observation.timestamp)

    print("\nObservation successfully created.")


if __name__ == "__main__":

    main()
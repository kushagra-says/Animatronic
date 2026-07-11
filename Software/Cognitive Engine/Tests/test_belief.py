"""
test_belief.py

End-to-end test for the Belief Engine.

Pipeline:

Raw Text
    ↓
Perception
    ↓
Observation
    ↓
Belief
"""

from Core.perception import PerceptionEngine
from Core.observation_engine import ObservationEngine
from Core.belief_engine import BeliefEngine


TEST_INPUTS = [

    "My name is Kushagra.",

    "I study AI & ML at Ramaiah Institute of Technology.",

    "My favorite band is Tame Impala.",

    "I love listening to Tame Impala.",

    "I am building an offline AI.",

    "My favorite band is Tame Impala."
]


def separator():

    print("=" * 80)


def main():

    separator()

    print("BELIEF ENGINE TEST")

    separator()

    perception_engine = PerceptionEngine()

    observation_engine = ObservationEngine()

    belief_engine = BeliefEngine()

    # ----------------------------------------------------------

    for index, text in enumerate(

        TEST_INPUTS,

        start=1

    ):

        print()

        separator()

        print(f"INPUT #{index}")

        separator()

        print(text)

        # ----------------------------------------------

        perception = perception_engine.perceive(text)

        observation = observation_engine.observe(

            perception

        )

        belief = belief_engine.process_observation(

            observation

        )

        # ----------------------------------------------

        print()

        print("Generated Belief")

        print("-" * 50)

        print(f"Category     : {belief.category}")

        print(f"Statement    : {belief.statement}")

        print(f"Importance   : {belief.importance:.3f}")

        print(f"Strength     : {belief.strength:.3f}")

        print(f"Confidence   : {belief.confidence:.3f}")

        print(f"Support      : {belief.reinforcement_count}")

        print(f"Contradiction: {belief.contradiction_count}")

    # ==========================================================

    print()

    separator()

    print("FINAL BELIEF STORE")

    separator()

    belief_engine.print_beliefs()


if __name__ == "__main__":

    main()
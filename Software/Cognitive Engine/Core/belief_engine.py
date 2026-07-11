"""
belief_engine.py

Creates and updates beliefs from observations.

Version 1:
- Creates beliefs from observations.
- Reinforces existing beliefs.
- Uses simple statement matching.

Future versions will use:
- Semantic similarity
- Memory retrieval
- Contradiction detection
- Belief merging
"""

from datetime import datetime

from Core.belief import Belief


class BeliefEngine:

    def __init__(self):

        # Prototype belief store
        self.beliefs = []

    # ==========================================================
    # Public API
    # ==========================================================

    def process_observation(self, observation):

        """
        Main entry point.

        Observation
            ↓
        Find Matching Belief
            ↓
        Update OR Create
        """

        statement = self.generate_statement(observation)

        belief = self.find_matching_belief(statement)

        if belief is None:

            belief = self.create_belief(

                observation,

                statement

            )

            self.beliefs.append(belief)

        else:

            self.reinforce_belief(

                belief,

                observation

            )

        return belief

    # ==========================================================
    # Belief Creation
    # ==========================================================

    def create_belief(

        self,

        observation,

        statement

    ):

        belief = Belief()

        belief.statement = statement

        belief.category = self.classify_belief(observation)

        belief.importance = observation.importance

        belief.supporting_observations.append(

            observation

        )

        belief.reinforcement_count = 1

        belief.update_strength()

        belief.update_confidence()

        return belief

    # ==========================================================
    # Reinforcement
    # ==========================================================

    def reinforce_belief(

        self,

        belief,

        observation

    ):

        belief.reinforce(observation)

        belief.update_strength()

        belief.update_confidence()

        belief.updated_at = datetime.now()

    # ==========================================================
    # Matching
    # ==========================================================

    def find_matching_belief(

        self,

        statement

    ):

        for belief in self.beliefs:

            if belief.statement == statement:

                return belief

        return None

    # ==========================================================
    # Statement Generation
    # ==========================================================

    def generate_statement(

        self,

        observation

    ):

        entities = observation.perception.entities

        # ----------------------------------------------

        for entity in entities:

            if entity.label == "person":

                return f"User name is {entity.text}"

        # ----------------------------------------------

        for entity in entities:

            if entity.label == "music artist":

                return f"User likes {entity.text}"

        # ----------------------------------------------

        for entity in entities:

            if entity.label == "university":

                return f"User studies at {entity.text}"

        # ----------------------------------------------

        text = observation.perception.raw_text.strip()

        if len(text) > 80:

            text = text[:80] + "..."

        return text

    # ==========================================================
    # Classification
    # ==========================================================

    def classify_belief(

        self,

        observation

    ):

        entities = observation.perception.entities

        for entity in entities:

            if entity.label == "person":

                return "Identity"

            if entity.label == "music artist":

                return "Preference"

            if entity.label == "university":

                return "Education"

            if entity.label == "technology":

                return "Knowledge"

        return "General"

    # ==========================================================
    # Utilities
    # ==========================================================

    def list_beliefs(self):

        return self.beliefs

    # ----------------------------------------------------------

    def print_beliefs(self):

        print()

        print("=" * 70)

        print("CURRENT BELIEFS")

        print("=" * 70)

        if len(self.beliefs) == 0:

            print("No beliefs stored.")

            return

        for i, belief in enumerate(

            self.beliefs,

            start=1

        ):

            print()

            print(f"Belief #{i}")

            print("-" * 50)

            print(f"ID               : {belief.id}")

            print(f"Category         : {belief.category}")

            print(f"Statement        : {belief.statement}")

            print(f"Importance       : {belief.importance:.3f}")

            print(f"Strength         : {belief.strength:.3f}")

            print(f"Confidence       : {belief.confidence:.3f}")

            print(f"Reinforcements   : {belief.reinforcement_count}")

            print(f"Contradictions   : {belief.contradiction_count}")

            print(
                f"Supporting Obs.  : "
                f"{len(belief.supporting_observations)}"
            )

            print(
                f"Contradicting Obs: "
                f"{len(belief.contradicting_observations)}"
            )

            print(f"Created          : {belief.created_at}")

            print(f"Updated          : {belief.updated_at}")
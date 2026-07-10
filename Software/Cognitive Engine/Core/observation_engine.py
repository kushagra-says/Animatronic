"""
observation_engine.py

Evaluates a Perception and produces an Observation.

Version 1 uses deterministic heuristics.
These heuristics will later be replaced with learned models.
"""

from datetime import datetime

from Core.observation import Observation


class ObservationEngine:

    def __init__(self):

        pass

    # ==========================================================
    # Individual Cognitive Attributes
    # ==========================================================

    def evaluate_novelty(self, perception):

        """
        Placeholder.

        Without memory we cannot determine true novelty.

        Returning a neutral score for now.
        """

        return 0.50

    # ----------------------------------------------------------

    def evaluate_persistence(self, perception):

        score = 0.20

        for entity in perception.entities:

            if entity.label in [

                "person",

                "music artist",

                "university",

                "field of study"

            ]:

                score += 0.20

        return min(score, 1.0)

    # ----------------------------------------------------------

    def evaluate_personal_relevance(self, perception):

        score = 0.0

        for entity in perception.entities:

            if entity.label in [

                "person",

                "music artist",

                "university",

                "field of study"

            ]:

                score += 0.25

        return min(score, 1.0)

    # ----------------------------------------------------------

    def evaluate_goal_relevance(self, perception):

        text = perception.raw_text.lower()

        keywords = [

            "project",

            "research",

            "build",

            "study",

            "learn",

            "ai",

            "memory",

            "cognitive",

            "offline"

        ]

        score = 0.0

        for word in keywords:

            if word in text:

                score += 0.10

        return min(score, 1.0)

    # ----------------------------------------------------------

    def evaluate_temporal_relevance(self, perception):

        text = perception.raw_text.lower()

        temporary_words = [

            "today",

            "tomorrow",

            "yesterday",

            "currently",

            "now",

            "this week"

        ]

        for word in temporary_words:

            if word in text:

                return 1.0

        return 0.20

    # ==========================================================
    # Importance
    # ==========================================================

    def calculate_importance(

        self,

        novelty,

        persistence,

        personal,

        goal,

        temporal

    ):

        weights = {

            "novelty": 0.20,

            "persistence": 0.25,

            "personal": 0.30,

            "goal": 0.15,

            "temporal": 0.10
        }

        importance = (

            novelty * weights["novelty"]

            + persistence * weights["persistence"]

            + personal * weights["personal"]

            + goal * weights["goal"]

            + temporal * weights["temporal"]

        )

        return round(min(importance, 1.0), 3)

    # ==========================================================
    # Observation
    # ==========================================================

    def observe(self, perception):

        novelty = self.evaluate_novelty(

            perception

        )

        persistence = self.evaluate_persistence(

            perception

        )

        personal = self.evaluate_personal_relevance(

            perception

        )

        goal = self.evaluate_goal_relevance(

            perception

        )

        temporal = self.evaluate_temporal_relevance(

            perception

        )

        importance = self.calculate_importance(

            novelty,

            persistence,

            personal,

            goal,

            temporal

        )

        reasoning = (

            f"Novelty={novelty:.2f}, "

            f"Persistence={persistence:.2f}, "

            f"Personal={personal:.2f}, "

            f"Goal={goal:.2f}, "

            f"Temporal={temporal:.2f}"

        )

        return Observation(

            perception=perception,

            timestamp=datetime.now(),

            novelty=novelty,

            persistence=persistence,

            personal_relevance=personal,

            goal_relevance=goal,

            temporal_relevance=temporal,

            importance=importance,

            reasoning=reasoning

        )
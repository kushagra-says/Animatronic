"""
belief.py

Defines the Belief object.

A Belief represents the Cognitive Engine's long-term understanding
of the world, formed from one or more observations.

Beliefs evolve over time.
They are reinforced, weakened, or replaced as new observations arrive.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from Core.observation import Observation


# ==========================================================
# Belief
# ==========================================================

@dataclass
class Belief:

    # ------------------------------------------------------
    # Identity
    # ------------------------------------------------------

    id: str = field(default_factory=lambda: str(uuid4()))

    # ------------------------------------------------------
    # Classification
    # ------------------------------------------------------

    category: str = "Unknown"

    statement: str = ""

    # ------------------------------------------------------
    # Cognitive State
    # ------------------------------------------------------

    importance: float = 0.0

    strength: float = 0.5

    confidence: float = 0.5

    # ------------------------------------------------------
    # Supporting Evidence
    # ------------------------------------------------------

    supporting_observations: list[Observation] = field(
        default_factory=list
    )

    contradicting_observations: list[Observation] = field(
        default_factory=list
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    reinforcement_count: int = 0

    contradiction_count: int = 0

    access_count: int = 0

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    created_at: datetime = field(
        default_factory=datetime.now
    )

    updated_at: datetime = field(
        default_factory=datetime.now
    )

    last_accessed: datetime = field(
        default_factory=datetime.now
    )

    # ======================================================
    # Utility Methods
    # ======================================================

    def reinforce(self, observation: Observation):

        self.supporting_observations.append(observation)

        self.reinforcement_count += 1

        self.updated_at = datetime.now()

    # ------------------------------------------------------

    def contradict(self, observation: Observation):

        self.contradicting_observations.append(observation)

        self.contradiction_count += 1

        self.updated_at = datetime.now()

    # ------------------------------------------------------

    def accessed(self):

        self.access_count += 1

        self.last_accessed = datetime.now()

    # ------------------------------------------------------

    def update_strength(self):

        """
        Placeholder implementation.

        Later this will include:

        - reinforcement
        - contradiction
        - recency
        - temporal decay
        """

        support = self.reinforcement_count + 1

        contradiction = self.contradiction_count

        self.strength = max(
            0.0,
            min(
                support / (support + contradiction + 1),
                1.0
            )
        )

    # ------------------------------------------------------

    def update_confidence(self):

        """
        Prototype implementation.

        Confidence is derived from the current strength
        and the original importance.
        """

        self.confidence = round(

            (self.strength + self.importance) / 2,

            3

        )

    # ------------------------------------------------------

    def summary(self):

        return {

            "id": self.id,

            "category": self.category,

            "statement": self.statement,

            "importance": self.importance,

            "strength": self.strength,

            "confidence": self.confidence,

            "reinforcements": self.reinforcement_count,

            "contradictions": self.contradiction_count,

            "supporting_observations": len(
                self.supporting_observations
            ),

            "contradicting_observations": len(
                self.contradicting_observations
            ),

            "created_at": self.created_at,

            "updated_at": self.updated_at
        }
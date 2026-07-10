"""
observation.py

Defines the Observation object.

An Observation represents the Cognitive Engine's evaluation
of a Perception.

Observation is immutable and contains no decision-making logic.
The ObservationEngine is responsible for constructing it.
"""

from dataclasses import dataclass
from datetime import datetime

from Core.perception import Perception


# ==========================================================
# Observation
# ==========================================================

@dataclass(frozen=True)
class Observation:
    """
    Represents a single evaluated perception.
    """

    # Original perception
    perception: Perception

    # Time the observation was created
    timestamp: datetime

    # ------------------------------------------------------
    # Cognitive Attributes
    # ------------------------------------------------------

    # How new is this information?
    novelty: float

    # How likely is this information to remain true?
    persistence: float

    # How relevant is this to the user?
    personal_relevance: float

    # How relevant is this to the AI's current objectives?
    goal_relevance: float

    # How time-sensitive is this information?
    temporal_relevance: float

    # ------------------------------------------------------
    # Overall Evaluation
    # ------------------------------------------------------

    # Overall importance assigned by the Observation Engine.
    importance: float

    # Optional explanation for debugging and testing.
    reasoning: str
"""
observation_engine.py

Evaluates a Perception and produces an Observation.

Version 1 uses deterministic heuristics.
These heuristics will later be replaced with learned models.
"""

from datetime import datetime

from Core.observation import Observation


# Entity labels that count toward persistence / personal relevance.
# Compared lowercased and prefix-stripped so GLiNER label drift
# ("Music Artist", "music", "artist"...) still matches.
_PERSISTENT_ENTITY_LABELS = {
    "person",
    "music artist",
    "music",
    "artist",
    "university",
    "field of study",
    "field",
    "study",
    "organization",
    "location",
}

# First-person / possessive phrases that signal personal relevance
# even when no entity matches the whitelist.
_PERSONAL_PHRASES = (
    "my ", "i am", "i'm", "i love", "i hate", "i like", "i prefer",
    "i think", "i feel", "i want", "i need", "i study", "i work",
    "i live", "favourite", "favorite", "i've ", "i have",
)

# Stopwords used for cheap content-word Jaccard in evaluate_novelty.
_NOVELTY_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "her", "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "has", "him", "let", "say", "she",
    "too", "use", "with", "from", "this", "that", "have", "will",
    "one", "two", "any", "was", "were", "been", "they", "them",
    "your", "yours", "user", "user:",
})

# Recent-episode window used for novelty. Cheap to compute, captures
# what the user has been saying over the last few minutes.
_NOVELTY_WINDOW = 8


def _content_words(text: str) -> set:
    """Lowercase tokens >3 chars, minus stopwords. Cheap content fingerprint."""
    if not text:
        return set()
    out = set()
    for tok in text.lower().split():
        cleaned = "".join(c for c in tok if c.isalnum())
        if len(cleaned) > 3 and cleaned not in _NOVELTY_STOPWORDS:
            out.add(cleaned)
    return out


def _normalise_label(label: str) -> str:
    """Lower-case + strip known GLiNER prefixes so label drift still matches."""
    s = label.strip().lower()
    for prefix in ("music ", "field "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


class ObservationEngine:

    def __init__(self):

        pass

    # ==========================================================
    # Individual Cognitive Attributes
    # ==========================================================

    def evaluate_novelty(self, perception):

        """
        Compute novelty against the recent episode tail (in-DB).

        Returns 1.0 if the perception is completely unlike the last
        few user utterances; 0.1 if it is a near-duplicate; 0.5 if
        there is no episode history to compare against (cold start).
        """

        text = perception.raw_text

        try:
            from Core.memory_store import MemoryStore  # noqa: F401
            from Core.memory import MemoryEngine

            # Lazy: only touch the DB when MemoryEngine is wired in.
            engine = getattr(self, "_memory_engine", None)
            if engine is None:
                return 0.50

            tail = engine.get_episode_tail(k=_NOVELTY_WINDOW)
        except Exception:
            return 0.50

        if not tail:
            return 0.50  # cold start — neutral

        current_words = _content_words(text)
        if not current_words:
            return 0.50

        # Max Jaccard similarity against the last K episodes.
        max_sim = 0.0

        for ep in tail:
            ep_words = _content_words(ep.summary)
            if not ep_words:
                continue
            inter = len(current_words & ep_words)
            union = len(current_words | ep_words)
            if union == 0:
                continue
            sim = inter / union
            if sim > max_sim:
                max_sim = sim

        # Map similarity -> novelty. Near-duplicate (~1.0) -> ~0.1.
        # Fully novel (~0.0) -> ~1.0. Slight floor so 0.0 similarity
        # still leaves a hair of room above 1.0.
        novelty = max(0.1, min(1.0, 1.0 - max_sim))

        return round(novelty, 3)

    # ----------------------------------------------------------

    def evaluate_persistence(self, perception):

        score = 0.20

        for entity in perception.entities:

            if _normalise_label(entity.label) in _PERSISTENT_ENTITY_LABELS:

                score += 0.20

        return min(score, 1.0)

    # ----------------------------------------------------------

    def evaluate_personal_relevance(self, perception):

        """
        Score how likely this perception is to be about the user.

        Two signals:
          1. Recognised persistent entities (people, places, music artists,
             fields of study) contribute +0.25 each, capped at 1.0.
          2. First-person / possessive language ("my favourite", "i love",
             "i study") contributes +0.20 per matched phrase, capped at 1.0.

        Either signal alone is enough to clear the 0.3 threshold used by
        EmotionRules.PERSONAL_HIGH.
        """

        entity_score = 0.0

        for entity in perception.entities:

            if _normalise_label(entity.label) in _PERSISTENT_ENTITY_LABELS:

                entity_score += 0.25

        entity_score = min(entity_score, 1.0)

        text = perception.raw_text.lower()

        phrase_score = 0.0

        for phrase in _PERSONAL_PHRASES:

            if phrase in text:

                phrase_score += 0.20

        phrase_score = min(phrase_score, 1.0)

        # Take the strongest of the two signals — either is enough to
        # matter. Don't average, because averaging punishes inputs that
        # only match one signal (which is the common case).
        return max(entity_score, phrase_score)

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
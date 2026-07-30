"""
memory.py

MemoryEngine — the durable store + decay semantics for the Cognitive Engine.

Responsibilities:
    - Persist perceptions, observations, beliefs, episodes
    - Apply ACT-R decay to beliefs on read
    - Provide the queries the ContextBuilder needs
    - Run periodic maintenance (apply_decay)

Decay model (ACT-R style):

    strength = base_strength * ln(1 + accesses) / (1 + age_hours)

We do NOT denormalize strength into the table. It is recomputed on read so
the source of truth stays in one place (base_strength + updated_at).

See: Docs/ARCHITECTURE.md
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

from Core.memory_store import MemoryStore

# Heavy perception-stack imports are deferred so this module loads
# without gliner/sentence-transformers installed. Methods that need
# them resolve the types lazily.
if TYPE_CHECKING:
    from Core.observation import Observation
    from Core.perception import Perception


# ==========================================================
# Constants
# ==========================================================

EMBEDDING_DIM = 384

# Beliefs whose current strength falls below this are soft-deleted.
PRUNE_THRESHOLD = 0.05

# Default ranking limits for the ContextBuilder.
DEFAULT_TOP_BELIEFS = 5
DEFAULT_RECENT_OBSERVATIONS = 8

# Minimum importance score for an observation to be surfaced in context.
importance_threshold_for_context = 0.30


# ==========================================================
# Row dataclasses (read-side convenience)
# ==========================================================

@dataclass
class BeliefRow:
    id: int
    created_at: float
    updated_at: float
    last_accessed: float
    accesses: int
    category: str
    statement: str
    base_strength: float
    importance: float
    confidence: float
    is_active: bool

    def current_strength(self) -> float:
        """ACT-R decay."""
        age_hours = max(0.0, (time.time() - self.updated_at) / 3600.0)
        return self.base_strength * math.log(1 + self.accesses) / (1.0 + age_hours)


@dataclass
class ObservationRow:
    id: int
    timestamp: float
    perception_id: int
    novelty: float
    persistence: float
    personal_relevance: float
    goal_relevance: float
    temporal_relevance: float
    importance: float
    reasoning: str


@dataclass
class EpisodeRow:
    id: int
    started_at: float
    ended_at: float
    summary: str
    importance: float


# ==========================================================
# MemoryEngine
# ==========================================================

class MemoryEngine:
    """
    High-level memory API used by the rest of the Cognitive Engine.

    Internally delegates SQL to MemoryStore. Owns the ACT-R decay semantics.
    """

    def __init__(self, store: MemoryStore):

        self.store = store

    # ==========================================================
    # Writes
    # ==========================================================

    def persist_perception(self, perception: "Perception") -> int:

        sql = """
            INSERT INTO perceptions (timestamp, raw_text, language, language_conf, semantics_id)
            VALUES (?, ?, ?, ?, NULL)
        """

        cur = self.store.conn.execute(
            sql,
            (
                perception.timestamp.timestamp(),
                perception.raw_text,
                perception.language.language,
                perception.language.confidence,
            ),
        )

        return cur.lastrowid

    # ------------------------------------------------------

    def persist_observation(
        self,
        observation: "Observation",
        perception_id: int,
    ) -> int:

        sql = """
            INSERT INTO observations (
                timestamp, perception_id,
                novelty, persistence, personal_relevance,
                goal_relevance, temporal_relevance,
                importance, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cur = self.store.conn.execute(
            sql,
            (
                observation.timestamp.timestamp(),
                perception_id,
                observation.novelty,
                observation.persistence,
                observation.personal_relevance,
                observation.goal_relevance,
                observation.temporal_relevance,
                observation.importance,
                observation.reasoning,
            ),
        )

        return cur.lastrowid

    # ------------------------------------------------------

    def persist_belief(
        self,
        *,
        statement: str,
        category: str,
        importance: float,
        base_strength: float = 0.7,
        confidence: float = 0.5,
        observation_id: Optional[int] = None,
        relation: str = "support",
    ) -> int:

        now = time.time()

        cur = self.store.conn.execute(
            """
            INSERT INTO beliefs (
                created_at, updated_at, last_accessed, accesses,
                category, statement, base_strength, importance,
                confidence, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                now, now, now, 0,
                category, statement, base_strength, importance,
                confidence,
            ),
        )

        belief_id = cur.lastrowid

        if observation_id is not None:

            self.store.conn.execute(
                """
                INSERT OR IGNORE INTO belief_evidence
                    (belief_id, observation_id, relation)
                VALUES (?, ?, ?)
                """,
                (belief_id, observation_id, relation),
            )

        return belief_id

    # ------------------------------------------------------

    def link_evidence(
        self,
        belief_id: int,
        observation_id: int,
        relation: str,
    ) -> None:

        self.store.conn.execute(
            """
            INSERT OR IGNORE INTO belief_evidence
                (belief_id, observation_id, relation)
            VALUES (?, ?, ?)
            """,
            (belief_id, observation_id, relation),
        )

    # ------------------------------------------------------

    def persist_episode(
        self,
        *,
        summary: str,
        importance: float,
        started_at: float,
        ended_at: Optional[float] = None,
    ) -> int:

        cur = self.store.conn.execute(
            """
            INSERT INTO episodes (started_at, ended_at, summary, importance)
            VALUES (?, ?, ?, ?)
            """,
            (started_at, ended_at or time.time(), summary, importance),
        )

        return cur.lastrowid

    # ------------------------------------------------------
    # Vector writes (best-effort)

    def store_embedding(self, table: str, row_id: int, embedding: np.ndarray) -> bool:

        if embedding.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim {embedding.shape[0]} != {EMBEDDING_DIM}"
            )

        try:
            self.store.conn.execute(
                f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
                (row_id, embedding.astype(np.float32).tobytes()),
            )
            return True
        except Exception:
            # sqlite-vec may be unavailable; fall back to FTS5 + numpy.
            return False

    # ==========================================================
    # Reads
    # ==========================================================

    def get_top_beliefs(
        self,
        k: int = DEFAULT_TOP_BELIEFS,
        min_strength: float = 0.0,
    ) -> list[BeliefRow]:

        rows = self.store.conn.execute(
            """
            SELECT * FROM beliefs
            WHERE is_active = 1
            ORDER BY base_strength DESC, updated_at DESC
            LIMIT ?
            """,
            (k * 4,),  # fetch extra, filter after decay
        ).fetchall()

        scored = []

        for row in rows:

            belief = self._row_to_belief(row)
            s = belief.current_strength()

            if s >= min_strength:
                scored.append((s, belief))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [b for _, b in scored[:k]]

    # ------------------------------------------------------

    def get_recent_observations(
        self,
        k: int = DEFAULT_RECENT_OBSERVATIONS,
        since_seconds: float = 3600.0,
        min_importance: float = importance_threshold_for_context,
    ) -> list[ObservationRow]:

        cutoff = time.time() - since_seconds

        rows = self.store.conn.execute(
            """
            SELECT * FROM observations
            WHERE timestamp >= ? AND importance >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff, min_importance, k),
        ).fetchall()

        return [self._row_to_observation(r) for r in rows]

    # ------------------------------------------------------

    def get_belief_by_id(self, belief_id: int) -> Optional[BeliefRow]:

        row = self.store.conn.execute(
            "SELECT * FROM beliefs WHERE id = ?",
            (belief_id,),
        ).fetchone()

        if row is None:
            return None

        belief = self._row_to_belief(row)
        self._touch_belief(belief_id)
        return belief

    # ------------------------------------------------------

    def search_beliefs_by_text(self, query: str, k: int = 5) -> list[BeliefRow]:

        rows = self.store.conn.execute(
            """
            SELECT b.* FROM beliefs b
            JOIN beliefs_fts f ON b.id = f.rowid
            WHERE beliefs_fts MATCH ?
            AND b.is_active = 1
            LIMIT ?
            """,
            (query, k),
        ).fetchall()

        return [self._row_to_belief(r) for r in rows]

    # ------------------------------------------------------

    def get_episode_tail(self, k: int) -> list[EpisodeRow]:

        rows = self.store.conn.execute(
            """
            SELECT * FROM episodes
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (k,),
        ).fetchall()

        return [self._row_to_episode(r) for r in rows]

    # ------------------------------------------------------
    # Belief lookup helpers (used by BeliefEngine)

    def find_belief_by_statement(self, statement: str) -> Optional[BeliefRow]:

        row = self.store.conn.execute(
            "SELECT * FROM beliefs WHERE statement = ? AND is_active = 1 LIMIT 1",
            (statement,),
        ).fetchone()

        return self._row_to_belief(row) if row else None

    # ==========================================================
    # Maintenance
    # ==========================================================

    def apply_decay(self, now: Optional[float] = None) -> int:

        """
        Soft-delete beliefs whose current strength falls below PRUNE_THRESHOLD.

        Returns the number of beliefs pruned.
        """

        now = now or time.time()

        rows = self.store.conn.execute(
            "SELECT id, base_strength, updated_at, accesses FROM beliefs WHERE is_active = 1"
        ).fetchall()

        pruned = 0

        for row in rows:

            age_hours = max(0.0, (now - row["updated_at"]) / 3600.0)
            strength = row["base_strength"] * math.log(1 + row["accesses"]) / (1.0 + age_hours)

            if strength < PRUNE_THRESHOLD:

                self.store.conn.execute(
                    "UPDATE beliefs SET is_active = 0 WHERE id = ?",
                    (row["id"],),
                )
                pruned += 1

        return pruned

    # ------------------------------------------------------

    def reflect(
        self,
        window_seconds: float = 3600.0,
        min_observations: int = 3,
        jaccard_threshold: float = 0.20,
    ) -> int:
        """
        Cluster recent observations by content-word overlap (Jaccard) and
        synthesise a "Pattern" belief for any cluster with at least
        `min_observations` members. Returns the number of new beliefs written.

        Cheap heuristic — no embeddings required. Embedding-based clustering
        can replace this once the perception stack is loaded in production.

        The synthesised belief links to every observation in its cluster via
        the `belief_evidence` table so its strength can grow as those
        observations get re-surfaced.
        """

        cutoff = time.time() - window_seconds

        rows = self.store.conn.execute(
            """
            SELECT id, reasoning, importance FROM observations
            WHERE timestamp >= ? AND importance >= ?
            ORDER BY id ASC
            """,
            (cutoff, importance_threshold_for_context),
        ).fetchall()

        if len(rows) < min_observations:
            return 0

        # Tokenise each observation's reasoning into content words.
        tokenised = [(row["id"], row["reasoning"], row["importance"],
                      _content_words(row["reasoning"])) for row in rows]

        # Simple union-find over pairwise Jaccard >= threshold.
        parent = list(range(len(tokenised)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(len(tokenised)):
            for j in range(i + 1, len(tokenised)):
                wi = tokenised[i][3]
                wj = tokenised[j][3]
                if not wi or not wj:
                    continue
                inter = len(wi & wj)
                union_size = len(wi) + len(wj) - inter
                if union_size == 0:
                    continue
                if inter / union_size >= jaccard_threshold:
                    union(i, j)

        # Group by root.
        groups: dict = {}
        for idx, (obs_id, reasoning, importance, _) in enumerate(tokenised):
            root = find(idx)
            groups.setdefault(root, []).append((obs_id, reasoning, importance))

        written = 0

        for cluster in groups.values():
            if len(cluster) < min_observations:
                continue

            # Skip if a Pattern belief already covers these observations.
            obs_ids = [c[0] for c in cluster]
            placeholders = ",".join("?" * len(obs_ids))
            existing = self.store.conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM belief_evidence be
                JOIN beliefs b ON be.belief_id = b.id
                WHERE be.observation_id IN ({placeholders})
                  AND b.category = 'Pattern'
                  AND b.is_active = 1
                """,
                obs_ids,
            ).fetchone()

            if existing["c"] >= len(obs_ids):
                # Every observation in this cluster is already covered.
                continue

            # Synthesise a candidate belief from the cluster.
            mean_importance = sum(c[2] for c in cluster) / len(cluster)
            top_words = _top_keywords(cluster, n=4)

            statement = f"Recurring theme: {', '.join(top_words)}"

            belief_id = self.persist_belief(
                statement=statement,
                category="Pattern",
                importance=min(1.0, mean_importance),
                base_strength=min(0.6, mean_importance),
                confidence=0.4,  # reflection is a guess; starts low
            )

            for obs_id, _, _ in cluster:
                self.link_evidence(belief_id, obs_id, relation="support")

            written += 1

        return written

    # ------------------------------------------------------

    def stats(self) -> dict:

        counts = {}

        for table in ("perceptions", "observations", "beliefs", "episodes"):

            row = self.store.conn.execute(
                f"SELECT COUNT(*) AS c FROM {table}"
            ).fetchone()
            counts[table] = row["c"]

        active = self.store.conn.execute(
            "SELECT COUNT(*) AS c FROM beliefs WHERE is_active = 1"
        ).fetchone()["c"]

        counts["active_beliefs"] = active

        return counts

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _touch_belief(self, belief_id: int) -> None:

        self.store.conn.execute(
            """
            UPDATE beliefs
            SET accesses = accesses + 1,
                last_accessed = ?
            WHERE id = ?
            """,
            (time.time(), belief_id),
        )

    # ------------------------------------------------------

    def _row_to_belief(self, row) -> BeliefRow:

        return BeliefRow(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            accesses=row["accesses"],
            category=row["category"],
            statement=row["statement"],
            base_strength=row["base_strength"],
            importance=row["importance"],
            confidence=row["confidence"],
            is_active=bool(row["is_active"]),
        )

    # ------------------------------------------------------

    def _row_to_observation(self, row) -> ObservationRow:

        return ObservationRow(
            id=row["id"],
            timestamp=row["timestamp"],
            perception_id=row["perception_id"],
            novelty=row["novelty"],
            persistence=row["persistence"],
            personal_relevance=row["personal_relevance"],
            goal_relevance=row["goal_relevance"],
            temporal_relevance=row["temporal_relevance"],
            importance=row["importance"],
            reasoning=row["reasoning"],
        )

    # ------------------------------------------------------

    def _row_to_episode(self, row) -> EpisodeRow:

        return EpisodeRow(
            id=row["id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            summary=row["summary"],
            importance=row["importance"],
        )


# ==========================================================
# Reflection helpers (module-level so the engine stays importable
# without numpy/sentence-transformers)
# ==========================================================

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "her", "his", "how", "its", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "has", "him", "let", "say", "she",
    "too", "use", "with", "from", "this", "that", "have", "will",
    "one", "two", "any", "was", "were", "been", "they", "them",
})


def _content_words(text: str) -> set:
    """Lowercase tokens >3 chars, minus stopwords. Cheap content fingerprint."""
    if not text:
        return set()
    out = set()
    for tok in text.lower().split():
        cleaned = "".join(c for c in tok if c.isalnum())
        if len(cleaned) > 3 and cleaned not in _STOPWORDS:
            out.add(cleaned)
    return out


def _top_keywords(cluster, n: int = 4) -> list:
    """Return the n most frequent content words across a cluster."""
    freq: dict = {}
    for _, reasoning, _ in cluster:
        for w in _content_words(reasoning):
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:n]]

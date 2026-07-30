"""
memory_store.py

SQLite connection wrapper for the Memory Engine.

Single source of truth for how the engine touches the database.
All other modules go through MemoryEngine (memory.py), which uses this
under the hood.

Schema and decay semantics live in Docs/ARCHITECTURE.md.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from Utilities.config import DATA_DIR


# ==========================================================
# Schema (with vector + FTS5 search)
# ==========================================================

SCHEMA_FULL = """
-- Perceptions: raw feature-extraction output
CREATE TABLE IF NOT EXISTS perceptions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL NOT NULL,
    raw_text      TEXT,
    language      TEXT,
    language_conf REAL,
    semantics_id  INTEGER
);

-- Observations: scored perceptions
CREATE TABLE IF NOT EXISTS observations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          REAL NOT NULL,
    perception_id      INTEGER REFERENCES perceptions(id),
    novelty            REAL,
    persistence        REAL,
    personal_relevance REAL,
    goal_relevance     REAL,
    temporal_relevance REAL,
    importance         REAL,
    reasoning          TEXT
);

-- Beliefs: long-term compiled statements
CREATE TABLE IF NOT EXISTS beliefs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    last_accessed  REAL,
    accesses       INTEGER DEFAULT 0,
    category       TEXT,
    statement      TEXT,
    base_strength  REAL,
    importance     REAL,
    confidence     REAL,
    is_active      INTEGER DEFAULT 1
);

-- Belief <-> observation evidence graph
CREATE TABLE IF NOT EXISTS belief_evidence (
    belief_id      INTEGER REFERENCES beliefs(id),
    observation_id INTEGER REFERENCES observations(id),
    relation       TEXT CHECK(relation IN ('support', 'contradict')),
    PRIMARY KEY (belief_id, observation_id)
);

-- Episodes: discrete moments worth remembering
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   REAL,
    ended_at     REAL,
    summary      TEXT,
    importance   REAL,
    embedding_id INTEGER
);

-- Vector tables (sqlite-vec). Dimensions match all-MiniLM-L6-v2 (384).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_perceptions
    USING vec0(embedding float[384]);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_beliefs
    USING vec0(embedding float[384]);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes
    USING vec0(embedding float[384]);

-- FTS5 for substring search fallback
CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts
    USING fts5(statement, content='beliefs', content_rowid='id');

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
    USING fts5(summary, content='episodes', content_rowid='id');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_observations_importance
    ON observations(importance DESC);

CREATE INDEX IF NOT EXISTS idx_observations_timestamp
    ON observations(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_beliefs_active
    ON beliefs(is_active, base_strength DESC);
"""


# ==========================================================
# Schema (fallback — vector + FTS5 unavailable)
# ==========================================================

SCHEMA_MINIMAL = """
CREATE TABLE IF NOT EXISTS perceptions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     REAL NOT NULL,
    raw_text      TEXT,
    language      TEXT,
    language_conf REAL,
    semantics_id  INTEGER
);

CREATE TABLE IF NOT EXISTS observations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          REAL NOT NULL,
    perception_id      INTEGER REFERENCES perceptions(id),
    novelty            REAL,
    persistence        REAL,
    personal_relevance REAL,
    goal_relevance     REAL,
    temporal_relevance REAL,
    importance         REAL,
    reasoning          TEXT
);

CREATE TABLE IF NOT EXISTS beliefs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    last_accessed  REAL,
    accesses       INTEGER DEFAULT 0,
    category       TEXT,
    statement      TEXT,
    base_strength  REAL,
    importance     REAL,
    confidence     REAL,
    is_active      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS belief_evidence (
    belief_id      INTEGER REFERENCES beliefs(id),
    observation_id INTEGER REFERENCES observations(id),
    relation       TEXT CHECK(relation IN ('support', 'contradict')),
    PRIMARY KEY (belief_id, observation_id)
);

CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   REAL,
    ended_at     REAL,
    summary      TEXT,
    importance   REAL,
    embedding_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_observations_importance
    ON observations(importance DESC);

CREATE INDEX IF NOT EXISTS idx_observations_timestamp
    ON observations(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_beliefs_active
    ON beliefs(is_active, base_strength DESC);
"""


# ==========================================================
# Store
# ==========================================================

class MemoryStore:
    """
    Owns the SQLite connection for the Cognitive Engine.

    Why a single class for the connection:
    - WAL mode requires a single writer-friendly connection per process.
    - All modules go through MemoryEngine, which uses this class.
    - Tests can instantiate with a temp path.
    """

    DEFAULT_DB_PATH = DATA_DIR / "memory.db"

    def __init__(self, db_path: Optional[Path] = None):

        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # tick loop runs threads
            isolation_level=None,     # autocommit; we manage txns explicitly
        )

        self.conn.row_factory = sqlite3.Row

        self._configure()
        self._init_schema()

    # ------------------------------------------------------

    def _configure(self):

        # WAL gives concurrent reads + a single writer; safe for the tick loop.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA temp_store=MEMORY")

    # ------------------------------------------------------

    def _init_schema(self):

        # Try the full schema first (sqlite-vec + FTS5).
        try:
            self.conn.executescript(SCHEMA_FULL)
            return
        except Exception as exc:
            # Save the diagnostic; fall back to the minimal schema.
            self._init_error = str(exc)

        # Fallback: minimal schema. Vector search degrades to brute-force
        # numpy at the engine layer; FTS5 is unavailable.
        try:
            self.conn.executescript(SCHEMA_MINIMAL)
        except Exception as exc:
            raise RuntimeError(
                f"MemoryStore failed to initialise even the minimal schema: {exc}"
            ) from exc

    # ------------------------------------------------------

    def close(self):

        self.conn.close()

    # ------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()

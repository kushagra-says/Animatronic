# Animatronic AI — Architecture

## Purpose

A personality-coded AI ("chalant JARVIS") that:

- Perceives input (text now, camera/mic later)
- Forms and prunes long-term beliefs from observations
- Reacts with a persistent emotional state
- Maintains a working context window that knows what matters
- Continuously ticks so it can act, stay silent, or reflect

It is not sentient. It is not a generic assistant. It has *opinions* backed by evidence that decay over time.

---

## Design Pillars

| Pillar | Choice |
|---|---|
| Decay model | ACT-R style: `strength = base * ln(1 + accesses) / (1 + age_hours)` |
| Storage | SQLite (single file) + `sqlite-vec` for embedding search |
| Tick cadence | Hybrid: event-driven for input, periodic loop for decay/reflection |
| Context window | Last K raw messages + top beliefs + recent relevant observations + emotion state |
| LLM | Local via Ollama (model chosen at runtime per persona) |
| Runtime | Offline-only. No internet calls anywhere. |

---

## End-to-End Data Flow

```
INPUT (text / camera / mic)
    │
    ▼
[PerceptionEngine]   ── language, entities, embedding
    │
    ▼
[ObservationEngine]  ── novelty, persistence, personal, goal, temporal → importance
    │
    ▼
[BeliefEngine]       ── match/reinforce/create belief, possibly contradict
    │
    ▼
[EmotionEngine]      ── perturb PAD state from observation
    │
    ▼
[MemoryEngine]       ── persist perception, observation, belief tick
    │
    ▼
[ContextBuilder]     ── assemble prompt: tail + top beliefs + recent + emotion
    │
    ▼
[LLM]                ── generate response (if any)
    │
    ▼
OUTPUT (text / TTS / animatronic)
    │
    ▼
[Decay loop]         ── runs on schedule, not on every event
```

The tick loop owns this sequence. Every step is independently testable.

---

## Module Layout

```
Software/Cognitive Engine/
├── Core/
│   ├── perception.py          ✅ existing
│   ├── perception.py          ✅ existing
│   ├── observation.py          ✅ existing
│   ├── observation_engine.py  ✅ existing
│   ├── belief.py              ✅ existing
│   ├── belief_engine.py       ✅ existing
│   ├── evidence.py            ✅ existing
│   ├── memory.py              🔜 SQLite store + ACT-R decay
│   ├── emotion.py             🔜 PAD state + traits
│   ├── context.py             🔜 prompt assembly
│   ├── consciousness.py       🔜 tick loop / orchestrator
│   └── llm.py                 🔜 engine-side LLM wrapper
├── Models/
│   ├── execution_profile.py
│   └── profiles.py
├── Utilities/
│   ├── config.py
│   ├── logger.py
│   └── performance_monitor.py
└── Docs/
    └── ARCHITECTURE.md        ← this file
```

---

## Memory Engine

### Schema

```sql
-- Perceptions: raw feature-extraction output
CREATE TABLE perceptions (
    id              INTEGER PRIMARY KEY,
    timestamp       REAL NOT NULL,
    raw_text        TEXT,
    language        TEXT,
    language_conf   REAL,
    semantics_id    INTEGER  -- FK into vec_perceptions
);

-- Observations: scored perceptions
CREATE TABLE observations (
    id                  INTEGER PRIMARY KEY,
    timestamp           REAL NOT NULL,
    perception_id       INTEGER REFERENCES perceptions(id),
    novelty             REAL,
    persistence         REAL,
    personal_relevance  REAL,
    goal_relevance      REAL,
    temporal_relevance  REAL,
    importance          REAL,
    reasoning           TEXT
);

-- Beliefs: long-term compiled statements
CREATE TABLE beliefs (
    id              INTEGER PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_accessed   REAL,
    accesses        INTEGER DEFAULT 0,
    category        TEXT,
    statement       TEXT,
    base_strength   REAL,
    importance      REAL,
    strength        REAL,  -- recomputed by decay
    confidence      REAL,
    is_active       INTEGER DEFAULT 1  -- soft-deleted
);

-- Belief ↔ observation graph
CREATE TABLE belief_evidence (
    belief_id       INTEGER REFERENCES beliefs(id),
    observation_id  INTEGER REFERENCES observations(id),
    relation        TEXT CHECK(relation IN ('support', 'contradict')),
    PRIMARY KEY (belief_id, observation_id)
);

-- Episodes: discrete moments worth remembering
CREATE TABLE episodes (
    id            INTEGER PRIMARY KEY,
    started_at    REAL,
    ended_at      REAL,
    summary       TEXT,
    importance    REAL,
    embedding_id  INTEGER
);

-- Vector tables (sqlite-vec)
CREATE VIRTUAL TABLE vec_perceptions USING vec0(embedding float[384]);
CREATE VIRTUAL TABLE vec_beliefs     USING vec0(embedding float[384]);
CREATE VIRTUAL TABLE vec_episodes    USING vec0(embedding float[384]);
```

### ACT-R Decay

```python
def current_strength(belief, now):
    age_hours = (now - belief.updated_at) / 3600
    accesses = belief.accesses
    return belief.base_strength * math.log(1 + accesses) / (1 + age_hours)
```

- `base_strength` is set when a belief is created.
- `accesses` increments every time the belief is retrieved into context.
- `updated_at` is touched on reinforcement or contradiction.
- Strength is recomputed on read by `MemoryEngine`. We do not denormalize it into the table — keeps the source of truth singular.

### Queries

| Query | Purpose |
|---|---|
| `get_top_beliefs(k, min_strength)` | Active beliefs by current strength |
| `get_recent_observations(since, min_importance)` | Within time window, above threshold |
| `get_episode_tail(k)` | Last K raw messages, always included |
| `search_by_embedding(vec, k)` | Cosine similarity fallback for long-tail recall |
| `apply_decay(now)` | Recompute all `base_strength * ...` and prune below threshold |

---

## Emotion Engine

### State

```python
@dataclass
class EmotionState:
    pleasure: float = 0.0    # -1.0 .. 1.0
    arousal: float = 0.0     # -1.0 .. 1.0
    dominance: float = 0.0   # -1.0 .. 1.0
    curiosity: float = 0.5   # 0.0 .. 1.0
    trust: float = 0.5       # 0.0 .. 1.0
    boredom: float = 0.0     # 0.0 .. 1.0
    updated_at: float
```

- PAD is the canonical affect space.
- `curiosity`, `trust`, `boredom` are derived scalars maintained for prompt injection.
- Decay toward neutral baseline at fixed rate per tick.
- Perturbed by observations through `ObservationEngine` impact mapping.

### Perturbation rules

| Observation signal | Effect |
|---|---|
| High novelty + high personal | `curiosity += 0.1`, `arousal += 0.05` |
| Contradiction detected | `dominance -= 0.1`, `arousal += 0.1` |
| Compliment entity | `pleasure += 0.1`, `trust += 0.05` |
| Threat language | `pleasure -= 0.2`, `fear` spike |
| Long silence (>5 min) | `boredom += 0.1`, `arousal -= 0.05` |

These are values, not hard rules — final state is clamped to `[-1, 1]` / `[0, 1]`.

---

## Context Builder

Assembles the prompt in a fixed order with a token budget:

```
[System prompt]               ~120 tokens
[Emotion state summary]       ~40 tokens
[Top 5 strongest beliefs]     ~200 tokens
[Recent relevant observations]~300 tokens
[Last K raw messages]         ~variable (tail-priority)
[New user input]              ~variable
```

If total exceeds budget, prune from the bottom of the "recent relevant observations" first, then compress beliefs to one-liners, then truncate the raw tail (only as a last resort — the raw tail is sacred).

Token estimates are conservative (chars / 4). Exact counting is future work.

---

## Tick Loop

### Two threads

1. **Event thread** — runs when input arrives.
   - Perceive → Observe → Update Belief → Perturb Emotion → Persist → Build Context → LLM → Output

2. **Maintenance thread** — runs every 60 s.
   - Snapshot each belief's current strength
   - Apply decay
   - Soft-delete beliefs below threshold (e.g. `0.05`)
   - Decay emotion toward baseline (10% pull per tick)
   - Optional: reflection — cluster recent observations into a new belief

Both threads synchronize via a single `MemoryEngine` instance with SQLite WAL mode.

---

## LLM Wrapper

Separate from `Chatbot/llm.py`. The engine-side wrapper:

- Accepts a context bundle and a `prompt_kind` (speak / reflect / decide)
- Injects persona/mood preamble
- Tracks call latency and reports via `PerformanceMonitor`
- Caches repeated prompts (single-hit cache for now)

---

## CLI Prototype

```
You: hello
AI: Hello. You look tired.

[ state ] pleasure=+0.1  arousal=+0.05  curiosity=0.7  trust=0.5
[ belief #1 ] User studies at MIT   (strength=0.74, confidence=0.81)
[ belief #2 ] User dislikes mornings (strength=0.55, confidence=0.62)
[ tick 47 ] 12:04:33
```

The CLI shows what the AI is thinking, not just what it says. That's the whole point — debugging a memory system requires seeing the memory.

---

## Future Work (not in prototype)

- Replace `ObservationEngine` heuristics with a learned model
- Tie `EmotionEngine` perturbations to the LLM's own self-report
- Camera perception (emotion-from-face)
- Microphone perception (tone, prosody)
- 3D animatronic head simulator (separate module, talks to engine via socket)
- TTS / lip-sync output

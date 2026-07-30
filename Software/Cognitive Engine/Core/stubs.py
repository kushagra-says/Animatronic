"""
stubs.py

Drop-in stand-ins for the heavy engine modules so the CLI can boot
without ollama, gliner, lingua, or sentence-transformers installed.

Used by:
    Software/Chatbot/main.py --stub
    Tests/test_integration.py

Same surface as the real engines — Consciousness cannot tell the difference.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from Core.llm import LLMResult


# ==========================================================
# Lightweight datetime shim
# ==========================================================

class _Now:
    """Mimics datetime.now() enough for `.timestamp()` calls."""
    def timestamp(self) -> float:
        return time.time()


# ==========================================================
# Perception stand-in
# ==========================================================

@dataclass
class StubLanguageEvidence:
    language: str = "en"
    confidence: float = 1.0


@dataclass
class StubPerception:
    timestamp: _Now
    raw_text: str
    language: StubLanguageEvidence
    entities: list = field(default_factory=list)
    semantics: object = None


class StubPerceptionEngine:
    """Returns a deterministic Perception for any input text."""
    def perceive(self, text: str) -> StubPerception:
        return StubPerception(
            timestamp=_Now(),
            raw_text=text,
            language=StubLanguageEvidence(),
        )


# ==========================================================
# Observation stand-in
# ==========================================================

@dataclass
class StubObservation:
    timestamp: _Now
    novelty: float = 0.5
    persistence: float = 0.5
    personal_relevance: float = 0.5
    goal_relevance: float = 0.5
    temporal_relevance: float = 0.5
    importance: float = 0.5
    reasoning: str = "stub observation"


class StubObservationEngine:
    """Returns a deterministic Observation."""
    def observe(self, perception) -> StubObservation:
        return StubObservation(timestamp=_Now())


# ==========================================================
# Belief stand-in
# ==========================================================

@dataclass
class StubBelief:
    statement: str = "user is talking to me"
    category: str = "Interaction"
    importance: float = 0.6
    confidence: float = 0.7
    contradiction_count: int = 0


class StubBeliefEngine:
    """Returns one belief on the first call, nothing afterwards."""
    call_count: int = 0

    def process_observation(self, observation) -> Optional[StubBelief]:
        self.call_count += 1
        if self.call_count == 1:
            return StubBelief()
        return None


# ==========================================================
# LLM stand-in
# ==========================================================

class StubLLM:
    """Returns a canned response without hitting Ollama."""
    text: str = (
        "[stub] I would normally reply here. Real responses require "
        "ollama + the perception stack to be installed. The plumbing "
        "(perception -> observation -> belief -> emotion -> context -> LLM "
        "-> output) is working though — see the [ state ] and [ tick ] lines."
    )

    def generate(self, system: str, user: str) -> LLMResult:
        return LLMResult(
            text=self.text,
            prompt_tokens=len(system) // 4,
            response_tokens=len(self.text) // 4,
        )
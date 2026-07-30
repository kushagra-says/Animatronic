"""
chatbot.py

Bridges the user-facing CLI to the Cognitive Engine.

The CLI is the prototype animatronic "body" — it captures user input,
hands it to the Consciousness tick, and renders the response.

The maintenance loop runs in a background thread on a fixed cadence.
"""

import sys
import threading
import time
from pathlib import Path

# Chatbot/ lives next to Cognitive Engine/. Core/ belongs to the Cognitive
# Engine package, so make sure that directory is on sys.path before we try
# to import from it — regardless of whether main.py, an integrator, or a
# direct invocation brought us here.
_ENGINE_DIR = Path(__file__).resolve().parent.parent / "Cognitive Engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from Core.consciousness import Consciousness  # noqa: E402
from Core.memory_store import MemoryStore  # noqa: E402


# ==========================================================
# Config
# ==========================================================

MAINTENANCE_INTERVAL_SEC = 60


# ==========================================================
# Chatbot
# ==========================================================

class Chatbot:
    """
    Entry point used by main.py.

    Holds the Consciousness instance and a maintenance thread.
    """

    def __init__(self, stub: bool = False, maintenance_interval_sec: int = MAINTENANCE_INTERVAL_SEC):

        self.store = MemoryStore()

        if stub:
            # Use fake engines so the CLI can boot without ollama / gliner /
            # lingua / sentence-transformers installed. Real wiring,
            # canned replies.
            from Core.stubs import (
                StubPerceptionEngine,
                StubObservationEngine,
                StubBeliefEngine,
                StubLLM,
            )
            from Core.emotion import EmotionEngine
            from Core.memory import MemoryEngine
            from Core.context import ContextBuilder

            memory = MemoryEngine(self.store)
            self.consciousness = Consciousness(
                store=self.store,
                perception=StubPerceptionEngine(),
                observation_engine=StubObservationEngine(),
                belief_engine=StubBeliefEngine(),
                memory=memory,
                emotion=EmotionEngine(),
                context_builder=ContextBuilder(memory, EmotionEngine()),
                llm=StubLLM(),
            )
        else:
            self.consciousness = Consciousness(store=self.store)

        self._stop_event = threading.Event()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="cog-engine-maintenance",
        )

        self._maintenance_thread.start()
        self._maintenance_interval = maintenance_interval_sec

    # ------------------------------------------------------

    def chat(self, user_input: str) -> str:

        result = self.consciousness.on_input(user_input)

        # Show the AI's state alongside the response so the user
        # can see what its memory looks like.
        print(f"\n[ state ] {result.emotion_summary}")
        print(f"[ tick  ] {result.elapsed_sec:.2f}s")

        return result.response_text

    # ------------------------------------------------------

    def shutdown(self) -> None:

        self._stop_event.set()
        self._maintenance_thread.join(timeout=2)
        self.store.close()

    # ------------------------------------------------------

    def _maintenance_loop(self) -> None:

        while not self._stop_event.is_set():

            self.consciousness.maintenance_tick()

            # Daemon thread sleep — wake up periodically to check the stop flag.
            self._stop_event.wait(self._maintenance_interval)

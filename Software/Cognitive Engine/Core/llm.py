"""
llm.py

Engine-side LLM wrapper.

Separate from Software/Chatbot/llm.py so we can evolve the engine's
prompting independently. The engine has very different needs from the
chatbot — it injects emotion, beliefs, observations, and sometimes
asks for *action* rather than a reply.

See: Docs/ARCHITECTURE.md
"""

import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

# Heavy runtime import (ollama) is deferred so this module loads even when
# ollama isn't installed in the active venv — useful for unit tests that
# substitute a fake LLM.
if TYPE_CHECKING:
    from ollama import Client


# ==========================================================
# Config
# ==========================================================

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512


# ==========================================================
# Prompt kinds
# ==========================================================

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    response_tokens: int


class EngineLLM:
    """
    Thin wrapper around the Ollama client for the Cognitive Engine.

    The engine does not stream — it needs a complete response to decide
    what to do next. Streaming belongs to the CLI/UI layer.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):

        # Import ollama only when we actually construct the client. Keeps
        # the module importable for tests that pass in a fake LLM.
        from ollama import Client
        self.client = Client(host=host)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Verify the model is present at startup. If it's missing we try
        # to pull it once; if that fails we raise a clear error so the
        # CLI doesn't later crash with a generic "model not found".
        self._verify_or_pull()

    # ------------------------------------------------------

    def _verify_or_pull(self) -> None:
        """
        Ensure the configured Ollama model is available locally.

        Two cases:
          1. Model is already pulled — fast path, no network.
          2. Model is missing — pull once, then continue.
          3. Ollama server is unreachable — raise with a helpful message
             (ConnectionError surfaces as ollama.RequestError).
        """

        try:
            available = {m.model for m in self.client.list().models}
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.client.host}. "
                f"Is the server running? (ollama serve)\n"
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

        if self.model in available:
            return  # fast path

        print(f"[llm] Model '{self.model}' not found locally — pulling now...")

        try:
            # Stream so the user sees progress rather than a silent hang.
            last_status = ""
            for chunk in self.client.pull(self.model, stream=True):
                status = getattr(chunk, "status", "")
                if status and status != last_status:
                    print(f"[llm]   {status}")
                    last_status = status
        except Exception as e:
            raise RuntimeError(
                f"Failed to pull Ollama model '{self.model}'. "
                f"Try running 'ollama pull {self.model}' manually.\n"
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------

    def health_check(self) -> bool:

        try:
            available = {m.model for m in self.client.list().models}
            return self.model in available
        except Exception:
            return False

    # ------------------------------------------------------

    def generate(self, system: str, user: str) -> LLMResult:

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        )

        text = response.message.content or ""

        # Ollama exposes counts when available; fall back to char estimates.
        prompt_tokens = getattr(response, "prompt_eval_count", None) or len(system) // 4
        response_tokens = getattr(response, "eval_count", None) or len(text) // 4

        return LLMResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )

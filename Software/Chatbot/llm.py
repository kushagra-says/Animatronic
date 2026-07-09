"""
llm.py

Handles all communication with the local Ollama server.
"""

from ollama import Client

from config import Config


class LLM:
    """Wrapper around the local Ollama server."""

    def __init__(self):
        self.client = Client(host=Config.OLLAMA_HOST)
        self.model = Config.DEFAULT_MODEL

    def _generation_options(self) -> dict:
        """Returns generation settings."""

        return {
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "num_predict": Config.MAX_TOKENS,
        }

    def health_check(self):
        """Checks whether the configured model exists."""

        models = self.client.list()

        available_models = [
            model.model
            for model in models.models
        ]

        if self.model not in available_models:
            raise ValueError(
                f"Model '{self.model}' is not installed."
            )

    def list_models(self):
        """Returns installed Ollama models."""

        models = self.client.list()

        return [
            model.model
            for model in models.models
        ]

    def generate(self, messages):
        """
        Generates a response from the model.

        Parameters
        ----------
        messages : list[Message]
        """

        ollama_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options=self._generation_options(),
        )

        return response.message.content

    def stream(self, messages):
        """
        Streams a response from the model.
        """

        ollama_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        stream = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=True,
            options=self._generation_options(),
        )

        for chunk in stream:
            yield chunk.message.content
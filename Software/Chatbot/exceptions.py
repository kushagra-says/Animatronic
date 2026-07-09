"""
exceptions.py

Custom exceptions used throughout the project.
"""


class ChatbotError(Exception):
    """
    Base exception for the entire project.
    """

    pass


class LLMConnectionError(ChatbotError):
    """
    Raised when Ollama cannot be reached.
    """

    pass


class ModelNotFoundError(ChatbotError):
    """
    Raised when the configured model is not installed.
    """

    pass


class GenerationError(ChatbotError):
    """
    Raised when the model fails to generate a response.
    """

    pass
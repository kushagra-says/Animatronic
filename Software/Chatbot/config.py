"""
config.py

Loads and stores all project configuration.
No other module should read environment variables directly.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()


class Config:
    """Central configuration class."""

    # --------------------------------------------------
    # Project Paths
    # --------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "Data"
    LOG_DIR = BASE_DIR / "Logs"
    MODEL_DIR = BASE_DIR / "Models"

    # --------------------------------------------------
    # Ollama
    # --------------------------------------------------

    OLLAMA_HOST = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434"
    )

    DEFAULT_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:3b"
    )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    TEMPERATURE = float(
        os.getenv("TEMPERATURE", "0.7")
    )

    TOP_P = float(
        os.getenv("TOP_P", "0.9")
    )

    MAX_TOKENS = int(
        os.getenv("MAX_TOKENS", "1024")
    )

    STREAM = (
        os.getenv("STREAM", "True").lower() == "true"
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )
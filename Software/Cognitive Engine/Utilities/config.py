"""
config.py

Central configuration for the Cognitive Engine.
"""

from pathlib import Path


# ==========================================================
# Root Directory
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent


# ==========================================================
# Project Directories
# ==========================================================

MODELS_DIR = ROOT_DIR / "Models"

DATA_DIR = ROOT_DIR / "Data"

TESTS_DIR = ROOT_DIR / "Tests"

LOGS_DIR = ROOT_DIR / "Logs"

DOCS_DIR = ROOT_DIR / "Docs"


# ==========================================================
# Runtime Logs
# ==========================================================

RUNTIME_LOGS_DIR = LOGS_DIR / "Runtime"

PERFORMANCE_LOGS_DIR = LOGS_DIR / "Performance"


# ==========================================================
# Ensure directories exist
# ==========================================================

directories = [

    MODELS_DIR,

    DATA_DIR,

    TESTS_DIR,

    LOGS_DIR,

    DOCS_DIR,

    RUNTIME_LOGS_DIR,

    PERFORMANCE_LOGS_DIR
]

for directory in directories:

    directory.mkdir(parents=True, exist_ok=True)
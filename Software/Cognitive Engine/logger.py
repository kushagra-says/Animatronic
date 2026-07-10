"""
logger.py

Shared logger used throughout the Cognitive Engine.
"""

import logging

from config import RUNTIME_LOGS_DIR


def get_logger(name: str):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(
        RUNTIME_LOGS_DIR / f"{name}.log",
        encoding="utf-8"
    )

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"

    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
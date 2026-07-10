"""
perception.py

The Perception Engine combines the outputs of all feature extractors
into a unified perception object.

This module DOES NOT perform reasoning or memory management.
Its only responsibility is to organize extracted features.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gliner import GLiNER
from lingua import LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer


# ==========================================================
# Perception Data
# ==========================================================

@dataclass
class Perception:

    raw_text: str

    timestamp: datetime

    language: str

    language_confidence: float

    entities: list

    embedding: Any


# ==========================================================
# Perception Engine
# ==========================================================

class PerceptionEngine:

    def __init__(self):

        print("Loading Perception Engine...")

        self.language_detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .build()
        )

        self.entity_extractor = GLiNER.from_pretrained(
            "urchade/gliner_medium-v2.1"
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Perception Engine Ready.")

    # ------------------------------------------------------

    def perceive(self, text: str) -> Perception:

        # ----------------------------
        # Language
        # ----------------------------

        language = self.language_detector.detect_language_of(
            text
        )

        confidence_values = (
            self.language_detector
            .compute_language_confidence_values(text)
        )

        language_confidence = 0.0

        for item in confidence_values:

            if item.language == language:

                language_confidence = item.value

                break

        # ----------------------------
        # Entities
        # ----------------------------

        labels = [

            "person",

            "organization",

            "location",

            "music artist",

            "university",

            "project",

            "field of study",

            "technology"
        ]

        entities = self.entity_extractor.predict_entities(

            text,

            labels
        )

        # ----------------------------
        # Embedding
        # ----------------------------

        embedding = self.embedding_model.encode(

            text,

            convert_to_tensor=True
        )

        return Perception(

            raw_text=text,

            timestamp=datetime.now(),

            language=str(language),

            language_confidence=language_confidence,

            entities=entities,

            embedding=embedding
        )
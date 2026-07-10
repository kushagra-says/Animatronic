"""
perception.py

Combines the outputs of all feature extractors into a unified
Perception object.
"""

from dataclasses import dataclass
from datetime import datetime

from gliner import GLiNER
from lingua import LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer

from Core.evidence import (
    LanguageEvidence,
    EntityEvidence,
    SemanticEvidence,
)


# ==========================================================
# Perception Object
# ==========================================================

@dataclass
class Perception:

    raw_text: str

    timestamp: datetime

    language: LanguageEvidence

    entities: list[EntityEvidence]

    semantics: SemanticEvidence


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

        timestamp = datetime.now()

        # ==================================================
        # Language
        # ==================================================

        language = self.language_detector.detect_language_of(text)

        confidence_values = (
            self.language_detector
            .compute_language_confidence_values(text)
        )

        language_confidence = 0.0

        for item in confidence_values:

            if item.language == language:

                language_confidence = item.value

                break

        language_evidence = LanguageEvidence(
            extractor="Language Detection",
            model="Lingua",
            version="1.0",
            confidence=language_confidence,
            timestamp=timestamp,
            language=str(language)
        )

        # ==================================================
        # Entities
        # ==================================================

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

        extracted_entities = self.entity_extractor.predict_entities(

            text,

            labels
        )

        entity_evidence = []

        for entity in extracted_entities:

            entity_evidence.append(

                EntityEvidence(
                    extractor="Entity Extraction",
                    model="GLiNER Medium",
                    version="2.1",
                    confidence=entity["score"],
                    timestamp=timestamp,
                    text=entity["text"],
                    label=entity["label"],
                    start=entity["start"],
                    end=entity["end"]
                )
            )

        # ==================================================
        # Semantic Embedding
        # ==================================================

        embedding = self.embedding_model.encode(

            text,

            convert_to_tensor=True
        )

        semantic_evidence = SemanticEvidence(
            extractor="Semantic Embedding",
            model="all-MiniLM-L6-v2",
            version="2",
            confidence=1.0,
            timestamp=timestamp,
            embedding=embedding,
            dimension=embedding.shape[0]
        )

        # ==================================================
        # Return Perception
        # ==================================================

        return Perception(

            raw_text=text,

            timestamp=timestamp,

            language=language_evidence,

            entities=entity_evidence,

            semantics=semantic_evidence
        )
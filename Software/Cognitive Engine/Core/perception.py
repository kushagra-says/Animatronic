"""
perception.py

Combines the outputs of all feature extractors into a unified
Perception object.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
# Model loading helpers
# ==========================================================

# Pin the model identifiers so they're easy to swap later.
GLINER_MODEL_ID = "urchade/gliner_medium-v2.1"
SENTENCE_TRANSFORMER_MODEL_ID = "all-MiniLM-L6-v2"


def _is_cached(model_id: str) -> bool:
    """
    True if the HuggingFace model is already in the local cache.

    Checks the canonical HF cache directory for a repo directory named
    models--{org}--{name}. If found, returns True and tells the loader
    to use local_files_only=True, which skips the metadata re-fetch and
    its noisy progress bars on cold start.
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    # HF escapes '/' in repo_ids as '--' in the on-disk folder name.
    expected = "models--" + model_id.replace("/", "--")
    return (cache_root / expected).is_dir()


def _load_gliner(model_id: str):
    """
    Load GLiNER from cache when available. Avoids the metadata re-fetch
    (and its noisy progress bars) that from_pretrained triggers every
    cold start even when the model is already on disk.
    """
    if _is_cached(model_id):
        return GLiNER.from_pretrained(model_id, local_files_only=True)
    return GLiNER.from_pretrained(model_id)


def _load_sentence_transformer(model_id: str):
    """Same caching strategy as _load_gliner."""
    if _is_cached(model_id):
        return SentenceTransformer(model_id, local_files_only=True)
    return SentenceTransformer(model_id)


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

        self.entity_extractor = _load_gliner(GLINER_MODEL_ID)
        self.embedding_model = _load_sentence_transformer(SENTENCE_TRANSFORMER_MODEL_ID)

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
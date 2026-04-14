"""
normalization_layer.py — Step 3: Normalize Output

Converts raw LLM responses into structured data:
    - final_answer: extracted core answer (if extractable)
    - semantic_embedding: numerical vector for similarity computation
    - refusal_flag: boolean indicating if the model refused
    - answer_hash: deterministic hash of extracted answer for exact comparison

This layer uses TF-IDF based embeddings from sklearn — no external
embedding model required. The embedding captures semantic similarity
between responses for contradiction detection.
"""

import re
import hashlib
import math
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

from .config import REFUSAL_PATTERNS, MAX_ANSWER_LENGTH


class NormalizedResponse:
    """Normalized representation of an LLM response."""

    def __init__(
        self,
        node_id: str,
        raw_content: str,
        final_answer: str,
        semantic_embedding: np.ndarray,
        refusal_flag: bool,
        answer_hash: str,
    ):
        self.node_id = node_id
        self.raw_content = raw_content
        self.final_answer = final_answer
        self.semantic_embedding = semantic_embedding
        self.refusal_flag = refusal_flag
        self.answer_hash = answer_hash

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "final_answer": self.final_answer,
            "refusal_flag": self.refusal_flag,
            "answer_hash": self.answer_hash,
        }

    def __repr__(self) -> str:
        status = "REFUSED" if self.refusal_flag else "ANSWERED"
        preview = self.final_answer[:60].replace("\n", " ")
        return f"NormalizedResponse(id={self.node_id}, {status}, '{preview}...')"


class EmbeddingEngine:
    """
    TF-IDF based semantic embedding engine.

    Uses sklearn TfidfVectorizer to create dense vector representations
    of LLM responses. The embeddings are used for:
        - Semantic similarity between prompts (Edge Type A)
        - Embedding distance for variance edges (Edge Type C)
        - Contradiction detection clustering

    The engine is stateless per batch — each call to embed() computes
    a fresh TF-IDF model over the provided texts. This ensures
    determinism and prevents drift.
    """

    def __init__(self):
        self.vectorizer = None
        self.embeddings = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Compute TF-IDF embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), n_features)
        """
        if not texts:
            return np.array([])

        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        return self.embeddings

    def cosine_similarity(self, idx_a: int, idx_b: int) -> float:
        """
        Compute cosine similarity between two embedded texts.

        Args:
            idx_a: Index of first text.
            idx_b: Index of second text.

        Returns:
            Float between -1 and 1 (typically 0 to 1 for TF-IDF).
        """
        if self.embeddings is None:
            return 0.0
        vec_a = self.embeddings[idx_a].reshape(1, -1)
        vec_b = self.embeddings[idx_b].reshape(1, -1)
        return float(sklearn_cosine_sim(vec_a, vec_b)[0][0])

    def embedding_distance(self, idx_a: int, idx_b: int) -> float:
        """
        Compute Euclidean distance between two embeddings.

        Args:
            idx_a: Index of first text.
            idx_b: Index of second text.

        Returns:
            Non-negative float representing distance.
        """
        if self.embeddings is None:
            return float("inf")
        vec_a = self.embeddings[idx_a]
        vec_b = self.embeddings[idx_b]
        return float(np.linalg.norm(vec_a - vec_b))


def _detect_refusal(text: str) -> bool:
    """
    Detect if a response is a refusal/safety completion.

    Checks against a list of known refusal patterns. Case-insensitive.
    A response is flagged as refusal if ANY pattern matches.

    Args:
        text: The LLM response text.

    Returns:
        True if the response appears to be a refusal.
    """
    text_lower = text.lower().strip()
    for pattern in REFUSAL_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def _extract_answer(text: str) -> str:
    """
    Extract the core answer from a response.

    Strategy:
        1. If response contains common answer markers (e.g., "Answer:", "Therefore,"),
           extract text after the last marker.
        2. Otherwise, take the last non-empty paragraph (models tend to reason first,
           then state the answer).
        3. Truncate to MAX_ANSWER_LENGTH.

    Args:
        text: The raw LLM response.

    Returns:
        Extracted answer string.
    """
    text = text.strip()
    if not text:
        return ""

    # Answer markers — find the last one
    answer_markers = [
        r"(?:^|\n)\s*(?:answer|conclusion|result|final answer|therefore|thus|so,)\s*[:.]?\s*",
        r"\[Answer\]\s*",
        r"###\s*(?:answer|result)\s*\n",
    ]

    extracted = None
    for marker in answer_markers:
        matches = list(re.finditer(marker, text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            candidate = text[last_match.end():].strip()
            if candidate:
                extracted = candidate
                break

    if extracted is None:
        # Take the last non-empty paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            extracted = paragraphs[-1]
        else:
            # Fall back to last sentence
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
            extracted = sentences[-1] if sentences else text

    # Clean up
    extracted = re.sub(r'\s+', ' ', extracted).strip()
    if len(extracted) > MAX_ANSWER_LENGTH:
        extracted = extracted[:MAX_ANSWER_LENGTH] + "..."

    return extracted


def _hash_answer(answer: str) -> str:
    """Generate a deterministic hash of the extracted answer."""
    normalized = answer.lower().strip()
    # Remove punctuation for hash normalization
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def normalize_response(response) -> NormalizedResponse:
    """
    Normalize a single LLM response.

    Args:
        response: An LLMResponse object.

    Returns:
        NormalizedResponse with extracted answer, refusal flag, and metadata.
    """
    refusal = _detect_refusal(response.content)
    answer = _extract_answer(response.content)
    answer_hash = _hash_answer(answer)

    return NormalizedResponse(
        node_id=response.node_id,
        raw_content=response.content,
        final_answer=answer,
        semantic_embedding=None,  # populated by embed_responses()
        refusal_flag=refusal,
        answer_hash=answer_hash,
    )


def embed_responses(
    normalized_responses: list[NormalizedResponse],
) -> EmbeddingEngine:
    """
    Compute semantic embeddings for all normalized responses.

    Creates a batch TF-IDF model over all response texts.
    Populates the semantic_embedding field on each NormalizedResponse.

    Args:
        normalized_responses: List of NormalizedResponse objects.

    Returns:
        EmbeddingEngine with computed embeddings and similarity methods.
    """
    engine = EmbeddingEngine()
    texts = [r.final_answer for r in normalized_responses]
    embeddings = engine.embed(texts)

    for i, resp in enumerate(normalized_responses):
        resp.semantic_embedding = embeddings[i]

    return engine


def normalize_and_embed(
    llm_responses: list,
) -> tuple[list[NormalizedResponse], EmbeddingEngine]:
    """
    Full normalization pipeline: normalize + embed.

    Args:
        llm_responses: List of LLMResponse objects.

    Returns:
        Tuple of (normalized_responses, embedding_engine).
    """
    normalized = [normalize_response(r) for r in llm_responses]
    engine = embed_responses(normalized)
    return normalized, engine

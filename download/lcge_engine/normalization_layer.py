"""
normalization_layer.py — Step 3: Normalize Output (v1.1)

Converts raw LLM responses into structured data for instability classification:
    - final_answer: extracted core answer
    - semantic_embedding: numerical vector for similarity computation
    - refusal_flag: boolean indicating if the model refused
    - answer_hash: deterministic hash of extracted answer
    - reasoning_trace: extracted reasoning steps (v1.1)
    - format_signature: structural format hash (v1.1)
    - semantic_family_id: family grouping identifier (v1.1)
"""

import re
import hashlib
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

from .config import REFUSAL_PATTERNS, MAX_ANSWER_LENGTH, REASONING_MARKERS, FORMAT_PATTERNS


class NormalizedResponse:
    """Normalized representation of an LLM response (v1.1 schema)."""

    def __init__(
        self,
        node_id: str,
        raw_content: str,
        final_answer: str,
        semantic_embedding: np.ndarray,
        refusal_flag: bool,
        answer_hash: str,
        reasoning_trace: str = "",
        format_signature: str = "",
        semantic_family_id: str = "",
    ):
        self.node_id = node_id
        self.raw_content = raw_content
        self.final_answer = final_answer
        self.semantic_embedding = semantic_embedding
        self.refusal_flag = refusal_flag
        self.answer_hash = answer_hash
        self.reasoning_trace = reasoning_trace
        self.format_signature = format_signature
        self.semantic_family_id = semantic_family_id

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "final_answer": self.final_answer,
            "refusal_flag": self.refusal_flag,
            "answer_hash": self.answer_hash,
            "reasoning_trace": self.reasoning_trace,
            "format_signature": self.format_signature,
            "semantic_family_id": self.semantic_family_id,
        }

    def __repr__(self) -> str:
        status = "REFUSED" if self.refusal_flag else "ANSWERED"
        preview = self.final_answer[:60].replace("\n", " ")
        fmt = self.format_signature[:8] if self.format_signature else "none"
        return f"NormalizedResponse(id={self.node_id}, {status}, fmt={fmt}, '{preview}...')"


class EmbeddingEngine:
    """
    TF-IDF based semantic embedding engine.

    Stateless per batch — each call to embed() computes a fresh TF-IDF model.
    """

    def __init__(self):
        self.vectorizer = None
        self.embeddings = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        # Replace empty strings with placeholder to avoid sklearn crash
        cleaned = [t.strip() if t.strip() else "__empty__" for t in texts]

        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        try:
            self.embeddings = self.vectorizer.fit_transform(cleaned).toarray()
        except ValueError:
            # If TF-IDF still fails (e.g., all identical), return zero vectors
            self.embeddings = np.zeros((len(cleaned), 1))
        return self.embeddings

    def cosine_similarity(self, idx_a: int, idx_b: int) -> float:
        if self.embeddings is None:
            return 0.0
        vec_a = self.embeddings[idx_a].reshape(1, -1)
        vec_b = self.embeddings[idx_b].reshape(1, -1)
        return float(sklearn_cosine_sim(vec_a, vec_b)[0][0])

    def embedding_distance(self, idx_a: int, idx_b: int) -> float:
        if self.embeddings is None:
            return float("inf")
        vec_a = self.embeddings[idx_a]
        vec_b = self.embeddings[idx_b]
        return float(np.linalg.norm(vec_a - vec_b))


# ============================================================
# Extraction functions
# ============================================================

def _detect_refusal(text: str) -> bool:
    text_lower = text.lower().strip()
    for pattern in REFUSAL_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def _extract_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

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
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            extracted = paragraphs[-1]
        else:
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
            extracted = sentences[-1] if sentences else text

    extracted = re.sub(r'\s+', ' ', extracted).strip()
    if len(extracted) > MAX_ANSWER_LENGTH:
        extracted = extracted[:MAX_ANSWER_LENGTH] + "..."

    return extracted


def _extract_reasoning_trace(text: str) -> str:
    """
    Extract the reasoning portion of a response.

    Strategy: split at the last answer marker or conclusion phrase.
    Everything before that split is the reasoning trace.
    If no clear marker, use all content except the final paragraph.
    """
    text = text.strip()
    if not text:
        return ""

    # Find the answer boundary
    answer_markers = [
        r"(?:^|\n)\s*(?:answer|conclusion|result|final answer|therefore|thus|so,)\s*[:.]?\s*",
        r"\[Answer\]\s*",
        r"###\s*(?:answer|result)\s*\n",
    ]

    split_point = len(text)
    for marker in answer_markers:
        matches = list(re.finditer(marker, text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            if last_match.start() < split_point:
                split_point = last_match.start()

    # Also split at explicit reasoning-answer boundaries
    boundary_patterns = [
        r"(?:^|\n)\s*---+\s*(?:^|\n)",
        r"(?:^|\n)\s*###\s+answer",
        r"\n\s*\[answer\]",
    ]
    for pattern in boundary_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            pos = matches[-1].start()
            if pos < split_point:
                split_point = pos

    trace = text[:split_point].strip()

    # If trace is most of the text, try paragraph-based split
    if len(trace) > len(text) * 0.85:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            trace = "\n\n".join(paragraphs[:-1])

    # Clean up
    trace = re.sub(r'\n{3,}', '\n\n', trace).strip()
    if len(trace) > 1000:
        trace = trace[:1000] + "..."

    return trace


def _compute_format_signature(text: str) -> str:
    """
    Compute a structural format signature for a response.

    Detects: bullet lists, numbered lists, code blocks, JSON, headings, tables.
    Returns a hash representing the detected format structure.
    """
    detected_formats = []

    for fmt_name, pattern in FORMAT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            detected_formats.append(fmt_name)

    # Detect paragraph structure
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        detected_formats.append("multi_paragraph")
    elif len(paragraphs) == 1 and len(text) < 200:
        detected_formats.append("single_short")

    # Detect line structure
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 5:
        # Check if most lines are short (list-like)
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 80:
            detected_formats.append("short_lines")

    # Sort for deterministic hash
    fmt_key = "|".join(sorted(detected_formats))
    return hashlib.sha256(fmt_key.encode("utf-8")).hexdigest()[:12]


def _hash_answer(answer: str) -> str:
    normalized = answer.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


# ============================================================
# Public API
# ============================================================

def normalize_response(response, semantic_family_id: str = "") -> NormalizedResponse:
    """
    Normalize a single LLM response with v1.1 fields.
    """
    refusal = _detect_refusal(response.content)
    answer = _extract_answer(response.content)
    answer_hash = _hash_answer(answer)
    reasoning_trace = _extract_reasoning_trace(response.content)
    format_signature = _compute_format_signature(response.content)

    return NormalizedResponse(
        node_id=response.node_id,
        raw_content=response.content,
        final_answer=answer,
        semantic_embedding=None,
        refusal_flag=refusal,
        answer_hash=answer_hash,
        reasoning_trace=reasoning_trace,
        format_signature=format_signature,
        semantic_family_id=semantic_family_id,
    )


def embed_responses(
    normalized_responses: list[NormalizedResponse],
    mode: str = "answer",
) -> EmbeddingEngine:
    """
    Compute semantic embeddings for all normalized responses.

    Args:
        normalized_responses: List of NormalizedResponse objects.
        mode: "answer" for final_answer text, "reasoning" for reasoning_trace text.

    Returns:
        EmbeddingEngine with computed embeddings.
    """
    engine = EmbeddingEngine()
    if mode == "reasoning":
        texts = [r.reasoning_trace or r.final_answer for r in normalized_responses]
    else:
        texts = [r.final_answer for r in normalized_responses]
    embeddings = engine.embed(texts)

    for i, resp in enumerate(normalized_responses):
        resp.semantic_embedding = embeddings[i]

    return engine


def normalize_and_embed(
    llm_responses: list,
    semantic_family_id: str = "",
) -> tuple[list[NormalizedResponse], EmbeddingEngine, EmbeddingEngine]:
    """
    Full normalization pipeline: normalize + embed answer + embed reasoning.

    Returns:
        Tuple of (normalized_responses, answer_engine, reasoning_engine).
    """
    normalized = [normalize_response(r, semantic_family_id) for r in llm_responses]

    # Embed answers (primary signal)
    answer_engine = embed_responses(normalized, mode="answer")

    # Embed reasoning traces (secondary signal)
    reasoning_engine = embed_responses(normalized, mode="reasoning")

    return normalized, answer_engine, reasoning_engine

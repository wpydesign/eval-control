"""
vector_store.py — Raw response → embedding vectors.

Converts raw LLM response texts into numerical vector representations.
No labels. No scores. No interpretation. Just embeddings.

Uses TF-IDF vectorization fitted on the entire response corpus.
"""

import json
import os
import sys
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

logger = logging.getLogger("lattice")


def load_run_records(jsonl_path: str) -> list[dict]:
    """
    Load all run records from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL output from run_lattice.py.

    Returns:
        List of record dicts.
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_embedding_matrix(records: list[dict]) -> tuple[np.ndarray, dict]:
    """
    Build TF-IDF embedding matrix from raw response texts.

    Fits the vectorizer on ALL texts simultaneously to ensure
    consistent feature space across all responses.

    Args:
        records: List of run records with "response" field.

    Returns:
        Tuple of:
            - embedding_matrix: np.ndarray of shape (n_records, n_features)
            - metadata: dict with vectorizer info
    """
    texts = []
    run_ids = []
    empty_count = 0

    for rec in records:
        text = rec.get("response", "").strip()
        if not text or text.startswith("["):
            text = "__empty__"
            empty_count += 1
        texts.append(text)
        run_ids.append(rec.get("run_id", ""))

    if empty_count > 0:
        logger.warning(f"  {empty_count}/{len(texts)} responses were empty/errored")

    # Fit TF-IDF on all texts (consistent feature space)
    vectorizer = TfidfVectorizer(
        max_features=1024,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    try:
        embedding_matrix = vectorizer.fit_transform(texts).toarray()
    except ValueError:
        # Fallback: zero vectors
        logger.error("  TF-IDF failed, using zero vectors")
        embedding_matrix = np.zeros((len(texts), 1))
        vectorizer = None

    metadata = {
        "n_records": len(records),
        "n_features": embedding_matrix.shape[1],
        "empty_count": empty_count,
        "vectorizer_fitted": vectorizer is not None,
    }

    logger.info(f"  Embedding matrix: {embedding_matrix.shape}")
    logger.info(f"  Features: {metadata['n_features']}")

    return embedding_matrix, metadata


def save_embeddings(
    embedding_matrix: np.ndarray,
    run_ids: list[str],
    output_path: str,
):
    """
    Save embeddings to numpy binary format + ID mapping.

    Args:
        embedding_matrix: The TF-IDF embedding matrix.
        run_ids: List of run_id strings (row ordering).
        output_path: Path to save .npz file.
    """
    np.savez_compressed(
        output_path,
        embeddings=embedding_matrix,
        run_ids=np.array(run_ids),
    )
    logger.info(f"  Saved embeddings to {output_path}")


def save_records_for_solver(
    records: list[dict],
    embedding_matrix: np.ndarray,
    output_path: str,
):
    """
    Save a minimal JSON file for the coordinate solver.
    Contains only: run_id, prompt_id, strategy, rep, projection_index.

    Args:
        records: List of run records.
        embedding_matrix: The embedding matrix (for dimension info).
        output_path: Path to save JSON file.
    """
    minimal = []
    for i, rec in enumerate(records):
        minimal.append({
            "run_id": rec.get("run_id", ""),
            "prompt_id": rec.get("prompt_id", -1),
            "strategy": rec.get("strategy", ""),
            "rep": rec.get("rep", -1),
            "row_index": i,
        })

    with open(output_path, "w") as f:
        json.dump({
            "n_records": len(records),
            "n_features": embedding_matrix.shape[1],
            "records": minimal,
        }, f, indent=2)

    logger.info(f"  Saved record index to {output_path}")

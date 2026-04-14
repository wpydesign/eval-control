"""
edge_builder.py — Step 4: Edge Builder (CRITICAL)

Creates edges between graph nodes based on three edge types:

    Edge Type A — semantic_similarity
        cosine_similarity(prompt_a, prompt_b)

    Edge Type B — contradiction_edge
        Triggered if: same task intent AND different final_answer OR refusal mismatch

    Edge Type C — variance_edge
        embedding_distance > threshold

Edge construction follows the Permission Inconsistency Graph Engine v6.0
methodology adapted for LLM response analysis.

Edge weights follow the security-weight model:
    - same_family:       0.0 (grouping only)
    - contradiction:     3.0 (direct violation)
    - variance:          2.0 (moderate indicator)
    - refusal_mismatch:  2.5 (significant inconsistency)
"""

from typing import Optional

import numpy as np

from .config import (
    SEMANTIC_SIMILARITY_THRESHOLD_HIGH,
    CONTRADICTION_PROMPT_SIMILARITY_MIN,
    CONTRADICTION_RESPONSE_DIVERGENCE_MIN,
    VARIANCE_DISTANCE_THRESHOLD,
    EDGE_WEIGHTS,
    RESPONSE_SIZE_DIVERGENCE_RATIO,
)
from .normalization_layer import (
    NormalizedResponse,
    EmbeddingEngine,
)
from .prompt_input_layer import PromptVariant


class Edge:
    """A directed edge between two graph nodes."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float,
        evidence: dict,
        edge_confidence: float = 1.0,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight
        self.evidence = evidence
        self.edge_confidence = edge_confidence

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "evidence": self.evidence,
            "edge_confidence": self.edge_confidence,
        }

    def __repr__(self) -> str:
        return (
            f"Edge({self.source_id} -> {self.target_id}, "
            f"type={self.edge_type}, weight={self.weight:.2f})"
        )


class EdgeBuilder:
    """
    Constructs edges between nodes based on prompt similarity,
    response divergence, and structural analysis.

    The builder operates in three passes:
        Pass 1: Build same_family edges (prompt similarity)
        Pass 2: Build contradiction edges (answer divergence + refusal mismatch)
        Pass 3: Build variance edges (embedding distance)
    """

    def __init__(self, prompt_engine: Optional[EmbeddingEngine] = None):
        self.prompt_engine = prompt_engine
        self.response_engine = None
        self.prompt_embeddings = None
        self.response_embeddings = None

    def set_response_engine(self, engine: EmbeddingEngine):
        """Set the response embedding engine for variance computation."""
        self.response_engine = engine
        self.response_embeddings = engine.embeddings

    def set_prompt_engine(self, engine: EmbeddingEngine):
        """Set the prompt embedding engine for prompt similarity computation."""
        self.prompt_engine = engine
        self.prompt_embeddings = engine.embeddings

    def build_all_edges(
        self,
        variants: list,
        normalized_responses: list[NormalizedResponse],
    ) -> list[Edge]:
        """
        Build all edges for a set of variants and their responses.

        Args:
            variants: List of PromptVariant objects.
            normalized_responses: Corresponding list of NormalizedResponse objects.

        Returns:
            List of Edge objects representing the graph's edge set.
        """
        edges = []

        # Create lookup maps
        variant_map = {v.node_id: v for v in variants}
        response_map = {r.node_id: r for r in normalized_responses}

        node_ids = list(variant_map.keys())
        n = len(node_ids)

        for i in range(n):
            for j in range(i + 1, n):
                node_a = node_ids[i]
                node_b = node_ids[j]

                variant_a = variant_map[node_a]
                variant_b = variant_map[node_b]
                resp_a = response_map.get(node_a)
                resp_b = response_map.get(node_b)

                if resp_a is None or resp_b is None:
                    continue

                # Pass 1: Semantic similarity (prompt-level)
                prompt_sim = self._compute_prompt_similarity(variant_a, variant_b)
                if prompt_sim >= SEMANTIC_SIMILARITY_THRESHOLD_HIGH:
                    edges.append(Edge(
                        source_id=node_a,
                        target_id=node_b,
                        edge_type="same_family",
                        weight=EDGE_WEIGHTS["same_family"],
                        evidence={"prompt_similarity": round(prompt_sim, 4)},
                        edge_confidence=prompt_sim,
                    ))

                # Pass 2: Contradiction detection
                contradiction_edges = self._build_contradiction_edges(
                    node_a, node_b, variant_a, variant_b, resp_a, resp_b, prompt_sim
                )
                edges.extend(contradiction_edges)

                # Pass 3: Variance edge
                variance_edge = self._build_variance_edge(
                    node_a, node_b, resp_a, resp_b
                )
                if variance_edge:
                    edges.append(variance_edge)

        return edges

    def _compute_prompt_similarity(
        self, variant_a: "PromptVariant", variant_b: "PromptVariant"
    ) -> float:
        """
        Compute semantic similarity between two prompt variants.

        Primary signal: same task field (structural identity).
        All variants generated from the same seed share the same task,
        so they belong to the same family by construction.

        Secondary signal: TF-IDF cosine similarity on prompt text,
        used to modulate within-family edge confidence.

        This follows the LCGE spec: the engine is about contradiction
        within a prompt family, not across different tasks.
        """
        # Primary: same task = same family (hard structural identity)
        if variant_a.task == variant_b.task:
            base_sim = 0.95  # high baseline for same-family variants
        else:
            base_sim = 0.1  # different tasks = different families

        # Secondary: use TF-IDF similarity to modulate confidence
        if self.prompt_engine and self.prompt_embeddings is not None:
            idx_a = variant_a.variant_index
            idx_b = variant_b.variant_index
            if idx_a < len(self.prompt_embeddings) and idx_b < len(self.prompt_embeddings):
                tfidf_sim = self.prompt_engine.cosine_similarity(idx_a, idx_b)
                # Blend: use TF-IDF as a confidence modulator, not as primary signal
                # If TF-IDF agrees (high), keep high base. If TF-IDF disagrees, reduce.
                if tfidf_sim > 0.3:
                    base_sim = max(base_sim, tfidf_sim * 0.6 + base_sim * 0.4)
                # For same-task variants, never drop below 0.80
                if variant_a.task == variant_b.task:
                    base_sim = max(base_sim, 0.80)

        return max(0.0, min(1.0, base_sim))

    def _build_contradiction_edges(
        self,
        node_a: str,
        node_b: str,
        variant_a: "PromptVariant",
        variant_b: "PromptVariant",
        resp_a: NormalizedResponse,
        resp_b: NormalizedResponse,
        prompt_sim: float,
    ) -> list[Edge]:
        """
        Build contradiction edges between two nodes.

        Contradiction requires:
            - Same task intent (prompt similarity above threshold)
            - Different final_answer OR refusal mismatch
        """
        edges = []

        # Gate: prompts must be sufficiently similar
        if prompt_sim < CONTRADICTION_PROMPT_SIMILARITY_MIN:
            return edges

        # Check answer divergence
        answer_divergence = 0.0
        if resp_a.answer_hash != resp_b.answer_hash:
            answer_divergence = 1.0  # Different answers = full divergence
        else:
            answer_divergence = 0.0  # Same answer = no divergence

        # Check refusal mismatch
        refusal_mismatch = 0.0
        if resp_a.refusal_flag != resp_b.refusal_flag:
            refusal_mismatch = 1.0  # One refused, other didn't

        # Check response size divergence
        size_ratio = 0.0
        len_a = len(resp_a.raw_content)
        len_b = len(resp_b.raw_content)
        if len_a > 0 and len_b > 0:
            size_ratio = max(len_a, len_b) / max(min(len_a, len_b), 1)
        size_divergence = 1.0 if size_ratio >= RESPONSE_SIZE_DIVERGENCE_RATIO else 0.0

        # Compute embedding-based response divergence
        embedding_divergence = 0.0
        if self.response_engine and resp_a.semantic_embedding is not None and resp_b.semantic_embedding is not None:
            cos_sim = float(np.dot(resp_a.semantic_embedding, resp_b.semantic_embedding) /
                          (np.linalg.norm(resp_a.semantic_embedding) * np.linalg.norm(resp_b.semantic_embedding) + 1e-10))
            # Low cosine similarity = high divergence
            embedding_divergence = max(0.0, 1.0 - cos_sim)

        # Build contradiction edge if conditions met
        total_divergence = (answer_divergence * 0.6 +
                           refusal_mismatch * 0.8 +
                           size_divergence * 0.3 +
                           embedding_divergence * 0.5)

        if total_divergence >= CONTRADICTION_RESPONSE_DIVERGENCE_MIN:
            evidence = {
                "prompt_similarity": round(prompt_sim, 4),
                "answer_divergence": answer_divergence,
                "refusal_mismatch": refusal_mismatch,
                "size_ratio": round(size_ratio, 2),
                "embedding_divergence": round(embedding_divergence, 4),
                "total_divergence": round(total_divergence, 4),
            }

            # Determine contradiction type
            if refusal_mismatch > 0:
                edge_type = "refusal_mismatch"
                weight = EDGE_WEIGHTS["refusal_mismatch"]
            elif answer_divergence > 0:
                edge_type = "contradiction"
                weight = EDGE_WEIGHTS["contradiction"]
            else:
                edge_type = "contradiction"
                weight = EDGE_WEIGHTS["contradiction"] * 0.7

            edge_confidence = min(1.0, prompt_sim * total_divergence)

            edges.append(Edge(
                source_id=node_a,
                target_id=node_b,
                edge_type=edge_type,
                weight=weight,
                evidence=evidence,
                edge_confidence=edge_confidence,
            ))

        return edges

    def _build_variance_edge(
        self,
        node_a: str,
        node_b: str,
        resp_a: NormalizedResponse,
        resp_b: NormalizedResponse,
    ) -> Optional[Edge]:
        """
        Build variance edge if embedding distance exceeds threshold.

        Variance edges indicate moderate divergence between responses
        that don't qualify as full contradictions.
        """
        if self.response_engine is None:
            return None

        # Compute distance via response embedding
        if resp_a.semantic_embedding is not None and resp_b.semantic_embedding is not None:
            distance = float(np.linalg.norm(resp_a.semantic_embedding - resp_b.semantic_embedding))

            if distance > VARIANCE_DISTANCE_THRESHOLD:
                return Edge(
                    source_id=node_a,
                    target_id=node_b,
                    edge_type="variance",
                    weight=EDGE_WEIGHTS["variance"],
                    evidence={"embedding_distance": round(distance, 4)},
                    edge_confidence=min(1.0, distance / (VARIANCE_DISTANCE_THRESHOLD * 2)),
                )

        return None

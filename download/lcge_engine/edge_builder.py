"""
edge_builder.py — Step 4: Edge Builder (v1.1)

Creates edges between graph nodes based on three instability-focused edge types:

    Edge Type A — behavioral_shift
        Detects response change between nodes (general divergence)

    Edge Type B — policy_flip_edge
        Refusal <-> answer transitions

    Edge Type C — semantic_drift_edge
        Meaning divergence above threshold

Replaces v1.0 contradiction/contradiction_edge/variance_edge model.
"""

from typing import Optional

import numpy as np

from .config import (
    EDGE_WEIGHTS,
    SEMANTIC_DRIFT_THRESHOLD,
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
    Constructs edges between nodes for instability analysis.

    Three-pass construction:
        Pass 1: family_link edges (prompt similarity — grouping)
        Pass 2: policy_flip_edge (refusal <-> answer transitions)
        Pass 3: behavioral_shift + semantic_drift (response divergence)
    """

    def __init__(self, prompt_engine: Optional[EmbeddingEngine] = None):
        self.prompt_engine = prompt_engine
        self.response_engine = None
        self.reasoning_engine = None
        self.prompt_embeddings = None

    def set_response_engine(self, engine: EmbeddingEngine):
        self.response_engine = engine

    def set_reasoning_engine(self, engine: EmbeddingEngine):
        self.reasoning_engine = engine

    def set_prompt_engine(self, engine: EmbeddingEngine):
        self.prompt_engine = engine
        self.prompt_embeddings = engine.embeddings

    def build_all_edges(
        self,
        variants: list,
        normalized_responses: list[NormalizedResponse],
    ) -> list[Edge]:
        edges = []
        variant_map = {v.node_id: v for v in variants}
        response_map = {r.node_id: r for r in normalized_responses}
        node_ids = list(variant_map.keys())
        n = len(node_ids)

        for i in range(n):
            for j in range(i + 1, n):
                node_a, node_b = node_ids[i], node_ids[j]
                variant_a, variant_b = variant_map[node_a], variant_map[node_b]
                resp_a = response_map.get(node_a)
                resp_b = response_map.get(node_b)

                if resp_a is None or resp_b is None:
                    continue

                # Pass 1: Family link (grouping)
                prompt_sim = self._compute_prompt_similarity(variant_a, variant_b)
                if prompt_sim >= 0.80:
                    edges.append(Edge(
                        source_id=node_a, target_id=node_b,
                        edge_type="family_link",
                        weight=EDGE_WEIGHTS["family_link"],
                        evidence={"prompt_similarity": round(prompt_sim, 4)},
                        edge_confidence=prompt_sim,
                    ))

                # Pass 2: Policy flip edge
                pf = self._build_policy_flip_edge(node_a, node_b, resp_a, resp_b, prompt_sim)
                if pf:
                    edges.append(pf)

                # Pass 3: Behavioral shift + semantic drift
                shift = self._build_behavioral_shift_edge(node_a, node_b, resp_a, resp_b, prompt_sim)
                if shift:
                    edges.append(shift)

                drift = self._build_semantic_drift_edge(node_a, node_b, resp_a, resp_b, prompt_sim)
                if drift:
                    edges.append(drift)

        return edges

    def _compute_prompt_similarity(
        self, variant_a: "PromptVariant", variant_b: "PromptVariant"
    ) -> float:
        if variant_a.task == variant_b.task:
            base_sim = 0.95
        else:
            base_sim = 0.1

        if self.prompt_engine and self.prompt_embeddings is not None:
            idx_a = variant_a.variant_index
            idx_b = variant_b.variant_index
            if idx_a < len(self.prompt_embeddings) and idx_b < len(self.prompt_embeddings):
                tfidf_sim = self.prompt_engine.cosine_similarity(idx_a, idx_b)
                if tfidf_sim > 0.3:
                    base_sim = max(base_sim, tfidf_sim * 0.6 + base_sim * 0.4)
                if variant_a.task == variant_b.task:
                    base_sim = max(base_sim, 0.80)

        return max(0.0, min(1.0, base_sim))

    def _build_policy_flip_edge(
        self, node_a: str, node_b: str,
        resp_a: NormalizedResponse, resp_b: NormalizedResponse,
        prompt_sim: float,
    ) -> Optional[Edge]:
        """Build policy_flip_edge when one node refuses and other answers."""
        if resp_a.refusal_flag != resp_b.refusal_flag:
            # Determine which refused and which answered
            if resp_a.refusal_flag:
                refused_node, answered_node = node_a, node_b
            else:
                refused_node, answered_node = node_b, node_a

            evidence = {
                "prompt_similarity": round(prompt_sim, 4),
                "refused_node": refused_node[:8],
                "answered_node": answered_node[:8],
            }

            # Size divergence between refusal and answer
            len_ref = len(resp_a.raw_content) if resp_a.refusal_flag else len(resp_b.raw_content)
            len_ans = len(resp_b.raw_content) if not resp_b.refusal_flag else len(resp_a.raw_content)
            if len_ref > 0:
                evidence["size_ratio"] = round(max(len_ref, len_ans) / max(min(len_ref, len_ans), 1), 2)

            return Edge(
                source_id=node_a, target_id=node_b,
                edge_type="policy_flip",
                weight=EDGE_WEIGHTS["policy_flip"],
                evidence=evidence,
                edge_confidence=min(1.0, prompt_sim),
            )
        return None

    def _build_behavioral_shift_edge(
        self, node_a: str, node_b: str,
        resp_a: NormalizedResponse, resp_b: NormalizedResponse,
        prompt_sim: float,
    ) -> Optional[Edge]:
        """Build behavioral_shift edge when response characteristics diverge."""
        # Skip if both refused — both refusing is consistent behavior
        if resp_a.refusal_flag and resp_b.refusal_flag:
            return None

        # Already handled by policy_flip
        if resp_a.refusal_flag != resp_b.refusal_flag:
            return None

        # Both answered — check for behavioral shift
        divergence_signals = []

        # Answer hash mismatch
        if resp_a.answer_hash != resp_b.answer_hash:
            divergence_signals.append("answer_mismatch")

        # Format signature mismatch
        if resp_a.format_signature != resp_b.format_signature:
            divergence_signals.append("format_mismatch")

        # Response size divergence
        len_a, len_b = len(resp_a.raw_content), len(resp_b.raw_content)
        size_ratio = 0.0
        if len_a > 0 and len_b > 0:
            size_ratio = max(len_a, len_b) / max(min(len_a, len_b), 1)
        if size_ratio >= RESPONSE_SIZE_DIVERGENCE_RATIO:
            divergence_signals.append("size_divergence")

        # Embedding divergence
        embedding_div = 0.0
        if resp_a.semantic_embedding is not None and resp_b.semantic_embedding is not None:
            norm_a = np.linalg.norm(resp_a.semantic_embedding)
            norm_b = np.linalg.norm(resp_b.semantic_embedding)
            if norm_a > 0 and norm_b > 0:
                cos_sim = float(np.dot(resp_a.semantic_embedding, resp_b.semantic_embedding) / (norm_a * norm_b))
                embedding_div = max(0.0, 1.0 - cos_sim)
                if embedding_div > SEMANTIC_DRIFT_THRESHOLD:
                    divergence_signals.append("embedding_drift")

        if not divergence_signals:
            return None

        evidence = {
            "prompt_similarity": round(prompt_sim, 4),
            "signals": divergence_signals,
            "embedding_divergence": round(embedding_div, 4),
            "size_ratio": round(size_ratio, 2),
            "format_match": resp_a.format_signature == resp_b.format_signature,
        }

        # Weight scales with number of divergence signals
        signal_weight = min(len(divergence_signals) / 3.0, 1.0)

        return Edge(
            source_id=node_a, target_id=node_b,
            edge_type="behavioral_shift",
            weight=EDGE_WEIGHTS["behavioral_shift"] * signal_weight,
            evidence=evidence,
            edge_confidence=min(1.0, prompt_sim * signal_weight),
        )

    def _build_semantic_drift_edge(
        self, node_a: str, node_b: str,
        resp_a: NormalizedResponse, resp_b: NormalizedResponse,
        prompt_sim: float,
    ) -> Optional[Edge]:
        """Build semantic_drift_edge when meaning diverges above threshold."""
        if resp_a.semantic_embedding is None or resp_b.semantic_embedding is None:
            return None

        norm_a = np.linalg.norm(resp_a.semantic_embedding)
        norm_b = np.linalg.norm(resp_b.semantic_embedding)
        if norm_a == 0 or norm_b == 0:
            return None

        cos_sim = float(np.dot(resp_a.semantic_embedding, resp_b.semantic_embedding) / (norm_a * norm_b))
        drift = max(0.0, 1.0 - cos_sim)

        if drift >= SEMANTIC_DRIFT_THRESHOLD:
            evidence = {
                "prompt_similarity": round(prompt_sim, 4),
                "cosine_similarity": round(cos_sim, 4),
                "drift_magnitude": round(drift, 4),
            }
            return Edge(
                source_id=node_a, target_id=node_b,
                edge_type="semantic_drift",
                weight=EDGE_WEIGHTS["semantic_drift"] * drift,
                evidence=evidence,
                edge_confidence=min(1.0, drift),
            )

        return None

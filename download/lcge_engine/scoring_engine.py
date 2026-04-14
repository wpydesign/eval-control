"""
scoring_engine.py — Step 7: Scoring

Computes confidence scores for contradiction clusters.

Formula (v1.0 — simple version):
    confidence = contradiction_strength
               + response_divergence
               + refusal_inconsistency_bonus

    diversity_bonus = 0.5 * (num_distinct_edge_types - 1)
    confidence = min(total_weight * (1 + diversity_bonus), 10.0)

Follows the Permission Inconsistency Graph Engine v6.0 confidence model:
    effective_weight = edge.weight * edge.edge_confidence
    total_weight = sum(effective_weight for all edges)
    diversity = number of distinct non-zero-weight edge types
    confidence = min(total_weight * (1 + 0.5 * (diversity - 1)), 10.0)

Reproducibility is computed by running detection multiple times
and measuring the stability of findings.
"""

import hashlib
from typing import Optional
from collections import Counter

from .graph_constructor import ConsistencyGraph
from .contradiction_detector import ContradictionCluster, ContradictionDetector
from .config import (
    CONFIDENCE_CAP,
    DIVERSITY_BONUS_FACTOR,
    REPRODUCIBILITY_RUNS,
    REPRODUCIBILITY_THRESHOLD,
    SUBMISSION_MIN_CONFIDENCE,
    SUBMISSION_MIN_EDGE_TYPES,
)


class ScoredCluster:
    """A contradiction cluster with full confidence scoring."""

    def __init__(
        self,
        cluster: ContradictionCluster,
        confidence: float,
        diversity: int,
        is_submittable: bool,
        rejection_reason: Optional[str] = None,
    ):
        self.cluster = cluster
        self.confidence = confidence
        self.diversity = diversity
        self.is_submittable = is_submittable
        self.rejection_reason = rejection_reason

    def to_dict(self) -> dict:
        d = self.cluster.to_dict()
        d["confidence"] = round(self.confidence, 2)
        d["diversity"] = self.diversity
        d["is_submittable"] = self.is_submittable
        if self.rejection_reason:
            d["rejection_reason"] = self.rejection_reason
        return d


class ScoringEngine:
    """
    Scores contradiction clusters and applies submission thresholds.

    Every score derives from graph structure. No heuristics.
    No human judgment. No probabilistic models.
    """

    def score_cluster(
        self,
        cluster: ContradictionCluster,
        graph: ConsistencyGraph,
    ) -> ScoredCluster:
        """
        Score a single contradiction cluster.

        Args:
            cluster: A ContradictionCluster from the detector.
            graph: The parent ConsistencyGraph.

        Returns:
            ScoredCluster with confidence, diversity, and submission eligibility.
        """
        # Collect all edges involved in this cluster
        cluster_node_set = set(cluster.nodes_involved)
        all_edges = graph.get_security_edges()

        # Find edges within this cluster
        cluster_edges = [
            e for e in all_edges
            if e.source_id in cluster_node_set and e.target_id in cluster_node_set
        ]

        if not cluster_edges:
            return ScoredCluster(
                cluster=cluster,
                confidence=0.0,
                diversity=0,
                is_submittable=False,
                rejection_reason="No security edges in cluster",
            )

        # Compute effective weight for each edge
        effective_weights = []
        for edge in cluster_edges:
            ew = edge.weight * edge.edge_confidence
            effective_weights.append(ew)

        # Sum effective weights
        total_weight = sum(effective_weights)

        # Count distinct edge types (diversity)
        edge_types = set(e.edge_type for e in cluster_edges)
        diversity = len(edge_types)

        # Apply diversity bonus
        diversity_multiplier = 1.0 + DIVERSITY_BONUS_FACTOR * (diversity - 1)

        # Final confidence (capped)
        raw_confidence = total_weight * diversity_multiplier
        confidence = min(raw_confidence, CONFIDENCE_CAP)

        # Submission gate check
        is_submittable = True
        rejection_reason = None

        if confidence < SUBMISSION_MIN_CONFIDENCE:
            is_submittable = False
            rejection_reason = (
                f"Confidence {confidence:.1f} below minimum {SUBMISSION_MIN_CONFIDENCE}"
            )

        if diversity < SUBMISSION_MIN_EDGE_TYPES:
            is_submittable = False
            existing = rejection_reason or ""
            rejection_reason = (
                f"{existing}; Only {diversity} edge type(s), "
                f"need {SUBMISSION_MIN_EDGE_TYPES}"
            ).lstrip("; ")

        return ScoredCluster(
            cluster=cluster,
            confidence=confidence,
            diversity=diversity,
            is_submittable=is_submittable,
            rejection_reason=rejection_reason,
        )

    def score_all(
        self,
        clusters: list[ContradictionCluster],
        graph: ConsistencyGraph,
    ) -> list[ScoredCluster]:
        """
        Score all clusters and return sorted results.

        Args:
            clusters: List of ContradictionCluster objects.
            graph: The parent ConsistencyGraph.

        Returns:
            List of ScoredCluster objects, sorted by confidence descending.
        """
        scored = [self.score_cluster(c, graph) for c in clusters]
        scored.sort(key=lambda s: s.confidence, reverse=True)
        return scored


def compute_reproducibility(
    run_results: list[list[ScoredCluster]],
    threshold: float = REPRODUCIBILITY_THRESHOLD,
) -> dict:
    """
    Compute reproducibility score across multiple runs.

    Reproducibility measures whether the same contradiction clusters
    appear consistently across repeated executions.

    Args:
        run_results: List of scored cluster lists from multiple runs.
        threshold: Minimum reproduction rate for a finding to be "reproducible".

    Returns:
        Dict with reproducibility statistics.
    """
    total_runs = len(run_results)
    if total_runs == 0:
        return {"overall_reproducibility": 0.0, "stable_clusters": []}

    # Normalize cluster identifiers across runs
    # Use node-set hashing as cluster fingerprint
    all_fingerprints = []
    for run in run_results:
        run_fingerprints = []
        for scored in run:
            # Create a stable fingerprint from node set
            node_key = "|".join(sorted(scored.cluster.nodes_involved))
            fp = hashlib.sha256(node_key.encode()).hexdigest()[:12]
            run_fingerprints.append(fp)
        all_fingerprints.append(set(run_fingerprints))

    # Count how often each fingerprint appears
    fingerprint_counts = Counter()
    for fps in all_fingerprints:
        for fp in fps:
            fingerprint_counts[fp] += 1

    # Compute reproducibility for each cluster
    stable_clusters = []
    for fp, count in fingerprint_counts.items():
        rate = count / total_runs
        if rate >= threshold:
            stable_clusters.append({
                "fingerprint": fp,
                "reproduction_rate": round(rate, 2),
                "appears_in_runs": count,
                "total_runs": total_runs,
            })

    # Overall reproducibility
    if all_fingerprints:
        total_clusters = sum(len(fps) for fps in all_fingerprints)
        stable_count = sum(1 for fps in all_fingerprints
                         for fp in fps if fingerprint_counts.get(fp, 0) / total_runs >= threshold)
        overall = stable_count / max(total_clusters, 1)
    else:
        overall = 0.0

    return {
        "overall_reproducibility": round(overall, 2),
        "stable_clusters": sorted(stable_clusters, key=lambda x: x["reproduction_rate"], reverse=True),
        "runs_performed": total_runs,
        "threshold": threshold,
    }

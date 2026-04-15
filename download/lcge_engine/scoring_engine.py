"""
scoring_engine.py — Step 7: Instability Scoring (v1.2)

Replaces binary contradiction scoring with 4-component instability scoring.

v1.2 changes:
    - Dual global score: peak (max) + mean (average) across families
    - Lightweight normalization stub: score / NORMALIZATION_DIVISOR
    - Normalized scores enable cross-task comparability (full calibration in v2)

Scoring:
    instability_score = weighted_sum(all components), cap at 10.0
        = policy_score * 3.5
        + reasoning_score * 2.5  (raised in v1.2)
        + knowledge_score * 2.0
        + formatting_score * 1.5
"""

from typing import Optional
from collections import Counter, defaultdict

import numpy as np

from .graph_constructor import ConsistencyGraph
from .instability_classifier import InstabilityCluster
from .config import (
    INSTABILITY_WEIGHTS,
    INSTABILITY_SCORE_CAP,
    INSTABILITY_TYPES,
    NORMALIZATION_DIVISOR,
)


class ScoredInstability:
    """A scored instability cluster with confidence metadata."""

    def __init__(
        self,
        cluster: InstabilityCluster,
        confidence: float,
        is_significant: bool,
        rejection_reason: Optional[str] = None,
    ):
        self.cluster = cluster
        self.confidence = confidence
        self.is_significant = is_significant
        self.rejection_reason = rejection_reason

    def to_dict(self) -> dict:
        d = self.cluster.to_dict()
        d["confidence"] = round(self.confidence, 2)
        d["is_significant"] = self.is_significant
        if self.rejection_reason:
            d["rejection_reason"] = self.rejection_reason
        return d


class ScoringEngine:
    """
    Scores instability clusters and produces global metrics.

    v1.2:
        - global_instability_peak = max score across significant clusters
        - global_instability_mean = mean score across significant clusters
        - normalized_peak = peak / NORMALIZATION_DIVISOR (0-1 scale)
        - normalized_mean = mean / NORMALIZATION_DIVISOR (0-1 scale)
    """

    # Threshold for "significant" instability
    SIGNIFICANCE_THRESHOLD = 1.0

    def score_cluster(
        self, cluster: InstabilityCluster, graph: ConsistencyGraph
    ) -> ScoredInstability:
        """Score a single instability cluster."""
        total_score = cluster.total_score

        # Confidence = how much of the cap is used
        confidence = total_score / INSTABILITY_SCORE_CAP if INSTABILITY_SCORE_CAP > 0 else 0.0

        # Significance check
        is_significant = total_score >= self.SIGNIFICANCE_THRESHOLD
        rejection_reason = None

        if not is_significant:
            rejection_reason = (
                f"Score {total_score:.2f} below significance threshold "
                f"{self.SIGNIFICANCE_THRESHOLD}"
            )

        return ScoredInstability(
            cluster=cluster,
            confidence=confidence,
            is_significant=is_significant,
            rejection_reason=rejection_reason,
        )

    def score_all(
        self, clusters: list[InstabilityCluster], graph: ConsistencyGraph
    ) -> list[ScoredInstability]:
        """Score all clusters and return sorted results."""
        scored = [self.score_cluster(c, graph) for c in clusters]
        scored.sort(key=lambda s: s.cluster.total_score, reverse=True)
        return scored

    def compute_global_metrics(
        self, scored_clusters: list[ScoredInstability]
    ) -> dict:
        """
        Compute global instability metrics across all clusters.

        v1.2: Returns both peak and mean global scores, plus normalized versions.

        Returns:
            dict with:
                - global_instability_peak: float (0-10) — max score
                - global_instability_mean: float (0-10) — mean score
                - normalized_peak: float (0-1) — peak / divisor
                - normalized_mean: float (0-1) — mean / divisor
                - dominant_failure_mode: str
                - instability_type_counts: dict
                - component_averages: dict
        """
        if not scored_clusters:
            return {
                "global_instability_peak": 0.0,
                "global_instability_mean": 0.0,
                "normalized_peak": 0.0,
                "normalized_mean": 0.0,
                "dominant_failure_mode": "stable",
                "instability_type_counts": {"stable": 1},
                "component_averages": {},
            }

        significant = [s for s in scored_clusters if s.is_significant]

        if not significant:
            return {
                "global_instability_peak": 0.0,
                "global_instability_mean": 0.0,
                "normalized_peak": 0.0,
                "normalized_mean": 0.0,
                "dominant_failure_mode": "stable",
                "instability_type_counts": {"stable": len(scored_clusters)},
                "component_averages": {},
            }

        # v1.2: Dual global score — peak AND mean
        scores = [s.cluster.total_score for s in significant]
        global_peak = max(scores)
        global_mean = sum(scores) / len(scores)

        # v1.2: Lightweight normalization (full calibration layer in v2)
        normalized_peak = global_peak / NORMALIZATION_DIVISOR
        normalized_mean = global_mean / NORMALIZATION_DIVISOR

        # Dominant failure mode = weighted by score
        type_weighted = defaultdict(float)
        for s in significant:
            type_weighted[s.cluster.instability_type] += s.cluster.total_score

        dominant_type = max(type_weighted, key=type_weighted.get) if type_weighted else "stable"

        # Type counts
        type_counts = Counter(s.cluster.instability_type for s in scored_clusters)

        # Component averages across significant clusters
        component_keys = ["policy", "reasoning", "knowledge", "formatting"]
        component_avgs = {}
        for key in component_keys:
            values = [s.cluster.component_scores.get(key, 0.0) for s in significant]
            if values:
                component_avgs[key] = round(float(sum(values) / len(values)), 4)
            else:
                component_avgs[key] = 0.0

        return {
            "global_instability_peak": round(global_peak, 2),
            "global_instability_mean": round(global_mean, 2),
            "normalized_peak": round(normalized_peak, 4),
            "normalized_mean": round(normalized_mean, 4),
            "dominant_failure_mode": dominant_type,
            "instability_type_counts": dict(type_counts),
            "component_averages": component_avgs,
        }


def normalize_score(score: float, task_type_stats: Optional[dict] = None) -> float:
    """
    Normalize a raw instability score to 0-1 range.

    v1.2: Simple stub — score / NORMALIZATION_DIVISOR.
    v2 will use task_type_stats with mean/std for z-score normalization.

    Args:
        score: Raw instability score (0-10).
        task_type_stats: Unused in v1.2 (reserved for v2 calibration).

    Returns:
        Normalized score in 0-1 range.
    """
    return score / NORMALIZATION_DIVISOR


def compute_reproducibility(
    run_results: list[list[ScoredInstability]],
    threshold: float = 0.6,
) -> dict:
    """
    Compute reproducibility across multiple runs.

    Uses instability_type as the fingerprint for stability measurement.
    """
    import hashlib

    total_runs = len(run_results)
    if total_runs == 0:
        return {"overall_reproducibility": 0.0, "stable_types": []}

    type_per_run = []
    for run in run_results:
        types = set()
        for scored in run:
            if scored.is_significant:
                types.add(scored.cluster.instability_type)
        type_per_run.append(types)

    # Check which types appear consistently
    all_types = set()
    for types in type_per_run:
        all_types.update(types)

    stable_types = []
    for t in all_types:
        count = sum(1 for types in type_per_run if t in types)
        rate = count / total_runs
        if rate >= threshold:
            stable_types.append({
                "type": t,
                "reproduction_rate": round(rate, 2),
            })

    return {
        "overall_reproducibility": round(
            len(stable_types) / max(len(all_types), 1), 2
        ),
        "stable_types": stable_types,
        "runs_performed": total_runs,
    }

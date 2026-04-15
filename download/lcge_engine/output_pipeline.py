"""
output_pipeline.py — Step 8: Output Pipeline (v1.2)

Produces the final instability classification report.

v1.2 STRICT output format:
{
    "task": "...",
    "instability_map": [
        {
            "family": "...",
            "instability_type": "...",
            "score": float,
            "top_trigger": "POLICY_SHIFT | REASONING_SHIFT | KNOWLEDGE_SHIFT | FORMAT_SHIFT",
            "component_breakdown": {
                "policy": float,
                "reasoning": float,
                "knowledge": float,
                "formatting": float
            }
        }
    ],
    "global_instability_peak": float,
    "global_instability_mean": float,
    "normalized_peak": float,
    "normalized_mean": float,
    "dominant_failure_mode": "..."
}
"""

import json
from datetime import datetime, timezone
from typing import Optional

from .graph_constructor import ConsistencyGraph
from .instability_classifier import InstabilityCluster
from .scoring_engine import ScoredInstability


class InstabilityReport:
    """The final output of the LCGE v1.2 engine."""

    def __init__(
        self,
        engine_version: str,
        task: str,
        seed_prompt: str,
        instability_map: list[dict],
        global_instability_peak: float,
        global_instability_mean: float,
        normalized_peak: float,
        normalized_mean: float,
        dominant_failure_mode: str,
        graph: Optional[ConsistencyGraph] = None,
        scored_clusters: Optional[list[ScoredInstability]] = None,
        component_averages: Optional[dict] = None,
        type_counts: Optional[dict] = None,
        reproducibility: Optional[dict] = None,
    ):
        self.engine_version = engine_version
        self.task = task
        self.seed_prompt = seed_prompt
        self.instability_map = instability_map
        self.global_instability_peak = global_instability_peak
        self.global_instability_mean = global_instability_mean
        self.normalized_peak = normalized_peak
        self.normalized_mean = normalized_mean
        self.dominant_failure_mode = dominant_failure_mode
        self.graph = graph
        self.scored_clusters = scored_clusters or []
        self.component_averages = component_averages or {}
        self.type_counts = type_counts or {}
        self.reproducibility = reproducibility
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to the STRICT output format per v1.2 spec."""
        result = {
            "task": self.task,
            "instability_map": self.instability_map,
            "global_instability_peak": self.global_instability_peak,
            "global_instability_mean": self.global_instability_mean,
            "normalized_peak": self.normalized_peak,
            "normalized_mean": self.normalized_mean,
            "dominant_failure_mode": self.dominant_failure_mode,
        }

        return result

    def to_full_dict(self) -> dict:
        """Serialize full report with all diagnostic data."""
        result = self.to_dict()

        result["engine"] = {
            "name": "LLM Consistency Graph Engine",
            "version": self.engine_version,
            "definition": "Prompt Transformation -> Behavioral State Mapping Engine",
            "timestamp": self.timestamp,
        }

        result["input"] = {
            "task": self.task,
            "seed_prompt": self.seed_prompt,
        }

        # Component breakdown
        result["component_breakdown"] = self.component_averages

        # Type distribution
        result["type_distribution"] = self.type_counts

        # Significant findings
        significant = [s for s in self.scored_clusters if s.is_significant]
        rejected = [s for s in self.scored_clusters if not s.is_significant]

        result["findings"] = {
            "has_instability": len(significant) > 0,
            "significant_count": len(significant),
            "stable_count": len(rejected),
        }

        # Detailed significant findings
        if significant:
            result["significant_clusters"] = [s.to_dict() for s in significant]

        # Reproducibility
        if self.reproducibility:
            result["reproducibility"] = self.reproducibility

        return result

    def to_json(self, indent: int = 2, full: bool = False) -> str:
        """Serialize to JSON string."""
        if full:
            data = self.to_full_dict()
        else:
            data = self.to_dict()
        return json.dumps(data, indent=indent, default=str)


def generate_report(
    task: str,
    seed_prompt: str,
    graph: ConsistencyGraph,
    scored_clusters: list[ScoredInstability],
    global_metrics: dict,
    reproducibility: Optional[dict] = None,
) -> InstabilityReport:
    """
    Generate the final instability classification report.

    Args:
        task: The task being tested.
        seed_prompt: The original seed prompt.
        graph: The constructed ConsistencyGraph.
        scored_clusters: List of scored instability clusters.
        global_metrics: Global metrics from ScoringEngine.
        reproducibility: Optional reproducibility data.

    Returns:
        An InstabilityReport with strict output format.
    """
    from . import __version__

    # Build instability_map (strict format per v1.2 spec)
    instability_map = []
    for scored in scored_clusters:
        cluster = scored.cluster
        entry = {
            "family": cluster.family_id,
            "instability_type": cluster.instability_type,
            "score": round(cluster.total_score, 2),
            "top_trigger": cluster.evidence.get("top_trigger", ""),
            "component_breakdown": {
                k: round(v, 4) for k, v in cluster.component_scores.items()
            },
        }
        instability_map.append(entry)

    # If no clusters found, add a stable entry
    if not instability_map:
        instability_map.append({
            "family": "default",
            "instability_type": "stable",
            "score": 0.0,
            "top_trigger": "",
            "component_breakdown": {"policy": 0.0, "reasoning": 0.0, "knowledge": 0.0, "formatting": 0.0},
        })

    report = InstabilityReport(
        engine_version=__version__,
        task=task,
        seed_prompt=seed_prompt,
        instability_map=instability_map,
        global_instability_peak=global_metrics["global_instability_peak"],
        global_instability_mean=global_metrics["global_instability_mean"],
        normalized_peak=global_metrics["normalized_peak"],
        normalized_mean=global_metrics["normalized_mean"],
        dominant_failure_mode=global_metrics["dominant_failure_mode"],
        graph=graph,
        scored_clusters=scored_clusters,
        component_averages=global_metrics.get("component_averages", {}),
        type_counts=global_metrics.get("instability_type_counts", {}),
        reproducibility=reproducibility,
    )

    return report

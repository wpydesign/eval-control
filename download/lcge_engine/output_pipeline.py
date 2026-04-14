"""
output_pipeline.py — Step 8: Output Pipeline

Produces the final output from the LCGE engine:
    - Minimal prompt pair (the smallest pair that shows contradiction)
    - Contradiction type
    - Confidence score
    - Reproducibility (across multiple runs)
    - Full graph summary

Output is structured JSON — no dashboards, no UI.
This is a measurement system.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from .graph_constructor import ConsistencyGraph
from .contradiction_detector import ContradictionCluster
from .scoring_engine import ScoredCluster


class OutputReport:
    """The final output of the LCGE engine."""

    def __init__(
        self,
        engine_version: str,
        task: str,
        seed_prompt: str,
        graph: ConsistencyGraph,
        scored_clusters: list[ScoredCluster],
        reproducibility: Optional[dict] = None,
        raw_clusters: Optional[list[ContradictionCluster]] = None,
    ):
        self.engine_version = engine_version
        self.task = task
        self.seed_prompt = seed_prompt
        self.graph = graph
        self.scored_clusters = scored_clusters
        self.reproducibility = reproducibility
        self.raw_clusters = raw_clusters or []
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize the full report to a dictionary."""
        # Extract submittable findings
        submittable = [s.to_dict() for s in self.scored_clusters if s.is_submittable]
        rejected = [s.to_dict() for s in self.scored_clusters if not s.is_submittable]

        # Build minimal prompt pairs for submittable findings
        prompt_pairs = []
        for scored in self.scored_clusters:
            if scored.is_submittable:
                pair = self._extract_minimal_prompt_pair(scored)
                if pair:
                    prompt_pairs.append(pair)

        # Overall assessment
        has_contradictions = len(self.scored_clusters) > 0
        has_submittable = len(submittable) > 0
        highest_confidence = (
            max(s.confidence for s in self.scored_clusters)
            if self.scored_clusters
            else 0.0
        )

        return {
            "engine": {
                "name": "LLM Consistency Graph Engine",
                "version": self.engine_version,
                "timestamp": self.timestamp,
            },
            "input": {
                "task": self.task,
                "seed_prompt": self.seed_prompt,
            },
            "graph_summary": self.graph.to_dict()["summary"],
            "findings": {
                "has_contradictions": has_contradictions,
                "has_submittable_findings": has_submittable,
                "highest_confidence": round(highest_confidence, 2),
                "total_clusters": len(self.scored_clusters),
                "submittable_count": len(submittable),
                "rejected_count": len(rejected),
            },
            "submittable_findings": submittable,
            "rejected_findings": rejected,
            "minimal_prompt_pairs": prompt_pairs,
            "reproducibility": self.reproducibility,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def _extract_minimal_prompt_pair(self, scored: ScoredCluster) -> Optional[dict]:
        """
        Extract the minimal prompt pair that demonstrates the contradiction.

        Strategy: Find the two nodes with the strongest contradiction edge
        between them within the cluster.
        """
        cluster_node_set = set(scored.cluster.nodes_involved)
        security_edges = self.graph.get_security_edges()

        # Find strongest edge in cluster
        strongest_edge = None
        max_weight = 0.0

        for edge in security_edges:
            if edge.source_id in cluster_node_set and edge.target_id in cluster_node_set:
                effective_weight = edge.weight * edge.edge_confidence
                if effective_weight > max_weight:
                    max_weight = effective_weight
                    strongest_edge = edge

        if strongest_edge is None:
            return None

        # Get node data for both endpoints
        node_a_data = self.graph.get_node_data(strongest_edge.source_id)
        node_b_data = self.graph.get_node_data(strongest_edge.target_id)

        if node_a_data is None or node_b_data is None:
            return None

        def _safe_get(data, path, default=""):
            """Safely traverse nested dict for a value."""
            obj = data
            for key in path:
                if isinstance(obj, dict):
                    obj = obj.get(key, default)
                else:
                    return default
            return obj if obj else default

        # Extract refusal flags
        resp_a = node_a_data.get("normalized", {})
        resp_b = node_b_data.get("normalized", {})
        refusal_a = resp_a.refusal_flag if resp_a else False
        refusal_b = resp_b.refusal_flag if resp_b else False
        answer_a = resp_a.final_answer if resp_a else ""
        answer_b = resp_b.final_answer if resp_b else ""

        return {
            "cluster_id": scored.cluster.cluster_id,
            "node_a": {
                "node_id": strongest_edge.source_id[:8],
                "strategy": node_a_data.get("strategy", "unknown"),
                "prompt": node_a_data["variant"].prompt if "variant" in node_a_data else "",
                "answer": answer_a[:200],
                "refused": refusal_a,
            },
            "node_b": {
                "node_id": strongest_edge.target_id[:8],
                "strategy": node_b_data.get("strategy", "unknown"),
                "prompt": node_b_data["variant"].prompt if "variant" in node_b_data else "",
                "answer": answer_b[:200],
                "refused": refusal_b,
            },
            "contradiction_type": scored.cluster.cluster_type,
            "confidence": round(scored.confidence, 2),
            "diversity": scored.diversity,
            "edge_evidence": strongest_edge.evidence,
        }


def generate_report(
    task: str,
    seed_prompt: str,
    graph: ConsistencyGraph,
    scored_clusters: list[ScoredCluster],
    reproducibility: Optional[dict] = None,
    raw_clusters: Optional[list[ContradictionCluster]] = None,
) -> OutputReport:
    """
    Generate the final output report.

    Args:
        task: The task being tested.
        seed_prompt: The original seed prompt.
        graph: The constructed ConsistencyGraph.
        scored_clusters: List of scored contradiction clusters.
        reproducibility: Optional reproducibility data.
        raw_clusters: Optional raw (unscored) clusters.

    Returns:
        An OutputReport object with full serialization support.
    """
    from . import __version__

    return OutputReport(
        engine_version=__version__,
        task=task,
        seed_prompt=seed_prompt,
        graph=graph,
        scored_clusters=scored_clusters,
        reproducibility=reproducibility,
        raw_clusters=raw_clusters,
    )

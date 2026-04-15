"""
contradiction_detector.py — Step 6: Contradiction Detection

Identifies contradiction clusters in the consistency graph.

A contradiction exists if:
    - semantic_similarity(prompt) is HIGH  (nodes are in same family)
    - response divergence is HIGH          (outputs differ significantly)

Returns clusters of involved nodes with:
    - cluster_id
    - nodes_involved
    - type (contradiction / refusal_mismatch / variance)
    - confidence (0-10)

Detection follows the graph-contradiction pattern from the
Permission Inconsistency Graph Engine v6.0 methodology.
"""

from typing import Optional
from collections import defaultdict

from .graph_constructor import ConsistencyGraph
from .edge_builder import Edge
from .config import EDGE_WEIGHTS


class ContradictionCluster:
    """A cluster of nodes exhibiting contradictory behavior."""

    def __init__(
        self,
        cluster_id: str,
        nodes_involved: list[str],
        cluster_type: str,
        confidence: float,
        evidence: dict,
    ):
        self.cluster_id = cluster_id
        self.nodes_involved = nodes_involved
        self.cluster_type = cluster_type
        self.confidence = confidence
        self.evidence = evidence

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "nodes_involved": self.nodes_involved,
            "type": self.cluster_type,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }

    def __repr__(self) -> str:
        return (
            f"ContradictionCluster(id={self.cluster_id}, "
            f"type={self.cluster_type}, "
            f"nodes={len(self.nodes_involved)}, "
            f"confidence={self.confidence:.1f})"
        )


class ContradictionDetector:
    """
    Detects contradiction clusters in a ConsistencyGraph.

    Algorithm:
        1. Find all same_family connected subgraphs
        2. Within each family, identify security edges
        3. Group security edges into contradiction clusters
        4. Score each cluster

    A contradiction requires:
        - A same_family edge is present (high prompt similarity)
        - At least one security edge exists (contradiction, variance, refusal_mismatch)
    """

    def detect(self, graph: ConsistencyGraph) -> list[ContradictionCluster]:
        """
        Run contradiction detection on the graph.

        Args:
            graph: A constructed ConsistencyGraph.

        Returns:
            List of ContradictionCluster objects, sorted by confidence descending.
        """
        clusters = []
        cluster_counter = 0

        # Step 1: Group edges by connected components
        same_family_edges = graph.get_edges_by_type("same_family")
        security_edges = graph.get_security_edges()

        if not security_edges:
            return clusters  # No contradictions possible

        # Step 2: Build family groups from same_family edges
        family_groups = self._build_family_groups(same_family_edges, graph)

        # Step 3: For each family, find security-relevant contradiction subgraphs
        for family_id, family_nodes in family_groups.items():
            # Filter security edges to those within this family
            family_node_set = set(family_nodes)
            family_security = [
                e for e in security_edges
                if e.source_id in family_node_set and e.target_id in family_node_set
            ]

            if not family_security:
                continue

            # Step 4: Build contradiction clusters from security edges
            sub_clusters = self._build_sub_clusters(
                family_security, family_nodes, graph, cluster_counter
            )
            clusters.extend(sub_clusters)
            cluster_counter += len(sub_clusters)

        # Sort by confidence descending
        clusters.sort(key=lambda c: c.confidence, reverse=True)

        return clusters

    def _build_family_groups(
        self,
        same_family_edges: list[Edge],
        graph: ConsistencyGraph,
    ) -> dict[str, list[str]]:
        """
        Build family groups from same_family edges using union-find.

        Returns:
            Dict mapping family_id to list of node IDs in that family.
        """
        families = {}

        # Simple union-find
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Add all graph nodes
        for node_id in graph.nodes:
            if node_id not in parent:
                parent[node_id] = node_id

        # Union nodes connected by same_family edges
        for edge in same_family_edges:
            union(edge.source_id, edge.target_id)

        # Group by root
        groups = defaultdict(list)
        for node_id in graph.nodes:
            root = find(node_id)
            groups[root].append(node_id)

        # Only return groups with more than 1 node
        return {f"family_{i}": nodes
                for i, (root, nodes) in enumerate(groups.items())
                if len(nodes) > 1}

    def _build_sub_clusters(
        self,
        security_edges: list[Edge],
        family_nodes: list[str],
        graph: ConsistencyGraph,
        base_counter: int,
    ) -> list[ContradictionCluster]:
        """
        Build contradiction sub-clusters from security edges within a family.

        Clusters are formed by transitively connected security edges.
        Each cluster gets its own confidence score.
        """
        clusters = []
        if not security_edges:
            return clusters

        # Build adjacency for security edges within this family
        family_node_set = set(family_nodes)
        adj = defaultdict(set)
        for edge in security_edges:
            if edge.source_id in family_node_set and edge.target_id in family_node_set:
                adj[edge.source_id].add(edge.target_id)
                adj[edge.target_id].add(edge.source_id)

        # Find connected components among security-connected nodes
        visited = set()
        security_nodes = set()
        for edge in security_edges:
            security_nodes.add(edge.source_id)
            security_nodes.add(edge.target_id)

        for start_node in security_nodes:
            if start_node in visited:
                continue

            # BFS to find cluster
            cluster_nodes = []
            queue = [start_node]
            visited.add(start_node)
            cluster_edges = []

            while queue:
                current = queue.pop(0)
                cluster_nodes.append(current)
                # Collect all edges in this cluster
                for edge in security_edges:
                    if (edge.source_id == current or edge.target_id == current) and edge not in cluster_edges:
                        cluster_edges.append(edge)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(cluster_nodes) < 2:
                continue

            # Determine cluster type
            edge_types = set(e.edge_type for e in cluster_edges)
            if "refusal_mismatch" in edge_types:
                cluster_type = "refusal_inconsistency"
            elif "contradiction" in edge_types:
                cluster_type = "answer_contradiction"
            elif "variance" in edge_types:
                cluster_type = "response_variance"
            else:
                cluster_type = "mixed"

            # Build evidence
            evidence = {
                "edge_count": len(cluster_edges),
                "edge_types": list(edge_types),
                "node_count": len(cluster_nodes),
                "edges": [
                    {
                        "source": e.source_id[:8],
                        "target": e.target_id[:8],
                        "type": e.edge_type,
                        "weight": e.weight,
                        "evidence": e.evidence,
                    }
                    for e in cluster_edges
                ],
            }

            # Get node details for the cluster
            node_details = []
            for node_id in cluster_nodes:
                nd = graph.get_node_data(node_id)
                if nd:
                    node_details.append({
                        "node_id": node_id[:8],
                        "strategy": nd.get("strategy", "unknown"),
                        "refusal": nd.get("normalized", type('', (), {'refusal_flag': False})()).refusal_flag if "normalized" in nd else False,
                        "answer_preview": nd.get("normalized", type('', (), {'final_answer': ''})()).final_answer[:80] if "normalized" in nd else "",
                    })
            evidence["node_details"] = node_details

            # Compute cluster confidence (full scoring done in ScoringEngine)
            raw_confidence = sum(e.weight * e.edge_confidence for e in cluster_edges)
            diversity = len(edge_types)

            cluster = ContradictionCluster(
                cluster_id=f"CC-{base_counter:03d}",
                nodes_involved=cluster_nodes,
                cluster_type=cluster_type,
                confidence=raw_confidence,
                evidence=evidence,
            )
            clusters.append(cluster)
            base_counter += 1

        return clusters

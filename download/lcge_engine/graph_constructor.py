"""
graph_constructor.py — Step 5: Graph Construction

Assembles nodes and edges into a queryable graph structure.
Uses networkx for graph operations.

The graph is the single source of truth for all findings.
Every contradiction, every confidence score, every detection
derives from graph structure — never from raw response output.

Graph structure:
    Nodes: endpoint variants (prompt + response pairs)
    Edges: security-relevant relationships between variants
"""

import networkx as nx
from typing import Optional

from .prompt_input_layer import PromptVariant
from .normalization_layer import NormalizedResponse
from .edge_builder import Edge


class ConsistencyGraph:
    """
    The core graph structure for LLM consistency analysis.

    This is NOT a knowledge graph. It is NOT a social graph.
    It is a contradiction detection graph — every edge represents
    a security-relevant relationship between LLM output variants.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self._node_data = {}   # node_id -> {variant, response, normalized}
        self._edge_data = []   # list of Edge objects

    @property
    def nodes(self) -> list[str]:
        """Return list of node IDs."""
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[Edge]:
        """Return list of Edge objects."""
        return self._edge_data

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def add_node(
        self,
        variant: PromptVariant,
        normalized_response: NormalizedResponse,
    ):
        """
        Add a node to the graph.

        Args:
            variant: The prompt variant for this node.
            normalized_response: The normalized response data.
        """
        node_id = variant.node_id
        self.graph.add_node(node_id)
        self._node_data[node_id] = {
            "variant": variant,
            "normalized": normalized_response,
            "strategy": variant.strategy,
            "task": variant.task,
        }

    def add_edge(self, edge: Edge):
        """
        Add an edge to the graph.

        Args:
            edge: An Edge object.
        """
        self.graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
        self._edge_data.append(edge)

    def get_node_data(self, node_id: str) -> Optional[dict]:
        """Retrieve metadata for a node."""
        return self._node_data.get(node_id)

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node."""
        return [e for e in self._edge_data
                if e.source_id == node_id or e.target_id == node_id]

    def get_edges_by_type(self, edge_type: str) -> list[Edge]:
        """Get all edges of a specific type."""
        return [e for e in self._edge_data if e.edge_type == edge_type]

    def get_edges_between(self, node_a: str, node_b: str) -> list[Edge]:
        """Get all edges between two specific nodes."""
        return [e for e in self._edge_data
                if (e.source_id == node_a and e.target_id == node_b)
                or (e.source_id == node_b and e.target_id == node_a)]

    def get_security_edges(self) -> list[Edge]:
        """
        Get all security-relevant edges (non-grouping).

        Security edges: contradiction, variance, refusal_mismatch
        Non-security: same_family (weight = 0.0)
        """
        return [e for e in self._edge_data if e.weight > 0.0]

    def get_connected_components(self) -> list[list[str]]:
        """Get connected components of the graph."""
        return [list(c) for c in nx.connected_components(self.graph)]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbors of a node."""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def to_dict(self) -> dict:
        """Serialize graph to dictionary for output."""
        nodes_list = []
        for node_id in self.graph.nodes():
            nd = self._node_data.get(node_id, {})
            nodes_list.append({
                "node_id": node_id,
                "strategy": nd.get("strategy", "unknown"),
                "task": nd.get("task", ""),
                "prompt": nd.get("variant", type('', (), {'prompt': ''})()).prompt if "variant" in nd else "",
                "final_answer": nd.get("normalized", type('', (), {'final_answer': ''})()).final_answer if "normalized" in nd else "",
                "refusal_flag": nd.get("normalized", type('', (), {'refusal_flag': False})()).refusal_flag if "normalized" in nd else False,
            })

        edges_list = [e.to_dict() for e in self._edge_data]

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "summary": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "security_edge_count": len(self.get_security_edges()),
                "connected_components": len(self.get_connected_components()),
            },
        }

    def __repr__(self) -> str:
        return (
            f"ConsistencyGraph(nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"security_edges={len(self.get_security_edges())})"
        )


def build_graph(
    variants: list[PromptVariant],
    normalized_responses: list[NormalizedResponse],
    edges: list[Edge],
) -> ConsistencyGraph:
    """
    Construct a ConsistencyGraph from variants, responses, and edges.

    Args:
        variants: List of PromptVariant objects.
        normalized_responses: Corresponding NormalizedResponse objects.
        edges: List of Edge objects.

    Returns:
        A fully constructed ConsistencyGraph.
    """
    graph = ConsistencyGraph()

    # Add all nodes
    response_map = {r.node_id: r for r in normalized_responses}
    for variant in variants:
        normalized = response_map.get(variant.node_id)
        if normalized:
            graph.add_node(variant, normalized)

    # Add all edges
    for edge in edges:
        # Only add edge if both nodes exist in graph
        if edge.source_id in graph.graph and edge.target_id in graph.graph:
            graph.add_edge(edge)

    return graph

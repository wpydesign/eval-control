"""
graph_constructor.py — Step 5: Graph Construction (v1.1)

Assembles nodes and edges into a queryable graph structure.
Updated node schema with v1.1 fields: reasoning_trace, format_signature, semantic_family_id.
"""

import networkx as nx
from typing import Optional

from .prompt_input_layer import PromptVariant
from .normalization_layer import NormalizedResponse
from .edge_builder import Edge


class ConsistencyGraph:
    """
    The core graph structure for LLM behavioral instability analysis.

    v1.1: Each node carries the full v1.1 schema including
    reasoning_trace, format_signature, and semantic_family_id.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self._node_data = {}
        self._edge_data = []

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[Edge]:
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
        semantic_family_id: str = "default",
    ):
        node_id = variant.node_id
        self.graph.add_node(node_id)
        self._node_data[node_id] = {
            "variant": variant,
            "normalized": normalized_response,
            "strategy": variant.strategy,
            "task": variant.task,
            "family_id": semantic_family_id,
        }

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
        self._edge_data.append(edge)

    def get_node_data(self, node_id: str) -> Optional[dict]:
        return self._node_data.get(node_id)

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        return [e for e in self._edge_data
                if e.source_id == node_id or e.target_id == node_id]

    def get_edges_by_type(self, edge_type: str) -> list[Edge]:
        return [e for e in self._edge_data if e.edge_type == edge_type]

    def get_edges_between(self, node_a: str, node_b: str) -> list[Edge]:
        return [e for e in self._edge_data
                if (e.source_id == node_a and e.target_id == node_b)
                or (e.source_id == node_b and e.target_id == node_a)]

    def get_instability_edges(self) -> list[Edge]:
        """Get all edges with non-zero weight (actual instability signals)."""
        return [e for e in self._edge_data if e.weight > 0.0]

    def get_connected_components(self) -> list[list[str]]:
        return [list(c) for c in nx.connected_components(self.graph)]

    def get_neighbors(self, node_id: str) -> list[str]:
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def to_dict(self) -> dict:
        nodes_list = []
        for node_id in self.graph.nodes():
            nd = self._node_data.get(node_id, {})
            normalized = nd.get("normalized")
            variant = nd.get("variant")
            nodes_list.append({
                "node_id": node_id,
                "strategy": nd.get("strategy", "unknown"),
                "task": nd.get("task", ""),
                "family_id": nd.get("family_id", "default"),
                "prompt": variant.prompt if variant else "",
                "final_answer": normalized.final_answer if normalized else "",
                "refusal_flag": normalized.refusal_flag if normalized else False,
                "format_signature": normalized.format_signature if normalized else "",
                "reasoning_trace_len": len(normalized.reasoning_trace) if normalized else 0,
            })

        edges_list = [e.to_dict() for e in self._edge_data]

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "summary": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "instability_edge_count": len(self.get_instability_edges()),
                "connected_components": len(self.get_connected_components()),
            },
        }

    def __repr__(self) -> str:
        return (
            f"ConsistencyGraph(nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"instability_edges={len(self.get_instability_edges())})"
        )


def build_graph(
    variants: list[PromptVariant],
    normalized_responses: list[NormalizedResponse],
    edges: list[Edge],
    semantic_family_id: str = "default",
) -> ConsistencyGraph:
    graph = ConsistencyGraph()

    response_map = {r.node_id: r for r in normalized_responses}
    for variant in variants:
        normalized = response_map.get(variant.node_id)
        if normalized:
            family_id = normalized.semantic_family_id or semantic_family_id
            graph.add_node(variant, normalized, family_id)

    for edge in edges:
        if edge.source_id in graph.graph and edge.target_id in graph.graph:
            graph.add_edge(edge)

    return graph

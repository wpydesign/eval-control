"""
instability_classifier.py — Step 6: Instability Classifier (v1.2)

Classifies LLM behavioral instability into typed categories:
    - policy_flip: refusal <-> answer transitions
    - reasoning_variance: different solution paths
    - knowledge_variance: factual disagreement
    - formatting_variance: structure changes affecting meaning
    - stable: no significant instability

v1.2 changes:
    - top_trigger now uses typed TriggerType (POLICY_SHIFT, REASONING_SHIFT,
      KNOWLEDGE_SHIFT, FORMAT_SHIFT) instead of free-text node ID
    - Reasoning dominance override: when reasoning_score > threshold AND
      reasoning_score > knowledge_score, reasoning wins dominance
    - _find_top_trigger() now classifies by signal source, not just edge weight
"""

from typing import Optional
from collections import defaultdict

import numpy as np

from .graph_constructor import ConsistencyGraph
from .edge_builder import Edge
from .config import (
    INSTABILITY_TYPES,
    POLICY_FLIP_SCORE_PER_PAIR,
    REASONING_DIVERGENCE_THRESHOLD,
    FORMAT_ENTROPY_THRESHOLD,
    TRIGGER_TYPES,
    REASONING_DOMINANCE_OVERRIDE_THRESHOLD,
)


# ============================================================
# TriggerType — typed trigger classification (v1.2)
# ============================================================

TRIGGER_TYPE_ENUM = [
    "POLICY_SHIFT",
    "REASONING_SHIFT",
    "KNOWLEDGE_SHIFT",
    "FORMAT_SHIFT",
]


def classify_trigger_type(component_scores: dict[str, float]) -> str:
    """
    Classify the top instability trigger from component scores.

    Returns one of: POLICY_SHIFT, REASONING_SHIFT, KNOWLEDGE_SHIFT, FORMAT_SHIFT.

    Logic: find the component with the highest raw score (not weighted).
    This identifies which SIGNAL SOURCE is driving the most instability,
    regardless of the weight applied to it.
    """
    type_map = {
        "policy": "POLICY_SHIFT",
        "reasoning": "REASONING_SHIFT",
        "knowledge": "KNOWLEDGE_SHIFT",
        "formatting": "FORMAT_SHIFT",
    }

    if not component_scores:
        return "FORMAT_SHIFT"

    # Find raw max (pre-weight)
    best_component = max(component_scores, key=component_scores.get)
    return type_map.get(best_component, "FORMAT_SHIFT")


class InstabilityCluster:
    """A cluster of nodes exhibiting a specific type of instability."""

    def __init__(
        self,
        cluster_id: str,
        family_id: str,
        instability_type: str,
        nodes_involved: list[str],
        component_scores: dict,
        total_score: float,
        evidence: dict,
    ):
        self.cluster_id = cluster_id
        self.family_id = family_id
        self.instability_type = instability_type
        self.nodes_involved = nodes_involved
        self.component_scores = component_scores
        self.total_score = total_score
        self.evidence = evidence

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "family_id": self.family_id,
            "instability_type": self.instability_type,
            "nodes_involved": self.nodes_involved,
            "component_scores": {k: round(v, 4) for k, v in self.component_scores.items()},
            "total_score": round(self.total_score, 2),
            "evidence": self.evidence,
        }

    def __repr__(self) -> str:
        return (
            f"InstabilityCluster(id={self.cluster_id}, "
            f"type={self.instability_type}, "
            f"nodes={len(self.nodes_involved)}, "
            f"score={self.total_score:.1f})"
        )


class InstabilityClassifier:
    """
    Classifies behavioral instability across prompt variant families.

    Algorithm:
        1. Group nodes by semantic_family_id
        2. For each family, compute 4 component instability scores:
           a. policy_instability — refusal/answer flip rate
           b. reasoning_instability — embedding divergence of reasoning traces
           c. knowledge_instability — entity-level answer disagreement
           d. formatting_instability — format signature diversity
        3. Classify dominant instability type (with reasoning override rule)
        4. If all components are below threshold, classify as "stable"
    """

    def classify(self, graph: ConsistencyGraph) -> list[InstabilityCluster]:
        """
        Run instability classification on the graph.

        Returns:
            List of InstabilityCluster objects, sorted by total_score descending.
        """
        clusters = []
        cluster_counter = 0

        # Step 1: Group nodes by semantic family
        families = self._group_by_family(graph)

        if not families:
            return clusters

        # Step 2: Classify each family
        for family_id, family_nodes in families.items():
            if len(family_nodes) < 2:
                continue

            component_scores = self._compute_component_scores(graph, family_nodes)

            # Step 3: Determine dominant type (v1.2: with reasoning override)
            instability_type = self._classify_type(component_scores)

            # Step 4: Compute total score (weighted sum, capped)
            from .config import INSTABILITY_WEIGHTS, INSTABILITY_SCORE_CAP
            total = sum(
                component_scores.get(k, 0.0) * INSTABILITY_WEIGHTS.get(k, 0.0)
                for k in INSTABILITY_WEIGHTS
            )
            total = min(total, INSTABILITY_SCORE_CAP)

            # Build evidence (v1.2: includes typed top_trigger)
            evidence = self._build_evidence(graph, family_nodes, component_scores)

            cluster = InstabilityCluster(
                cluster_id=f"IC-{cluster_counter:03d}",
                family_id=family_id,
                instability_type=instability_type,
                nodes_involved=family_nodes,
                component_scores=component_scores,
                total_score=total,
                evidence=evidence,
            )
            clusters.append(cluster)
            cluster_counter += 1

        clusters.sort(key=lambda c: c.total_score, reverse=True)
        return clusters

    def _group_by_family(self, graph: ConsistencyGraph) -> dict[str, list[str]]:
        """Group nodes by semantic_family_id."""
        families = defaultdict(list)
        for node_id in graph.nodes:
            nd = graph.get_node_data(node_id)
            family_id = nd.get("family_id", "default") if nd else "default"
            families[family_id].append(node_id)
        return dict(families)

    def _compute_component_scores(
        self, graph: ConsistencyGraph, family_nodes: list[str]
    ) -> dict[str, float]:
        """Compute all 4 instability component scores for a family."""
        policy = self._score_policy_instability(graph, family_nodes)
        reasoning = self._score_reasoning_instability(graph, family_nodes)
        knowledge = self._score_knowledge_instability(graph, family_nodes)
        formatting = self._score_formatting_instability(graph, family_nodes)

        return {
            "policy": policy,
            "reasoning": reasoning,
            "knowledge": knowledge,
            "formatting": formatting,
        }

    def _score_policy_instability(
        self, graph: ConsistencyGraph, family_nodes: list[str]
    ) -> float:
        """
        Score A: Policy instability.

        Measures refusal <-> answer flip frequency.
        Each flip pair contributes POLICY_FLIP_SCORE_PER_PAIR.
        """
        # Count refusal/answer distribution
        refusal_count = 0
        answer_count = 0
        node_data_map = {}

        for node_id in family_nodes:
            nd = graph.get_node_data(node_id)
            if nd and "normalized" in nd:
                resp = nd["normalized"]
                node_data_map[node_id] = resp
                if resp.refusal_flag:
                    refusal_count += 1
                else:
                    answer_count += 1

        if refusal_count == 0 or answer_count == 0:
            return 0.0  # All same behavior = no policy instability

        # Count flip pairs
        flip_pairs = 0
        refused_nodes = [nid for nid, resp in node_data_map.items() if resp.refusal_flag]
        answered_nodes = [nid for nid, resp in node_data_map.items() if not resp.refusal_flag]

        for r_node in refused_nodes:
            for a_node in answered_nodes:
                # Check if there's a policy_flip edge
                edges = graph.get_edges_between(r_node, a_node)
                has_flip = any(e.edge_type == "policy_flip" for e in edges)
                if has_flip:
                    flip_pairs += 1

        # Normalize by max possible flip pairs
        max_pairs = len(refused_nodes) * len(answered_nodes)
        if max_pairs == 0:
            return 0.0

        flip_rate = flip_pairs / max_pairs
        # Score: flip_rate * max_possible_score
        return flip_rate * min(refusal_count, answer_count) * POLICY_FLIP_SCORE_PER_PAIR

    def _score_reasoning_instability(
        self, graph: ConsistencyGraph, family_nodes: list[str]
    ) -> float:
        """
        Score B: Reasoning instability.

        Measures divergence of REASONING TRACES (not final answers) between
        non-refused nodes. Only counts nodes with substantial reasoning traces.
        Short answers without reasoning = no reasoning instability signal.

        Uses a fresh TF-IDF model over reasoning traces to compute
        trace-specific embeddings, independent of answer embeddings.
        """
        # Collect non-refused nodes with SUBSTANTIAL reasoning traces
        answered_nodes = []
        for node_id in family_nodes:
            nd = graph.get_node_data(node_id)
            if nd and "normalized" in nd:
                resp = nd["normalized"]
                # Only count nodes with actual reasoning (trace > 200 chars)
                if not resp.refusal_flag and resp.reasoning_trace and len(resp.reasoning_trace) > 200:
                    answered_nodes.append((node_id, resp))

        if len(answered_nodes) < 2:
            return 0.0

        # v1.2: Coverage guard — if only a small fraction of nodes produce
        # substantial reasoning, the divergence is likely format-driven noise,
        # not genuine reasoning path instability. Require at least 40% of
        # non-refused nodes to have substantial reasoning traces.
        # This prevents factual questions from being classified as reasoning_variance
        # just because format_change variants produce longer explanations.
        total_answered = 0
        for nid in family_nodes:
            nd = graph.get_node_data(nid)
            if nd and "normalized" in nd and not nd["normalized"].refusal_flag:
                total_answered += 1

        if total_answered > 0:
            coverage_ratio = len(answered_nodes) / total_answered
            if coverage_ratio < 0.4:
                return 0.0

        # Build a fresh TF-IDF model over reasoning traces
        traces = [resp.reasoning_trace for _, resp in answered_nodes]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=256, stop_words="english",
                ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
            )
            trace_embeddings = vectorizer.fit_transform(traces).toarray()
        except Exception:
            return 0.0

        # Compute pairwise trace divergence
        divergences = []
        for i in range(len(answered_nodes)):
            for j in range(i + 1, len(answered_nodes)):
                vec_a = trace_embeddings[i]
                vec_b = trace_embeddings[j]
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                if norm_a > 0 and norm_b > 0:
                    cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                    div = max(0.0, 1.0 - cos)
                    divergences.append(div)

        if not divergences:
            return 0.0

        avg_div = sum(divergences) / len(divergences)
        # Reasoning requires meaningful divergence to count
        if avg_div < REASONING_DIVERGENCE_THRESHOLD:
            return 0.0

        return min((avg_div - REASONING_DIVERGENCE_THRESHOLD) * 3.0, 2.5)

    def _score_knowledge_instability(
        self, graph: ConsistencyGraph, family_nodes: list[str]
    ) -> float:
        """
        Score C: Knowledge instability.

        Measures factual disagreement across non-refused nodes.
        Uses word-level Jaccard similarity on NORMALIZED core answers.
        TF-IDF embeddings are unreliable for short factual answers,
        so we use direct word overlap instead.

        A pair is a "knowledge conflict" only if:
            1. Their normalized core answers share < 50% word overlap
               AND
            2. At least one answer is substantive (> 3 content words)
        """
        answered_nodes = []
        for node_id in family_nodes:
            nd = graph.get_node_data(node_id)
            if nd and "normalized" in nd:
                resp = nd["normalized"]
                if not resp.refusal_flag:
                    answered_nodes.append((node_id, resp))

        if len(answered_nodes) < 2:
            return 0.0

        # Normalize answers to word sets for Jaccard comparison
        def _normalize_to_words(text: str) -> set:
            """Extract content words from text."""
            import re
            text = text.lower().strip()
            # Remove common function words
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                        "being", "have", "has", "had", "do", "does", "did", "will",
                        "would", "could", "should", "may", "might", "shall", "can",
                        "to", "of", "in", "for", "on", "with", "at", "by", "from",
                        "it", "its", "this", "that", "these", "those", "and", "or",
                        "but", "not", "no", "so", "if", "as"}
            words = set(re.findall(r'[a-z]+', text))
            return words - stopwords

        knowledge_conflicts = 0.0
        total_pairs = 0

        for i in range(len(answered_nodes)):
            for j in range(i + 1, len(answered_nodes)):
                _, resp_a = answered_nodes[i]
                _, resp_b = answered_nodes[j]
                total_pairs += 1

                words_a = _normalize_to_words(resp_a.final_answer)
                words_b = _normalize_to_words(resp_b.final_answer)

                if not words_a or not words_b:
                    continue  # Skip empty/meaningless answers

                # Jaccard similarity
                intersection = words_a & words_b
                union = words_a | words_b
                jaccard = len(intersection) / len(union) if union else 0.0

                # Knowledge conflict: low word overlap AND substantive answers
                # (both have > 3 content words = substantive enough to compare)
                if jaccard < 0.5 and len(words_a) > 2 and len(words_b) > 2:
                    # Severity: lower jaccard = stronger conflict
                    severity = 1.0 - jaccard
                    knowledge_conflicts += severity

        if total_pairs == 0:
            return 0.0

        conflict_rate = knowledge_conflicts / total_pairs
        return min(conflict_rate * 3.0, 2.5)

    def _score_formatting_instability(
        self, graph: ConsistencyGraph, family_nodes: list[str]
    ) -> float:
        """
        Score D: Formatting instability.

        Measures format signature diversity across nodes.
        Uses Shannon entropy of format signatures.
        """
        format_sigs = []
        for node_id in family_nodes:
            nd = graph.get_node_data(node_id)
            if nd and "normalized" in nd:
                resp = nd["normalized"]
                if not resp.refusal_flag:
                    format_sigs.append(resp.format_signature)

        if len(format_sigs) < 2:
            return 0.0

        # Count format signature frequencies
        sig_counts = defaultdict(int)
        for sig in format_sigs:
            sig_counts[sig] += 1

        # Compute Shannon entropy
        total = len(format_sigs)
        entropy = 0.0
        for count in sig_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # Maximum entropy = log2(n)
        max_entropy = np.log2(len(set(format_sigs))) if len(set(format_sigs)) > 1 else 1.0

        # Normalized entropy
        if max_entropy > 0:
            norm_entropy = entropy / max_entropy
        else:
            norm_entropy = 0.0

        # Only count as instability if above threshold and there's actual diversity
        if norm_entropy > FORMAT_ENTROPY_THRESHOLD and len(sig_counts) > 1:
            return norm_entropy * 2.0  # Scale to 0-2 range

        return 0.0

    def _classify_type(self, scores: dict[str, float]) -> str:
        """
        Classify the dominant instability type from component scores.

        v1.2: Added reasoning dominance override.
            if reasoning_score > 0.6 AND reasoning_score > knowledge_score:
                allow reasoning to win dominance even if knowledge weighted higher
        """
        from .config import INSTABILITY_WEIGHTS, INSTABILITY_SCORE_CAP
        total = sum(
            scores.get(k, 0.0) * INSTABILITY_WEIGHTS.get(k, 0.0)
            for k in INSTABILITY_WEIGHTS
        )
        if total < 0.5:
            return "stable"

        # Find dominant component by weighted score
        weighted = {k: scores.get(k, 0.0) * INSTABILITY_WEIGHTS.get(k, 0.0) for k in scores}
        dominant = max(weighted, key=weighted.get)

        # v1.2: Reasoning dominance override
        # Conditions (all must be true):
        #   1. Reasoning raw score > threshold (substantial signal)
        #   2. Reasoning raw score > knowledge raw score (reasoning is stronger)
        #   3. Reasoning raw score is the maximum across ALL components
        #      (don't override when formatting or policy is the real driver)
        # This prevents the override from hijacking factual tasks where
        # formatting dominates.
        reasoning_raw = scores.get("reasoning", 0.0)
        knowledge_raw = scores.get("knowledge", 0.0)

        if (reasoning_raw > REASONING_DOMINANCE_OVERRIDE_THRESHOLD
                and reasoning_raw > knowledge_raw
                and reasoning_raw >= max(scores.values())):
            dominant = "reasoning"

        type_map = {
            "policy": "policy_flip",
            "reasoning": "reasoning_variance",
            "knowledge": "knowledge_variance",
            "formatting": "formatting_variance",
        }
        return type_map.get(dominant, "stable")

    def _build_evidence(
        self, graph: ConsistencyGraph, family_nodes: list[str], scores: dict
    ) -> dict:
        """
        Build evidence dict for a cluster.

        v1.2: Now includes typed top_trigger using classify_trigger_type().
        """
        node_details = []
        for node_id in family_nodes:
            nd = graph.get_node_data(node_id)
            if nd and "normalized" in nd:
                resp = nd["normalized"]
                node_details.append({
                    "node_id": node_id[:8],
                    "strategy": nd.get("strategy", "unknown"),
                    "refused": resp.refusal_flag,
                    "format_sig": resp.format_signature[:8],
                    "answer_preview": resp.final_answer[:80],
                })

        # Get policy flip edges
        flip_edges = []
        for e in graph.get_edges_by_type("policy_flip"):
            if e.source_id in family_nodes and e.target_id in family_nodes:
                flip_edges.append({
                    "refused": e.evidence.get("refused_node", "?"),
                    "answered": e.evidence.get("answered_node", "?"),
                })

        # v1.2 FIX: Compute typed top_trigger from component scores
        top_trigger = classify_trigger_type(scores)

        evidence = {
            "node_count": len(family_nodes),
            "node_details": node_details,
            "policy_flip_edges": flip_edges,
            "top_trigger": top_trigger,
        }
        return evidence

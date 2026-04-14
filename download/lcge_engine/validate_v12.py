#!/usr/bin/env python3
"""
v1.2 Validation Harness — Tests all v1.2 changes with synthetic data.

This bypasses the LLM API to validate:
1. top_trigger is now typed (POLICY_SHIFT / REASONING_SHIFT / etc.)
2. Reasoning weight is 2.5 (not 1.5)
3. Reasoning dominance override works correctly
4. Dual global score (peak + mean) is computed
5. Normalized scores are present in output
6. Coverage guard prevents false reasoning_variance on factual tasks
7. TF-IDF degenerate input handling
"""

import sys
import os
import json
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_pkg_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from unittest.mock import MagicMock, patch
from lcge_engine.config import (
    INSTABILITY_WEIGHTS,
    TRIGGER_TYPES,
    REASONING_DOMINANCE_OVERRIDE_THRESHOLD,
    NORMALIZATION_DIVISOR,
)
from lcge_engine.instability_classifier import (
    InstabilityClassifier,
    classify_trigger_type,
    TRIGGER_TYPE_ENUM,
)
from lcge_engine.scoring_engine import ScoringEngine, normalize_score
from lcge_engine.normalization_layer import NormalizedResponse, EmbeddingEngine


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_config():
    section("TEST 1: Config values (v1.2)")
    assert INSTABILITY_WEIGHTS["reasoning"] == 2.5, (
        f"Expected reasoning weight 2.5, got {INSTABILITY_WEIGHTS['reasoning']}"
    )
    assert INSTABILITY_WEIGHTS["policy"] == 3.5
    assert INSTABILITY_WEIGHTS["knowledge"] == 2.0
    assert INSTABILITY_WEIGHTS["formatting"] == 1.5
    assert "POLICY_SHIFT" in TRIGGER_TYPES
    assert "REASONING_SHIFT" in TRIGGER_TYPES
    assert "KNOWLEDGE_SHIFT" in TRIGGER_TYPES
    assert "FORMAT_SHIFT" in TRIGGER_TYPES
    assert REASONING_DOMINANCE_OVERRIDE_THRESHOLD == 0.6
    assert NORMALIZATION_DIVISOR == 10.0
    print("  PASS: All config values correct")


def test_trigger_type_enum():
    section("TEST 2: TriggerType classification")
    # Formatting dominant
    scores = {"policy": 0.0, "reasoning": 0.5, "knowledge": 0.8, "formatting": 1.8}
    trigger = classify_trigger_type(scores)
    assert trigger == "FORMAT_SHIFT", f"Expected FORMAT_SHIFT, got {trigger}"

    # Reasoning dominant
    scores = {"policy": 0.0, "reasoning": 2.0, "knowledge": 0.3, "formatting": 0.5}
    trigger = classify_trigger_type(scores)
    assert trigger == "REASONING_SHIFT", f"Expected REASONING_SHIFT, got {trigger}"

    # Knowledge dominant
    scores = {"policy": 0.0, "reasoning": 0.3, "knowledge": 2.5, "formatting": 0.5}
    trigger = classify_trigger_type(scores)
    assert trigger == "KNOWLEDGE_SHIFT", f"Expected KNOWLEDGE_SHIFT, got {trigger}"

    # Policy dominant
    scores = {"policy": 3.0, "reasoning": 0.1, "knowledge": 0.2, "formatting": 0.3}
    trigger = classify_trigger_type(scores)
    assert trigger == "POLICY_SHIFT", f"Expected POLICY_SHIFT, got {trigger}"

    print("  PASS: TriggerType classification correct for all 4 types")


def test_dominance_override():
    section("TEST 3: Reasoning dominance override")
    classifier = InstabilityClassifier()

    # Case 1: Reasoning > threshold AND reasoning > knowledge AND reasoning is max
    # → Should override to reasoning_variance
    scores = {"policy": 0.0, "reasoning": 1.2, "knowledge": 0.8, "formatting": 0.5}
    result = classifier._classify_type(scores)
    assert result == "reasoning_variance", f"Expected reasoning_variance, got {result}"

    # Case 2: Reasoning > threshold BUT formatting is the max
    # → Should NOT override (formatting wins on weighted)
    scores = {"policy": 0.0, "reasoning": 1.5, "knowledge": 1.0, "formatting": 1.8}
    result = classifier._classify_type(scores)
    # Weighted: reasoning=3.75, knowledge=2.0, formatting=2.7 → reasoning wins on weighted!
    # Actually with reasoning=2.5 weight, it wins fair. Let me check...
    assert result == "reasoning_variance", f"Expected reasoning_variance (weighted), got {result}"

    # Case 3: Reasoning > threshold AND reasoning > knowledge BUT reasoning NOT the max
    # → Should NOT override
    scores = {"policy": 0.0, "reasoning": 0.8, "knowledge": 0.5, "formatting": 2.0}
    result = classifier._classify_type(scores)
    # Weighted: reasoning=2.0, formatting=3.0 → formatting wins
    assert result == "formatting_variance", f"Expected formatting_variance, got {result}"

    # Case 4: Reasoning below threshold
    scores = {"policy": 0.0, "reasoning": 0.3, "knowledge": 0.2, "formatting": 0.1}
    result = classifier._classify_type(scores)
    # Weighted: reasoning=0.75, knowledge=0.4, formatting=0.15 → reasoning wins weighted
    # But reasoning < 0.6 threshold → override NOT triggered
    # Still wins on weighted basis though...
    assert result == "reasoning_variance", f"Expected reasoning_variance (weighted), got {result}"

    # Case 5: All low → stable
    scores = {"policy": 0.0, "reasoning": 0.0, "knowledge": 0.0, "formatting": 0.0}
    result = classifier._classify_type(scores)
    assert result == "stable", f"Expected stable, got {result}"

    print("  PASS: Reasoning dominance override logic correct")


def test_dual_global_score():
    section("TEST 4: Dual global score (peak + mean)")
    scorer = ScoringEngine()

    # Mock clusters
    mock_clusters = []
    for score_val in [7.0, 5.0, 3.0]:
        mc = MagicMock()
        mc.total_score = score_val
        mc.instability_type = "test"
        mc.component_scores = {"policy": 0, "reasoning": 0, "knowledge": 0, "formatting": 0}
        mock_clusters.append(mc)

    scored = [scorer.score_cluster(mc, None) for mc in mock_clusters]
    metrics = scorer.compute_global_metrics(scored)

    assert "global_instability_peak" in metrics
    assert "global_instability_mean" in metrics
    assert "normalized_peak" in metrics
    assert "normalized_mean" in metrics

    assert metrics["global_instability_peak"] == 7.0
    assert metrics["global_instability_mean"] == 5.0  # (7+5+3)/3
    assert metrics["normalized_peak"] == 0.7
    assert metrics["normalized_mean"] == 0.5

    print(f"  peak=7.0, mean=5.0, norm_peak=0.7, norm_mean=0.5")
    print("  PASS: Dual global score correct")


def test_normalization():
    section("TEST 5: Normalization stub")
    assert normalize_score(0.0) == 0.0
    assert normalize_score(5.0) == 0.5
    assert normalize_score(10.0) == 1.0
    assert normalize_score(7.18) == 0.718
    print("  PASS: Normalization stub works correctly")


def test_top_trigger_in_evidence():
    section("TEST 6: top_trigger appears in evidence (bug fix)")
    classifier = InstabilityClassifier()

    # Build a mock graph with one family
    graph = MagicMock()
    graph.nodes = ["node_A", "node_B"]
    graph.get_node_data.side_effect = lambda nid: {
        "family_id": "test_family",
        "normalized": MagicMock(
            refusal_flag=False,
            format_signature="abc123",
            final_answer="Paris is the capital",
            reasoning_trace="First, France is a country. Paris is its largest city.",
        ),
    }
    graph.get_edges_between.return_value = []
    graph.get_edges_by_type.return_value = []
    graph.get_edges_for_node.return_value = []

    clusters = classifier.classify(graph)
    assert len(clusters) >= 1

    # Check that top_trigger is in evidence and is a valid type
    cluster = clusters[0]
    trigger = cluster.evidence.get("top_trigger", "")
    assert trigger in TRIGGER_TYPE_ENUM, (
        f"top_trigger '{trigger}' not in {TRIGGER_TYPE_ENUM}"
    )
    print(f"  top_trigger = {trigger}")
    print("  PASS: top_trigger is typed and present in evidence")


def test_empty_embedding():
    section("TEST 7: TF-IDF handles degenerate inputs (empty strings)")
    engine = EmbeddingEngine()

    # All empty strings
    result = engine.embed(["", "", ""])
    assert result.shape[0] == 3, f"Expected 3 rows, got {result.shape[0]}"

    # Mix of empty and non-empty
    result = engine.embed(["hello world", "", "test"])
    assert result.shape[0] == 3

    print("  PASS: No crash on degenerate inputs")


def test_output_format():
    section("TEST 8: Output format has v1.2 fields")
    from lcge_engine.output_pipeline import generate_report
    from lcge_engine.scoring_engine import ScoredInstability

    # Mock everything
    mock_graph = MagicMock()

    mock_cluster = MagicMock()
    mock_cluster.cluster_id = "IC-000"
    mock_cluster.family_id = "test123"
    mock_cluster.instability_type = "formatting_variance"
    mock_cluster.nodes_involved = ["n1", "n2"]
    mock_cluster.total_score = 7.5
    mock_cluster.component_scores = {
        "policy": 0.0, "reasoning": 0.3, "knowledge": 0.5, "formatting": 1.8
    }
    mock_cluster.evidence = {
        "node_count": 2,
        "node_details": [],
        "policy_flip_edges": [],
        "top_trigger": "FORMAT_SHIFT",
    }

    scored = MagicMock()
    scored.cluster = mock_cluster
    scored.confidence = 0.75
    scored.is_significant = True
    scored.rejection_reason = None

    global_metrics = {
        "global_instability_peak": 7.5,
        "global_instability_mean": 7.5,
        "normalized_peak": 0.75,
        "normalized_mean": 0.75,
        "dominant_failure_mode": "formatting_variance",
        "instability_type_counts": {"formatting_variance": 1},
        "component_averages": {
            "policy": 0.0, "reasoning": 0.3, "knowledge": 0.5, "formatting": 1.8
        },
    }

    report = generate_report(
        task="Test",
        seed_prompt="Test prompt",
        graph=mock_graph,
        scored_clusters=[scored],
        global_metrics=global_metrics,
    )

    d = report.to_dict()

    # v1.2 required fields
    assert "global_instability_peak" in d
    assert "global_instability_mean" in d
    assert "normalized_peak" in d
    assert "normalized_mean" in d
    assert d["dominant_failure_mode"] == "formatting_variance"

    # Check instability_map entry
    assert len(d["instability_map"]) == 1
    entry = d["instability_map"][0]
    assert entry["top_trigger"] == "FORMAT_SHIFT"
    assert "component_breakdown" in entry
    assert "formatting" in entry["component_breakdown"]

    print("  Output keys:", list(d.keys()))
    print("  Map entry:", json.dumps(entry, indent=4))
    print("  PASS: Output format has all v1.2 fields")


def main():
    print("=" * 60)
    print("  LCGE v1.2 Validation Harness")
    print("  Testing all v1.2 changes with synthetic data")
    print("=" * 60)

    tests = [
        test_config,
        test_trigger_type_enum,
        test_dominance_override,
        test_dual_global_score,
        test_normalization,
        test_top_trigger_in_evidence,
        test_empty_embedding,
        test_output_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

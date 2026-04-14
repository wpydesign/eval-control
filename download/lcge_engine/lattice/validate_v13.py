#!/usr/bin/env python3
"""
validate_v13.py — Validates the v1.3.1 lattice infrastructure.

Tests all components with synthetic data (no API calls needed):
1.  Frozen prompts loaded correctly
2.  Lattice index generation (115 points = 5 prompts x 23 strategies)
3.  Axis membership and structure
4.  Variant generator produces one variant per strategy
5.  Synthetic manifold generator
6.  Vector store builds embedding matrix
7.  Coordinate solver: distance metrics (all 5)
8.  Coordinate solver: PCA decomposition
9.  Coordinate solver: strategy + axis displacement
10. Multi-metric comparison
11. Full pipeline (lattice -> vector -> solver)
12. Lattice simulator end-to-end
13. NO classification artifacts anywhere
"""

import sys
import os
import json
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Tests 1-4: Structural validation
# ============================================================

def test_frozen_prompts():
    section("TEST 1: Frozen prompts")
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    assert len(FROZEN_PROMPTS) == 5
    assert "What is the capital of France?" in FROZEN_PROMPTS
    assert "What is 2 + 2?" in FROZEN_PROMPTS
    assert "Explain how quicksort works." in FROZEN_PROMPTS
    assert "Is it ethical to lie to protect someone?" in FROZEN_PROMPTS
    assert "Write a short summary of climate change." in FROZEN_PROMPTS
    print(f"  {len(FROZEN_PROMPTS)} frozen prompts loaded")
    print("  PASS")


def test_lattice_index():
    section("TEST 2: Lattice index generation (v1.3.1 expanded)")
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import (
        generate_lattice_index,
        ALL_STRATEGY_NAMES,
        get_all_axes,
        AXIS_STRATEGY_COUNTS,
    )

    lattice = generate_lattice_index(FROZEN_PROMPTS)
    expected = len(FROZEN_PROMPTS) * len(ALL_STRATEGY_NAMES)
    assert len(lattice) == expected, f"Expected {expected}, got {len(lattice)}"

    # Each prompt has all strategies
    for pid in range(len(FROZEN_PROMPTS)):
        point_strategies = set(l['strategy'] for l in lattice if l['prompt_id'] == pid)
        assert point_strategies == set(ALL_STRATEGY_NAMES), \
            f"Prompt {pid} missing strategies: {set(ALL_STRATEGY_NAMES) - point_strategies}"

    # Each point has axis field
    for point in lattice:
        assert 'axis' in point, f"Missing axis in {point['run_key']}"
        assert point['axis'] != 'unknown', f"Unknown axis in {point['run_key']}"

    # No classification artifacts
    for point in lattice:
        assert 'instability_type' not in point
        assert 'score' not in point
        assert 'trigger' not in point

    # Axis structure
    axes = get_all_axes()
    assert len(axes) == 7, f"Expected 7 axes, got {len(axes)}"
    total_by_axis = sum(AXIS_STRATEGY_COUNTS.values())
    assert total_by_axis == 23, f"Expected 23 strategies across axes, got {total_by_axis}"

    print(f"  {len(lattice)} lattice points ({len(FROZEN_PROMPTS)} prompts x {len(ALL_STRATEGY_NAMES)} strategies)")
    print(f"  7 axes: {axes}")
    print(f"  Axis strategy counts: {dict(AXIS_STRATEGY_COUNTS)}")
    print("  All points have axis membership")
    print("  No classification artifacts present")
    print("  PASS")


def test_axis_membership():
    section("TEST 3: Axis membership queries")
    from lcge_engine.lattice.variant_generator import (
        get_strategies_by_axis,
        get_axis_for_strategy,
        resolve_legacy_strategy,
    )

    # Get strategies by axis
    constraint_strats = get_strategies_by_axis("constraint_intensity")
    assert len(constraint_strats) == 4
    assert "constraint_none" in constraint_strats
    assert "constraint_heavy" in constraint_strats

    role_strats = get_strategies_by_axis("role_instability")
    assert len(role_strats) == 5
    assert "role_skeptic" in role_strats

    format_strats = get_strategies_by_axis("format_manifold")
    assert len(format_strats) == 6

    # Get axis for strategy
    assert get_axis_for_strategy("token_1sentence") == "token_pressure"
    assert get_axis_for_strategy("instruction_conflict") == "instruction_hierarchy"
    assert get_axis_for_strategy("format_json") == "format_manifold"

    # Legacy mapping
    assert resolve_legacy_strategy("constraint_add") == "constraint_light"
    assert resolve_legacy_strategy("role_change") == "role_expert"
    assert resolve_legacy_strategy("format_change") == "format_bullet"
    assert resolve_legacy_strategy("step_by_step") == "format_step_by_step"
    assert resolve_legacy_strategy("paraphrase") == "paraphrase"  # unchanged

    print(f"  constraint_intensity: {constraint_strats}")
    print(f"  role_instability: {role_strats}")
    print(f"  format_manifold: {format_strats}")
    print(f"  Legacy mappings resolved correctly")
    print("  PASS")


def test_variant_generator():
    section("TEST 4: Variant generator (v1.3.1)")
    from lcge_engine.lattice.variant_generator import (
        generate_strategy_variants,
        ALL_STRATEGY_NAMES,
    )

    variants = generate_strategy_variants("What is 2+2?")
    assert len(variants) == 23, f"Expected 23 variants, got {len(variants)}"

    strategies = [v['strategy'] for v in variants]
    assert set(strategies) == set(ALL_STRATEGY_NAMES)

    # All variants contain the seed prompt
    for v in variants:
        assert "What is 2+2?" in v['prompt']

    # All have axis field
    for v in variants:
        assert v['axis'] != 'unknown'

    print(f"  23 variants generated (one per strategy)")
    print("  All contain seed prompt text")
    print("  All have axis membership")
    print("  PASS")


# ============================================================
# Tests 5-6: Synthetic manifold
# ============================================================

def test_synthetic_manifold():
    section("TEST 5: Synthetic manifold generator")
    from lcge_engine.lattice.synthetic_manifold import (
        SyntheticManifold,
        RESPONSE_CLASSES,
    )
    from lcge_engine.lattice.variant_generator import generate_lattice_index
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS

    assert len(RESPONSE_CLASSES) == 5
    assert "deterministic" in RESPONSE_CLASSES
    assert "divergent" in RESPONSE_CLASSES

    manifold = SyntheticManifold(seed=42)

    # Deterministic assignment
    cls1 = manifold.assign_response_class(0, "paraphrase", "semantic_reformulation", 0)
    cls2 = manifold.assign_response_class(0, "paraphrase", "semantic_reformulation", 5)
    assert cls1 == cls2, "Same (prompt, strategy) should yield same class across reps"

    # Different strategies can have different classes
    cls3 = manifold.assign_response_class(0, "adversarial", "adversarial_probe", 0)
    # (just verify it doesn't crash and returns a valid class)
    assert cls3 in RESPONSE_CLASSES

    # Response generation
    response = manifold.generate_response(0, "paraphrase", "semantic_reformulation", 0)
    assert isinstance(response, str)
    assert len(response) > 0

    # Full lattice generation
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    records = manifold.generate_full_lattice(lattice, num_reps=3)
    assert len(records) == 115 * 3  # 115 points x 3 reps

    # All records have response_class
    for rec in records:
        assert "response_class" in rec
        assert rec["response_class"] in RESPONSE_CLASSES

    # Class distribution
    dist = manifold.get_class_distribution(records)
    assert len(dist) == 5, f"Expected 5 classes, got {len(dist)}"

    print(f"  5 response classes: {RESPONSE_CLASSES}")
    print(f"  Deterministic class assignment: OK")
    print(f"  Full lattice: {len(records)} records (115 x 3)")
    print(f"  Class distribution: {dist}")
    print("  PASS")


# ============================================================
# Test 7: Vector store
# ============================================================

def test_vector_store():
    section("TEST 6: Vector store")
    from lcge_engine.lattice.vector_store import load_run_records, build_embedding_matrix

    records = []
    for i in range(50):
        records.append({
            'run_id': f'test_{i}',
            'response': f'This is synthetic response number {i} with some varied content about topic A.',
            'row_index': i,
        })

    jsonl_path = '/tmp/test_v131_vectors.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    loaded = load_run_records(jsonl_path)
    assert len(loaded) == 50

    embeddings, meta = build_embedding_matrix(loaded)
    assert embeddings.shape[0] == 50
    assert embeddings.shape[1] > 0
    assert meta['n_features'] > 0

    # Handle empty responses
    records_with_empty = records + [{'run_id': 'empty_1', 'response': '[ERROR] test', 'row_index': 50}]
    with open(jsonl_path, 'w') as f:
        for r in records_with_empty:
            f.write(json.dumps(r) + '\n')

    loaded2 = load_run_records(jsonl_path)
    embeddings2, meta2 = build_embedding_matrix(loaded2)
    assert embeddings2.shape[0] == 51

    print(f"  Embedding matrix: {embeddings.shape}")
    print(f"  Empty handling: {embeddings2.shape}")
    print("  PASS")


# ============================================================
# Tests 7-8: Coordinate solver
# ============================================================

def test_distance_metrics():
    section("TEST 7: Distance metrics (all 5)")
    from lcge_engine.lattice.coordinate_solver import (
        compute_distance_matrix,
        SUPPORTED_METRICS,
    )

    np.random.seed(42)
    embeddings = np.random.randn(30, 20)

    for metric in SUPPORTED_METRICS:
        dist = compute_distance_matrix(embeddings, metric)
        assert dist.shape == (30, 30), f"{metric}: wrong shape {dist.shape}"
        assert np.allclose(np.diag(dist), 0), f"{metric}: diagonal not zero"
        assert dist.min() >= -0.01, f"{metric}: negative distances"
        print(f"    {metric:20s}: shape={dist.shape}, mean={dist.mean():.4f}, max={dist.max():.4f}")

    print(f"  All {len(SUPPORTED_METRICS)} distance metrics computed successfully")
    print("  PASS")


def test_coordinate_solver():
    section("TEST 8: Coordinate solver (PCA + displacements)")
    from lcge_engine.lattice.coordinate_solver import (
        compute_distance_matrix,
        compute_pca,
        compute_strategy_displacement,
        compute_axis_displacement,
    )

    # Build synthetic embedding matrix (strategy-separated clusters)
    np.random.seed(42)
    n_per_strategy = 15
    strategies = [
        'paraphrase', 'constraint_none', 'constraint_light', 'constraint_heavy',
        'constraint_conflicting', 'instruction_system_only', 'instruction_user_only',
        'instruction_conflict', 'role_neutral', 'role_expert', 'role_skeptic',
        'role_adversarial_eval', 'role_obedient', 'format_paragraph', 'format_bullet',
        'format_json', 'format_step_by_step', 'format_compressed', 'format_verbose',
        'token_1sentence', 'token_5lines', 'token_full', 'adversarial',
    ]
    n_strategies = len(strategies)

    centers = np.random.randn(n_strategies, 20) * 2

    all_embeddings = []
    records = []
    axis_map = {
        'paraphrase': 'semantic_reformulation',
        'constraint_none': 'constraint_intensity', 'constraint_light': 'constraint_intensity',
        'constraint_heavy': 'constraint_intensity', 'constraint_conflicting': 'constraint_intensity',
        'instruction_system_only': 'instruction_hierarchy', 'instruction_user_only': 'instruction_hierarchy',
        'instruction_conflict': 'instruction_hierarchy',
        'role_neutral': 'role_instability', 'role_expert': 'role_instability',
        'role_skeptic': 'role_instability', 'role_adversarial_eval': 'role_instability',
        'role_obedient': 'role_instability',
        'format_paragraph': 'format_manifold', 'format_bullet': 'format_manifold',
        'format_json': 'format_manifold', 'format_step_by_step': 'format_manifold',
        'format_compressed': 'format_manifold', 'format_verbose': 'format_manifold',
        'token_1sentence': 'token_pressure', 'token_5lines': 'token_pressure',
        'token_full': 'token_pressure', 'adversarial': 'adversarial_probe',
    }

    for s_idx, strategy in enumerate(strategies):
        for r_idx in range(n_per_strategy):
            sample = centers[s_idx] + np.random.randn(20) * 0.5
            all_embeddings.append(sample)
            records.append({
                'run_id': f'test_{s_idx}_{r_idx}',
                'strategy': strategy,
                'axis': axis_map.get(strategy, 'unknown'),
                'row_index': len(records),
            })

    embeddings = np.array(all_embeddings)

    # Distance matrix
    dist = compute_distance_matrix(embeddings, "cosine")
    assert dist.shape == (len(records), len(records))
    assert np.allclose(np.diag(dist), 0)

    # PCA
    pca_result = compute_pca(embeddings, n_components=8)
    assert len(pca_result['principal_components']) > 0
    assert pca_result['projections'].shape[0] == len(records)

    total_var = sum(pc['variance_ratio'] for pc in pca_result['principal_components'])
    # Note: variance sum may be < 1.0 when PCA components < data dimensionality
    assert total_var > 0.5, f"Variance ratios sum too low: {total_var}"

    # Strategy displacement
    displacements = compute_strategy_displacement(records, pca_result['projections'])
    assert len(displacements) == n_strategies
    for s, d in displacements.items():
        assert 'displacement_from_global' in d
        assert 'n_samples' in d

    # Axis displacement (v1.3.1)
    axis_disp = compute_axis_displacement(records, pca_result['projections'])
    assert len(axis_disp) == 7  # 7 axes
    for axis, info in axis_disp.items():
        assert 'displacement_from_global' in info
        assert 'total_within_axis_spread' in info
        assert 'n_strategies' in info

    print(f"  Distance matrix: {dist.shape}")
    print(f"  PCA components: {len(pca_result['principal_components'])}")
    print(f"  Total variance captured: {total_var:.4f}")
    print(f"  Strategy displacements: {len(displacements)}")
    print(f"  Axis displacements: {len(axis_disp)}")

    disps = [d['displacement_from_global'] for d in displacements.values()]
    print(f"  Displacement range: {min(disps):.4f} - {max(disps):.4f}")

    # No classification artifacts
    assert 'instability_type' not in displacements
    assert 'dominant_mode' not in displacements
    print("  No classification artifacts")
    print("  PASS")


# ============================================================
# Test 9: Multi-metric comparison
# ============================================================

def test_multi_metric_comparison():
    section("TEST 9: Multi-metric comparison")
    from lcge_engine.lattice.coordinate_solver import compare_metrics

    np.random.seed(42)
    embeddings = np.random.randn(60, 25)

    results = compare_metrics(embeddings, metrics=["cosine", "euclidean"], n_components=5)

    assert "cosine" in results
    assert "euclidean" in results
    assert "_cross_metric_alignment" in results

    alignment = results["_cross_metric_alignment"]
    assert len(alignment) == 1  # Only one pair: cosine vs euclidean
    assert "metric_1" in alignment[0]
    assert "metric_2" in alignment[0]
    assert "mean_correlation" in alignment[0]

    print(f"  Cosine variance ratios: {results['cosine']['variance_ratios'][:3]}")
    print(f"  Euclidean variance ratios: {results['euclidean']['variance_ratios'][:3]}")
    print(f"  Cross-metric alignment: {alignment[0]['metric_1']} vs {alignment[0]['metric_2']}, corr={alignment[0]['mean_correlation']:.4f}")
    print("  PASS")


# ============================================================
# Test 10: Full pipeline
# ============================================================

def test_full_pipeline():
    section("TEST 10: Full pipeline (lattice -> vector -> solver)")
    from lcge_engine.lattice.coordinate_solver import solve_behavioral_space
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import ALL_STRATEGY_NAMES
    from lcge_engine.lattice.synthetic_manifold import SyntheticManifold
    from lcge_engine.lattice.variant_generator import generate_lattice_index

    # Generate synthetic data (115 lattice points x 3 reps = 345)
    manifold = SyntheticManifold(seed=99)
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    records = manifold.generate_full_lattice(lattice, num_reps=3)

    jsonl_path = '/tmp/test_v131_full.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    output_path = '/tmp/test_v131_behavioral_space.json'
    result_path = solve_behavioral_space(
        jsonl_path,
        n_components=8,
        metric="cosine",
        output_path=output_path,
    )

    assert os.path.exists(result_path)
    with open(result_path) as f:
        result = json.load(f)

    # Output format
    assert 'principal_components' in result
    assert 'projections' in result
    assert len(result['projections']) == 345

    for pc in result['principal_components']:
        assert 'variance_ratio' in pc
        assert 'cumulative_variance' in pc

    # v1.3.1: axis displacements present
    assert 'axis_displacements' in result
    assert len(result['axis_displacements']) == 7

    # v1.3.1: metric in parameters
    assert result['parameters']['distance_metric'] == 'cosine'

    # NO classification artifacts
    assert 'instability_type' not in result
    assert 'dominant_failure_mode' not in result
    assert 'trigger' not in result
    assert 'component_scores' not in result
    assert 'score' not in result

    # Auxiliary files
    output_dir = os.path.dirname(result_path)
    assert os.path.exists(os.path.join(output_dir, 'embeddings.npz'))
    assert os.path.exists(os.path.join(output_dir, 'distance_matrix_cosine.npz'))
    assert os.path.exists(os.path.join(output_dir, 'record_index.json'))

    print(f"  Output: {result_path}")
    print(f"  Projections: {len(result['projections'])} records")
    print(f"  PCs: {len(result['principal_components'])}")
    print(f"  Strategy displacements: {len(result['strategy_displacements'])}")
    print(f"  Axis displacements: {len(result['axis_displacements'])}")
    print(f"  Distance metric: {result['parameters']['distance_metric']}")
    print("  NO classification artifacts detected")
    print("  PASS")


# ============================================================
# Test 11: Lattice simulator
# ============================================================

def test_lattice_simulator():
    section("TEST 11: Lattice simulator (end-to-end)")
    from lcge_engine.lattice.lattice_simulator import LatticeSimulator

    sim = LatticeSimulator(seed=42, num_reps=3)
    output_dir = '/tmp/test_v131_sim'

    results = sim.run(output_dir=output_dir, metrics=["cosine", "euclidean"], save_intermediates=True)

    assert results is not None
    assert 'config' in results
    assert results['config']['n_lattice_points'] == 115
    assert results['config']['total_records'] == 115 * 3
    assert results['config']['n_strategies'] == 23

    # Manifold shape
    assert 'manifold_shape' in results
    assert results['manifold_shape']['shape'] in [
        'spherical', 'anisotropic', 'linear', 'planar', 'clustered', 'degenerate'
    ]

    # Response class distribution
    assert 'response_class_distribution' in results
    assert len(results['response_class_distribution']) == 5

    # Strategy + axis displacements
    assert len(results['strategy_displacements']) == 23
    assert len(results['axis_displacements']) == 7

    # Metric comparison
    assert 'metric_comparison' in results
    assert 'cosine' in results['metric_comparison']
    assert 'euclidean' in results['metric_comparison']

    # Response conditioning
    assert 'response_conditioning' in results
    assert 'per_strategy' in results['response_conditioning']
    assert 'per_axis' in results['response_conditioning']

    # Saved files
    assert os.path.exists(os.path.join(output_dir, 'simulation_results.json'))
    assert os.path.exists(os.path.join(output_dir, 'synthetic_runs.jsonl'))

    # NO classification artifacts
    assert 'instability_type' not in results
    assert 'dominant_failure_mode' not in results

    print(f"  Lattice points: {results['config']['n_lattice_points']}")
    print(f"  Total records: {results['config']['total_records']}")
    print(f"  Manifold shape: {results['manifold_shape']['shape']}")
    print(f"  Effective dimensionality: {results['manifold_shape'].get('effective_dimensionality', 'N/A')}")
    print(f"  Response classes: {results['response_class_distribution']}")
    print(f"  Strategy displacements: {len(results['strategy_displacements'])}")
    print(f"  Axis displacements: {len(results['axis_displacements'])}")
    print(f"  Metrics compared: 2")
    print(f"  Response conditioning: computed")
    print(f"  Output saved: {output_dir}")
    print("  NO classification artifacts")
    print("  PASS")


# ============================================================
# Test 12: Stress tests (subset)
# ============================================================

def test_stress_tests():
    section("TEST 12: Geometry stress tests")
    from lcge_engine.lattice.geometry_stress_test import (
        bootstrap_stability_test,
        cross_metric_consistency_test,
    )
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import generate_lattice_index
    from lcge_engine.lattice.synthetic_manifold import SyntheticManifold
    from lcge_engine.lattice.vector_store import build_embedding_matrix

    # Generate data
    manifold = SyntheticManifold(seed=42)
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    records = manifold.generate_full_lattice(lattice, num_reps=5)
    embeddings, meta = build_embedding_matrix(records)

    # Bootstrap (small n for speed)
    boot = bootstrap_stability_test(embeddings, records, n_bootstrap=10)
    assert 'overall_stability_score' in boot
    assert 'axis_stability' in boot
    print(f"    Bootstrap stability: {boot['overall_stability_score']:.4f}")

    # Cross-metric (subset of metrics for speed)
    cm = cross_metric_consistency_test(
        embeddings, records, metrics=["cosine", "euclidean", "manhattan"]
    )
    assert 'mean_alignment' in cm
    assert len(cm['pairwise_alignments']) == 3  # C(3,2)
    print(f"    Cross-metric alignment: {cm['mean_alignment']:.4f}")

    print("  PASS")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  LCGE v1.3.1 Validation — Overcomplete Lattice")
    print("=" * 60)

    tests = [
        test_frozen_prompts,
        test_lattice_index,
        test_axis_membership,
        test_variant_generator,
        test_synthetic_manifold,
        test_vector_store,
        test_distance_metrics,
        test_coordinate_solver,
        test_multi_metric_comparison,
        test_full_pipeline,
        test_lattice_simulator,
        test_stress_tests,
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
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed (out of {len(tests)})")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

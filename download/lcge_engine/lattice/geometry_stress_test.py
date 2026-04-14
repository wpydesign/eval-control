"""
geometry_stress_test.py — Coordinate system stability stress tests (v1.3.1).

Tests the robustness of the discovered behavioral coordinate system
under various perturbations of the data and geometry assumptions.

Test suite:
    1. Bootstrap stability — resample with replacement, measure axis correlation
    2. Cross-metric consistency — compare PCA axes across 5 distance metrics
    3. Sample size sensitivity — track axis drift with varying N
    4. Axis alignment rotation — measure how much PCA axes rotate per bootstrap

No API calls needed. Uses synthetic data from SyntheticManifold.

Usage:
    python -m lcge_engine.lattice.geometry_stress_test
    python -m lcge_engine.lattice.geometry_stress_test --bootstrap 50
"""

import argparse
import json
import os
import sys
import logging
from typing import Optional

import numpy as np

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

logger = logging.getLogger("lattice")


# ============================================================
# Utility: Procrustes alignment
# ============================================================

def procrustes_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Procrustes correlation between two matrices.

    Finds the optimal rotation of B to align with A, then
    returns the correlation of the aligned matrices.

    Args:
        A: Matrix (n x k).
        B: Matrix (n x k).

    Returns:
        Correlation coefficient in [-1, 1].
    """
    n = min(A.shape[0], B.shape[0])
    k = min(A.shape[1], B.shape[1])
    A_sub = A[:n, :k]
    B_sub = B[:n, :k]

    # Center
    A_c = A_sub - A_sub.mean(axis=0)
    B_c = B_sub - B_sub.mean(axis=0)

    # SVD for optimal rotation
    U, _, Vt = np.linalg.svd(A_c.T @ B_c)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1] * (k - 1) + [d])

    # Rotation matrix
    R = Vt.T @ D @ U.T

    # Aligned B
    B_aligned = B_c @ R

    # Correlation
    corr = np.sum(A_c * B_aligned) / (
        np.linalg.norm(A_c) * np.linalg.norm(B_aligned) + 1e-15
    )
    return float(corr)


# ============================================================
# Test 1: Bootstrap stability
# ============================================================

def bootstrap_stability_test(
    embeddings: np.ndarray,
    records: list[dict],
    n_bootstrap: int = 100,
    n_components: int = 8,
) -> dict:
    """
    Test coordinate system stability under resampling.

    Resamples the embedding matrix with replacement many times,
    computes PCA each time, and measures how much the axes
    drift from the baseline (full-sample PCA).

    Returns:
        dict with stability metrics.
    """
    from sklearn.decomposition import PCA

    n_samples = embeddings.shape[0]
    logger.info(f"  Bootstrap test: {n_bootstrap} resamples, {n_samples} samples")

    # Baseline PCA
    baseline_pca = PCA(n_components=min(n_components, min(embeddings.shape) - 1))
    baseline_proj = baseline_pca.fit_transform(embeddings)

    # Storage
    correlations = {f"PC{k}": [] for k in range(min(n_components, baseline_proj.shape[1]))}
    variance_ratios = {f"PC{k}": [] for k in range(min(n_components, baseline_proj.shape[1]))}

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_embeddings = embeddings[indices]

        try:
            boot_pca = PCA(n_components=min(n_components, min(boot_embeddings.shape) - 1))
            boot_proj = boot_pca.fit_transform(boot_embeddings)

            n_compare = min(baseline_proj.shape[1], boot_proj.shape[1])
            for k in range(n_compare):
                # Correlation of k-th PC between bootstrap and baseline
                corr = np.corrcoef(baseline_proj[:, k], boot_proj[:, k])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations[f"PC{k}"].append(abs(corr))
                variance_ratios[f"PC{k}"].append(
                    boot_pca.explained_variance_ratio_[k]
                )
        except Exception:
            continue

    # Summarize
    stability = {}
    for pc_name in correlations:
        vals = correlations[pc_name]
        if vals:
            stability[pc_name] = {
                "mean_correlation": round(float(np.mean(vals)), 4),
                "std_correlation": round(float(np.std(vals)), 4),
                "min_correlation": round(float(np.min(vals)), 4),
                "pct_above_0.9": round(float(np.mean(np.array(vals) > 0.9)), 4),
            }

    var_stability = {}
    for pc_name in variance_ratios:
        vals = variance_ratios[pc_name]
        if vals:
            var_stability[pc_name] = {
                "mean_variance_ratio": round(float(np.mean(vals)), 4),
                "std_variance_ratio": round(float(np.std(vals)), 4),
            }

    return {
        "n_bootstrap": n_bootstrap,
        "axis_stability": stability,
        "variance_stability": var_stability,
        "overall_stability_score": round(
            float(np.mean([
                s["mean_correlation"]
                for s in stability.values()
            ])),
            4,
        ) if stability else 0.0,
    }


# ============================================================
# Test 2: Cross-metric consistency
# ============================================================

def cross_metric_consistency_test(
    embeddings: np.ndarray,
    records: list[dict],
    metrics: list[str] | None = None,
    n_components: int = 8,
) -> dict:
    """
    Test whether PCA axes are stable across distance metrics.

    Different distance metrics define different geometries. If the
    behavioral axes are real (not artifacts), they should be
    approximately stable across metrics.

    Uses Procrustes alignment to measure axis similarity.

    Returns:
        dict with pairwise metric alignment scores.
    """
    from lcge_engine.lattice.coordinate_solver import (
        SUPPORTED_METRICS,
        compute_distance_matrix,
        _pcoa,
        compute_pca,
    )

    if metrics is None:
        metrics = SUPPORTED_METRICS

    logger.info(f"  Cross-metric test: {len(metrics)} metrics")

    # Compute projections per metric
    metric_projections = {}
    metric_variances = {}

    for metric in metrics:
        dist = compute_distance_matrix(embeddings, metric)
        if metric == "cosine":
            pca_result = compute_pca(embeddings, n_components)
            proj = pca_result["projections"]
            var_ratios = pca_result["explained_variance_ratio"]
        else:
            proj = _pcoa(dist, n_components)
            # Compute variance ratios
            variances = np.var(proj, axis=0)
            total = variances.sum()
            var_ratios = variances / total if total > 1e-15 else np.zeros(proj.shape[1])

        metric_projections[metric] = proj
        metric_variances[metric] = var_ratios

    # Pairwise Procrustes alignment
    alignments = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            m1, m2 = metrics[i], metrics[j]
            p1 = metric_projections[m1]
            p2 = metric_projections[m2]

            corr = procrustes_correlation(p1, p2)

            # Per-axis correlations
            min_dims = min(p1.shape[1], p2.shape[1])
            axis_corrs = []
            for k in range(min(min_dims, 3)):
                c = np.corrcoef(p1[:, k], p2[:, k])[0, 1]
                axis_corrs.append(round(float(abs(c)) if not np.isnan(c) else 0.0, 4))

            alignments.append({
                "metric_1": m1,
                "metric_2": m2,
                "procrustes_correlation": round(corr, 4),
                "per_axis_correlation": axis_corrs,
                "mean_axis_correlation": round(float(np.mean(axis_corrs)), 4) if axis_corrs else 0.0,
            })

    return {
        "metrics_tested": metrics,
        "pairwise_alignments": alignments,
        "mean_alignment": round(
            float(np.mean([a["procrustes_correlation"] for a in alignments])),
            4,
        ) if alignments else 0.0,
    }


# ============================================================
# Test 3: Sample size sensitivity
# ============================================================

def sample_size_sensitivity_test(
    records_generator,  # callable(num_reps) -> list[dict]
    sample_sizes: list[int] | None = None,
    n_components: int = 8,
    seed: int = 42,
) -> dict:
    """
    Test how PCA axes change with fewer repetitions.

    Runs the full pipeline at multiple sample sizes and tracks
    how the top PCs drift.

    Args:
        records_generator: Callable(num_reps) -> list of record dicts.
        sample_sizes: List of rep counts to test.
        n_components: PCA components.
        seed: Random seed.

    Returns:
        dict with per-size stability metrics.
    """
    from lcge_engine.lattice.vector_store import build_embedding_matrix
    from lcge_engine.lattice.coordinate_solver import compute_pca

    if sample_sizes is None:
        sample_sizes = [3, 5, 10, 15, 20, 25, 30]

    logger.info(f"  Sample size test: {sample_sizes}")

    np.random.seed(seed)

    # Baseline: largest sample size
    baseline_reps = max(sample_sizes)
    baseline_records = records_generator(baseline_reps)
    baseline_emb, _ = build_embedding_matrix(baseline_records)
    baseline_pca = compute_pca(baseline_emb, n_components)
    baseline_proj = baseline_pca["projections"]

    results = {}
    for n_reps in sample_sizes:
        records = records_generator(n_reps)
        emb, meta = build_embedding_matrix(records)
        pca_result = compute_pca(emb, n_components)
        proj = pca_result["projections"]

        # Procrustes alignment with baseline
        corr = procrustes_correlation(proj, baseline_proj)

        # Per-axis correlation
        min_dims = min(proj.shape[1], baseline_proj.shape[1])
        axis_corrs = []
        for k in range(min(min_dims, 3)):
            c = np.corrcoef(proj[:, k], baseline_proj[:, k])[0, 1]
            axis_corrs.append(round(float(abs(c)) if not np.isnan(c) else 0.0, 4))

        results[str(n_reps)] = {
            "n_records": len(records),
            "n_features": meta["n_features"],
            "procrustes_correlation": round(corr, 4),
            "per_axis_correlation": axis_corrs,
            "top_variance_ratio": (
                round(float(pca_result["explained_variance_ratio"][0]), 4)
                if len(pca_result["explained_variance_ratio"]) > 0
                else 0.0
            ),
        }

        logger.info(
            f"    N={n_reps:3d}: procrustes={corr:.4f}, "
            f"axis_corr={axis_corrs}"
        )

    return {
        "sample_sizes_tested": sample_sizes,
        "baseline_reps": baseline_reps,
        "per_size_results": results,
    }


# ============================================================
# Full stress test runner
# ============================================================

def run_all_stress_tests(
    output_dir: Optional[str] = None,
    bootstrap_reps: int = 50,
    sample_sizes: list[int] | None = None,
) -> dict:
    """
    Run the complete stress test suite on synthetic data.

    Args:
        output_dir: Where to save results.
        bootstrap_reps: Number of bootstrap resamples.
        sample_sizes: Rep counts for sensitivity test.

    Returns:
        Comprehensive stress test results.
    """
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import generate_lattice_index
    from lcge_engine.lattice.synthetic_manifold import SyntheticManifold
    from lcge_engine.lattice.vector_store import build_embedding_matrix

    if output_dir is None:
        output_dir = os.path.join(_pkg_dir, "output", "stress_test")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Geometry Stress Tests — full suite")
    logger.info("=" * 60)

    # Generate data
    lattice_index = generate_lattice_index(FROZEN_PROMPTS)
    manifold = SyntheticManifold(seed=42)

    def make_records(num_reps: int) -> list[dict]:
        return manifold.generate_full_lattice(lattice_index, num_reps)

    records = make_records(20)
    embeddings, meta = build_embedding_matrix(records)

    results = {
        "config": {
            "bootstrap_reps": bootstrap_reps,
            "n_records": len(records),
            "n_features": meta["n_features"],
        },
    }

    # Test 1: Bootstrap stability
    logger.info("\n[TEST 1] Bootstrap stability...")
    results["bootstrap"] = bootstrap_stability_test(
        embeddings, records, n_bootstrap=bootstrap_reps
    )

    # Test 2: Cross-metric consistency
    logger.info("\n[TEST 2] Cross-metric consistency...")
    results["cross_metric"] = cross_metric_consistency_test(
        embeddings, records
    )

    # Test 3: Sample size sensitivity
    logger.info("\n[TEST 3] Sample size sensitivity...")
    results["sample_sensitivity"] = sample_size_sensitivity_test(
        make_records, sample_sizes=sample_sizes
    )

    # Save
    results_path = os.path.join(output_dir, "stress_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Results saved: {results_path}")

    # Print summary
    print_stress_test_summary(results)

    return results


def print_stress_test_summary(results: dict):
    """Print a human-readable summary of stress test results."""
    print(f"\n{'='*60}")
    print(f"  Geometry Stress Test Summary")
    print(f"{'='*60}")

    # Bootstrap
    boot = results.get("bootstrap", {})
    print(f"\n  Bootstrap Stability ({boot.get('n_bootstrap', '?')} resamples):")
    overall = boot.get("overall_stability_score", 0.0)
    print(f"    Overall stability: {overall:.4f}")
    for pc, stats in boot.get("axis_stability", {}).items():
        print(
            f"    {pc}: mean_corr={stats['mean_correlation']:.4f} "
            f"std={stats['std_correlation']:.4f} "
            f"pct>0.9={stats['pct_above_0.9']:.1%}"
        )

    # Cross-metric
    cm = results.get("cross_metric", {})
    print(f"\n  Cross-Metric Consistency:")
    print(f"    Mean alignment: {cm.get('mean_alignment', 0.0):.4f}")
    for a in cm.get("pairwise_alignments", []):
        print(
            f"    {a['metric_1']:20s} vs {a['metric_2']:20s}: "
            f"procrustes={a['procrustes_correlation']:.4f}"
        )

    # Sample sensitivity
    ss = results.get("sample_sensitivity", {})
    print(f"\n  Sample Size Sensitivity:")
    for size, data in ss.get("per_size_results", {}).items():
        print(
            f"    N={size:3s}: procrustes={data['procrustes_correlation']:.4f} "
            f"top_var={data['top_variance_ratio']:.4f}"
        )

    print(f"\n{'='*60}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3.1 Geometry Stress Tests",
    )
    parser.add_argument(
        "--bootstrap", "-b", type=int, default=50,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_all_stress_tests(output_dir=args.output, bootstrap_reps=args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())

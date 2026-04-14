"""
coordinate_solver.py — Distance matrix + PCA/SVD decomposition (v1.3.1).

Takes embedding vectors and discovers the empirical coordinate system
of the behavioral response manifold.

Supported distance metrics:
    cosine          — 1 - cosine_similarity (default)
    euclidean       — L2 distance (sklearn)
    manhattan       — L1 distance (sklearn)
    centered_cosine — cosine on mean-centered vectors
    rank_distance   — 1 - Spearman rank correlation (ordering-based)

Outputs:
    1. Pairwise response distance matrix (configurable metric)
    2. PCA decomposition (principal components + variance ratios)
    3. Projection of each run into the discovered coordinate space
    4. Strategy/axis displacement analysis
    5. (NEW) Multi-metric comparison for basis stability testing

NO classification. NO instability types. NO triggers.
Only empirical geometry.
"""

import json
import os
import sys
import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import (
    cosine_similarity as sklearn_cosine_sim,
    euclidean_distances,
    manhattan_distances,
)

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

logger = logging.getLogger("lattice")

SUPPORTED_METRICS = ["cosine", "euclidean", "manhattan", "centered_cosine", "rank_distance"]


# ============================================================
# Distance metrics
# ============================================================

def _cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance: D = 1 - cos(a, b)."""
    sim = sklearn_cosine_sim(embeddings)
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _euclidean_distance(embeddings: np.ndarray) -> np.ndarray:
    """L2 (Euclidean) distance."""
    dist = euclidean_distances(embeddings)
    np.fill_diagonal(dist, 0.0)
    return dist


def _manhattan_distance(embeddings: np.ndarray) -> np.ndarray:
    """L1 (Manhattan/Cityblock) distance."""
    dist = manhattan_distances(embeddings)
    np.fill_diagonal(dist, 0.0)
    return dist


def _centered_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance on mean-centered vectors."""
    centered = embeddings - embeddings.mean(axis=0)
    return _cosine_distance(centered)


def _rank_distance(embeddings: np.ndarray) -> np.ndarray:
    """
    Rank-based distance: 1 - cosine of column-wise ranks.

    Approximates Spearman rank correlation structure without
    requiring pairwise scipy calls. Ranks each feature column
    independently, then applies cosine distance on the rank matrix.

    This metric is invariant to monotonic transformations of
    individual features — it captures only ordinal structure.
    """
    n, p = embeddings.shape
    ranked = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        order = np.argsort(embeddings[:, j])
        ranked[order, j] = np.arange(1, n + 1, dtype=np.float64)
    return _cosine_distance(ranked)


_DISTANCE_FUNCS = {
    "cosine": _cosine_distance,
    "euclidean": _euclidean_distance,
    "manhattan": _manhattan_distance,
    "centered_cosine": _centered_cosine_distance,
    "rank_distance": _rank_distance,
}


def compute_distance_matrix(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise distance matrix using the specified metric.

    Args:
        embeddings: Matrix of shape (n_samples, n_features).
        metric: One of SUPPORTED_METRICS.

    Returns:
        Distance matrix of shape (n_samples, n_samples).
        Diagonal is always 0.
    """
    if metric not in _DISTANCE_FUNCS:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {SUPPORTED_METRICS}"
        )

    logger.info(f"  Computing {metric} distance ({embeddings.shape[0]} samples)...")
    distance_matrix = _DISTANCE_FUNCS[metric](embeddings)

    logger.info(f"  Distance matrix stats ({metric}):")
    nonzero = distance_matrix[distance_matrix > 0]
    logger.info(f"    mean: {distance_matrix.mean():.4f}")
    logger.info(f"    std:  {distance_matrix.std():.4f}")
    if nonzero.size > 0:
        logger.info(f"    min (nonzero): {nonzero.min():.4f}")
    logger.info(f"    max:  {distance_matrix.max():.4f}")

    return distance_matrix


# ============================================================
# PCA decomposition
# ============================================================

def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 8,
) -> dict:
    """
    Compute PCA decomposition of the embedding matrix.

    Discovers principal behavioral directions from data.

    Args:
        embeddings: Matrix of shape (n_samples, n_features).
        n_components: Number of principal components to compute.

    Returns:
        dict with:
            - "principal_components": list of component info dicts
            - "components_matrix": np.ndarray (n_components, n_features)
            - "projections": np.ndarray (n_samples, n_components)
            - "explained_variance": np.ndarray
            - "explained_variance_ratio": np.ndarray
    """
    logger.info(f"  Running PCA (k={n_components}) on {embeddings.shape}...")

    max_components = min(embeddings.shape[0], embeddings.shape[1], n_components)
    if max_components < 2:
        logger.warning(f"  Not enough data for PCA (max {max_components} components)")
        return {
            "principal_components": [],
            "components_matrix": np.zeros((1, embeddings.shape[1])),
            "projections": np.zeros((embeddings.shape[0], 1)),
            "explained_variance": np.array([0.0]),
            "explained_variance_ratio": np.array([0.0]),
        }

    pca = PCA(n_components=max_components)
    projections = pca.fit_transform(embeddings)

    pc_info = []
    cumulative = 0.0
    for i in range(max_components):
        ratio = pca.explained_variance_ratio_[i]
        cumulative += ratio
        pc_info.append({
            "index": i,
            "variance_ratio": round(float(ratio), 6),
            "cumulative_variance": round(float(cumulative), 6),
            "eigenvalue": round(float(pca.explained_variance_[i]), 6),
        })

    logger.info(f"  PCA results:")
    for pc in pc_info:
        logger.info(
            f"    PC{pc['index']}: {pc['variance_ratio']:.4f} variance "
            f"(cum: {pc['cumulative_variance']:.4f})"
        )

    return {
        "principal_components": pc_info,
        "components_matrix": pca.components_,
        "projections": projections,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


# ============================================================
# Displacement analysis
# ============================================================

def compute_strategy_displacement(
    records: list[dict],
    projections: np.ndarray,
) -> dict:
    """
    Compute mean displacement per perturbation strategy.

    For each strategy, compute centroid of projections,
    then measure distance from global centroid.

    Args:
        records: Run records with "strategy" and "row_index".
        projections: PCA projection matrix (n_samples, n_components).

    Returns:
        dict mapping strategy name to displacement info.
    """
    from collections import defaultdict

    strategy_indices = defaultdict(list)
    for i, rec in enumerate(records):
        strategy_indices[rec.get("strategy", "unknown")].append(i)

    global_centroid = projections.mean(axis=0)

    displacements = {}
    for strategy, indices in strategy_indices.items():
        if not indices:
            continue
        strategy_centroid = projections[indices].mean(axis=0)
        displacement = np.linalg.norm(strategy_centroid - global_centroid)
        displacements[strategy] = {
            "n_samples": len(indices),
            "centroid": [round(float(x), 6) for x in strategy_centroid],
            "displacement_from_global": round(float(displacement), 6),
        }

    sorted_strategies = sorted(
        displacements.items(),
        key=lambda x: x[1]["displacement_from_global"],
        reverse=True,
    )

    logger.info(f"  Strategy displacements (from global centroid):")
    for strategy, info in sorted_strategies:
        logger.info(
            f"    {strategy:25s}: displacement={info['displacement_from_global']:.4f} "
            f"(n={info['n_samples']})"
        )

    return dict(sorted_strategies)


def compute_axis_displacement(
    records: list[dict],
    projections: np.ndarray,
) -> dict:
    """
    Compute mean displacement per perturbation axis (v1.3.1).

    Aggregates all strategies within an axis into a single centroid.

    Args:
        records: Run records with "axis" and "row_index".
        projections: PCA projection matrix (n_samples, n_components).

    Returns:
        dict mapping axis name to displacement info.
    """
    from collections import defaultdict

    axis_indices = defaultdict(list)
    for i, rec in enumerate(records):
        axis_indices[rec.get("axis", "unknown")].append(i)

    global_centroid = projections.mean(axis=0)

    displacements = {}
    for axis, indices in axis_indices.items():
        if not indices:
            continue
        axis_centroid = projections[indices].mean(axis=0)
        displacement = np.linalg.norm(axis_centroid - global_centroid)
        # Also compute within-axis spread (std of per-rep projections)
        axis_projections = projections[indices]
        axis_spread = np.std(axis_projections, axis=0)
        displacements[axis] = {
            "n_samples": len(indices),
            "n_strategies": len(set(
                records[i].get("strategy", "") for i in indices
            )),
            "centroid": [round(float(x), 6) for x in axis_centroid],
            "displacement_from_global": round(float(displacement), 6),
            "within_axis_spread": [round(float(x), 6) for x in axis_spread],
            "total_within_axis_spread": round(float(np.linalg.norm(axis_spread)), 6),
        }

    sorted_axes = sorted(
        displacements.items(),
        key=lambda x: x[1]["displacement_from_global"],
        reverse=True,
    )

    logger.info(f"  Axis displacements (from global centroid):")
    for axis, info in sorted_axes:
        logger.info(
            f"    {axis:25s}: displacement={info['displacement_from_global']:.4f} "
            f"spread={info['total_within_axis_spread']:.4f} "
            f"(n={info['n_samples']}, strategies={info['n_strategies']})"
        )

    return dict(sorted_axes)


# ============================================================
# Multi-metric comparison (v1.3.1)
# ============================================================

def compare_metrics(
    embeddings: np.ndarray,
    metrics: Optional[list[str]] = None,
    n_components: int = 8,
) -> dict:
    """
    Run PCA with multiple distance metrics and compare axes.

    This is the key function for basis stability testing.
    Different metrics define different geometries; if PCA axes
    are stable across metrics, the discovered basis is robust.

    For each metric, computes:
        1. Distance matrix
        2. PCA decomposition
        3. Top-k variance ratios
        4. Strategy displacements

    Then computes cross-metric axis alignment via
    component correlation (subspace overlap).

    Args:
        embeddings: Embedding matrix (n_samples, n_features).
        metrics: List of metrics to compare (default: all 5).
        n_components: Number of PCA components per metric.

    Returns:
        dict with per-metric results and cross-metric alignment.
    """
    if metrics is None:
        metrics = SUPPORTED_METRICS

    results = {}

    for metric in metrics:
        logger.info(f"  --- Metric: {metric} ---")

        # Distance matrix
        dist_matrix = compute_distance_matrix(embeddings, metric)

        # PCA on distance matrix
        # Note: PCA is applied to the embeddings themselves (not the distance matrix),
        # but the distance matrix is saved for downstream use.
        # For distance-based PCA, we use MDS-style projection:
        # double-center the distance matrix, then eigendecompose.
        if metric != "cosine":
            # Use classical MDS (PCoA) on the distance matrix
            projections = _pcoa(dist_matrix, n_components)
            # Compute variance ratios from projection spread
            var_ratios = _variance_ratios_from_projections(projections)
        else:
            # For cosine, use standard PCA on embeddings
            pca_result = compute_pca(embeddings, n_components)
            projections = pca_result["projections"]
            var_ratios = pca_result["explained_variance_ratio"]

        # Distance matrix summary
        nonzero = dist_matrix[dist_matrix > 0]
        results[metric] = {
            "distance_matrix_summary": {
                "mean": round(float(dist_matrix.mean()), 6),
                "std": round(float(dist_matrix.std()), 6),
                "min_nonzero": round(float(nonzero.min()), 6) if nonzero.size > 0 else 0.0,
                "max": round(float(dist_matrix.max()), 6),
            },
            "variance_ratios": [round(float(v), 6) for v in var_ratios[:n_components]],
            "cumulative_variance": round(float(sum(var_ratios[:n_components])), 6),
            "projection_shape": list(projections.shape),
            "projections_sample": [
                [round(float(x), 4) for x in row]
                for row in projections[:5]
            ],
        }

    # Cross-metric alignment: correlation of top-k projection axes
    alignment = _compute_cross_metric_alignment(results, metrics)
    results["_cross_metric_alignment"] = alignment

    return results


def _pcoa(
    distance_matrix: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Principal Coordinates Analysis (classical MDS).

    Takes a distance matrix and projects into Euclidean space
    via eigendecomposition of the double-centered inner product matrix.

    Args:
        distance_matrix: Pairwise distance matrix (n x n).
        n_components: Number of dimensions to keep.

    Returns:
        Projection matrix (n_samples, n_components).
    """
    n = distance_matrix.shape[0]

    # Double-centering: B = -0.5 * H * D^2 * H
    H = np.eye(n) - np.ones((n, n)) / n
    D_sq = distance_matrix ** 2
    B = -0.5 * H @ D_sq @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep positive eigenvalues only
    k = min(n_components, np.sum(eigenvalues > 1e-10))
    if k < 1:
        return np.zeros((n, n_components))

    L = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    projections = eigenvectors[:, :k] @ L

    # Pad if fewer components than requested
    if k < n_components:
        pad = np.zeros((n, n_components - k))
        projections = np.hstack([projections, pad])

    return projections


def _variance_ratios_from_projections(projections: np.ndarray) -> np.ndarray:
    """Compute variance ratio per projection column."""
    variances = np.var(projections, axis=0)
    total = variances.sum()
    if total < 1e-15:
        return np.zeros(projections.shape[1])
    return variances / total


def _compute_cross_metric_alignment(
    metric_results: dict,
    metrics: list[str],
) -> list[dict]:
    """
    Measure axis alignment between pairs of metrics.

    For each pair (m1, m2), compute the correlation between
    their top-3 projection columns. High correlation means
    the metrics discover similar axes.

    Returns:
        List of pairwise alignment dicts.
    """
    alignments = []

    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            m1, m2 = metrics[i], metrics[j]
            p1 = np.array(metric_results[m1]["projections_sample"])
            p2 = np.array(metric_results[m2]["projections_sample"])

            if p1.shape[1] == 0 or p2.shape[1] == 0:
                alignments.append({
                    "metric_1": m1,
                    "metric_2": m2,
                    "correlation": 0.0,
                    "note": "empty projections",
                })
                continue

            # Correlation of first projection axis
            min_cols = min(p1.shape[1], p2.shape[1])
            correlations = []
            for k in range(min(min_cols, 3)):
                c1 = p1[:, k] - p1[:, k].mean()
                c2 = p2[:, k] - p2[:, k].mean()
                norm1 = np.linalg.norm(c1)
                norm2 = np.linalg.norm(c2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    corr = float(np.dot(c1, c2) / (norm1 * norm2))
                else:
                    corr = 0.0
                correlations.append(round(corr, 4))

            alignments.append({
                "metric_1": m1,
                "metric_2": m2,
                "axis_correlations": correlations,
                "mean_correlation": round(float(np.mean(correlations)), 4) if correlations else 0.0,
            })

    return alignments


# ============================================================
# Full pipeline
# ============================================================

def solve_behavioral_space(
    jsonl_path: str,
    n_components: int = 8,
    metric: str = "cosine",
    output_path: Optional[str] = None,
) -> str:
    """
    Full pipeline: load records -> embed -> distance -> PCA -> output.

    Args:
        jsonl_path: Path to JSONL run records.
        n_components: Number of PCA components.
        metric: Distance metric to use.
        output_path: Path to save behavioral_space.json.

    Returns:
        Path to the output file.
    """
    from lcge_engine.lattice.vector_store import (
        load_run_records,
        build_embedding_matrix,
        save_embeddings,
        save_records_for_solver,
    )

    logger.info("=" * 60)
    logger.info(f"Coordinate Solver — discovering behavioral basis ({metric})")
    logger.info("=" * 60)

    # Load records
    logger.info("[1/6] Loading run records...")
    records = load_run_records(jsonl_path)
    logger.info(f"  Loaded {len(records)} records")

    # Build embeddings
    logger.info("[2/6] Building embedding matrix...")
    embeddings, meta = build_embedding_matrix(records)

    # Output directory
    output_dir = (
        os.path.dirname(output_path)
        if output_path
        else os.path.join(_pkg_dir, "output")
    )
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npz")
    save_embeddings(
        embeddings,
        [r.get("run_id", "") for r in records],
        emb_path,
    )

    idx_path = os.path.join(output_dir, "record_index.json")
    save_records_for_solver(records, embeddings, idx_path)

    # Distance matrix
    logger.info("[3/6] Computing distance matrix...")
    distance_matrix = compute_distance_matrix(embeddings, metric)

    dist_path = os.path.join(output_dir, f"distance_matrix_{metric}.npz")
    np.savez_compressed(dist_path, distance_matrix=distance_matrix)
    logger.info(f"  Saved distance matrix to {dist_path}")

    # PCA
    logger.info("[4/6] Computing PCA decomposition...")
    pca_result = compute_pca(embeddings, n_components)

    # Strategy displacement
    logger.info("[5/6] Computing strategy displacements...")
    strategy_displacement = compute_strategy_displacement(
        records, pca_result["projections"]
    )

    # Axis displacement (v1.3.1)
    logger.info("[5.5/6] Computing axis displacements...")
    axis_displacement = compute_axis_displacement(
        records, pca_result["projections"]
    )

    # Build output
    if output_path is None:
        output_path = os.path.join(output_dir, "behavioral_space.json")

    projections_dict = {}
    for i, rec in enumerate(records):
        run_id = rec.get("run_id", f"run_{i}")
        proj = pca_result["projections"][i]
        projections_dict[run_id] = [round(float(x), 6) for x in proj]

    nonzero = distance_matrix[distance_matrix > 0]

    output = {
        "method": f"PCA on TF-IDF response embeddings ({metric})",
        "parameters": {
            "n_records": meta["n_records"],
            "n_features": meta["n_features"],
            "n_components": len(pca_result["principal_components"]),
            "distance_metric": metric,
            "empty_responses": meta["empty_count"],
        },
        "principal_components": pca_result["principal_components"],
        "distance_matrix_summary": {
            "metric": metric,
            "n_samples": len(records),
            "mean_distance": round(float(distance_matrix.mean()), 6),
            "std_distance": round(float(distance_matrix.std()), 6),
            "min_nonzero_distance": (
                round(float(nonzero.min()), 6) if nonzero.size > 0 else 0.0
            ),
            "max_distance": round(float(distance_matrix.max()), 6),
        },
        "strategy_displacements": strategy_displacement,
        "axis_displacements": axis_displacement,
        "projections": projections_dict,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3.1 Coordinate Solver — discover behavioral basis from data",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to JSONL run records",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output path for behavioral_space.json",
    )
    parser.add_argument(
        "--components", "-k", type=int, default=8,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--metric", "-m", type=str, default="cosine",
        choices=SUPPORTED_METRICS,
        help="Distance metric to use",
    )

    args = parser.parse_args()

    path = solve_behavioral_space(
        jsonl_path=args.input,
        n_components=args.components,
        metric=args.metric,
        output_path=args.output,
    )

    print(f"\nBehavioral space saved: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

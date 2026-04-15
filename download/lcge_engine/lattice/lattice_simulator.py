"""
lattice_simulator.py — Pre-LLM lattice simulator (v1.3.1).

Runs the full LCGE lattice pipeline with synthetic data.
No API calls needed. Pure local computation.

Capabilities:
    1. Full pipeline: lattice index -> synthetic responses -> embeddings
       -> distance matrix -> PCA -> displacement analysis
    2. Multi-metric comparison: run pipeline with all 5 distance metrics
    3. Manifold shape assessment: spherical, anisotropic, clustered, curved
    4. Strategy/axis sensitivity ranking
    5. Response class distribution analysis

Usage:
    python -m lcge_engine.lattice.lattice_simulator
    python -m lcge_engine.lattice.lattice_simulator --reps 20 --metrics all
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone
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
# Response metadata extraction
# ============================================================

REFUSAL_PATTERNS = [
    "cannot provide", "unable to", "i must decline",
    "i'm sorry", "not something i", "outside the scope",
    "don't have enough", "i can't", "i cannot",
]


def extract_response_metadata(response: str) -> dict:
    """
    Extract free metadata signals from a response string.

    These are signals that cost nothing to capture and provide
    additional dimensions for behavioral analysis.

    Args:
        response: Raw response text.

    Returns:
        dict with metadata fields.
    """
    lower = response.lower()
    word_count = len(response.split())
    char_count = len(response)
    sentence_count = max(1, response.count(".") + response.count("!") + response.count("?"))

    is_refusal = any(p in lower for p in REFUSAL_PATTERNS)
    starts_with_disclaimer = lower.startswith(("i'm", "i am", "as an", "i cannot"))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(word_count / sentence_count, 2),
        "is_refusal": is_refusal,
        "starts_with_disclaimer": starts_with_disclaimer,
    }


# ============================================================
# Manifold shape analysis
# ============================================================

def assess_manifold_shape(projections: np.ndarray) -> dict:
    """
    Assess the geometric shape of the response manifold in PCA space.

    Classifies the manifold as one of:
        - spherical: similar variance across all PCs
        - anisotropic: one or a few PCs dominate
        - clustered: multimodal distribution (multiple dense regions)
        - planar: most variance in 2D subspace
        - linear: most variance along single axis

    Also computes quantitative shape descriptors.

    Args:
        projections: PCA projection matrix (n_samples, n_components).

    Returns:
        dict with shape assessment and quantitative descriptors.
    """
    n_samples, n_components = projections.shape

    # Variance per PC
    variances = np.var(projections, axis=0)
    total_var = variances.sum()
    if total_var < 1e-15:
        return {"shape": "degenerate", "note": "zero variance"}

    # Normalized variance ratios
    var_ratios = variances / total_var

    # Top-k cumulative variance
    cum_var = np.cumsum(var_ratios)

    # Shape classification
    shape = "anisotropic"
    details = []

    if n_components >= 2:
        # Anisotropy ratio: PC1 variance / mean of remaining PCs
        if var_ratios[0] > 0.8:
            shape = "linear"
            details.append(f"PC1 captures {var_ratios[0]:.1%} of variance")
        elif cum_var[1] > 0.9:
            shape = "planar"
            details.append(f"PC1+PC2 capture {cum_var[1]:.1%} of variance")
        elif cum_var[min(3, n_components - 1)] > 0.95:
            shape = "anisotropic"
            details.append(f"Top-4 PCs capture {cum_var[min(3, n_components - 1)]:.1%}")
        else:
            shape = "spherical"
            details.append("Variance distributed across many components")

        # Check for clustering: kurtosis of each PC
        from scipy.stats import kurtosis as _kurtosis
        kurtosis_values = []
        for k in range(min(n_components, 5)):
            try:
                kurt = _kurtosis(projections[:, k])
                kurtosis_values.append(float(kurt))
            except Exception:
                kurtosis_values.append(0.0)

        # High kurtosis indicates clustering (heavy tails / multimodal)
        max_kurtosis = max(abs(k) for k in kurtosis_values)
        if max_kurtosis > 3.0:
            shape = "clustered"
            details.append(
                f"High kurtosis detected ({max_kurtosis:.2f}), suggesting multimodal structure"
            )

    # Quantitative descriptors
    # Global centroid
    centroid = projections.mean(axis=0)

    # Pairwise distances between projections (in PCA space)
    from sklearn.metrics.pairwise import euclidean_distances as _euc
    pca_dist = _euc(projections)
    np.fill_diagonal(pca_dist, 0.0)

    descriptors = {
        "eigenvalue_spectrum": [round(float(v), 6) for v in variances],
        "variance_ratios": [round(float(v), 6) for v in var_ratios],
        "cumulative_variance": [round(float(v), 6) for v in cum_var],
        "effective_dimensionality": round(
            float(np.exp(-np.sum(var_ratios * np.log(var_ratios + 1e-15)))),
            4,
        ),
        "max_pca_distance": round(float(pca_dist.max()), 4),
        "mean_pca_distance": round(float(pca_dist.mean()), 4),
        "std_pca_distance": round(float(pca_dist.std()), 4),
        "centroid": [round(float(x), 4) for x in centroid],
    }

    return {
        "shape": shape,
        "details": details,
        **descriptors,
    }


# ============================================================
# Response conditioning analysis
# ============================================================

def analyze_response_conditioning(records: list[dict]) -> dict:
    """
    Analyze free response metadata across the lattice.

    Computes per-strategy and per-axis distributions of:
        - response length (char/word count)
        - refusal frequency
        - verbosity (words per sentence)

    Args:
        records: Run records (with optional metadata fields).

    Returns:
        dict with conditioning analysis results.
    """
    from collections import defaultdict

    # Extract metadata for all records
    enriched = []
    for rec in records:
        response = rec.get("response", "")
        meta = extract_response_metadata(response)
        enriched.append({**rec, "response_meta": meta})

    # Per-strategy stats
    strategy_stats = defaultdict(lambda: {
        "char_counts": [], "word_counts": [], "refusal_count": 0,
        "disclaimer_count": 0, "n": 0,
    })

    axis_stats = defaultdict(lambda: {
        "char_counts": [], "word_counts": [], "refusal_count": 0,
        "disclaimer_count": 0, "n": 0,
    })

    for rec in enriched:
        meta = rec["response_meta"]
        strategy = rec.get("strategy", "unknown")
        axis = rec.get("axis", "unknown")

        for stats in [strategy_stats[strategy], axis_stats[axis]]:
            stats["char_counts"].append(meta["char_count"])
            stats["word_counts"].append(meta["word_count"])
            if meta["is_refusal"]:
                stats["refusal_count"] += 1
            if meta["starts_with_disclaimer"]:
                stats["disclaimer_count"] += 1
            stats["n"] += 1

    # Summarize
    def summarize(name, stats_dict):
        result = {}
        for key, stats in sorted(stats_dict.items()):
            if stats["n"] == 0:
                continue
            chars = np.array(stats["char_counts"])
            words = np.array(stats["word_counts"])
            result[key] = {
                "n_samples": stats["n"],
                "mean_char_count": round(float(chars.mean()), 2),
                "std_char_count": round(float(chars.std()), 2),
                "mean_word_count": round(float(words.mean()), 2),
                "std_word_count": round(float(words.std()), 2),
                "refusal_rate": round(stats["refusal_count"] / stats["n"], 4),
                "disclaimer_rate": round(stats["disclaimer_count"] / stats["n"], 4),
            }
        return result

    return {
        "per_strategy": summarize("strategy", strategy_stats),
        "per_axis": summarize("axis", axis_stats),
    }


# ============================================================
# Simulator
# ============================================================

class LatticeSimulator:
    """
    Pre-LLM lattice simulator.

    Runs the complete lattice pipeline using synthetic data,
    enabling pipeline validation and geometry analysis without
    any API calls.

    Args:
        seed: Random seed for reproducibility.
        num_reps: Repetitions per lattice point.
    """

    def __init__(self, seed: int = 42, num_reps: int = 20):
        self.seed = seed
        self.num_reps = num_reps
        self.results = {}

    def run(
        self,
        output_dir: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        save_intermediates: bool = True,
    ) -> dict:
        """
        Run the full synthetic lattice pipeline.

        Args:
            output_dir: Directory to save intermediate files.
            metrics: Distance metrics to compute (default: cosine, euclidean).
            save_intermediates: Whether to save .npz and .json files.

        Returns:
            Comprehensive results dict.
        """
        from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
        from lcge_engine.lattice.variant_generator import (
            generate_lattice_index,
            ALL_STRATEGY_NAMES,
            get_all_axes,
            AXIS_STRATEGY_COUNTS,
        )
        from lcge_engine.lattice.synthetic_manifold import SyntheticManifold
        from lcge_engine.lattice.vector_store import (
            build_embedding_matrix,
            save_embeddings,
            save_records_for_solver,
        )
        from lcge_engine.lattice.coordinate_solver import (
            compute_distance_matrix,
            compute_pca,
            compute_strategy_displacement,
            compute_axis_displacement,
            compare_metrics,
            SUPPORTED_METRICS,
        )

        if output_dir is None:
            output_dir = os.path.join(_pkg_dir, "output", "simulation")
        os.makedirs(output_dir, exist_ok=True)

        if metrics is None:
            metrics = ["cosine", "euclidean"]

        logger.info("=" * 60)
        logger.info("Lattice Simulator — full synthetic pipeline")
        logger.info(f"  Seed: {self.seed}, Reps: {self.num_reps}")
        logger.info(f"  Metrics: {metrics}")
        logger.info("=" * 60)

        # Step 1: Generate lattice index
        logger.info("[1/7] Generating lattice index...")
        lattice_index = generate_lattice_index(FROZEN_PROMPTS)
        n_lattice_points = len(lattice_index)
        total_records = n_lattice_points * self.num_reps
        logger.info(f"  Lattice: {n_lattice_points} points ({len(FROZEN_PROMPTS)} prompts x {len(ALL_STRATEGY_NAMES)} strategies)")
        logger.info(f"  Total records: {total_records}")

        # Step 2: Generate synthetic responses
        logger.info("[2/7] Generating synthetic responses...")
        manifold = SyntheticManifold(seed=self.seed)
        records = manifold.generate_full_lattice(lattice_index, self.num_reps)

        class_dist = manifold.get_class_distribution(records)
        logger.info(f"  Response class distribution: {class_dist}")

        # Step 3: Build embeddings
        logger.info("[3/7] Building embedding matrix...")
        embeddings, meta = build_embedding_matrix(records)
        logger.info(f"  Embedding matrix: {embeddings.shape}")

        # Step 4: Response conditioning analysis
        logger.info("[4/7] Analyzing response conditioning...")
        conditioning = analyze_response_conditioning(records)

        # Step 5: Primary metric (cosine) — full pipeline
        primary_metric = metrics[0]
        logger.info(f"[5/7] Primary pipeline ({primary_metric})...")

        dist_matrix = compute_distance_matrix(embeddings, primary_metric)
        pca_result = compute_pca(embeddings, n_components=8)
        strategy_disp = compute_strategy_displacement(records, pca_result["projections"])
        axis_disp = compute_axis_displacement(records, pca_result["projections"])
        manifold_shape = assess_manifold_shape(pca_result["projections"])

        # Step 6: Multi-metric comparison
        logger.info("[6/7] Multi-metric comparison...")
        if len(metrics) > 1:
            metric_comparison = compare_metrics(embeddings, metrics)
        else:
            metric_comparison = None

        # Step 7: Response class geometry
        logger.info("[7/7] Response class geometry analysis...")
        class_geometry = self._analyze_class_geometry(
            records, pca_result["projections"]
        )

        # Build results
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "seed": self.seed,
                "num_reps": self.num_reps,
                "n_prompts": len(FROZEN_PROMPTS),
                "n_strategies": len(ALL_STRATEGY_NAMES),
                "n_lattice_points": n_lattice_points,
                "total_records": total_records,
                "n_features": meta["n_features"],
                "metrics": metrics,
            },
            "axes": {name: count for name, count in AXIS_STRATEGY_COUNTS.items()},
            "response_class_distribution": class_dist,
            "manifold_shape": manifold_shape,
            "strategy_displacements": strategy_disp,
            "axis_displacements": axis_disp,
            "response_conditioning": conditioning,
            "class_geometry": class_geometry,
            "metric_comparison": metric_comparison,
            "principal_components": pca_result["principal_components"],
        }

        # Save intermediates
        if save_intermediates:
            # JSONL records
            jsonl_path = os.path.join(output_dir, "synthetic_runs.jsonl")
            with open(jsonl_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"  Saved records: {jsonl_path}")

            # Embeddings
            emb_path = os.path.join(output_dir, "synthetic_embeddings.npz")
            save_embeddings(
                embeddings,
                [r.get("run_id", "") for r in records],
                emb_path,
            )

            # Record index
            idx_path = os.path.join(output_dir, "synthetic_record_index.json")
            save_records_for_solver(records, embeddings, idx_path)

            # Full results
            results_path = os.path.join(output_dir, "simulation_results.json")
            # Remove numpy arrays for JSON serialization
            serializable = self._make_serializable(self.results)
            with open(results_path, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"  Saved results: {results_path}")

        logger.info("=" * 60)
        logger.info("Simulation complete")
        logger.info("=" * 60)

        return self.results

    def _analyze_class_geometry(
        self, records: list[dict], projections: np.ndarray
    ) -> dict:
        """Analyze how response classes cluster in PCA space."""
        from collections import defaultdict

        class_indices = defaultdict(list)
        for i, rec in enumerate(records):
            rc = rec.get("response_class", "unknown")
            class_indices[rc].append(i)

        results = {}
        global_centroid = projections.mean(axis=0)

        for cls, indices in class_indices.items():
            cls_proj = projections[indices]
            cls_centroid = cls_proj.mean(axis=0)
            displacement = np.linalg.norm(cls_centroid - global_centroid)
            within_spread = np.std(cls_proj, axis=0)
            results[cls] = {
                "n_samples": len(indices),
                "displacement_from_global": round(float(displacement), 6),
                "within_class_spread": round(float(np.linalg.norm(within_spread)), 6),
                "centroid": [round(float(x), 4) for x in cls_centroid],
            }

        # Inter-class distances
        centroids = {cls: np.array(info["centroid"]) for cls, info in results.items()}
        inter_class = {}
        classes = list(centroids.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1, c2 = classes[i], classes[j]
                dist = np.linalg.norm(centroids[c1] - centroids[c2])
                inter_class[f"{c1}_vs_{c2}"] = round(float(dist), 4)

        results["_inter_class_distances"] = inter_class
        return results

    def _make_serializable(self, obj):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    def print_report(self):
        """Print a human-readable summary of simulation results."""
        r = self.results
        if not r:
            print("No results. Run simulate() first.")
            return

        print(f"\n{'='*60}")
        print(f"  LCGE v1.3.1 Lattice Simulation Report")
        print(f"{'='*60}")
        print(f"\n  Config:")
        print(f"    Prompts: {r['config']['n_prompts']}")
        print(f"    Strategies: {r['config']['n_strategies']}")
        print(f"    Lattice points: {r['config']['n_lattice_points']}")
        print(f"    Repetitions: {r['config']['num_reps']}")
        print(f"    Total records: {r['config']['total_records']}")
        print(f"    Embedding features: {r['config']['n_features']}")

        print(f"\n  Manifold shape: {r['manifold_shape']['shape'].upper()}")
        for detail in r['manifold_shape'].get('details', []):
            print(f"    - {detail}")
        eff_dim = r['manifold_shape'].get('effective_dimensionality', '?')
        print(f"    Effective dimensionality: {eff_dim}")

        print(f"\n  Top 5 strategy displacements:")
        sorted_disp = sorted(
            r['strategy_displacements'].items(),
            key=lambda x: x[1]["displacement_from_global"],
            reverse=True,
        )[:5]
        for strategy, info in sorted_disp:
            print(f"    {strategy:25s}: {info['displacement_from_global']:.4f}")

        print(f"\n  Axis displacements:")
        for axis, info in r['axis_displacements'].items():
            print(
                f"    {axis:25s}: displacement={info['displacement_from_global']:.4f} "
                f"spread={info['total_within_axis_spread']:.4f}"
            )

        print(f"\n  Response class distribution:")
        for cls, count in r['response_class_distribution'].items():
            print(f"    {cls:20s}: {count}")

        if r.get('class_geometry'):
            print(f"\n  Response class geometry:")
            for cls, geo in r['class_geometry'].items():
                if cls.startswith("_"):
                    continue
                print(
                    f"    {cls:20s}: displacement={geo['displacement_from_global']:.4f} "
                    f"spread={geo['within_class_spread']:.4f}"
                )

        if r.get('metric_comparison') and r['metric_comparison'].get('_cross_metric_alignment'):
            print(f"\n  Cross-metric axis alignment:")
            for alignment in r['metric_comparison']['_cross_metric_alignment']:
                print(
                    f"    {alignment['metric_1']:20s} vs {alignment['metric_2']:20s}: "
                    f"corr={alignment['mean_correlation']:.4f}"
                )

        print(f"\n{'='*60}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3.1 Lattice Simulator — synthetic pipeline testing",
    )
    parser.add_argument("--reps", "-n", type=int, default=20, help="Repetitions per lattice point")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument(
        "--metrics", "-m", type=str, default="cosine,euclidean",
        help="Comma-separated distance metrics, or 'all' for all 5",
    )

    args = parser.parse_args()

    if args.metrics == "all":
        from lcge_engine.lattice.coordinate_solver import SUPPORTED_METRICS
        metrics = SUPPORTED_METRICS
    else:
        metrics = [m.strip() for m in args.metrics.split(",")]

    sim = LatticeSimulator(seed=args.seed, num_reps=args.reps)
    sim.run(output_dir=args.output, metrics=metrics)
    sim.print_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())

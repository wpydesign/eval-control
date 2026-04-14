"""
solve_real_space.py — Adapt real runs.jsonl → vector_store → coordinate_solver.

Maps field names from run_real_lattice output to what vector_store/coordinate_solver
expect, then runs the full geometry pipeline.

Input:  lattice/output/runs.jsonl (real data, field: raw_response_text)
Output: lattice/output/behavioral_space.json
"""

import json
import os
import sys
import logging

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lattice.solve_real")

OUTPUT_DIR = os.path.join(_pkg_dir, "output")


def adapt_real_records(input_path: str) -> list[dict]:
    """
    Load real runs.jsonl and adapt to vector_store format.

    Maps: raw_response_text → response
    """
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            # Map to vector_store expected format
            adapted = {
                "run_id": raw.get("run_id", ""),
                "prompt_id": raw.get("prompt_id", -1),
                "strategy": raw.get("strategy", ""),
                "axis": raw.get("axis", "unknown"),
                "rep": raw.get("rep", -1),
                "seed_prompt": "",  # Not needed for embedding
                "variant_prompt": "",  # Not needed for embedding
                "response": raw.get("raw_response_text", ""),
                "response_length": raw.get("response_length", 0),
                "word_count": raw.get("word_count", 0),
                "is_refusal": raw.get("is_refusal", False),
                "token_count": raw.get("token_count", 0),
                "finish_reason": raw.get("finish_reason", ""),
                "latency_ms": raw.get("latency_ms", 0),
                "temperature": 0.7,
                "metadata": {
                    "model": raw.get("provider_used", "unknown"),
                    "timestamp": raw.get("timestamp", ""),
                },
            }
            records.append(adapted)

    return records


def main():
    runs_path = os.path.join(OUTPUT_DIR, "runs.jsonl")
    adapted_path = os.path.join(OUTPUT_DIR, "runs_adapted.jsonl")
    space_path = os.path.join(OUTPUT_DIR, "behavioral_space.json")

    logger.info("=" * 60)
    logger.info("STEP 1: Adapting real records for vector_store")
    logger.info("=" * 60)

    records = adapt_real_records(runs_path)
    logger.info(f"  Loaded {len(records)} records from {runs_path}")

    # Write adapted records
    with open(adapted_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote adapted records to {adapted_path}")

    # Verify
    empty = sum(1 for r in records if not r["response"].strip() or r["response"].strip().startswith("["))
    logger.info(f"  Empty/error records: {empty}/{len(records)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: vector_store — building embeddings")
    logger.info("=" * 60)

    from lcge_engine.lattice.vector_store import (
        build_embedding_matrix,
        save_embeddings,
        save_records_for_solver,
    )

    embeddings, meta = build_embedding_matrix(records)
    logger.info(f"  Embedding matrix: {embeddings.shape}")

    emb_path = os.path.join(OUTPUT_DIR, "embeddings.npz")
    save_embeddings(
        embeddings,
        [r.get("run_id", "") for r in records],
        emb_path,
    )

    idx_path = os.path.join(OUTPUT_DIR, "record_index.json")
    save_records_for_solver(records, embeddings, idx_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: coordinate_solver — discovering geometry")
    logger.info("=" * 60)

    from lcge_engine.lattice.coordinate_solver import (
        compute_distance_matrix,
        compute_pca,
        compute_strategy_displacement,
        compute_axis_displacement,
    )

    import numpy as np

    # Distance matrix (cosine)
    distance_matrix = compute_distance_matrix(embeddings, "cosine")

    dist_path = os.path.join(OUTPUT_DIR, "distance_matrix_cosine.npz")
    np.savez_compressed(dist_path, distance_matrix=distance_matrix)
    logger.info(f"  Saved: {dist_path}")

    # PCA
    n_components = min(8, min(embeddings.shape) - 1)
    pca_result = compute_pca(embeddings, n_components)

    # Strategy displacement
    strategy_displacement = compute_strategy_displacement(
        records, pca_result["projections"]
    )

    # Axis displacement
    axis_displacement = compute_axis_displacement(
        records, pca_result["projections"]
    )

    # Build behavioral_space.json
    projections_dict = {}
    for i, rec in enumerate(records):
        run_id = rec.get("run_id", f"run_{i}")
        proj = pca_result["projections"][i]
        projections_dict[run_id] = [round(float(x), 6) for x in proj]

    nonzero = distance_matrix[distance_matrix > 0]

    output = {
        "method": "PCA on TF-IDF response embeddings (cosine)",
        "source": "real data — ollama_local (Qwen2.5-0.5B-Instruct, Q2_K, CPU)",
        "parameters": {
            "n_records": meta["n_records"],
            "n_features": meta["n_features"],
            "n_components": len(pca_result["principal_components"]),
            "distance_metric": "cosine",
            "empty_responses": meta["empty_count"],
            "reps_per_point": 5,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "principal_components": pca_result["principal_components"],
        "distance_matrix_summary": {
            "metric": "cosine",
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

    with open(space_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"")
    logger.info(f"  Output: {space_path}")
    logger.info(f"  File size: {os.path.getsize(space_path)/1024:.0f} KB")
    logger.info(f"")
    logger.info(f"  Top 5 strategy displacements:")
    for s, info in list(strategy_displacement.items())[:5]:
        logger.info(f"    {s:30s}: {info['displacement_from_global']:.4f}")
    logger.info(f"")
    logger.info(f"  Axis displacements:")
    for a, info in axis_displacement.items():
        logger.info(f"    {a:30s}: disp={info['displacement_from_global']:.4f}, spread={info['total_within_axis_spread']:.4f}")
    logger.info(f"")
    logger.info(f"  PCA variance explained:")
    for pc in pca_result["principal_components"]:
        logger.info(f"    PC{pc['index']}: {pc['variance_ratio']:.4f} (cum: {pc['cumulative_variance']:.4f})")

    logger.info(f"")
    logger.info("=" * 60)
    logger.info("BEHAVIORAL SPACE GENERATED")
    logger.info("=" * 60)
    logger.info(f"  runs.jsonl:              {runs_path}")
    logger.info(f"  behavioral_space.json:   {space_path}")
    logger.info(f"  embeddings.npz:          {emb_path}")
    logger.info(f"  record_index.json:       {idx_path}")
    logger.info(f"  distance_matrix_cosine:  {dist_path}")

    return space_path


if __name__ == "__main__":
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""Solve behavioral space for DeepSeek runs."""
import json, os, sys, numpy as np

OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"
sys.path.insert(0, "/home/z/my-project/download")

from lcge_engine.lattice.vector_store import build_embedding_matrix, save_embeddings, save_records_for_solver
from lcge_engine.lattice.coordinate_solver import compute_distance_matrix, compute_pca, compute_strategy_displacement, compute_axis_displacement

def solve(input_jsonl, output_json, label):
    print(f"=== Solving: {label} ===", flush=True)
    
    records = []
    with open(input_jsonl) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                records.append({
                    "run_id": raw.get("run_id", ""),
                    "prompt_id": raw.get("prompt_id", -1),
                    "strategy": raw.get("strategy", ""),
                    "axis": raw.get("axis", "unknown"),
                    "rep": raw.get("rep", -1),
                    "seed_prompt": "",
                    "variant_prompt": "",
                    "response": raw.get("raw_response_text", ""),
                    "response_length": raw.get("response_length", 0),
                    "word_count": raw.get("word_count", 0),
                    "is_refusal": raw.get("is_refusal", False),
                    "token_count": raw.get("token_count", 0),
                    "finish_reason": raw.get("finish_reason", ""),
                    "latency_ms": raw.get("latency_ms", 0),
                    "temperature": 0.7,
                    "metadata": {"model": raw.get("provider_used", "unknown"), "timestamp": raw.get("timestamp", "")},
                })
    
    print(f"  Records: {len(records)}", flush=True)
    
    # Embed
    embeddings, meta = build_embedding_matrix(records)
    print(f"  Embeddings: {embeddings.shape}", flush=True)
    
    # Distance
    dm = compute_distance_matrix(embeddings, "cosine")
    nonzero = dm[dm > 0]
    
    # PCA
    n_components = min(8, min(embeddings.shape) - 1)
    pca_result = compute_pca(embeddings, n_components)
    
    # Displacements
    sd = compute_strategy_displacement(records, pca_result["projections"])
    ad = compute_axis_displacement(records, pca_result["projections"])
    
    # Projections
    proj_dict = {}
    for i, rec in enumerate(records):
        proj_dict[rec.get("run_id", f"run_{i}")] = [round(float(x), 6) for x in pca_result["projections"][i]]
    
    output = {
        "method": "PCA on TF-IDF response embeddings (cosine)",
        "source": f"real data — {label}",
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
            "mean_distance": round(float(dm.mean()), 6),
            "std_distance": round(float(dm.std()), 6),
            "min_nonzero_distance": round(float(nonzero.min()), 6) if nonzero.size > 0 else 0.0,
            "max_distance": round(float(dm.max()), 6),
        },
        "strategy_displacements": sd,
        "axis_displacements": ad,
        "projections": proj_dict,
    }
    
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"  Cosine mean: {dm.mean():.4f}", flush=True)
    print(f"  PCA cum var: {pca_result['principal_components'][-1]['cumulative_variance']:.4f}", flush=True)
    print(f"  Top 5 strategies:", flush=True)
    for s, info in list(sd.items())[:5]:
        print(f"    {s:30s}: {info['displacement_from_global']:.4f}", flush=True)
    print(f"  Output: {output_json}", flush=True)
    return output_json

if __name__ == "__main__":
    solve(
        os.path.join(OUTPUT_DIR, "runs_deepseek.jsonl"),
        os.path.join(OUTPUT_DIR, "behavioral_space_deepseek.json"),
        "deepseek-v3",
    )

#!/usr/bin/env python3
"""Solve all DeepSeek behavioral spaces: run1, repeat, + 4 metric variants."""
import json, os, sys, numpy as np

OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"
sys.path.insert(0, "/home/z/my-project/download")

from lcge_engine.lattice.vector_store import build_embedding_matrix
from lcge_engine.lattice.coordinate_solver import compute_distance_matrix, compute_pca, compute_strategy_displacement, compute_axis_displacement

def load_and_adapt(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                records.append({
                    "run_id": raw.get("run_id", ""),
                    "prompt_id": raw.get("prompt_id", -1),
                    "strategy": raw.get("strategy", ""),
                    "axis": raw.get("axis", "unknown"),
                    "rep": raw.get("rep", -1),
                    "seed_prompt": "", "variant_prompt": "",
                    "response": raw.get("raw_response_text", ""),
                    "response_length": raw.get("response_length", 0),
                    "word_count": raw.get("word_count", 0),
                    "is_refusal": raw.get("is_refusal", False),
                    "token_count": 0, "finish_reason": "", "latency_ms": 0,
                    "temperature": 0.7,
                    "metadata": {"model": raw.get("provider_used", "unknown")},
                })
    return records

def solve(records, metric, label):
    embeddings, meta = build_embedding_matrix(records)
    dm = compute_distance_matrix(embeddings, metric)
    nonzero = dm[dm > 0]
    n_components = min(8, min(embeddings.shape) - 1)

    if metric == "cosine":
        pca_result = compute_pca(embeddings, n_components)
        projections = pca_result["projections"]
    else:
        # MDS-style for non-cosine
        from lcge_engine.lattice.coordinate_solver import _pcoa, _variance_ratios_from_projections
        projections = _pcoa(dm, n_components)
        var_ratios = _variance_ratios_from_projections(projections)
        pca_result = {
            "principal_components": [
                {"index": i, "variance_ratio": round(float(var_ratios[i]), 6),
                 "cumulative_variance": round(float(sum(var_ratios[:i+1])), 6)}
                for i in range(len(var_ratios))
            ],
            "projections": projections,
        }

    sd = compute_strategy_displacement(records, projections)
    ad = compute_axis_displacement(records, projections)
    proj_dict = {}
    for i, rec in enumerate(records):
        proj_dict[rec.get("run_id", f"run_{i}")] = [round(float(x), 6) for x in projections[i]]

    output = {
        "method": f"PCA on TF-IDF response embeddings ({metric})",
        "source": f"real data — {label}",
        "parameters": {
            "n_records": meta["n_records"], "n_features": meta["n_features"],
            "n_components": len(pca_result["principal_components"]),
            "distance_metric": metric, "empty_responses": meta["empty_count"],
            "reps_per_point": 5, "temperature": 0.7, "top_p": 0.9,
        },
        "principal_components": pca_result["principal_components"],
        "distance_matrix_summary": {
            "metric": metric, "n_samples": len(records),
            "mean_distance": round(float(dm.mean()), 6),
            "std_distance": round(float(dm.std()), 6),
            "min_nonzero_distance": round(float(nonzero.min()), 6) if nonzero.size > 0 else 0.0,
            "max_distance": round(float(dm.max()), 6),
        },
        "strategy_displacements": sd, "axis_displacements": ad, "projections": proj_dict,
    }
    return output

# 1. Solve repeat (cosine)
print("=== Solving repeat (cosine) ===", flush=True)
recs_repeat = load_and_adapt(os.path.join(OUTPUT_DIR, "runs_deepseek_repeat.jsonl"))
result = solve(recs_repeat, "cosine", "deepseek-v3 (repeat)")
out_path = os.path.join(OUTPUT_DIR, "behavioral_space_deepseek_repeat.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
dm = result["distance_matrix_summary"]
pc = result["principal_components"]
print(f"  Cosine mean: {dm['mean_distance']:.4f}, PCA cum: {pc[-1]['cumulative_variance']:.4f}", flush=True)
for s, info in list(result["strategy_displacements"].items())[:5]:
    print(f"    {s:30s}: {info['displacement_from_global']:.4f}", flush=True)

# 2. Solve metric variants on run1 data
recs_run1 = load_and_adapt(os.path.join(OUTPUT_DIR, "runs_deepseek.jsonl"))
for metric in ["euclidean", "manhattan", "centered_cosine", "rank_distance"]:
    print(f"\n=== Solving run1 ({metric}) ===", flush=True)
    result = solve(recs_run1, metric, "deepseek-v3")
    out_path = os.path.join(OUTPUT_DIR, f"behavioral_space_deepseek_{metric}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    dm = result["distance_matrix_summary"]
    pc = result["principal_components"]
    print(f"  Mean dist: {dm['mean_distance']:.4f}, PCA cum: {pc[-1]['cumulative_variance']:.4f}", flush=True)
    for s, info in list(result["strategy_displacements"].items())[:3]:
        print(f"    {s:30s}: {info['displacement_from_global']:.4f}", flush=True)

print("\n=== ALL SPACES SOLVED ===", flush=True)

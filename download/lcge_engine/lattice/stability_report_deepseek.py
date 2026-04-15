#!/usr/bin/env python3
"""stability_report_deepseek.py — Full stability report for DeepSeek-V3 runs.

TEST 1: Same model repeat (run1 vs repeat)
TEST 2: Same data, different metrics (cosine vs euclidean/manhattan/centered_cosine/rank_distance)
Note: No Test 3 (cross-model) — only one real API model available.

Uses the same comparison logic as stability_report.py but with DeepSeek spaces.
"""
import json, os, sys, numpy as np

OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"

def load(path):
    with open(path) as f:
        return json.load(f)

def compare_pca(s1, s2):
    pcs_a = s1["principal_components"]
    pcs_b = s2["principal_components"]
    k = min(len(pcs_a), len(pcs_b))
    var_diffs = [round(abs(pcs_a[i]["variance_ratio"] - pcs_b[i]["variance_ratio"]), 6) for i in range(k)]
    order_a = sorted(range(k), key=lambda i: -pcs_a[i]["variance_ratio"])
    order_b = sorted(range(k), key=lambda i: -pcs_b[i]["variance_ratio"])
    return {
        "pc_variance_diffs": var_diffs,
        "max_variance_diff": round(max(var_diffs), 6),
        "mean_variance_diff": round(sum(var_diffs)/len(var_diffs), 6),
        "pc_ordering_preserved": order_a == order_b,
    }

def compare_strategy_overlap(s1, s2, top_k=5):
    sd_a = s1["strategy_displacements"]
    sd_b = s2["strategy_displacements"]
    top_a = set(sorted(sd_a.keys(), key=lambda s: sd_a[s]["displacement_from_global"], reverse=True)[:top_k])
    top_b = set(sorted(sd_b.keys(), key=lambda s: sd_b[s]["displacement_from_global"], reverse=True)[:top_k])
    overlap = top_a & top_b
    return {
        f"top_{top_k}_run1": sorted(top_a),
        f"top_{top_k}_run2": sorted(top_b),
        f"top_{top_k}_overlap": sorted(overlap),
        "overlap_count": len(overlap),
        "overlap_ratio": round(len(overlap)/top_k, 4),
    }

def compute_axis_alignment(s1, s2):
    proj_a = s1.get("projections", {})
    proj_b = s2.get("projections", {})
    common = set(proj_a.keys()) & set(proj_b.keys())
    if len(common) < 10:
        return {"axis_correlations": [], "mean_correlation": 0.0, "n_common_ids": len(common)}
    k = min(len(s1["principal_components"]), len(s2["principal_components"]))
    mat_a = np.array([proj_a[rid] for rid in sorted(common)])
    mat_b = np.array([proj_b[rid] for rid in sorted(common)])
    corrs = []
    for col in range(min(k, mat_a.shape[1], mat_b.shape[1])):
        ca = mat_a[:, col] - mat_a[:, col].mean()
        cb = mat_b[:, col] - mat_b[:, col].mean()
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        corrs.append(round(float(np.dot(ca, cb)/(na*nb)), 4) if na > 1e-10 and nb > 1e-10 else 0.0)
    return {"axis_correlations": corrs, "mean_correlation": round(float(np.mean(corrs)), 4) if corrs else 0.0, "n_common_ids": len(common)}

def main():
    report = {}

    # TEST 1: Same model repeat
    print("TEST 1: Same model, repeat run (DeepSeek-V3)...", flush=True)
    s1 = load(os.path.join(OUTPUT_DIR, "behavioral_space_deepseek.json"))
    s1r = load(os.path.join(OUTPUT_DIR, "behavioral_space_deepseek_repeat.json"))

    dm1 = s1["distance_matrix_summary"]["mean_distance"]
    dm1r = s1r["distance_matrix_summary"]["mean_distance"]

    pca_comp = compare_pca(s1, s1r)
    strat_comp = compare_strategy_overlap(s1, s1r)
    axis_align = compute_axis_alignment(s1, s1r)

    report["test1_repeat_run"] = {
        "model": "DeepSeek-V3",
        "cosine_mean_run1": dm1,
        "cosine_mean_run2": dm1r,
        "cosine_mean_diff": round(abs(dm1 - dm1r), 6),
        "cosine_mean_diff_threshold": 0.02,
        "cosine_mean_diff_pass": abs(dm1 - dm1r) <= 0.02,
        "pca_comparison": pca_comp,
        "strategy_overlap": strat_comp,
        "axis_alignment": axis_align,
        "overall_pass": abs(dm1 - dm1r) <= 0.02 and strat_comp["overlap_count"] >= 3 and pca_comp["pc_ordering_preserved"],
    }

    # TEST 2: Different metrics
    print("TEST 2: Different metrics...", flush=True)
    metrics = ["euclidean", "manhattan", "centered_cosine", "rank_distance"]
    metric_spaces = {}
    for m in metrics:
        path = os.path.join(OUTPUT_DIR, f"behavioral_space_deepseek_{m}.json")
        if os.path.exists(path):
            metric_spaces[m] = load(path)

    test2 = {}
    for m, ms in metric_spaces.items():
        pca_c = compare_pca(s1, ms)
        strat_c = compare_strategy_overlap(s1, ms)
        axis_c = compute_axis_alignment(s1, ms)
        test2[m] = {
            "distance_mean": ms["distance_matrix_summary"]["mean_distance"],
            "pca_max_var_diff": pca_c["max_variance_diff"],
            "pca_ordering_preserved": pca_c["pc_ordering_preserved"],
            "strategy_overlap_count": strat_c["overlap_count"],
            "strategy_overlap_ratio": strat_c["overlap_ratio"],
            "axis_mean_correlation": axis_c["mean_correlation"],
            "geometry_collapsed": axis_c["mean_correlation"] < 0.1,
        }

    any_collapse = any(v["geometry_collapsed"] for v in test2.values())
    mean_corr = np.mean([v["axis_mean_correlation"] for v in test2.values()])

    report["test2_metric_stability"] = {
        "metrics_tested": list(metrics),
        "per_metric": test2,
        "mean_axis_correlation_across_metrics": round(float(mean_corr), 4),
        "any_metric_collapse": any_collapse,
        "overall_pass": not any_collapse and mean_corr > 0.1,
    }

    # VERDICT
    t1 = report["test1_repeat_run"]["overall_pass"]
    t2 = report["test2_metric_stability"]["overall_pass"]

    report["verdict"] = {
        "test1_stable_across_runs": t1,
        "test2_not_collapsing_across_metrics": t2,
        "geometry_is_real": t1 and t2,
        "geometry_is_artifact": not (t1 and t2),
    }

    # Write
    out_path = os.path.join(OUTPUT_DIR, "stability_report_deepseek.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print
    print(f"\n{'='*60}", flush=True)
    print("DEEPSEEK-V3 STABILITY REPORT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nTEST 1 (same model, repeat run):", flush=True)
    print(f"  cosine_mean: run1={dm1:.4f}, run2={dm1r:.4f}, diff={abs(dm1-dm1r):.4f} (threshold: 0.02)", flush=True)
    print(f"  strategy overlap: {strat_comp['overlap_count']}/5 (threshold: 3)", flush=True)
    print(f"  PCA ordering preserved: {pca_comp['pc_ordering_preserved']}", flush=True)
    print(f"  axis alignment: {axis_align['mean_correlation']}", flush=True)
    print(f"  axis correlations per PC: {axis_align['axis_correlations']}", flush=True)
    print(f"  PASS: {t1}", flush=True)

    print(f"\nTEST 2 (different metrics):", flush=True)
    print(f"  mean axis correlation: {mean_corr:.4f}", flush=True)
    print(f"  any collapse: {any_collapse}", flush=True)
    for m, res in test2.items():
        print(f"    {m}: corr={res['axis_mean_correlation']:.4f}, overlap={res['strategy_overlap_count']}/5, collapsed={res['geometry_collapsed']}", flush=True)
    print(f"  PASS: {t2}", flush=True)

    print(f"\nVERDICT: {'GEOMETRY IS REAL' if report['verdict']['geometry_is_real'] else 'GEOMETRY IS ARTIFACT'}", flush=True)
    print(f"{'='*60}", flush=True)

    # Also compare with Qwen results for reference
    print(f"\n--- Cross-reference with Qwen2.5-0.5B ---", flush=True)
    try:
        sq = load(os.path.join(OUTPUT_DIR, "behavioral_space.json"))
        print(f"  Qwen cosine mean:  {sq['distance_matrix_summary']['mean_distance']:.4f}", flush=True)
        print(f"  DeepSeek cosine mean: {dm1:.4f}", flush=True)
        cross_overlap = compare_strategy_overlap(sq, s1)
        print(f"  Strategy overlap Qwen vs DeepSeek: {cross_overlap['overlap_count']}/5", flush=True)
        print(f"  Qwen top 5:     {cross_overlap['top_5_run1']}", flush=True)
        print(f"  DeepSeek top 5: {cross_overlap['top_5_run2']}", flush=True)
        cross_axis = compute_axis_alignment(sq, s1)
        print(f"  Axis alignment Qwen vs DeepSeek: {cross_axis['mean_correlation']:.4f}", flush=True)
    except Exception as e:
        print(f"  Could not load Qwen space: {e}", flush=True)

    return out_path

if __name__ == "__main__":
    main()

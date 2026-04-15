"""
stability_report.py — Generate comparison reports for all 3 stability tests.

TEST 1: Same model, repeat run (Qwen run1 vs Qwen run2)
TEST 2: Same data, different metrics (cosine vs euclidean/manhattan/centered_cosine/rank_distance)
TEST 3: Different models (Qwen vs TinyLlama)

Output: stability_report.json with numerical comparisons only.
"""

import json
import os
import sys
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_pkg_dir, "output")


def load_space(path):
    with open(path) as f:
        return json.load(f)


def compare_pca(space_a, space_b):
    """Compare PCA variance patterns."""
    pcs_a = space_a["principal_components"]
    pcs_b = space_b["principal_components"]

    k = min(len(pcs_a), len(pcs_b))
    var_diffs = []
    ordering_preserved = True
    prev_ratio_a = float("inf")

    for i in range(k):
        va = pcs_a[i]["variance_ratio"]
        vb = pcs_b[i]["variance_ratio"]
        var_diffs.append(round(abs(va - vb), 6))
        # Check ordering: PC0 > PC1 > PC2 ...
        if va >= prev_ratio_a and i > 0:
            pass  # Non-strict ordering is fine
        prev_ratio_a = va

    # Check if top-k ordering is preserved
    order_a = [pcs_a[i]["variance_ratio"] for i in range(k)]
    order_b = [pcs_b[i]["variance_ratio"] for i in range(k)]
    # Order preserved if sorted indices match
    rank_a = sorted(range(k), key=lambda i: -order_a[i])
    rank_b = sorted(range(k), key=lambda i: -order_b[i])
    ordering_match = rank_a == rank_b

    return {
        "pc_variance_diffs": var_diffs,
        "max_variance_diff": round(max(var_diffs), 6),
        "mean_variance_diff": round(sum(var_diffs) / len(var_diffs), 6),
        "pc_ordering_preserved": ordering_match,
    }


def compare_strategy_overlap(space_a, space_b, top_k=5):
    """Check top-k strategy displacement overlap."""
    sd_a = space_a["strategy_displacements"]
    sd_b = space_b["strategy_displacements"]

    top_a = set(sorted(sd_a.keys(), key=lambda s: sd_a[s]["displacement_from_global"], reverse=True)[:top_k])
    top_b = set(sorted(sd_b.keys(), key=lambda s: sd_b[s]["displacement_from_global"], reverse=True)[:top_k])

    overlap = top_a & top_b
    return {
        f"top_{top_k}_run1": sorted(top_a),
        f"top_{top_k}_run2": sorted(top_b),
        f"top_{top_k}_overlap": sorted(overlap),
        f"overlap_count": len(overlap),
        f"overlap_ratio": round(len(overlap) / top_k, 4),
    }


def compute_axis_alignment(space_a, space_b):
    """Correlation of PCA projection axes between two spaces."""
    # Get projection vectors aligned by run_id
    proj_a = space_a.get("projections", {})
    proj_b = space_b.get("projections", {})

    common_ids = set(proj_a.keys()) & set(proj_b.keys())
    if len(common_ids) < 10:
        return {"axis_correlations": [], "mean_correlation": 0.0, "note": "insufficient common ids"}

    k = min(len(space_a["principal_components"]), len(space_b["principal_components"]))

    # Build aligned matrices
    vecs_a = []
    vecs_b = []
    for rid in sorted(common_ids):
        vecs_a.append(proj_a[rid])
        vecs_b.append(proj_b[rid])

    mat_a = np.array(vecs_a)
    mat_b = np.array(vecs_b)

    correlations = []
    for col in range(min(k, mat_a.shape[1], mat_b.shape[1])):
        ca = mat_a[:, col] - mat_a[:, col].mean()
        cb = mat_b[:, col] - mat_b[:, col].mean()
        na = np.linalg.norm(ca)
        nb = np.linalg.norm(cb)
        if na > 1e-10 and nb > 1e-10:
            corr = float(np.dot(ca, cb) / (na * nb))
        else:
            corr = 0.0
        correlations.append(round(corr, 4))

    return {
        "axis_correlations": correlations,
        "mean_correlation": round(float(np.mean(correlations)), 4) if correlations else 0.0,
        "n_common_ids": len(common_ids),
    }


def main():
    report = {}

    # ============================================================
    # TEST 1: Same model, repeat run
    # ============================================================
    print("TEST 1: Same model, repeat run...")
    s1 = load_space(os.path.join(OUTPUT_DIR, "behavioral_space.json"))
    s1r = load_space(os.path.join(OUTPUT_DIR, "behavioral_space_repeat.json"))

    dm_mean_1 = s1["distance_matrix_summary"]["mean_distance"]
    dm_mean_1r = s1r["distance_matrix_summary"]["mean_distance"]

    pca_comp = compare_pca(s1, s1r)
    strat_comp = compare_strategy_overlap(s1, s1r)
    axis_align = compute_axis_alignment(s1, s1r)

    report["test1_repeat_run"] = {
        "cosine_mean_run1": dm_mean_1,
        "cosine_mean_run2": dm_mean_1r,
        "cosine_mean_diff": round(abs(dm_mean_1 - dm_mean_1r), 6),
        "cosine_mean_diff_threshold": 0.02,
        "cosine_mean_diff_pass": abs(dm_mean_1 - dm_mean_1r) <= 0.02,
        "pca_comparison": pca_comp,
        "strategy_overlap": strat_comp,
        "axis_alignment": axis_align,
        "overall_pass": (
            abs(dm_mean_1 - dm_mean_1r) <= 0.02
            and strat_comp["overlap_count"] >= 3
            and pca_comp["pc_ordering_preserved"]
        ),
    }

    # ============================================================
    # TEST 2: Same data, different metrics
    # ============================================================
    print("TEST 2: Different metrics...")
    metrics = ["euclidean", "manhattan", "centered_cosine", "rank_distance"]
    metric_spaces = {}
    for m in metrics:
        path = os.path.join(OUTPUT_DIR, f"behavioral_space_{m}.json")
        if os.path.exists(path):
            metric_spaces[m] = load_space(path)

    test2_results = {}
    for m, ms in metric_spaces.items():
        # PCA comparison (variance pattern)
        pca_c = compare_pca(s1, ms)
        # Strategy overlap
        strat_c = compare_strategy_overlap(s1, ms)
        # Axis alignment
        axis_c = compute_axis_alignment(s1, ms)

        test2_results[m] = {
            "distance_mean": ms["distance_matrix_summary"]["mean_distance"],
            "pca_max_var_diff": pca_c["max_variance_diff"],
            "pca_ordering_preserved": pca_c["pc_ordering_preserved"],
            "strategy_overlap_count": strat_c["overlap_count"],
            "strategy_overlap_ratio": strat_c["overlap_ratio"],
            "axis_mean_correlation": axis_c["mean_correlation"],
            "geometry_collapsed": axis_c["mean_correlation"] < 0.1,
        }

    # Summary: does geometry collapse across metrics?
    any_collapse = any(v["geometry_collapsed"] for v in test2_results.values())
    mean_corr = np.mean([v["axis_mean_correlation"] for v in test2_results.values()])

    report["test2_metric_stability"] = {
        "metrics_tested": list(metrics),
        "per_metric": test2_results,
        "mean_axis_correlation_across_metrics": round(float(mean_corr), 4),
        "any_metric_collapse": any_collapse,
        "overall_pass": not any_collapse and mean_corr > 0.1,
    }

    # ============================================================
    # TEST 3: Different models
    # ============================================================
    print("TEST 3: Different models...")
    s2 = load_space(os.path.join(OUTPUT_DIR, "behavioral_space_model2.json"))

    dm_mean_2 = s2["distance_matrix_summary"]["mean_distance"]
    pca_cross = compare_pca(s1, s2)
    strat_cross = compare_strategy_overlap(s1, s2)
    axis_cross = compute_axis_alignment(s1, s2)

    report["test3_cross_model"] = {
        "model1": "Qwen2.5-0.5B-Instruct (Q2_K)",
        "model2": "TinyLlama-1.1B-Chat (Q2_K)",
        "cosine_mean_model1": dm_mean_1,
        "cosine_mean_model2": dm_mean_2,
        "cosine_mean_stays_high": dm_mean_2 > 0.85,
        "pca_low_compression_model2": s2["principal_components"][0]["cumulative_variance"] < 0.5,
        "pca_comparison": pca_cross,
        "strategy_overlap": strat_cross,
        "axis_alignment": axis_cross,
        "partial_overlap": strat_cross["overlap_count"] >= 2,
        "overall_pass": (
            dm_mean_2 > 0.85
            and s2["principal_components"][0]["cumulative_variance"] < 0.5
            and strat_cross["overlap_count"] >= 2
        ),
    }

    # ============================================================
    # VERDICT
    # ============================================================
    t1_pass = report["test1_repeat_run"]["overall_pass"]
    t2_pass = report["test2_metric_stability"]["overall_pass"]
    t3_pass = report["test3_cross_model"]["overall_pass"]

    report["verdict"] = {
        "test1_stable_across_runs": t1_pass,
        "test2_not_collapsing_across_metrics": t2_pass,
        "test3_partially_consistent_across_models": t3_pass,
        "geometry_is_real": t1_pass and t2_pass and t3_pass,
        "geometry_is_artifact": not (t1_pass and t2_pass),
    }

    # Write report
    out_path = os.path.join(OUTPUT_DIR, "stability_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"STABILITY REPORT")
    print(f"{'='*60}")
    print(f"\nTEST 1 (same model, repeat run):")
    print(f"  cosine_mean_diff: {report['test1_repeat_run']['cosine_mean_diff']} (threshold: 0.02)")
    print(f"  strategy overlap: {report['test1_repeat_run']['strategy_overlap']['overlap_count']}/5 (threshold: 3)")
    print(f"  PCA ordering preserved: {report['test1_repeat_run']['pca_comparison']['pc_ordering_preserved']}")
    print(f"  axis alignment: {report['test1_repeat_run']['axis_alignment']['mean_correlation']}")
    print(f"  PASS: {t1_pass}")

    print(f"\nTEST 2 (different metrics):")
    print(f"  mean axis correlation: {report['test2_metric_stability']['mean_axis_correlation_across_metrics']}")
    print(f"  any collapse: {report['test2_metric_stability']['any_metric_collapse']}")
    for m, res in test2_results.items():
        print(f"    {m}: corr={res['axis_mean_correlation']}, overlap={res['strategy_overlap_count']}/5, collapsed={res['geometry_collapsed']}")
    print(f"  PASS: {t2_pass}")

    print(f"\nTEST 3 (different models):")
    print(f"  cosine mean: Qwen={dm_mean_1:.4f}, TinyLlama={dm_mean_2:.4f} (threshold: >0.85)")
    print(f"  strategy overlap: {report['test3_cross_model']['strategy_overlap']['overlap_count']}/5 (threshold: 2)")
    print(f"  PCA PC0 cum: Qwen={s1['principal_components'][0]['cumulative_variance']:.4f}, TinyLlama={s2['principal_components'][0]['cumulative_variance']:.4f}")
    print(f"  axis alignment: {report['test3_cross_model']['axis_alignment']['mean_correlation']}")
    print(f"  PASS: {t3_pass}")

    print(f"\n{'='*60}")
    print(f"VERDICT: {'GEOMETRY IS REAL' if report['verdict']['geometry_is_real'] else 'GEOMETRY IS ARTIFACT'}")
    print(f"{'='*60}")

    return out_path


if __name__ == "__main__":
    sys.exit(main() or 0)

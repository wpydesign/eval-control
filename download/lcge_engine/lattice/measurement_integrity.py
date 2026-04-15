"""
measurement_integrity.py — Measurement Integrity Layer for LCGE.

This module does NOT measure models.
It measures the stability of the measurement pipeline itself.

Inputs: stability_report.json (output of 3-test stability verification)
Output: Measurement Stability Index (MSI) — single scalar 0–1

MSI = same_model_reproducibility - metric_sensitivity - cross_model_consistency_loss

Interpretation:
    MSI > 0.7  → measurement system is reliable
    0.4–0.7    → partially reliable, conditional use
    < 0.4      → system mainly reflects its own projection bias
"""

import json
import os
import sys
from datetime import datetime, timezone


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)


# ============================================================
# Component 1: Same-model reproducibility (R)
# ============================================================

def compute_reproducibility(test1: dict) -> dict:
    """
    How stable is the measurement under identical conditions?

    Sub-scores:
        distance_stability — cosine mean diff, 0=identical, 1=threshold breached
        axis_stability     — mean PC axis correlation between runs
        structure_stability — top-5 strategy overlap ratio
    """
    cosine_diff = test1["cosine_mean_diff"]
    cosine_threshold = test1.get("cosine_mean_diff_threshold", 0.02)

    # Distance stability: 1.0 = perfect (diff=0), 0.0 = at/beyond threshold
    distance_stability = max(0.0, 1.0 - cosine_diff / cosine_threshold)

    # Axis stability: mean correlation of PC axes between runs
    axis_correlations = test1["axis_alignment"]["axis_correlations"]
    # Use top-5 PCs (the meaningful ones), ignore noise axes
    top_corrs = [abs(c) for c in axis_correlations[:5]]
    axis_stability = sum(top_corrs) / len(top_corrs) if top_corrs else 0.0

    # Structure stability: strategy overlap ratio
    structure_stability = test1["strategy_overlap"]["overlap_ratio"]

    # Weighted composite
    # Weight axis stability highest — it captures full geometric reproducibility
    r = (
        0.20 * distance_stability
        + 0.50 * axis_stability
        + 0.30 * structure_stability
    )

    return {
        "distance_stability": round(distance_stability, 4),
        "axis_stability": round(axis_stability, 4),
        "structure_stability": round(structure_stability, 4),
        "composite": round(r, 4),
        "raw": {
            "cosine_mean_diff": cosine_diff,
            "cosine_threshold": cosine_threshold,
            "mean_axis_correlation": test1["axis_alignment"]["mean_correlation"],
            "strategy_overlap": test1["strategy_overlap"]["overlap_count"],
            "strategy_overlap_ratio": test1["strategy_overlap"]["overlap_ratio"],
            "top5_axis_correlations": axis_correlations[:5],
        },
    }


# ============================================================
# Component 2: Metric sensitivity (S)
# ============================================================

def compute_metric_sensitivity(test2: dict) -> dict:
    """
    How much does the discovered structure depend on the distance metric?

    High sensitivity = bad. The measurement is creating its own signal.

    Sub-scores:
        collapse_rate     — fraction of metrics that cause geometry collapse
        axis_drift        — mean absolute axis correlation across metrics (lower = worse)
        structure_drift   — mean strategy overlap ratio across alternative metrics
    """
    per_metric = test2["per_metric"]
    n_metrics = len(per_metric)

    if n_metrics == 0:
        return {"collapse_rate": 0.0, "axis_drift": 0.0, "structure_drift": 0.0, "composite": 0.0}

    # Collapse rate: fraction of alternative metrics that collapse
    n_collapsed = sum(1 for v in per_metric.values() if v["geometry_collapsed"])
    collapse_rate = n_collapsed / n_metrics

    # Axis drift: mean |correlation| across metrics
    axis_corrs = [abs(v["axis_mean_correlation"]) for v in per_metric.values()]
    # Normalize: 0.0 = zero correlation (max drift), 1.0 = perfect correlation
    axis_drift_raw = sum(axis_corrs) / len(axis_corrs) if axis_corrs else 0.0
    axis_drift_score = axis_drift_raw  # already 0–1 ish

    # Structure drift: mean strategy overlap across alternative metrics
    overlap_ratios = [v["strategy_overlap_ratio"] for v in per_metric.values()]
    structure_drift = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0

    # Composite sensitivity (0 = no problem, 1 = severe)
    # High collapse rate, low axis drift, low structure overlap = high sensitivity
    s = (
        0.40 * collapse_rate
        + 0.35 * (1.0 - axis_drift_score)
        + 0.25 * (1.0 - structure_drift)
    )

    return {
        "collapse_rate": round(collapse_rate, 4),
        "axis_drift_raw": round(axis_drift_raw, 4),
        "structure_drift": round(structure_drift, 4),
        "composite": round(s, 4),
        "raw": {
            "metrics_tested": test2["metrics_tested"],
            "n_collapsed": n_collapsed,
            "n_total": n_metrics,
            "per_metric_correlations": {
                k: round(v["axis_mean_correlation"], 4)
                for k, v in per_metric.items()
            },
            "per_metric_overlap": {
                k: v["strategy_overlap_count"]
                for k, v in per_metric.items()
            },
            "mean_axis_correlation": test2["mean_axis_correlation_across_metrics"],
        },
    }


# ============================================================
# Component 3: Cross-model consistency loss (C)
# ============================================================

def compute_cross_model_loss(test3: dict) -> dict:
    """
    How much behavioral structure diverges across models?

    If the measurement pipeline captures real structure,
    different models should show SOME shared geometric signal
    (same perturbation axes, same top displacement strategies).

    Low loss = geometry is model-independent (good).
    High loss = geometry is model-specific (measurement is capturing model artifacts).

    If test3 is absent (only one model tested), returns N/A.
    """
    if test3 is None:
        return {
            "axis_divergence": None,
            "structure_divergence": None,
            "composite": None,
            "note": "cross_model_test_not_run",
        }

    # Axis divergence: 1.0 - mean |correlation| between models
    axis_corrs = [abs(c) for c in test3["axis_alignment"]["axis_correlations"]]
    top_corrs = [abs(c) for c in axis_corrs[:5]]
    axis_divergence_raw = 1.0 - (sum(top_corrs) / len(top_corrs) if top_corrs else 0.0)

    # Structure divergence: 1.0 - strategy overlap ratio
    structure_divergence_raw = 1.0 - test3["strategy_overlap"]["overlap_ratio"]

    # Composite (0 = identical across models, 1 = completely different)
    c = (
        0.50 * axis_divergence_raw
        + 0.50 * structure_divergence_raw
    )

    return {
        "axis_divergence": round(axis_divergence_raw, 4),
        "structure_divergence": round(structure_divergence_raw, 4),
        "composite": round(c, 4),
        "raw": {
            "model1": test3.get("model1", "unknown"),
            "model2": test3.get("model2", "unknown"),
            "mean_axis_correlation": test3["axis_alignment"]["mean_correlation"],
            "strategy_overlap": test3["strategy_overlap"]["overlap_count"],
            "strategy_overlap_ratio": test3["strategy_overlap"]["overlap_ratio"],
            "top5_axis_correlations": axis_corrs[:5],
        },
    }


# ============================================================
# MSI: Measurement Stability Index
# ============================================================

def compute_msi(stability_report: dict) -> dict:
    """
    Compute Measurement Stability Index from a stability report.

    MSI = R - S - C

    Where:
        R = same_model_reproducibility (0–1, higher = better)
        S = metric_sensitivity        (0–1, higher = worse)
        C = cross_model_loss          (0–1, higher = worse, optional)

    Clamped to [0, 1].
    """
    test1 = stability_report.get("test1_repeat_run")
    test2 = stability_report.get("test2_metric_stability")
    test3 = stability_report.get("test3_cross_model")

    if test1 is None or test2 is None:
        return {"error": "stability report missing test1 or test2"}

    # Compute components
    reproducibility = compute_reproducibility(test1)
    sensitivity = compute_metric_sensitivity(test2)
    cross_model = compute_cross_model_loss(test3)

    R = reproducibility["composite"]
    S = sensitivity["composite"]
    C = cross_model["composite"] if cross_model["composite"] is not None else 0.0

    # MSI = R - S - C, clamped
    msi_raw = R - S - C
    msi = round(max(0.0, min(1.0, msi_raw)), 4)

    # Classification
    if msi >= 0.7:
        classification = "RELIABLE"
        description = (
            "Measurement system produces stable, metric-invariant structure. "
            "Cross-model comparisons are meaningful within this pipeline."
        )
    elif msi >= 0.4:
        classification = "CONDITIONAL"
        description = (
            "Measurement system is partially reliable. Same-model results "
            "are reproducible, but discovered structure depends on metric "
            "choice. Use with caution; do not treat PCA axes as ground truth."
        )
    else:
        classification = "DISTORTED"
        description = (
            "Measurement system mainly reflects its own projection bias. "
            "Discovered geometry is metric-dependent and not model-invariant. "
            "Results are descriptive of the pipeline, not of model behavior."
        )

    return {
        "msi": msi,
        "classification": classification,
        "description": description,
        "components": {
            "reproducibility": reproducibility,
            "metric_sensitivity": sensitivity,
            "cross_model_loss": cross_model,
        },
        "formula": "MSI = R - S - C  (clamped to [0, 1])",
        "weights": {
            "R": "0.20*distance + 0.50*axis + 0.30*structure",
            "S": "0.40*collapse + 0.35*(1-axis_drift) + 0.25*(1-structure_drift)",
            "C": "0.50*axis_divergence + 0.50*structure_divergence",
        },
        "thresholds": {
            "reliable": "MSI >= 0.7",
            "conditional": "0.4 <= MSI < 0.7",
            "distorted": "MSI < 0.4",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def classify_reliability(msi_result: dict) -> dict:
    """
    Produce a human-readable diagnostic summary.

    Answers the key questions:
        1. Is this evaluation method trustworthy?
        2. Does changing distance metric fake structure?
        3. Is cross-model comparison meaningful here?
    """
    msi = msi_result["msi"]
    classification = msi_result["classification"]
    R = msi_result["components"]["reproducibility"]["composite"]
    S = msi_result["components"]["metric_sensitivity"]["composite"]
    C_comp = msi_result["components"]["cross_model_loss"]["composite"]
    collapse_rate = msi_result["components"]["metric_sensitivity"]["collapse_rate"]

    q1_trustworthy = classification == "RELIABLE"
    q2_metric_fakes = collapse_rate > 0.3
    q3_cross_model = C_comp is not None and C_comp < 0.5

    return {
        "q1_is_evaluation_trustworthy": q1_trustworthy,
        "q1_detail": (
            f"{'Yes.' if q1_trustworthy else 'No.'} "
            f"MSI={msi} ({classification}). "
            f"Same-model reproducibility={R:.2f}, "
            f"metric sensitivity={S:.2f}."
        ),
        "q2_does_metric_choice_fake_structure": q2_metric_fakes,
        "q2_detail": (
            f"{'Yes.' if q2_metric_fakes else 'No.'} "
            f"{collapse_rate*100:.0f}% of alternative metrics caused geometry collapse. "
            f"Discovered PCA axes are not metric-invariant."
        ),
        "q3_is_cross_model_comparison_meaningful": q3_cross_model,
        "q3_detail": (
            f"{'Partially.' if q3_cross_model else 'No.'} "
            f"Cross-model loss={C_comp if C_comp is not None else 'N/A'}. "
            f"{'Some shared structure survives.' if q3_cross_model else 'Models produce different behavioral shapes; comparison is not valid.'}"
        ) if C_comp is not None else "Cross-model test not run.",
        "overall": f"MSI = {msi} → {classification}",
    }


# ============================================================
# Main: load report, compute MSI, output
# ============================================================

def evaluate(input_path: str, output_path: str = None) -> dict:
    """
    Load a stability report and produce the full integrity diagnostic.

    Args:
        input_path: Path to stability_report.json
        output_path: Path to write diagnostic output (default: auto)

    Returns:
        Full diagnostic dict.
    """
    with open(input_path) as f:
        report = json.load(f)

    msi_result = compute_msi(report)
    reliability = classify_reliability(msi_result)

    diagnostic = {
        "system_identity": "Measurement Distortion Detector over LLM Outputs",
        "input_report": input_path,
        "msi": msi_result,
        "reliability_questions": reliability,
    }

    if output_path is None:
        base = os.path.basename(input_path).replace(".json", "")
        output_path = os.path.join(OUTPUT_DIR, f"integrity_{base}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diagnostic, f, indent=2)

    return diagnostic


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LCGE Measurement Integrity Layer — "
                    "Measures stability of the measurement pipeline, not models."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to stability_report.json"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to write integrity diagnostic JSON"
    )

    args = parser.parse_args()

    diagnostic = evaluate(args.input, args.output)

    msi = diagnostic["msi"]
    print(f"\n{'='*60}")
    print(f"MEASUREMENT INTEGRITY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"\n  MSI = {msi['msi']:.4f}  →  {msi['classification']}")
    print(f"\n  Reproducibility (R):   {msi['components']['reproducibility']['composite']:.4f}")
    print(f"  Metric Sensitivity (S): {msi['components']['metric_sensitivity']['composite']:.4f}")
    c = msi['components']['cross_model_loss']
    c_str = f"{c['composite']:.4f}" if c['composite'] is not None else "N/A"
    print(f"  Cross-model Loss (C):   {c_str}")
    print(f"\n  {msi['description']}")

    rq = diagnostic["reliability_questions"]
    print(f"\n{'='*60}")
    print(f"KEY QUESTIONS")
    print(f"{'='*60}")
    print(f"  Q1: Is this evaluation method trustworthy?")
    print(f"      {rq['q1_detail']}")
    print(f"  Q2: Does changing distance metric fake structure?")
    print(f"      {rq['q2_detail']}")
    print(f"  Q3: Is cross-model comparison meaningful?")
    print(f"      {rq['q3_detail']}")

    print(f"\n  Output: {args.output or 'auto'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""
drift_monitor.py — Track calibration drift and distribution shift over time [v2.2.2]

Monitors whether the failure predictor's calibration degrades between runs.
If ECE drifts from 0.05 -> 0.15, the model needs retraining before it
silently produces unreliable probabilities.

Tracks per-run:
  - ECE (Expected Calibration Error)
  - Brier Score
  - Risk score distribution (mean, std, percentiles)
  - Threshold hit rates (how many samples hit review/escalate)
  - Labeled sample count and class balance

Outputs:
  logs/drift_log.jsonl          — one entry per run
  model/drift_baseline.json      — reference snapshot for comparison

Usage:
  python scripts/drift_monitor.py              # run check, append to log
  python scripts/drift_monitor.py --baseline    # set current run as new baseline
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
DRIFT_LOG_PATH = os.path.join(BASE, "logs", "drift_log.jsonl")
BASELINE_PATH = os.path.join(BASE, "model", "drift_baseline.json")

N_BINS = 10
ECE_WARNING_THRESHOLD = 0.08   # warn if ECE exceeds this
ECE_CRITICAL_THRESHOLD = 0.12  # critical if ECE exceeds this


def _make_v4(s):
    """Extract v4 feature dict from a sample row."""
    return {"S": s["S_v4"], "kappa": s["kappa_v4"],
            "delta_G": s["delta_G_v4"], "delta_L": s["delta_L_v4"]}


def _make_v1(s):
    """Extract v1 feature dict from a sample row."""
    return {"S": s["S_v1"]}


def load_labeled_data():
    """Load samples from failure_dataset.jsonl, split into labeled/unlabeled."""
    if not os.path.exists(DATASET_PATH):
        return [], []
    labeled, unlabeled = [], []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_wrong") is not None:
                labeled.append(row)
            else:
                unlabeled.append(row)
    return labeled, unlabeled


def compute_ece(y_true, y_prob, n_bins=N_BINS):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        pred_mean = y_prob[mask].mean()
        actual = y_true[mask].mean()
        ece += abs(pred_mean - actual) * n / total
    return round(float(ece), 4)


def compute_run_stats(predictor, labeled, unlabeled):
    """Compute all drift metrics for the current run."""
    from sklearn.metrics import brier_score_loss

    y_true_list = []
    y_prob_list = []
    all_risks = []

    # Compute risks for ALL samples
    for s in labeled + unlabeled:
        risk = predictor.predict(_make_v4(s), _make_v1(s))
        all_risks.append(risk["risk_score"])

    # Labeled: ECE + Brier
    if labeled:
        for s in labeled:
            risk = predictor.predict(_make_v4(s), _make_v1(s))
            y_true_list.append(s["is_wrong"])
            y_prob_list.append(risk["risk_score"])

        y_true = np.array(y_true_list)
        y_prob = np.array(y_prob_list)
        ece = compute_ece(y_true, y_prob)
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except ValueError:
            brier = None

        n_wrong = int(y_true.sum())
        n_correct = int((y_true == 0).sum())
    else:
        ece = None
        brier = None
        n_wrong = 0
        n_correct = 0

    all_risks = np.array(all_risks) if all_risks else np.array([0.0])

    rt = predictor._review_threshold
    et = predictor._escalate_threshold

    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "n_total": len(labeled) + len(unlabeled),
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "class_ratio": round(n_correct / max(1, n_wrong), 2),
        "ece": ece,
        "brier_score": round(brier, 4) if brier is not None else None,
        "risk_mean": round(float(all_risks.mean()), 4),
        "risk_std": round(float(all_risks.std()), 4),
        "risk_p10": round(float(np.percentile(all_risks, 10)), 4),
        "risk_p50": round(float(np.percentile(all_risks, 50)), 4),
        "risk_p90": round(float(np.percentile(all_risks, 90)), 4),
        "risk_min": round(float(all_risks.min()), 4),
        "risk_max": round(float(all_risks.max()), 4),
        "review_hit_rate": round(float((all_risks >= rt).mean()), 4),
        "escalate_hit_rate": round(float((all_risks >= et).mean()), 4),
        "review_threshold": rt,
        "escalate_threshold": et,
        "ece_status": "ok",
    }

    # ECE status
    if ece is not None:
        if ece >= ECE_CRITICAL_THRESHOLD:
            stats["ece_status"] = "CRITICAL"
        elif ece >= ECE_WARNING_THRESHOLD:
            stats["ece_status"] = "WARNING"

    return stats


def load_baseline():
    """Load baseline stats for comparison."""
    if not os.path.exists(BASELINE_PATH):
        return None
    try:
        with open(BASELINE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_baseline(stats):
    """Save current stats as new baseline."""
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Baseline saved to {BASELINE_PATH}")


def compare_with_baseline(current, baseline):
    """Compare current run stats against baseline, flag drifts."""
    if baseline is None:
        return

    print(f"\n  {'Metric':<25s} {'Baseline':>10s} {'Current':>10s} {'Delta':>10s} {'Status':>12s}")
    print(f"  {'-'*70}")

    drift_fields = [
        ("ECE", "ece", True),
        ("Brier Score", "brier_score", True),
        ("Risk mean", "risk_mean", False),
        ("Risk std", "risk_std", False),
        ("Review hit rate", "review_hit_rate", False),
        ("Escalate hit rate", "escalate_hit_rate", False),
        ("Labeled samples", "n_labeled", False),
    ]

    for label, key, lower_is_better in drift_fields:
        cur = current.get(key)
        base = baseline.get(key)
        if cur is None or base is None:
            continue
        delta = cur - base
        if abs(delta) < 0.0001:
            status = "stable"
        elif lower_is_better:
            status = "WORSE" if delta > 0.02 else ("ok" if delta > -0.02 else "IMPROVED")
        else:
            status = "shifted" if abs(delta) > 0.05 else "ok"

        if status == "stable":
            status_str = f"  {status}"
        elif status != "ok":
            status_str = f"! {status}"
        else:
            status_str = f"  {status}"
        print(f"  {label:<25s} {base:>10.4f} {cur:>10.4f} {delta:>+10.4f} {status_str:>12s}")


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from predict_failure import FailurePredictor

    print("Loading predictor...")
    predictor = FailurePredictor()
    if not predictor.is_loaded:
        print("ERROR: No trained model.")
        sys.exit(1)

    print("Loading data...")
    labeled, unlabeled = load_labeled_data()
    print(f"  {len(labeled)} labeled, {len(unlabeled)} unlabeled")

    print("Computing run stats...")
    stats = compute_run_stats(predictor, labeled, unlabeled)

    # Display
    print(f"\n{'='*60}")
    print(f"  DRIFT MONITOR - Run Report")
    print(f"{'='*60}")
    print(f"  Labeled: {stats['n_labeled']} (wrong={stats['n_wrong']}, correct={stats['n_correct']})")
    if stats["ece"] is not None:
        ece_tag = stats["ece_status"]
        if ece_tag == "CRITICAL":
            ece_tag += " - retrain immediately"
        elif ece_tag == "WARNING":
            ece_tag += " - calibration degrading"
        print(f"  ECE: {stats['ece']:.4f} [{ece_tag}]")
    if stats["brier_score"] is not None:
        print(f"  Brier: {stats['brier_score']:.4f}")
    print(f"  Risk distribution: mean={stats['risk_mean']:.3f}, std={stats['risk_std']:.3f}")
    print(f"    P10={stats['risk_p10']:.3f}  P50={stats['risk_p50']:.3f}  P90={stats['risk_p90']:.3f}")
    print(f"  Threshold hits: review={stats['review_hit_rate']:.1%}, escalate={stats['escalate_hit_rate']:.1%}")

    # Compare with baseline
    set_baseline = "--baseline" in sys.argv
    baseline = load_baseline()
    if set_baseline:
        print(f"\n  Setting new baseline...")
        save_baseline(stats)
    elif baseline:
        print(f"\n  Comparing with baseline ({baseline.get('timestamp', 'unknown')}):")
        compare_with_baseline(stats, baseline)
    else:
        print(f"\n  No baseline found. Run with --baseline to set one.")

    # Actionable recommendations
    print(f"\n  Recommendations:")
    if stats["ece"] is not None and stats["ece"] >= ECE_CRITICAL_THRESHOLD:
        print(f"  ! RETRAIN: ECE={stats['ece']:.3f} exceeds critical ({ECE_CRITICAL_THRESHOLD})")
    elif stats["ece"] is not None and stats["ece"] >= ECE_WARNING_THRESHOLD:
        print(f"  ! WATCH: ECE={stats['ece']:.3f} in warning zone ({ECE_WARNING_THRESHOLD})")
    else:
        print(f"  OK Calibration stable (ECE={stats['ece']:.3f})")

    if stats["n_labeled"] < 50:
        print(f"  ! LABEL: Only {stats['n_labeled']} labeled samples - acquire more")
    elif stats["n_labeled"] < 200:
        print(f"  ~ LABEL: {stats['n_labeled']} labeled samples - good, more is better")

    if baseline:
        base_review = baseline.get("review_hit_rate", 0)
        cur_review = stats["review_hit_rate"]
        if abs(cur_review - base_review) > 0.10:
            print(f"  ! DISTRIBUTION SHIFT: review hit rate {base_review:.1%} -> {cur_review:.1%}")

    # Append to drift log
    os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
    with open(DRIFT_LOG_PATH, "a") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    print(f"\n  Drift log appended to {DRIFT_LOG_PATH}")


if __name__ == "__main__":
    main()

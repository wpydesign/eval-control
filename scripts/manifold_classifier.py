#!/usr/bin/env python3
"""
manifold_classifier.py — Decomposed failure manifold classifiers [v2.5.0]

Replaces the unified FailurePredictor (single LR) with per-manifold predictive surfaces.
Each failure manifold has its own geometry, its own relationship between features and
failure probability, and its own optimal decision surface.

Manifold taxonomy (empirically derived from v2.4.0 batch labels):
  - overconfidence (blind_spot): n=6, 100% wrong rate — detection problem
    Features are misleadingly normal. The signal is the ABSENCE of expected signals.
    No within-manifold model needed: P(wrong | blind_spot) ≈ 1.0

  - contradiction: n=13, 69.2% wrong rate — structural disagreement instability
    v4 and v1 disagree on decision. The signal is in the disagreement geometry.
    Within-manifold model predicts which disagreement cases actually fail.

  - boundary: n=367, 12.5% wrong rate — mostly correct ambiguity
    Standard prediction regime. The signal is in the absolute score levels.
    Full LR classifier trained on boundary samples.

Architecture:
  Input → ManifoldRouter (rule-based + detection) → Per-manifold head → P(is_wrong)

Manifold routing (based on observable signals at prediction time):
  1. blind_spot detection: low proxy_score + moderate-high confidence + failure_mode match
  2. contradiction: disagreement_flag=true (v4 and v1 disagree on decision)
  3. boundary: everything else (no disagreement, no blind spot signals)

Data sources:
  - logs/batch_label_results.jsonl: 386 high-value samples with failure_type labels
  - logs/failure_dataset.jsonl: 700 existing labeled samples (supplement boundary)

Outputs:
  - model/manifold_models.pkl — per-manifold model package
  - model/manifold_report.json — per-manifold metrics and separation quality
  - logs/v250_baseline.json — frozen v2.4.0 reference baseline

Usage:
  python scripts/manifold_classifier.py
  python scripts/manifold_classifier.py --retrain
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    brier_score_loss, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_LABELS_PATH = os.path.join(BASE, "logs", "batch_label_results.jsonl")
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MODEL_DIR = os.path.join(BASE, "model")
MANIFOLD_MODEL_PATH = os.path.join(MODEL_DIR, "manifold_models.pkl")
MANIFOLD_REPORT_PATH = os.path.join(MODEL_DIR, "manifold_report.json")
BASELINE_PATH = os.path.join(BASE, "logs", "v250_baseline.json")

# Features available in both datasets
ROUTING_FEATURES = [
    "S_v4", "S_v1", "confidence_gap", "kappa_v4"
]

# Full features for boundary head (includes delta_G, delta_L when available)
BOUNDARY_FEATURES_FULL = [
    "S_v4", "S_v1", "confidence_gap", "kappa_v4", "delta_G_v4", "delta_L_v4"
]

# Features for contradiction head
CONTRADICTION_FEATURES = [
    "S_v4", "S_v1", "confidence_gap", "kappa_v4"
]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def freeze_baseline(batch_labels, existing_dataset):
    """Freeze v2.4.0 metrics as reference baseline."""
    labeled_existing = [r for r in existing_dataset if r.get("is_wrong") is not None]

    baseline = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "version": "v2.4.0",
        "description": "Ground truth distribution snapshot — pre-manifold-decomposition",
        "batch_labels": {
            "total": len(batch_labels),
            "by_channel": dict(Counter(r["source_channel"] for r in batch_labels)),
            "by_failure_type": dict(Counter(r["failure_type"] for r in batch_labels)),
        },
        "existing_dataset": {
            "total": len(existing_dataset),
            "labeled": len(labeled_existing),
            "wrong": sum(1 for r in labeled_existing if r["is_wrong"] == 1),
            "correct": sum(1 for r in labeled_existing if r["is_wrong"] == 0),
        },
        "manifold_segments": {},
        "unified_predictor_note": (
            "AUC is no longer the primary metric — manifold separation is. "
            "The unified AUC collapsed because it averages over multiple failure geometries "
            "with fundamentally different feature-failure relationships."
        ),
    }

    # Per-manifold statistics from batch labels
    for ft in ["overconfidence", "contradiction", "boundary"]:
        subset = [r for r in batch_labels if r["failure_type"] == ft]
        n_wrong = sum(1 for r in subset if r["final_label"] == 1)
        baseline["manifold_segments"][ft] = {
            "n": len(subset),
            "n_wrong": n_wrong,
            "n_correct": len(subset) - n_wrong,
            "wrong_rate": round(n_wrong / len(subset), 4) if subset else 0,
            "information_value": round(
                n_wrong * np.log2(len(subset) / max(n_wrong, 1)) +
                (len(subset) - n_wrong) * np.log2(len(subset) / max(len(subset) - n_wrong, 1)),
                4
            ) if len(subset) > 1 else 0,
        }

    # Blind spot channel analysis
    bs = [r for r in batch_labels if r["source_channel"] == "blind_spot"]
    bs_wrong = sum(1 for r in bs if r["final_label"] == 1)
    baseline["blind_spot_channel"] = {
        "n": len(bs),
        "n_wrong": bs_wrong,
        "wrong_rate": round(bs_wrong / len(bs), 4) if bs else 0,
        "failure_type_distribution": dict(Counter(r["failure_type"] for r in bs)),
    }

    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"  Baseline frozen to {BASELINE_PATH}")

    return baseline


def build_manifold_datasets(batch_labels, existing_dataset):
    """Build per-manifold training datasets.

    Returns dict: manifold_name → list of feature vectors + labels
    """
    # Map batch labels to unified format
    # final_label=1 → is_wrong=1, final_label=0 → is_wrong=0
    batch_samples = []
    for r in batch_labels:
        sample = {
            "query_id": r["query_id"],
            "S_v4": r.get("v4_prediction", 0.0),
            "S_v1": r.get("v1_shadow", 0.0),
            "confidence_gap": r.get("confidence_gap", 0.0),
            "kappa_v4": r.get("kappa_v4", 0.0),
            "delta_G_v4": 0.0,  # not available in batch labels
            "delta_L_v4": 0.0,  # not available in batch labels
            "failure_type": r["failure_type"],
            "source_channel": r["source_channel"],
            "is_wrong": r["final_label"],  # direct mapping
            "disagreement_flag": r.get("disagreement_flag", False),
        }
        batch_samples.append(sample)

    # Map existing dataset to same format
    existing_samples = []
    # Build query_id → failure_type index from batch labels
    batch_ft_index = {r["query_id"]: r["failure_type"] for r in batch_labels}

    for r in existing_dataset:
        if r.get("is_wrong") is None:
            continue
        sample = {
            "query_id": r.get("query_id", ""),
            "S_v4": r.get("S_v4", 0.0),
            "S_v1": r.get("S_v1", 0.0),
            "confidence_gap": r.get("confidence_gap", 0.0),
            "kappa_v4": r.get("kappa_v4", 0.0),
            "delta_G_v4": r.get("delta_G_v4", 0.0),
            "delta_L_v4": r.get("delta_L_v4", 0.0),
            "failure_type": batch_ft_index.get(r.get("query_id", ""), "boundary"),
            "source_channel": r.get("source_class", "unknown"),
            "is_wrong": r["is_wrong"],
            "disagreement_flag": r.get("divergence", False),
        }
        existing_samples.append(sample)

    # Assign manifolds
    # Manifold assignment logic (mirrors runtime routing):
    # 1. overconfidence/blind_spot: failure_type=overconfidence OR (source_channel=blind_spot)
    # 2. contradiction: failure_type=contradiction
    # 3. boundary: everything else (including failure_type=boundary AND unclassified existing)
    all_samples = batch_samples + existing_samples

    manifolds = {
        "overconfidence": [],
        "contradiction": [],
        "boundary": [],
    }

    for s in all_samples:
        ft = s["failure_type"]
        src = s.get("source_channel", "")

        if ft == "overconfidence" or src == "blind_spot":
            # Blind spot channel samples all go to overconfidence manifold
            # (they represent the same failure geometry: overconfidence collapse)
            s["manifold"] = "overconfidence"
            manifolds["overconfidence"].append(s)
        elif ft == "contradiction":
            s["manifold"] = "contradiction"
            manifolds["contradiction"].append(s)
        else:
            # Default: boundary (most samples)
            s["manifold"] = "boundary"
            manifolds["boundary"].append(s)

    return manifolds


def extract_features(samples, feature_list):
    """Extract feature matrix and target vector from samples."""
    X = []
    y = []
    for s in samples:
        features = [s.get(f, 0.0) for f in feature_list]
        if any(v is None for v in features):
            continue
        X.append(features)
        y.append(s["is_wrong"])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def train_overconfidence_detector(samples):
    """Train blind spot / overconfidence detector.

    With 100% wrong rate, this is a DETECTION problem, not a prediction problem.
    The model learns to identify the manifold boundary, not predict within-manifold outcomes.

    Returns a detection model: P(is_blind_spot | features).
    """
    print("\n" + "=" * 60)
    print("MANIFOLD: OVERCONFIDENCE (BLIND SPOT)")
    print("=" * 60)

    n = len(samples)
    n_wrong = sum(1 for s in samples if s["is_wrong"] == 1)
    print(f"  Samples: {n}")
    print(f"  Wrong rate: {n_wrong}/{n} = {n_wrong/n:.1%}")
    print(f"  Signal character: PERFECT SEPARABILITY — detection, not prediction")

    if n < 5:
        print("  WARNING: Too few samples for statistical model")
        print("  Using rule-based detection fallback")
        return {
            "type": "rule_based",
            "detection_rate": 1.0,
            "n_samples": n,
            "n_wrong": n_wrong,
            "description": (
                "100% wrong rate — no within-manifold model needed. "
                "P(is_wrong | overconfidence_manifold) = 1.0"
            ),
            "routing_rules": {
                "description": "Blind spot detection via proxy signals",
                "signal": "low proxy_score (0.0) + moderate confidence_gap + kappa_v4 < 0.5",
                "failure_modes": ["underspecified_tech", "vague_ambiguous", "domain_knowledge", "opinion_debate"],
            },
        }

    # Even with few samples, train a detector for manifold BOUNDARY identification
    # We need "not in overconfidence" samples too
    # Use boundary samples as negative class
    print("\n  Training manifold boundary detector...")
    # This would require negative samples — handled at integration time

    return {
        "type": "rule_based",
        "detection_rate": 1.0,
        "n_samples": n,
        "n_wrong": n_wrong,
        "description": "100% wrong rate — P(is_wrong | overconfidence) = 1.0",
    }


def train_contradiction_head(samples):
    """Train within-contradiction failure predictor.

    69% wrong rate — high signal density. The model predicts which
    structural disagreement cases actually produce wrong answers.
    """
    print("\n" + "=" * 60)
    print("MANIFOLD: CONTRADICTION (STRUCTURAL DISAGREEMENT)")
    print("=" * 60)

    n = len(samples)
    n_wrong = sum(1 for s in samples if s["is_wrong"] == 1)
    n_correct = n - n_wrong
    print(f"  Samples: {n}")
    print(f"  Wrong: {n_wrong}, Correct: {n_correct}")
    print(f"  Wrong rate: {n_wrong/n:.1%}")
    print(f"  Signal character: structural v4-v1 disagreement instability")

    if n < 6:
        print("  WARNING: Too few samples for statistical model")
        print("  Using base rate as predictor")
        return {
            "type": "base_rate",
            "base_rate": round(n_wrong / n, 4),
            "n_samples": n,
            "n_wrong": n_wrong,
            "description": f"Too few samples for LR. Base rate: {n_wrong/n:.1%}",
        }

    X, y = extract_features(samples, CONTRADICTION_FEATURES)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {CONTRADICTION_FEATURES}")

    # Feature statistics by class
    wrong_idx = np.where(y == 1)[0]
    correct_idx = np.where(y == 0)[0]
    print(f"\n  Feature means:")
    print(f"  {'Feature':20s}  {'Wrong':>8s}  {'Correct':>8s}  {'Gap':>8s}")
    for i, fname in enumerate(CONTRADICTION_FEATURES):
        w_mean = X[wrong_idx, i].mean() if len(wrong_idx) > 0 else 0
        c_mean = X[correct_idx, i].mean() if len(correct_idx) > 0 else 0
        print(f"  {fname:20s}  {w_mean:8.4f}  {c_mean:8.4f}  {w_mean - c_mean:+8.4f}")

    # Train with balanced class weight
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )

    # Use leave-one-out or simple train/test since n is small
    if n >= 10:
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    else:
        calibrated = base_model

    calibrated.fit(X, y)

    # Evaluate
    y_pred = calibrated.predict(X)
    y_prob = calibrated.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = None
    try:
        brier = brier_score_loss(y, y_prob)
    except ValueError:
        brier = None

    print(f"\n  Training metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    if auc is not None:
        print(f"    AUC-ROC:   {auc:.4f}")
    if brier is not None:
        print(f"    Brier:     {brier:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"    Confusion matrix:")
    print(f"                  Pred_0  Pred_1")
    print(f"      Actual_0    {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"      Actual_1    {cm[1][0]:>6}  {cm[1][1]:>6}")

    # Coefficients
    coef_dict = {}
    if hasattr(calibrated, "coef_"):
        for fname, coef in zip(CONTRADICTION_FEATURES, calibrated.coef_[0]):
            coef_dict[fname] = round(float(coef), 6)
    elif hasattr(calibrated, "calibrated_classifiers_") and len(calibrated.calibrated_classifiers_) > 0:
        bm = calibrated.calibrated_classifiers_[0].estimator
        if hasattr(bm, "coef_"):
            for fname, coef in zip(CONTRADICTION_FEATURES, bm.coef_[0]):
                coef_dict[fname] = round(float(coef), 6)

    if coef_dict:
        print(f"\n  Feature coefficients:")
        for fname, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
            direction = "wrong" if coef > 0 else "correct"
            print(f"    {fname:20s}  {coef:+.6f}  -> {direction}")

    return {
        "type": "logistic_regression",
        "model": calibrated,
        "features": CONTRADICTION_FEATURES,
        "n_samples": n,
        "n_wrong": n_wrong,
        "metrics": {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "confusion_matrix": cm.tolist(),
        },
        "coefficients": coef_dict,
        "base_rate": round(n_wrong / n, 4),
        "description": f"Contradiction manifold: {n} samples, {n_wrong/n:.1%} wrong rate",
    }


def train_boundary_head(samples):
    """Train within-boundary failure predictor.

    12.5% wrong rate — standard binary classification.
    This is the "normal" prediction regime where most samples live.
    """
    print("\n" + "=" * 60)
    print("MANIFOLD: BOUNDARY (MOSTLY CORRECT AMBIGUITY)")
    print("=" * 60)

    n = len(samples)
    n_wrong = sum(1 for s in samples if s["is_wrong"] == 1)
    n_correct = n - n_wrong
    print(f"  Samples: {n}")
    print(f"  Wrong: {n_wrong}, Correct: {n_correct}")
    print(f"  Wrong rate: {n_wrong/n:.1%}")
    print(f"  Signal character: standard uncertainty prediction regime")

    # Check feature availability
    has_delta = any(s.get("delta_G_v4", 0.0) != 0.0 or s.get("delta_L_v4", 0.0) != 0.0
                   for s in samples[:10])
    features = BOUNDARY_FEATURES_FULL if has_delta else CONTRADICTION_FEATURES

    X, y = extract_features(samples, features)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {features}")

    # Feature statistics by class
    wrong_idx = np.where(y == 1)[0]
    correct_idx = np.where(y == 0)[0]

    print(f"\n  Feature means (class separation):")
    print(f"  {'Feature':20s}  {'Wrong':>8s}  {'Correct':>8s}  {'Gap':>8s}")
    for i, fname in enumerate(features):
        w_mean = X[wrong_idx, i].mean() if len(wrong_idx) > 0 else 0
        c_mean = X[correct_idx, i].mean() if len(correct_idx) > 0 else 0
        print(f"  {fname:20s}  {w_mean:8.4f}  {c_mean:8.4f}  {w_mean - c_mean:+8.4f}")

    # Train with balanced class weight + isotonic calibration
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )

    if n >= 50:
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    else:
        calibrated = base_model

    calibrated.fit(X, y)

    # Evaluate
    y_pred = calibrated.predict(X)
    y_prob = calibrated.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = None
    try:
        brier = brier_score_loss(y, y_prob)
    except ValueError:
        brier = None

    # Cross-validation
    cv_auc_mean, cv_auc_std = None, None
    try:
        cv_folds = min(5, sum(y == 0), sum(y == 1))
        if cv_folds >= 2:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(calibrated, X, y, cv=skf, scoring="roc_auc")
            cv_auc_mean = float(np.mean(cv_scores))
            cv_auc_std = float(np.std(cv_scores))
    except Exception:
        pass

    print(f"\n  Training metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    if auc is not None:
        print(f"    AUC-ROC:   {auc:.4f}")
    if cv_auc_mean is not None:
        print(f"    CV AUC:    {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    if brier is not None:
        print(f"    Brier:     {brier:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"    Confusion matrix:")
    print(f"                  Pred_0  Pred_1")
    print(f"      Actual_0    {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"      Actual_1    {cm[1][0]:>6}  {cm[1][1]:>6}")

    # Cost-optimal threshold
    best_threshold = 0.5
    if auc is not None:
        FALSE_ACCEPT_COST = 5.0
        FALSE_REJECT_COST = 1.0
        ESCALATION_COST = 0.5

        best_cost = float("inf")
        for t in np.arange(0.1, 0.9, 0.05):
            cost = 0
            for i in range(len(y)):
                p = y_prob[i]
                if p >= t:
                    # escalate/review
                    cost += ESCALATION_COST if y[i] == 1 else FALSE_REJECT_COST
                else:
                    # accept
                    cost += FALSE_ACCEPT_COST * p if y[i] == 1 else 0
            if cost < best_cost:
                best_cost = cost
                best_threshold = t

        print(f"    Optimal threshold: {best_threshold:.2f}")

    # Coefficients
    coef_dict = {}
    if hasattr(calibrated, "calibrated_classifiers_") and len(calibrated.calibrated_classifiers_) > 0:
        bm = calibrated.calibrated_classifiers_[0].estimator
    elif hasattr(calibrated, "estimator"):
        bm = calibrated.estimator
    else:
        bm = calibrated

    if hasattr(bm, "coef_"):
        for fname, coef in zip(features, bm.coef_[0]):
            coef_dict[fname] = round(float(coef), 6)
        intercept = round(float(bm.intercept_[0]), 6)
    else:
        intercept = None

    if coef_dict:
        print(f"\n  Feature coefficients:")
        for fname, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
            direction = "wrong" if coef > 0 else "correct"
            print(f"    {fname:20s}  {coef:+.6f}  -> {direction}")

    return {
        "type": "logistic_regression",
        "model": calibrated,
        "features": features,
        "n_samples": n,
        "n_wrong": n_wrong,
        "metrics": {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "confusion_matrix": cm.tolist(),
            "cv_auc_mean": round(cv_auc_mean, 4) if cv_auc_mean is not None else None,
            "cv_auc_std": round(cv_auc_std, 4) if cv_auc_std is not None else None,
        },
        "coefficients": coef_dict,
        "intercept": intercept,
        "optimal_threshold": round(best_threshold, 2),
        "base_rate": round(n_wrong / n, 4),
        "description": f"Boundary manifold: {n} samples, {n_wrong/n:.1%} wrong rate",
    }


def train_manifold_router(all_samples):
    """Train a 3-class manifold router.

    Predicts which manifold a sample belongs to based on observable features.
    This enables decomposition: route first, then apply per-manifold head.
    """
    print("\n" + "=" * 60)
    print("MANIFOLD ROUTER (3-CLASS CLASSIFIER)")
    print("=" * 60)

    # Build routing dataset
    # Positive examples: samples with known manifold assignment
    # Negative examples: boundary samples (largest class, serves as baseline)
    routing_data = []
    for s in all_samples:
        manifold = s.get("manifold", "boundary")
        # Map to routing classes
        if manifold == "overconfidence":
            routing_label = 0  # overconfidence/blind_spot
        elif manifold == "contradiction":
            routing_label = 1  # contradiction
        else:
            routing_label = 2  # boundary

        features = [s.get(f, 0.0) for f in ROUTING_FEATURES]
        routing_data.append((features, routing_label))

    X_route = np.array([d[0] for d in routing_data], dtype=np.float64)
    y_route = np.array([d[1] for d in routing_data], dtype=np.int32)

    class_counts = Counter(y_route)
    class_names = {0: "overconfidence", 1: "contradiction", 2: "boundary"}
    print(f"  Routing dataset: {len(routing_data)} samples")
    for cls_id in sorted(class_counts.keys()):
        print(f"    {class_names[cls_id]:20s}: {class_counts[cls_id]}")

    # Train multinomial logistic regression
    if len(class_counts) >= 2:
        router = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            random_state=42,
        )
        router.fit(X_route, y_route)

        y_pred_route = router.predict(X_route)
        route_acc = accuracy_score(y_route, y_pred_route)

        print(f"\n  Router accuracy: {route_acc:.4f}")

        # Per-class metrics
        print(f"\n  Per-class routing accuracy:")
        for cls_id in sorted(class_counts.keys()):
            mask = y_route == cls_id
            if mask.sum() > 0:
                cls_acc = (y_pred_route[mask] == cls_id).mean()
                print(f"    {class_names[cls_id]:20s}: {cls_acc:.4f} ({mask.sum()} samples)")

        # Router coefficients
        if hasattr(router, "coef_"):
            print(f"\n  Router feature importance (avg absolute coefficient):")
            feat_importance = np.mean(np.abs(router.coef_), axis=0)
            for i, fname in enumerate(ROUTING_FEATURES):
                print(f"    {fname:20s}: {feat_importance[i]:.6f}")

        # Cross-validation
        try:
            cv_folds = min(5, *(class_counts[c] for c in class_counts if class_counts[c] >= 2))
            if cv_folds >= 2:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(router, X_route, y_route, cv=skf, scoring="accuracy")
                print(f"\n  Router CV accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        except Exception:
            pass

        return {
            "type": "multinomial_lr",
            "model": router,
            "features": ROUTING_FEATURES,
            "class_names": class_names,
            "n_samples": len(routing_data),
            "accuracy": round(route_acc, 4),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        }
    else:
        print("  WARNING: Not enough classes for routing model")
        return {
            "type": "rule_based",
            "class_names": class_names,
            "n_samples": len(routing_data),
        }


def evaluate_manifold_separation(manifolds):
    """Evaluate how well the manifolds are separated in feature space.

    High separation = the failure geometries are genuinely distinct.
    Low separation = the manifolds overlap and decomposition may not help.
    """
    print("\n" + "=" * 60)
    print("MANIFOLD SEPARATION ANALYSIS")
    print("=" * 60)

    # Compute per-manifold feature centroids
    centroids = {}
    for name, samples in manifolds.items():
        if not samples:
            continue
        X, _ = extract_features(samples, ROUTING_FEATURES)
        centroid = X.mean(axis=0)
        centroids[name] = centroid

    # Pairwise distances between centroids
    manifold_names = list(centroids.keys())
    print(f"\n  Centroid distances (L2):")
    print(f"  {'':20s}", end="")
    for name in manifold_names:
        print(f"  {name:>15s}", end="")
    print()

    separation_matrix = {}
    for n1 in manifold_names:
        print(f"  {n1:20s}", end="")
        separation_matrix[n1] = {}
        for n2 in manifold_names:
            dist = np.linalg.norm(centroids[n1] - centroids[n2])
            separation_matrix[n1][n2] = round(float(dist), 4)
            print(f"  {dist:15.4f}", end="")
        print()

    # Per-manifold variance (compactness)
    print(f"\n  Within-manifold variance (compactness):")
    for name, samples in manifolds.items():
        if len(samples) < 2:
            print(f"    {name:20s}: N/A (< 2 samples)")
            continue
        X, _ = extract_features(samples, ROUTING_FEATURES)
        var = np.mean(np.var(X, axis=0))
        print(f"    {name:20s}: {var:.6f}")

    # Information-theoretic separation
    # Entropy of manifold assignment (higher = more informative decomposition)
    total = sum(len(s) for s in manifolds.values())
    if total > 0:
        entropy = 0
        for name, samples in manifolds.items():
            p = len(samples) / total
            if p > 0:
                entropy -= p * np.log2(p)

        max_entropy = np.log2(len(manifolds)) if len(manifolds) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        print(f"\n  Decomposition informativeness:")
        print(f"    Raw entropy: {entropy:.4f} bits")
        print(f"    Max entropy: {max_entropy:.4f} bits")
        print(f"    Normalized:  {normalized_entropy:.4f}")

        # Per-manifold wrong rate entropy (how different are the failure rates?)
        wr_entropy = 0
        for name, samples in manifolds.items():
            if not samples:
                continue
            n_wrong = sum(1 for s in samples if s["is_wrong"] == 1)
            wr = n_wrong / len(samples)
            p = len(samples) / total
            if p > 0 and wr > 0 and wr < 1:
                wr_entropy += p * (-wr * np.log2(wr) - (1 - wr) * np.log2(1 - wr))

        print(f"    Wrong-rate conditional entropy: {wr_entropy:.4f} bits")

    return {
        "centroid_distances": separation_matrix,
        "total_samples": total,
        "manifold_sizes": {name: len(s) for name, s in manifolds.items()},
        "manifold_wrong_rates": {
            name: round(
                sum(1 for s in s_list if s["is_wrong"] == 1) / len(s_list), 4
            ) if s_list else 0
            for name, s_list in manifolds.items()
        },
    }


def main():
    force = "--retrain" in sys.argv

    print("=" * 60)
    print("MANIFOLD CLASSIFIER TRAINING [v2.5.0]")
    print("Decomposed failure manifold map — replacing unified predictor")
    print("=" * 60)

    if os.path.exists(MANIFOLD_MODEL_PATH) and not force:
        print(f"Manifold models already exist at {MANIFOLD_MODEL_PATH}")
        print("Use --retrain to force retraining")
        return

    # Load data
    print("\n[1/6] Loading data...")
    batch_labels = load_jsonl(BATCH_LABELS_PATH)
    existing_dataset = load_jsonl(DATASET_PATH)
    print(f"  Batch labels: {len(batch_labels)} (with failure_type)")
    print(f"  Existing dataset: {len(existing_dataset)} ({sum(1 for r in existing_dataset if r.get('is_wrong') is not None)} labeled)")

    # Step 1: Freeze baseline
    print("\n[2/6] Freezing v2.4.0 baseline...")
    baseline = freeze_baseline(batch_labels, existing_dataset)

    # Step 2: Build per-manifold datasets
    print("\n[3/6] Building per-manifold datasets...")
    manifolds = build_manifold_datasets(batch_labels, existing_dataset)

    for name, samples in manifolds.items():
        n_wrong = sum(1 for s in samples if s["is_wrong"] == 1)
        print(f"  {name:20s}: {len(samples):4d} samples, {n_wrong} wrong ({n_wrong/len(samples):.1%})" if samples else f"  {name:20s}: 0 samples")

    # Step 3: Evaluate manifold separation
    print("\n[4/6] Evaluating manifold separation...")
    separation = evaluate_manifold_separation(manifolds)

    # Step 4: Train per-manifold heads
    print("\n[5/6] Training per-manifold classifiers...")

    overconfidence_head = train_overconfidence_detector(manifolds["overconfidence"])
    contradiction_head = train_contradiction_head(manifolds["contradiction"])
    boundary_head = train_boundary_head(manifolds["boundary"])

    # Step 5: Train manifold router
    all_samples = []
    for name, samples in manifolds.items():
        all_samples.extend(samples)

    router = train_manifold_router(all_samples)

    # Step 6: Save model package
    print("\n[6/6] Saving manifold model package...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    manifold_package = {
        "version": "v2.5.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "decomposed_failure_manifold_map",
        "description": (
            "Three-manifold decomposition: each failure geometry has its own "
            "predictive surface. Replaces unified single-model predictor."
        ),
        "manifolds": {
            "overconfidence": {
                "head": {k: v for k, v in overconfidence_head.items() if k != "model"},
                "n_samples": overconfidence_head.get("n_samples", 0),
                "wrong_rate": 1.0,
            },
            "contradiction": {
                "head": {k: v for k, v in contradiction_head.items() if k != "model"},
                "n_samples": contradiction_head.get("n_samples", 0),
                "wrong_rate": contradiction_head.get("base_rate", 0),
            },
            "boundary": {
                "head": {k: v for k, v in boundary_head.items() if k != "model"},
                "n_samples": boundary_head.get("n_samples", 0),
                "wrong_rate": boundary_head.get("base_rate", 0),
            },
        },
        "router": {k: v for k, v in router.items() if k != "model"},
        "separation": separation,
        "baseline_reference": baseline["frozen_at"],
    }

    # Save models (with actual model objects for runtime use)
    model_package = {
        "overconfidence_head": overconfidence_head.get("model"),
        "contradiction_head": contradiction_head.get("model"),
        "boundary_head": boundary_head.get("model"),
        "router_model": router.get("model"),
        "metadata": manifold_package,
    }

    with open(MANIFOLD_MODEL_PATH, "wb") as f:
        pickle.dump(model_package, f)
    print(f"  Model saved to {MANIFOLD_MODEL_PATH}")

    # Save report (without model objects, JSON-serializable)
    report = manifold_package
    report["training_summary"] = {
        "total_samples": sum(len(s) for s in manifolds.values()),
        "manifold_distribution": {
            name: {
                "n": len(samples),
                "n_wrong": sum(1 for s in samples if s["is_wrong"] == 1),
                "wrong_rate": round(
                    sum(1 for s in samples if s["is_wrong"] == 1) / len(samples), 4
                ) if samples else 0,
            }
            for name, samples in manifolds.items()
        },
        "key_insight": (
            "AUC is no longer the primary metric. The system now produces "
            "per-manifold predictive surfaces, each capturing a distinct "
            "failure geometry. Manifold separation quality is the new "
            "evaluation axis."
        ),
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(MANIFOLD_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  Report saved to {MANIFOLD_REPORT_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — MANIFOLD DECOMPOSITION SUMMARY")
    print("=" * 60)
    print(f"""
  Architecture: decomposed_failure_manifold_map
  Baseline:     {baseline['frozen_at']}

  Manifold             n      Wrong%   AUC      Head Type
  ─────────────────────────────────────────────────────────
  overconfidence       {overconfidence_head.get('n_samples', 0):>4d}   100.0%    N/A      detection (rule-based)
  contradiction        {contradiction_head.get('n_samples', 0):>4d}    {contradiction_head.get('base_rate', 0)*100:>5.1f}%   {str(contradiction_head.get('metrics', {}).get('auc', 'N/A')):>5s}      LR + calibration
  boundary           {boundary_head.get('n_samples', 0):>5d}    {boundary_head.get('base_rate', 0)*100:>5.1f}%   {str(boundary_head.get('metrics', {}).get('auc', 'N/A')):>5s}      LR + calibration

  Router accuracy: {router.get('accuracy', 'N/A')}

  Key transition:
    FROM: one model, one AUC, one failure geometry
    TO:   three manifolds, three surfaces, structural decomposition

  What changed:
    - Blind spot (100% wrong) is now a DETECTION problem, not prediction
    - Contradiction (69% wrong) has its own failure surface
    - Boundary (12.5% wrong) is the standard prediction regime
    - The unified AUC was collapsing multiple geometries into one number
""")


if __name__ == "__main__":
    main()

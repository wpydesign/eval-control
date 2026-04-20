#!/usr/bin/env python3
"""
train_failure_predictor.py — Train + calibrate + optimize failure predictor [v2.2.1]

Reads logs/failure_dataset.jsonl, trains on labeled samples, saves model + schema.

Features (6 numeric):
  S_v4, S_v1, confidence_gap, kappa_v4, delta_G_v4, delta_L_v4

Model:
  LogisticRegression(class_weight='balanced', max_iter=1000)
  + CalibratedClassifierCV(isotonic) for probability calibration
  Outputs calibrated P(is_wrong) ∈ [0, 1]

Threshold optimization:
  Cost-based: minimizes expected cost given:
    C(false_accept) = 5.0  — letting a wrong answer through
    C(false_reject) = 1.0  — blocking a correct answer
    C(escalation)   = 0.5  — cost of human review per escalation
  Finds optimal review/escalate thresholds via grid search.

Outputs:
  model/failure_predictor.pkl   — trained model + feature list + metadata
  model/training_report.json    — accuracy, AUC, coefficients, confusion matrix

Usage:
  python scripts/train_failure_predictor.py
  python scripts/train_failure_predictor.py --retrain   # force retrain even if model exists
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import cross_val_score, cross_val_predict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "failure_predictor.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")

NUMERIC_FEATURES = [
    "S_v4", "S_v1", "confidence_gap", "kappa_v4", "delta_G_v4", "delta_L_v4"
]

# Cost matrix for threshold optimization
# These represent relative costs of different decision outcomes
FALSE_ACCEPT_COST = 5.0   # letting a wrong answer through (highest cost)
FALSE_REJECT_COST = 1.0   # blocking a correct answer (nuisance, not catastrophic)
ESCALATION_COST = 0.5      # cost per escalation (human review time)


def load_dataset():
    """Load failure_dataset.jsonl, return labeled samples only."""
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run build_failure_dataset.py first.")
        sys.exit(1)

    labeled = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_wrong") is not None:
                labeled.append(row)

    return labeled


def extract_X_y(samples):
    """Extract feature matrix X and target vector y from samples."""
    X = []
    y = []
    missing_count = 0
    for s in samples:
        features = [s.get(f, 0.0) for f in NUMERIC_FEATURES]
        # Check for any missing/None values
        if any(v is None for v in features):
            missing_count += 1
            continue
        X.append(features)
        y.append(s["is_wrong"])

    if missing_count > 0:
        print(f"  WARNING: {missing_count} samples skipped due to missing features")

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def train(X, y):
    """Train logistic regression + isotonic calibration."""
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    # Isotonic calibration: non-parametric, fits P(y|X) more accurately
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    # Split: fit base on all, calibrate via cross-val to avoid overfitting
    # Use 3-fold cross-validation for calibration fitting
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X, y)
    return calibrated


def evaluate(model, X, y):
    """Compute evaluation metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = np.mean(y_pred == y)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = None  # only one class in y
    try:
        brier = brier_score_loss(y, y_prob)
    except ValueError:
        brier = None

    cm = confusion_matrix(y, y_pred).tolist()

    # Cross-validation (5-fold)
    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)), scoring="roc_auc")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_mean, cv_std = None, None

    return {
        "accuracy": round(float(accuracy), 4),
        "auc": round(auc, 4) if auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "confusion_matrix": cm,
        "cv_auc_mean": round(cv_mean, 4) if cv_mean is not None else None,
        "cv_auc_std": round(cv_std, 4) if cv_std is not None else None,
    }


def optimize_thresholds(y_true, y_prob):
    """Find cost-optimal review and escalate thresholds via grid search.

    Decision framework:
      risk_score < review_threshold    → accept (no intervention)
      review_threshold <= risk < escalate → shadow_review (flag for attention)
      risk_score >= escalate_threshold  → escalate (force review / block)

    Cost model per sample:
      - If actually wrong (y=1):
          accept it     → FALSE_ACCEPT_COST (5.0)
          shadow_review → ESCALATION_COST (0.5, caught some)
          escalate      → ESCALATION_COST (0.5, caught most)
      - If actually correct (y=0):
          accept it     → 0 (correct decision)
          shadow_review → ESCALATION_COST * 0.3 (minor nuisance)
          escalate      → FALSE_REJECT_COST (1.0, unnecessary block)
    """
    n = len(y_true)
    best_cost = float('inf')
    best_review = 0.2
    best_escalate = 0.4

    # Grid search over threshold pairs
    for review_t in np.arange(0.05, 0.60, 0.05):
        for escalate_t in np.arange(review_t + 0.05, 1.0, 0.05):
            total_cost = 0.0
            for i in range(n):
                p = y_prob[i]
                actual_wrong = y_true[i]

                if p < review_t:
                    action = "accept"
                elif p < escalate_t:
                    action = "shadow_review"
                else:
                    action = "escalate"

                if actual_wrong == 1:
                    if action == "accept":
                        total_cost += FALSE_ACCEPT_COST * p  # weighted by confidence
                    elif action == "shadow_review":
                        total_cost += ESCALATION_COST
                    else:  # escalate
                        total_cost += ESCALATION_COST * 0.5  # caught, lower residual risk
                else:  # actual correct
                    if action == "accept":
                        total_cost += 0
                    elif action == "shadow_review":
                        total_cost += ESCALATION_COST * 0.3
                    else:  # escalate
                        total_cost += FALSE_REJECT_COST

            if total_cost < best_cost:
                best_cost = total_cost
                best_review = review_t
                best_escalate = escalate_t

    return {
        "review_threshold": round(float(best_review), 2),
        "escalate_threshold": round(float(best_escalate), 2),
        "total_cost": round(float(best_cost), 2),
        "cost_per_sample": round(float(best_cost / n), 4),
        "false_accept_cost": FALSE_ACCEPT_COST,
        "false_reject_cost": FALSE_REJECT_COST,
        "escalation_cost": ESCALATION_COST,
    }


def main():
    force = "--retrain" in sys.argv

    if os.path.exists(MODEL_PATH) and not force:
        print(f"Model already exists at {MODEL_PATH}")
        print("Use --retrain to force retraining")
        return

    print("Loading labeled dataset...")
    samples = load_dataset()
    print(f"  {len(samples)} labeled samples")

    if len(samples) < 10:
        print("ERROR: Need at least 10 labeled samples to train. Get more labeled data.")
        sys.exit(1)

    print(f"Extracting {len(NUMERIC_FEATURES)} features: {NUMERIC_FEATURES}")
    X, y = extract_X_y(samples)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution: is_wrong=0: {sum(y==0)}, is_wrong=1: {sum(y==1)}")

    print("\nTraining logistic regression...")
    model = train(X, y)

    print("Evaluating...")
    metrics = evaluate(model, X, y)

    print("\nOptimizing decision thresholds (cost-based)...")
    y_prob_all = model.predict_proba(X)[:, 1]
    thresholds = optimize_thresholds(y, y_prob_all)
    print(f"  Optimal review threshold:   {thresholds['review_threshold']}")
    print(f"  Optimal escalate threshold: {thresholds['escalate_threshold']}")
    print(f"  Expected cost per sample:   {thresholds['cost_per_sample']:.4f}")
    print(f"  Cost model: FA={FALSE_ACCEPT_COST}, FR={FALSE_REJECT_COST}, Esc={ESCALATION_COST}")

    # Feature coefficients (from fitted base model)
    coef_dict = {}
    # CalibratedClassifierCV with cv>0 stores fitted estimators in calibrated_classifiers_
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        base_model = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
        base_model = model.estimator
    else:
        base_model = model
    if hasattr(base_model, "coef_"):
        for fname, coef in zip(NUMERIC_FEATURES, base_model.coef_[0]):
            coef_dict[fname] = round(float(coef), 6)

    # Print results
    print(f"\n  Accuracy:    {metrics['accuracy']:.2%}")
    if metrics["auc"] is not None:
        print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    if metrics["brier_score"] is not None:
        print(f"  Brier score: {metrics['brier_score']:.4f}")
    if metrics["cv_auc_mean"] is not None:
        print(f"  CV AUC:      {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")

    print(f"\n  Confusion matrix:")
    print(f"                predicted_0  predicted_1")
    for i, label in enumerate(["actual_0 (correct)", "actual_1 (wrong)"]):
        print(f"  {label:20s}  {metrics['confusion_matrix'][i][0]:>10}  {metrics['confusion_matrix'][i][1]:>10}")

    print(f"\n  Feature coefficients (higher = more predictive of 'wrong'):")
    for fname, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
        direction = "wrong" if coef > 0 else "correct"
        print(f"    {fname:20s}  {coef:+.6f}  (→ {direction})")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_package = {
        "model": model,
        "features": NUMERIC_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(y),
        "metrics": metrics,
        "coefficients": coef_dict,
        "thresholds": thresholds,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save training report
    report = {
        "trained_at": model_package["trained_at"],
        "n_samples": len(y),
        "n_features": len(NUMERIC_FEATURES),
        "features": NUMERIC_FEATURES,
        "metrics": metrics,
        "coefficients": coef_dict,
        "intercept": round(float(base_model.intercept_[0]), 6) if hasattr(base_model, "intercept_") and base_model.intercept_ is not None else None,
        "thresholds": thresholds,
        "calibration_method": "isotonic",
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Training report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

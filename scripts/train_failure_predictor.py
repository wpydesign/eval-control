#!/usr/bin/env python3
"""
train_failure_predictor.py — Train logistic regression to predict P(is_wrong | signals) [v2.2.0]

Reads logs/failure_dataset.jsonl, trains on labeled samples, saves model + schema.

Features (6 numeric):
  S_v4, S_v1, confidence_gap, kappa_v4, delta_G_v4, delta_L_v4

Model:
  LogisticRegression(class_weight='balanced', max_iter=1000)
  Outputs calibrated P(is_wrong) ∈ [0, 1]

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
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import cross_val_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "failure_predictor.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")

NUMERIC_FEATURES = [
    "S_v4", "S_v1", "confidence_gap", "kappa_v4", "delta_G_v4", "delta_L_v4"
]


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
    """Train logistic regression with balanced class weights."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model


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

    # Feature coefficients (model insight)
    coef_dict = {}
    for fname, coef in zip(NUMERIC_FEATURES, model.coef_[0]):
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
        "intercept": round(float(model.intercept_[0]), 6),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Training report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()

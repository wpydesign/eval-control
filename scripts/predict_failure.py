#!/usr/bin/env python3
"""
predict_failure.py — Runtime failure probability predictor [v2.2.0]

Loads trained logistic regression model and computes P(is_wrong | signals)
for each evaluation result.

This is the prediction layer that turns reactive monitoring into proactive
failure prediction. Instead of "we detect problems after they happen,"
this gives: "this answer has 23% chance of being wrong."

Usage in pipeline:
  from scripts.predict_failure import FailurePredictor
  pred = FailurePredictor()
  risk_score = pred.predict(v4_scores)  # → 0.23
  if risk_score > 0.4:
      ...  # escalate

Risk thresholds:
  risk_score > 0.2 → shadow review (flag for human attention)
  risk_score > 0.4 → force review / escalate (probable failure)

Model retraining:
  pred = FailurePredictor()
  pred.retrain()  # rebuild dataset + retrain + reload

Outputs:
  model/failure_predictor.pkl  — trained model (created by train_failure_predictor.py)
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "model", "failure_predictor.pkl")
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")

# Risk thresholds (configurable)
RISK_REVIEW_THRESHOLD = 0.2    # above this → shadow review
RISK_ESCALATE_THRESHOLD = 0.4  # above this → force escalate

# Minimum samples for valid prediction
MIN_TRAIN_SAMPLES = 10


class FailurePredictor:
    """Loads and serves the failure probability predictor.

    Provides P(is_wrong | score_signals) for each evaluation.
    Falls back gracefully if no model is available (returns 0.0).
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self._model = None
        self._features = None
        self._metadata = None
        self._loaded = False
        self._load()

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def metadata(self):
        return self._metadata

    def _load(self):
        """Load trained model from disk."""
        if not os.path.exists(self.model_path):
            print(f"  [PREDICTOR] No model at {self.model_path} — risk_score will be 0.0")
            print(f"  [PREDICTOR] Run scripts/build_failure_dataset.py + scripts/train_failure_predictor.py")
            return False

        try:
            with open(self.model_path, "rb") as f:
                package = pickle.load(f)

            self._model = package["model"]
            self._features = package["features"]
            self._metadata = {
                "trained_at": package.get("trained_at", "unknown"),
                "n_samples": package.get("n_samples", 0),
                "metrics": package.get("metrics", {}),
                "coefficients": package.get("coefficients", {}),
            }
            self._loaded = True

            n = self._metadata["n_samples"]
            auc = self._metadata["metrics"].get("auc", "N/A")
            print(f"  [PREDICTOR] Loaded model (trained on {n} samples, AUC={auc})")
            return True
        except Exception as e:
            print(f"  [PREDICTOR] Failed to load model: {e}")
            return False

    def predict(self, v4_scores: dict, v1_scores: dict = None) -> dict:
        """Compute P(is_wrong | signals) for a single evaluation.

        Args:
            v4_scores: dict with keys S, kappa, delta_G, delta_L (from v4 evaluation)
            v1_scores: dict with keys S (from v1 evaluation). Optional.

        Returns:
            dict with:
              risk_score: float ∈ [0, 1] — P(is_wrong | signals)
              has_model: bool — whether predictor is available
              action: str — "none" | "shadow_review" | "escalate"
        """
        if not self._loaded or self._model is None:
            return {"risk_score": 0.0, "has_model": False, "action": "none"}

        try:
            # Build feature vector in the same order as training
            s_v4 = v4_scores.get("S", 0.0)
            s_v1 = v1_scores.get("S", 0.0) if v1_scores else 0.0
            confidence_gap = abs(s_v4 - s_v1)

            feature_values = [
                s_v4,                                    # S_v4
                s_v1,                                    # S_v1
                confidence_gap,                          # confidence_gap
                v4_scores.get("kappa", 0.0),            # kappa_v4
                v4_scores.get("delta_G", 0.0),          # delta_G_v4
                v4_scores.get("delta_L", 0.0),          # delta_L_v4
            ]

            X = np.array([feature_values], dtype=np.float64)
            prob = self._model.predict_proba(X)[0][1]  # P(class=1) = P(is_wrong)
            risk_score = float(round(prob, 4))

            # Determine action based on risk thresholds
            if risk_score >= RISK_ESCALATE_THRESHOLD:
                action = "escalate"
            elif risk_score >= RISK_REVIEW_THRESHOLD:
                action = "shadow_review"
            else:
                action = "none"

            return {"risk_score": risk_score, "has_model": True, "action": action}

        except Exception as e:
            return {"risk_score": 0.0, "has_model": False, "action": "none", "error": str(e)}

    def predict_batch(self, results: list) -> list:
        """Predict risk for a batch of evaluation results.

        Args:
            results: list of eval result dicts (from run_live_batch)

        Returns:
            list of prediction dicts (one per result)
        """
        predictions = []
        for r in results:
            v4 = r.get("v4", {})
            v1 = r.get("v1", {})
            pred = self.predict(v4, v1)
            predictions.append(pred)
        return predictions

    def retrain(self):
        """Rebuild dataset + retrain model + reload.

        This closes the loop: new labeled data → better predictions.
        """
        print("  [PREDICTOR] Retraining...")
        sys.path.insert(0, os.path.dirname(__file__))

        # Step 1: rebuild dataset
        try:
            from build_failure_dataset import main as build_main
            print("  [PREDICTOR] Rebuilding failure dataset...")
            build_main()
        except Exception as e:
            print(f"  [PREDICTOR] Dataset rebuild failed: {e}")
            return False

        # Step 2: retrain model
        try:
            from train_failure_predictor import main as train_main
            # Temporarily set --retrain flag
            old_argv = sys.argv
            sys.argv = ["train_failure_predictor.py", "--retrain"]
            train_main()
            sys.argv = old_argv
        except Exception as e:
            print(f"  [PREDICTOR] Retrain failed: {e}")
            return False

        # Step 3: reload model
        return self._load()

    def print_status(self):
        """Print predictor status summary."""
        if not self._loaded:
            print("  [PREDICTOR] Status: NOT LOADED (no model)")
            return

        m = self._metadata
        n = m["n_samples"]
        auc = m["metrics"].get("auc", "N/A")
        acc = m["metrics"].get("accuracy", "N/A")
        trained = m["trained_at"][:19] if m["trained_at"] != "unknown" else "unknown"
        print(f"  [PREDICTOR] Status: LOADED")
        print(f"  [PREDICTOR] Trained: {trained} on {n} samples")
        print(f"  [PREDICTOR] AUC={auc}, Accuracy={acc}")
        print(f"  [PREDICTOR] Thresholds: review>{RISK_REVIEW_THRESHOLD}, escalate>{RISK_ESCALATE_THRESHOLD}")


def main():
    """CLI: predict risk for a single sample or batch."""
    pred = FailurePredictor()
    pred.print_status()

    # Demo: test with known bad/good signals
    print("\nDemo predictions:")
    demos = [
        ("bad (low S, high delta_G)", {"S": 0.25, "kappa": 0.15, "delta_G": 0.85, "delta_L": 0.10}, {"S": 0.18}),
        ("good (high S, low delta_G)", {"S": 0.72, "kappa": 0.60, "delta_G": 0.45, "delta_L": 0.02}, {"S": 0.66}),
        ("borderline (mid S)", {"S": 0.48, "kappa": 0.35, "delta_G": 0.65, "delta_L": 0.05}, {"S": 0.42}),
    ]
    for label, v4, v1 in demos:
        result = pred.predict(v4, v1)
        action_tag = f" → {result['action']}" if result['action'] != 'none' else ""
        print(f"  {label:35s}  risk={result['risk_score']:.3f}{action_tag}")


if __name__ == "__main__":
    main()

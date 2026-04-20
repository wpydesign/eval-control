#!/usr/bin/env python3
"""
manifold_predict.py — Runtime decomposed failure manifold predictor [v2.5.0]

Replaces the unified FailurePredictor with a three-manifold decomposition:
  Input features → ManifoldRouter → Per-manifold head → P(is_wrong)

Each failure manifold has its own predictive surface:
  1. Overconfidence (blind spot): P(wrong | overconfidence) = 1.0 (detection)
  2. Contradiction: LR classifier, AUC=0.88 (structural disagreement)
  3. Boundary: LR classifier, AUC=0.71 (standard uncertainty)

The routing uses both the learned multinomial LR router and rule-based
fallbacks for signals that the router may miss (e.g., explicit disagreement).

Usage:
  from scripts.manifold_predict import ManifoldPredictor

  mp = ManifoldPredictor()
  result = mp.predict(v4_scores, v1_scores, disagreement=False)
  # → {
  #     "risk_score": 0.23,
  #     "manifold": "boundary",
  #     "manifold_confidence": 0.85,
  #     "per_manifold_scores": {
  #         "overconfidence": 0.02,
  #         "contradiction": 0.05,
  #         "boundary": 0.93,
  #     },
  #     "action": "shadow_review",
  #     "has_model": True,
  # }

Thresholds:
  - overconfidence manifold → always escalate (100% wrong rate)
  - contradiction manifold → escalate if P(wrong) > 0.5
  - boundary manifold → standard cost-optimized threshold (0.35)
"""

import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFOLD_MODEL_PATH = os.path.join(BASE, "model", "manifold_models.pkl")
MANIFOLD_REPORT_PATH = os.path.join(BASE, "model", "manifold_report.json")

# Per-manifold decision thresholds (reflecting failure geometry)
OVERCONFIDENCE_THRESHOLD = 0.01  # anything routed here is almost certainly wrong
CONTRADICTION_THRESHOLD = 0.50   # 69% base rate — moderate threshold
BOUNDARY_THRESHOLD = 0.35        # cost-optimized from training

# Global risk thresholds for backward compatibility
RISK_REVIEW_THRESHOLD = 0.20
RISK_ESCALATE_THRESHOLD = 0.40


class ManifoldPredictor:
    """Decomposed failure manifold predictor.

    Instead of one model trying to represent three different failure geometries,
    this routes each sample to its appropriate manifold and applies the
    corresponding predictive surface.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or MANIFOLD_MODEL_PATH
        self._models = None
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
        """Load trained manifold models from disk."""
        if not os.path.exists(self.model_path):
            print(f"  [MANIFOLD] No models at {self.model_path}")
            print(f"  [MANIFOLD] Run scripts/manifold_classifier.py --retrain")
            return False

        try:
            with open(self.model_path, "rb") as f:
                package = pickle.load(f)

            self._models = {
                "overconfidence_head": package.get("overconfidence_head"),
                "contradiction_head": package.get("contradiction_head"),
                "boundary_head": package.get("boundary_head"),
                "router": package.get("router_model"),
            }
            self._metadata = package.get("metadata", {})

            if not self._metadata:
                # Try loading from report
                if os.path.exists(MANIFOLD_REPORT_PATH):
                    with open(MANIFOLD_REPORT_PATH) as f:
                        self._metadata = json.load(f)

            self._loaded = True

            # Print status
            manifolds = self._metadata.get("manifolds", {})
            router_acc = self._metadata.get("router", {}).get("accuracy", "N/A")
            print(f"  [MANIFOLD] Loaded decomposed failure manifold predictor")
            print(f"  [MANIFOLD] Version: {self._metadata.get('version', 'unknown')}")
            print(f"  [MANIFOLD] Router accuracy: {router_acc}")
            for name, info in manifolds.items():
                head = info.get("head", {})
                n = info.get("n_samples", 0)
                wr = info.get("wrong_rate", 0)
                auc = head.get("metrics", {}).get("auc", "N/A")
                print(f"  [MANIFOLD]   {name:20s}: {n} samples, {wr:.1%} wrong, AUC={auc}")

            return True
        except Exception as e:
            print(f"  [MANIFOLD] Failed to load: {e}")
            return False

    def _route(self, features, disagreement_flag=False):
        """Route a sample to its failure manifold.

        Uses the learned router with rule-based override for disagreement signals.

        Args:
            features: array-like of [S_v4, S_v1, confidence_gap, kappa_v4]
            disagreement_flag: bool — explicit v4/v1 disagreement

        Returns:
            dict with manifold assignment and routing probabilities
        """
        X = np.array([features], dtype=np.float64)
        class_names = {0: "overconfidence", 1: "contradiction", 2: "boundary"}

        # Rule-based override: explicit disagreement → contradiction manifold
        # This catches cases the router might miss due to feature overlap
        if disagreement_flag and features[2] > 0.05:  # confidence_gap > 5%
            # Strong disagreement signal — route to contradiction
            return {
                "manifold": "contradiction",
                "confidence": 0.9,
                "probabilities": {
                    "overconfidence": 0.05,
                    "contradiction": 0.85,
                    "boundary": 0.10,
                },
                "routing_method": "rule_based_disagreement",
            }

        # Use learned router if available
        router = self._models.get("router")
        if router is not None:
            try:
                probs = router.predict_proba(X)[0]
                pred_class = int(router.predict(X)[0])
                manifold_name = class_names.get(pred_class, "boundary")

                return {
                    "manifold": manifold_name,
                    "confidence": float(max(probs)),
                    "probabilities": {
                        class_names[i]: float(probs[i])
                        for i in range(len(probs))
                    },
                    "routing_method": "learned_router",
                }
            except Exception:
                pass

        # Fallback: heuristic routing
        confidence_gap = features[2]
        kappa = features[3]

        if confidence_gap > 0.25 and kappa < 0.35:
            return {
                "manifold": "overconfidence",
                "confidence": 0.7,
                "probabilities": {"overconfidence": 0.7, "contradiction": 0.2, "boundary": 0.1},
                "routing_method": "heuristic",
            }
        elif confidence_gap > 0.10:
            return {
                "manifold": "contradiction",
                "confidence": 0.6,
                "probabilities": {"overconfidence": 0.1, "contradiction": 0.6, "boundary": 0.3},
                "routing_method": "heuristic",
            }
        else:
            return {
                "manifold": "boundary",
                "confidence": 0.8,
                "probabilities": {"overconfidence": 0.05, "contradiction": 0.15, "boundary": 0.8},
                "routing_method": "heuristic",
            }

    def _predict_overconfidence(self):
        """Overconfidence manifold: 100% wrong rate → detection, not prediction."""
        return 1.0  # P(is_wrong | overconfidence) = 1.0

    def _predict_contradiction(self, features):
        """Contradiction manifold: predict P(wrong | contradiction features).

        Within contradiction cases, features indicate which disagreements
        actually produce wrong answers.
        """
        model = self._models.get("contradiction_head")
        if model is None:
            return 0.765  # base rate fallback

        try:
            X = np.array([features], dtype=np.float64)
            prob = model.predict_proba(X)[0][1]
            return float(round(prob, 4))
        except Exception:
            return 0.765  # base rate fallback

    def _predict_boundary(self, features):
        """Boundary manifold: standard P(wrong | boundary features).

        This is the "normal" prediction regime — mostly correct ambiguity.
        """
        model = self._models.get("boundary_head")
        if model is None:
            return 0.211  # base rate fallback

        try:
            X = np.array([features], dtype=np.float64)
            prob = model.predict_proba(X)[0][1]
            return float(round(prob, 4))
        except Exception:
            return 0.211  # base rate fallback

    def predict(self, v4_scores: dict, v1_scores: dict = None,
                disagreement_flag: bool = False) -> dict:
        """Compute P(is_wrong | manifold, features) using decomposed prediction.

        Args:
            v4_scores: dict with keys S, kappa, delta_G, delta_L
            v1_scores: dict with keys S. Optional.
            disagreement_flag: bool — whether v4 and v1 explicitly disagree

        Returns:
            dict with:
              risk_score: float — P(is_wrong | manifold, features)
              manifold: str — assigned manifold name
              manifold_confidence: float — routing confidence
              per_manifold_scores: dict — P(wrong) per manifold
              action: str — "none" | "shadow_review" | "escalate"
              has_model: bool
        """
        if not self._loaded:
            return {
                "risk_score": 0.0, "manifold": "unknown",
                "manifold_confidence": 0.0,
                "per_manifold_scores": {}, "action": "none", "has_model": False,
            }

        try:
            # Extract features
            s_v4 = v4_scores.get("S", 0.0)
            s_v1 = v1_scores.get("S", 0.0) if v1_scores else 0.0
            confidence_gap = abs(s_v4 - s_v1)
            kappa_v4 = v4_scores.get("kappa", 0.0)

            features = [s_v4, s_v1, confidence_gap, kappa_v4]

            # Step 1: Route to manifold
            routing = self._route(features, disagreement_flag)
            manifold = routing["manifold"]

            # Step 2: Compute per-manifold P(wrong)
            overconfidence_score = self._predict_overconfidence()
            contradiction_score = self._predict_contradiction(features)
            boundary_score = self._predict_boundary(features)

            per_manifold = {
                "overconfidence": overconfidence_score,
                "contradiction": contradiction_score,
                "boundary": boundary_score,
            }

            # Step 3: Select score based on routed manifold
            risk_score = per_manifold[manifold]

            # Step 4: Determine action
            action = self._determine_action(risk_score, manifold)

            return {
                "risk_score": risk_score,
                "manifold": manifold,
                "manifold_confidence": routing["confidence"],
                "per_manifold_scores": per_manifold,
                "routing_method": routing["routing_method"],
                "action": action,
                "has_model": True,
            }

        except Exception as e:
            return {
                "risk_score": 0.0, "manifold": "error",
                "manifold_confidence": 0.0,
                "per_manifold_scores": {},
                "action": "none", "has_model": False,
                "error": str(e),
            }

    def _determine_action(self, risk_score: float, manifold: str) -> str:
        """Determine action based on risk score and manifold geometry.

        Each manifold has its own decision threshold reflecting its failure rate.
        """
        if manifold == "overconfidence":
            # 100% wrong rate — always escalate
            return "escalate"
        elif manifold == "contradiction":
            # 69% wrong rate — escalate if P(wrong) > contradiction threshold
            if risk_score >= CONTRADICTION_THRESHOLD:
                return "escalate"
            elif risk_score >= CONTRADICTION_THRESHOLD * 0.6:
                return "shadow_review"
            return "none"
        else:
            # Boundary: standard cost-optimized threshold
            if risk_score >= RISK_ESCALATE_THRESHOLD:
                return "escalate"
            elif risk_score >= BOUNDARY_THRESHOLD:
                return "shadow_review"
            elif risk_score >= RISK_REVIEW_THRESHOLD:
                return "shadow_review"
            return "none"

    def predict_batch(self, results: list) -> list:
        """Predict risk for a batch of evaluation results."""
        predictions = []
        for r in results:
            v4 = r.get("v4", {})
            v1 = r.get("v1", {})
            disc = r.get("divergence", False)
            pred = self.predict(v4, v1, disc)
            predictions.append(pred)
        return predictions

    def print_status(self):
        """Print manifold predictor status summary."""
        if not self._loaded:
            print("  [MANIFOLD] Status: NOT LOADED (no models)")
            return

        print("  [MANIFOLD] Status: LOADED")
        print(f"  [MANIFOLD] Architecture: {self._metadata.get('architecture', 'unknown')}")
        print(f"  [MANIFOLD] Version: {self._metadata.get('version', 'unknown')}")

        trained = self._metadata.get("trained_at", "unknown")
        if isinstance(trained, str) and len(trained) > 19:
            trained = trained[:19]
        print(f"  [MANIFOLD] Trained: {trained}")

        manifolds = self._metadata.get("manifolds", {})
        print(f"\n  [MANIFOLD] Per-manifold surfaces:")
        for name, info in manifolds.items():
            head = info.get("head", {})
            n = info.get("n_samples", 0)
            wr = info.get("wrong_rate", 0)
            auc = head.get("metrics", {}).get("auc", "N/A")
            ht = head.get("type", "unknown")
            print(f"    {name:20s}: n={n:>5d}, wrong={wr:.1%}, AUC={auc}, head={ht}")

        sep = self._metadata.get("separation", {})
        cd = sep.get("centroid_distances", {})
        if cd:
            print(f"\n  [MANIFOLD] Centroid distances:")
            for n1, dists in cd.items():
                for n2, d in dists.items():
                    if n1 != n2:
                        print(f"    {n1} <-> {n2}: {d:.4f}")

        print(f"\n  [MANIFOLD] Decision thresholds:")
        print(f"    Overconfidence: always escalate (100% wrong)")
        print(f"    Contradiction:  escalate if P(wrong) >= {CONTRADICTION_THRESHOLD}")
        print(f"    Boundary:       escalate if P(wrong) >= {RISK_ESCALATE_THRESHOLD}, review >= {BOUNDARY_THRESHOLD}")


def main():
    """CLI: demo manifold prediction."""
    mp = ManifoldPredictor()
    mp.print_status()

    if not mp.is_loaded:
        return

    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)

    demos = [
        (
            "blind spot (high confidence, wrong domain)",
            {"S": 0.63, "kappa": 0.60, "delta_G": 0.50, "delta_L": 0.05},
            {"S": 0.16},
            False,
        ),
        (
            "contradiction (v4 accept, v1 reject)",
            {"S": 0.72, "kappa": 0.55, "delta_G": 0.45, "delta_L": 0.03},
            {"S": 0.28},
            True,
        ),
        (
            "boundary (moderate uncertainty, probably correct)",
            {"S": 0.42, "kappa": 0.35, "delta_G": 0.60, "delta_L": 0.04},
            {"S": 0.39},
            False,
        ),
        (
            "clear correct (high S, no disagreement)",
            {"S": 0.75, "kappa": 0.65, "delta_G": 0.30, "delta_L": 0.02},
            {"S": 0.70},
            False,
        ),
        (
            "mild disagreement (near boundary)",
            {"S": 0.40, "kappa": 0.30, "delta_G": 0.55, "delta_L": 0.05},
            {"S": 0.50},
            True,
        ),
    ]

    for label, v4, v1, disc in demos:
        r = mp.predict(v4, v1, disc)
        print(f"\n  {label}")
        print(f"    manifold: {r['manifold']} (confidence: {r['manifold_confidence']:.2f})")
        print(f"    risk_score: {r['risk_score']:.4f} -> {r['action']}")
        print(f"    per-manifold: oc={r['per_manifold_scores'].get('overconfidence', 0):.3f}, "
              f"cd={r['per_manifold_scores'].get('contradiction', 0):.3f}, "
              f"bd={r['per_manifold_scores'].get('boundary', 0):.3f}")


if __name__ == "__main__":
    main()

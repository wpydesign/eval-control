#!/usr/bin/env python3
"""
build_failure_dataset.py — Extract features + is_wrong label from live eval log [v2.2.0]

Reads shadow_eval_live.jsonl + disagreement_cases.jsonl and produces a flat
training table: logs/failure_dataset.jsonl

Each row has:
  - query_id, prompt (identity)
  - S_v4, S_v1, confidence_gap, kappa_v4, delta_G_v4, delta_L_v4 (score signals)
  - divergence (bool)
  - failure_mode (from disagreement_cases, or "none")
  - factuality_risk_flag (from disagreement_cases, default false)
  - source_class (from live log)
  - v4_decision (accept/review/reject)
  - is_wrong (target: 1 if source_class=="bad", 0 if source_class=="good", null otherwise)

Label derivation:
  - source_class == "bad"  → is_wrong = 1  (answer is wrong/unsafe)
  - source_class == "good" → is_wrong = 0  (answer is correct/safe)
  - source_class == "borderline" → is_wrong = null (excluded from training)
  - source_class == "unknown" → is_wrong = null (no ground truth)

Usage:
  python scripts/build_failure_dataset.py
"""

import json
import os
import sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_LOG_PATH = os.path.join(BASE, "logs", "shadow_eval_live.jsonl")
DISAGREEMENT_PATH = os.path.join(BASE, "logs", "disagreement_cases.jsonl")
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")


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


def load_disagreement_index(disc_cases):
    """Build query_id → disagreement metadata index."""
    idx = {}
    for c in disc_cases:
        qid = c.get("query_id")
        if qid:
            idx[qid] = c
    return idx


def extract_features(rec, disc_meta):
    """Extract feature dict from a live eval record."""
    v4 = rec.get("v4", {})
    v1 = rec.get("v1", {})
    s_v4 = v4.get("S", 0.0)
    s_v1 = v1.get("S", 0.0)

    # Derive confidence gap from v1/v4 scores (available for all records)
    confidence_gap = abs(s_v4 - s_v1)
    # Also prefer the disagreement_cases value if available
    if disc_meta and "confidence_gap" in disc_meta:
        confidence_gap = abs(disc_meta["confidence_gap"])

    features = {
        "query_id": rec.get("query_id", ""),
        "prompt": rec.get("prompt", "")[:200],  # truncated for dataset size
        # Score signals
        "S_v4": round(s_v4, 6),
        "S_v1": round(s_v1, 6),
        "confidence_gap": round(confidence_gap, 6),
        "kappa_v4": round(v4.get("kappa", 0.0), 6),
        "delta_G_v4": round(v4.get("delta_G", 0.0), 6),
        "delta_L_v4": round(v4.get("delta_L", 0.0), 6),
        # Disagreement signals
        "divergence": rec.get("divergence", False),
        "S_delta": round(rec.get("S_delta", 0.0), 6),
        # Failure mode (from disagreement_cases)
        "failure_mode": disc_meta.get("failure_mode", "none") if disc_meta else "none",
        "factuality_risk_flag": disc_meta.get("factuality_risk_flag", False) if disc_meta else False,
        "is_high_impact": disc_meta.get("is_high_impact", False) if disc_meta else False,
        # Context
        "v4_decision": v4.get("decision", ""),
        "v1_decision": v1.get("decision", ""),
        "source_class": rec.get("source_class", "unknown"),
    }

    # Derive is_wrong label
    sc = rec.get("source_class", "unknown")
    if sc == "bad":
        features["is_wrong"] = 1
    elif sc == "good":
        features["is_wrong"] = 0
    else:
        features["is_wrong"] = None  # no ground truth

    return features


def main():
    print("Loading live eval log...")
    live_records = load_jsonl(LIVE_LOG_PATH)
    print(f"  {len(live_records)} records")

    print("Loading disagreement cases...")
    disc_cases = load_jsonl(DISAGREEMENT_PATH)
    disc_index = load_disagreement_index(disc_cases)
    print(f"  {len(disc_cases)} cases indexed by query_id")

    # Build dataset
    dataset = []
    stats = {"total": 0, "labeled": 0, "wrong": 0, "correct": 0, "unlabeled": 0, "with_disc": 0}

    for rec in live_records:
        qid = rec.get("query_id", "")
        disc_meta = disc_index.get(qid)
        if disc_meta:
            stats["with_disc"] += 1

        features = extract_features(rec, disc_meta)
        dataset.append(features)
        stats["total"] += 1

        if features["is_wrong"] is not None:
            stats["labeled"] += 1
            if features["is_wrong"] == 1:
                stats["wrong"] += 1
            else:
                stats["correct"] += 1
        else:
            stats["unlabeled"] += 1

    # Write dataset
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDataset written to {DATASET_PATH}")
    print(f"  Total rows:     {stats['total']}")
    print(f"  Labeled:        {stats['labeled']} (is_wrong is known)")
    print(f"    Wrong (1):    {stats['wrong']}")
    print(f"    Correct (0):  {stats['correct']}")
    print(f"  Unlabeled:      {stats['unlabeled']} (no ground truth)")
    print(f"  With disc meta: {stats['with_disc']}")

    if stats["labeled"] > 0:
        print(f"\n  Base error rate: {stats['wrong']}/{stats['labeled']} = {stats['wrong']/stats['labeled']*100:.1f}%")
        print(f"  Class balance:   {stats['correct']}/{stats['wrong']} = {stats['correct']/max(1,stats['wrong']):.1f}:1 (correct:wrong)")

    # Feature summary for labeled data
    if stats["labeled"] >= 5:
        print(f"\n  Feature means (labeled data):")
        labeled = [r for r in dataset if r["is_wrong"] is not None]
        wrong = [r for r in labeled if r["is_wrong"] == 1]
        correct = [r for r in labeled if r["is_wrong"] == 0]
        for feat in ["S_v4", "S_v1", "confidence_gap", "kappa_v4", "delta_G_v4"]:
            w_mean = sum(r[feat] for r in wrong) / len(wrong) if wrong else 0
            c_mean = sum(r[feat] for r in correct) / len(correct) if correct else 0
            print(f"    {feat:20s}  wrong={w_mean:.4f}  correct={c_mean:.4f}  gap={w_mean-c_mean:+.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
acquisition_policy.py — Adaptive data acquisition controller [v2.3.0]

This is the control system that decides WHAT REALITY TO ASK FOR NEXT.

It is NOT a data merger. It is an epistemic budget optimizer that allocates
finite labeling resources across three ORTHOGONAL failure manifolds:

  1. UNCERTAINTY POLICY (boundary exploration)
     - Selects: high entropy / boundary samples
     - Fixes: decision boundary fuzziness
     - Signal: 1 - uncertainty_score (rank-normalized)
     - Proven: ~3-10x label efficiency vs random

  2. BLIND-SPOT POLICY (overconfidence exposure)
     - Selects: high kappa x gap_norm (structurally unstable confidence)
     - Fixes: calibration tail failures, confident false negatives
     - Signal: kappa_v4 x |S_v4 - S_v1| / max(S_v4, eps)
     - Proven: orthogonal to risk_score (rho=+0.23), AUC=0.57

  3. COST POLICY (impact prioritization)
     - Selects: high-impact / high-consequence domains
     - Fixes: real-world relevance weighting
     - Signal: failure_mode severity x is_high_impact

Architecture (v2.3.0):
  - Channel-forced allocation: each policy gets guaranteed slots
  - Adaptive weighting: alpha/beta/gamma evolve based on realized outcomes
  - Closed loop: label -> retrain -> measure per-channel efficiency -> rebalance

Combined acquisition function:
    score = alpha * uncertainty_norm + beta * blind_spot_norm + gamma * cost_norm

Adaptive weight update:
    After each retrain cycle, measure per-channel label efficiency:
        efficiency(channel) = (fraction of labels that were is_wrong=1) / (budget allocated)
    Then: weight_new(channel) = weight_old * (1 + efficiency_gain)
    Normalized to sum to 1.0.

This ensures:
    - Channels that find more real failures get more budget next cycle
    - No channel can be reduced below MIN_WEIGHT (exploration floor)
    - System self-corrects if one channel becomes saturated

Outputs:
    logs/acquisition_queue.jsonl       - definitive "label this next" list
    logs/acquisition_budget.json       - budget allocation + adaptive weights
    logs/channel_performance.jsonl     - per-channel outcome tracking

Usage:
    python scripts/acquisition_policy.py                # compute full queue
    python scripts/acquisition_policy.py --budget 50     # allocate 50 labels
    python scripts/acquisition_policy.py --show 20       # show top 20
    python scripts/acquisition_policy.py --update-weights # adaptive reweight
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
ACTIVE_LEARNING_PATH = os.path.join(BASE, "logs", "active_learning_queue.jsonl")
MINING_QUEUE_PATH = os.path.join(BASE, "logs", "failure_mining_queue.jsonl")
ACQUISITION_QUEUE_PATH = os.path.join(BASE, "logs", "acquisition_queue.jsonl")
ACQUISITION_BUDGET_PATH = os.path.join(BASE, "logs", "acquisition_budget.json")
CHANNEL_PERF_PATH = os.path.join(BASE, "logs", "channel_performance.jsonl")
REWARD_BUFFER_PATH = os.path.join(BASE, "logs", "reward_buffer.jsonl")

# Lag compensation: weight updates use performance from t-1, not t.
# This prevents oscillation when the retrain loop runs at high throughput.
# On first cycle (no t-1 data), weights stay at defaults.
ATTRIBUTION_LAG = 1

# Default policy weights
DEFAULT_WEIGHTS = {
    "uncertainty": 0.50,
    "blind_spot": 0.35,
    "cost": 0.15,
}

# Adaptive constraints
MIN_WEIGHT = 0.10       # no channel below 10% (exploration floor)
MAX_WEIGHT = 0.60       # no channel above 60% (diversity cap)
LEARNING_RATE = 0.3     # how fast weights adapt (0=static, 1=fully reactive)
SMOOTHING = 0.7         # exponential smoothing for weight updates

# Cost severity mapping
COST_SEVERITY = {
    "hallucination": 5.0,
    "reasoning_error": 4.0,
    "safety_violation": 5.0,
    "factual_error": 3.0,
    "omission": 2.0,
    "bias": 4.0,
    "format_error": 1.0,
    "none": 1.0,
}


def load_queue(path):
    """Load a JSONL queue file."""
    if not os.path.exists(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_channel_performance():
    """Load channel performance history."""
    if not os.path.exists(CHANNEL_PERF_PATH):
        return []
    entries = []
    with open(CHANNEL_PERF_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def rank_normalize(values):
    """Normalize by rank (percentile). Robust to outliers."""
    from scipy.stats import rankdata
    ranks = rankdata(values)
    return ranks / len(ranks)


def compute_cost_score(sample):
    """Compute cost priority based on impact and failure mode severity."""
    score = 1.0
    if sample.get("is_high_impact"):
        score *= 2.0
    fm = sample.get("failure_mode", "none")
    score *= COST_SEVERITY.get(fm, 1.0)
    return score


def compute_acquisition_scores(uncertainty_queue, mining_queue, acquired_ids):
    """Compute per-channel scores for all candidate samples.

    Each channel gets its own independent score (not combined yet).
    The combination happens during allocation.

    Returns:
        dict mapping query_id -> {all component scores + metadata}
    """
    mining_lookup = {}
    for item in mining_queue:
        mining_lookup[item["query_id"]] = item

    unc_lookup = {}
    for item in uncertainty_queue:
        unc_lookup[item["query_id"]] = item

    # Union of all query IDs not yet acquired
    all_qids = set()
    for item in uncertainty_queue + mining_queue:
        if item["query_id"] not in acquired_ids:
            all_qids.add(item["query_id"])

    if not all_qids:
        return {}

    # Extract raw scores per channel
    unc_raw = []
    bs_raw = []
    cost_raw = []
    metadata = {}

    for qid in all_qids:
        unc_item = unc_lookup.get(qid, {})
        min_item = mining_lookup.get(qid, {})
        item_data = unc_item or min_item

        unc_raw.append(unc_item.get("uncertainty_score", 1.0))
        bs_raw.append(min_item.get("proxy_score", 0.0))
        cost_raw.append(compute_cost_score(item_data))

        metadata[qid] = {
            "unc_item": unc_item,
            "min_item": min_item,
            "item_data": item_data,
            "risk_score": min_item.get("risk_score",
                                       unc_item.get("risk_score", 0)),
            "source_class": item_data.get("source_class", "unknown"),
            "failure_mode": item_data.get("failure_mode", "none"),
            "is_high_impact": item_data.get("is_high_impact", False),
        }

    # Rank-normalize each channel independently
    # Uncertainty: invert (low uncertainty_score = high priority)
    unc_norm = rank_normalize([-u for u in unc_raw])
    bs_norm = rank_normalize(bs_raw)
    cost_norm = rank_normalize(cost_raw)

    qid_list = list(all_qids)
    results = {}
    for i, qid in enumerate(qid_list):
        meta = metadata[qid]
        results[qid] = {
            "query_id": qid,
            "prompt": meta["item_data"].get("prompt", ""),
            "uncertainty_raw": round(float(unc_raw[i]), 4),
            "uncertainty_norm": round(float(unc_norm[i]), 4),
            "blind_spot_raw": round(float(bs_raw[i]), 4),
            "blind_spot_norm": round(float(bs_norm[i]), 4),
            "cost_raw": round(float(cost_raw[i]), 4),
            "cost_norm": round(float(cost_norm[i]), 4),
            "risk_score": meta["risk_score"],
            "source_class": meta["source_class"],
            "failure_mode": meta["failure_mode"],
            "is_high_impact": meta["is_high_impact"],
        }

    return results


def allocate_forced_channels(score_map, budget, weights):
    """Channel-forced allocation: each policy gets guaranteed slots.

    This is the critical design choice:
    - Each channel independently selects its top-N samples
    - Budget is split: n_ch = int(budget * weight_ch)
    - De-duplicate: if a sample appears in multiple channels, keep it in the
      channel where it ranks highest (most unique value)
    - Remaining budget goes to unified-score ranking

    This guarantees no single channel can dominate the acquisition queue.
    """
    # Sort each channel independently
    by_uncertainty = sorted(
        score_map.values(), key=lambda x: x["uncertainty_norm"], reverse=True
    )
    by_blind_spot = sorted(
        score_map.values(), key=lambda x: x["blind_spot_norm"], reverse=True
    )
    by_cost = sorted(
        score_map.values(), key=lambda x: x["cost_norm"], reverse=True
    )

    # Calculate slot allocations
    n_unc = max(1, int(budget * weights["uncertainty"]))
    n_bs = max(1, int(budget * weights["blind_spot"]))
    n_cost = max(1, int(budget * weights["cost"]))

    # Phase 1: fill each channel's quota from its own ranking
    # Track which channel "owns" each sample
    channel_slots = {"uncertainty": {}, "blind_spot": {}, "cost": {}}

    # Uncertainty channel
    assigned = set()
    for s in by_uncertainty:
        if len(channel_slots["uncertainty"]) >= n_unc:
            break
        qid = s["query_id"]
        if qid not in assigned:
            channel_slots["uncertainty"][qid] = s
            assigned.add(qid)

    # Blind-spot channel
    for s in by_blind_spot:
        if len(channel_slots["blind_spot"]) >= n_bs:
            break
        qid = s["query_id"]
        if qid not in assigned:
            channel_slots["blind_spot"][qid] = s
            assigned.add(qid)

    # Cost channel
    for s in by_cost:
        if len(channel_slots["cost"]) >= n_cost:
            break
        qid = s["query_id"]
        if qid not in assigned:
            channel_slots["cost"][qid] = s
            assigned.add(qid)

    # Phase 2: compute unified score for ALL remaining samples
    remaining = []
    for qid, s in score_map.items():
        if qid not in assigned:
            unified = (weights["uncertainty"] * s["uncertainty_norm"]
                       + weights["blind_spot"] * s["blind_spot_norm"]
                       + weights["cost"] * s["cost_norm"])
            s["acquisition_score"] = round(unified, 6)
            # Dominant policy for remaining
            contributions = {
                "uncertainty": weights["uncertainty"] * s["uncertainty_norm"],
                "blind_spot": weights["blind_spot"] * s["blind_spot_norm"],
                "cost": weights["cost"] * s["cost_norm"],
            }
            s["dominant_policy"] = max(contributions, key=contributions.get)
            remaining.append(s)

    remaining.sort(key=lambda x: x["acquisition_score"], reverse=True)

    # Fill remaining budget from unified score
    leftover = budget - len(assigned)
    for s in remaining[:max(0, leftover)]:
        assigned.add(s["query_id"])

    # Build final allocation: channel slots + remaining
    final = []
    for channel_name, slots in channel_slots.items():
        for qid, s in slots.items():
            s_copy = dict(s)
            s_copy["assigned_channel"] = channel_name
            s_copy["dominant_policy"] = channel_name
            # Unified score for ranking within final list
            unified = (weights["uncertainty"] * s["uncertainty_norm"]
                       + weights["blind_spot"] * s["blind_spot_norm"]
                       + weights["cost"] * s["cost_norm"])
            s_copy["acquisition_score"] = round(unified, 6)
            final.append(s_copy)

    # Add remaining (unified-score) samples
    for s in remaining[:max(0, leftover)]:
        s_copy = dict(s)
        s_copy["assigned_channel"] = "unified_fallback"
        final.append(s_copy)

    # Sort by acquisition_score for final presentation
    final.sort(key=lambda x: x["acquisition_score"], reverse=True)

    # Compute stats
    channel_counts = {"uncertainty": 0, "blind_spot": 0, "cost": 0,
                      "unified_fallback": 0}
    for s in final:
        ch = s["assigned_channel"]
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

    return {
        "total_budget": budget,
        "allocation": {
            "uncertainty": n_unc,
            "blind_spot": n_bs,
            "cost": n_cost,
        },
        "channel_counts": channel_counts,
        "samples": final,
    }


def adapt_weights(weights, channel_performance, lag=ATTRIBUTION_LAG):
    """Adapt policy weights using LAG-COMPENSATED reward attribution.

    Weight updates use performance from cycle t-lag, not cycle t.
    This prevents feedback oscillation when the loop runs at high throughput.

    For each channel, compute:
        efficiency = (n_wrong_found / n_labeled_from_channel) * weight

    Channels that found more actual failures (per label) get more budget.
    Uses exponential smoothing to avoid wild swings.

    Constraints:
        MIN_WEIGHT <= weight <= MAX_WEIGHT
        weights sum to 1.0
    """
    if not channel_performance:
        return weights

    # Lag compensation: use performance from t-lag, exclude the last `lag` entries
    # which are from the current cycle (not yet reflected in AUC)
    if len(channel_performance) <= lag:
        # Not enough history for lagged attribution — keep current weights
        return weights
    recent = channel_performance[-(5 + lag):-lag] if len(channel_performance) > 5 + lag else channel_performance[:-lag]

    # Compute per-channel efficiency: fraction of labels that were wrong
    efficiencies = {}
    for channel in ["uncertainty", "blind_spot", "cost"]:
        total_labeled = 0
        total_wrong = 0
        for entry in recent:
            ch_data = entry.get("channels", {}).get(channel, {})
            total_labeled += ch_data.get("n_labeled", 0)
            total_wrong += ch_data.get("n_wrong", 0)
        if total_labeled > 0:
            efficiencies[channel] = total_wrong / total_labeled
        else:
            efficiencies[channel] = 0.0

    # Compute raw weight updates
    raw_weights = {}
    for channel in weights:
        eff = efficiencies.get(channel, 0.0)
        # Multiplicative update: weight * (1 + lr * efficiency)
        raw_weights[channel] = weights[channel] * (1 + LEARNING_RATE * eff)

    # Clamp to bounds
    for channel in raw_weights:
        raw_weights[channel] = max(MIN_WEIGHT, min(MAX_WEIGHT, raw_weights[channel]))

    # Normalize to sum to 1.0
    total = sum(raw_weights.values())
    if total > 0:
        new_weights = {ch: w / total for ch, w in raw_weights.items()}
    else:
        new_weights = dict(DEFAULT_WEIGHTS)

    # Exponential smoothing: blend old and new
    smoothed = {}
    for channel in weights:
        smoothed[channel] = (SMOOTHING * weights[channel]
                             + (1 - SMOOTHING) * new_weights[channel])

    # Final clamp and normalize
    for channel in smoothed:
        smoothed[channel] = max(MIN_WEIGHT, min(MAX_WEIGHT, smoothed[channel]))
    total = sum(smoothed.values())
    smoothed = {ch: w / total for ch, w in smoothed.items()}

    return smoothed


def record_channel_outcomes(acquisition_queue, failure_dataset_path):
    """Record which channel's selections turned out to be right/wrong.

    Cross-references the acquisition queue against the failure dataset to
    measure per-channel label efficiency.

    Returns:
        dict with per-channel {n_labeled, n_wrong, n_correct, efficiency}
    """
    # Load acquisition queue -> channel mapping
    channel_map = {}
    if os.path.exists(ACQUISITION_QUEUE_PATH):
        with open(ACQUISITION_QUEUE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    qid = entry["query_id"]
                    channel = entry.get("assigned_channel", "unknown")
                    channel_map[qid] = channel

    # Load failure dataset for outcomes
    outcomes = {}
    if os.path.exists(failure_dataset_path):
        with open(failure_dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    qid = entry["query_id"]
                    if entry.get("is_wrong") is not None and qid in channel_map:
                        outcomes[qid] = {
                            "is_wrong": entry["is_wrong"],
                            "channel": channel_map[qid],
                        }

    if not outcomes:
        return None

    # Aggregate per channel
    channels = {}
    for qid, o in outcomes.items():
        ch = o["channel"]
        if ch not in channels:
            channels[ch] = {"n_labeled": 0, "n_wrong": 0, "n_correct": 0}
        channels[ch]["n_labeled"] += 1
        if o["is_wrong"] == 1:
            channels[ch]["n_wrong"] += 1
        else:
            channels[ch]["n_correct"] += 1

    # Compute efficiency
    for ch in channels:
        n = channels[ch]["n_labeled"]
        if n > 0:
            channels[ch]["efficiency"] = round(channels[ch]["n_wrong"] / n, 4)
        else:
            channels[ch]["efficiency"] = 0.0

    return channels


def update_weights_cli():
    """CLI entry point for --update-weights.

    Implements lag-compensated reward attribution:
      1. Record current cycle's outcomes (time t) to reward buffer
      2. Load performance history
      3. Adapt weights using outcomes from time t-1 (not t)
      4. This one-cycle delay prevents oscillation at high throughput
    """
    perf = load_channel_performance()
    pre_update_len = len(perf)

    # Load current weights
    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(ACQUISITION_BUDGET_PATH):
        try:
            with open(ACQUISITION_BUDGET_PATH) as f:
                saved = json.load(f)
            if "weights" in saved:
                weights = saved["weights"]
        except (json.JSONDecodeError, OSError):
            pass

    print(f"  Current weights: unc={weights['uncertainty']:.3f}, "
          f"bs={weights['blind_spot']:.3f}, cost={weights['cost']:.3f}")

    # Step 1: Record current outcomes to reward buffer (time t)
    outcomes = record_channel_outcomes(
        ACQUISITION_QUEUE_PATH, DATASET_PATH
    )
    buffered = False
    if outcomes:
        print(f"\n  [t={pre_update_len}] Channel outcomes (stored for next cycle):")
        for ch, stats in outcomes.items():
            print(f"    {ch:>20s}: {stats['n_labeled']} labeled, "
                  f"{stats['n_wrong']} wrong ({stats['efficiency']:.1%} efficiency)")

        # Append to performance log (reward buffer)
        perf_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": pre_update_len,
            "channels": outcomes,
            "prev_weights": weights,
        }
        os.makedirs(os.path.dirname(CHANNEL_PERF_PATH), exist_ok=True)
        with open(CHANNEL_PERF_PATH, "a") as f:
            f.write(json.dumps(perf_entry, ensure_ascii=False) + "\n")
        perf.append(perf_entry)

        # Also write to explicit reward buffer for traceability
        os.makedirs(os.path.dirname(REWARD_BUFFER_PATH), exist_ok=True)
        with open(REWARD_BUFFER_PATH, "a") as f:
            f.write(json.dumps(perf_entry, ensure_ascii=False) + "\n")
        buffered = True
    else:
        print("  No new labeled outcomes found since last update.")

    # Step 2: Adapt weights using PREVIOUS cycle's outcomes (time t-1)
    # This is the lag compensation: we stored t, now we use t-1 for the update.
    new_weights = adapt_weights(weights, perf, lag=ATTRIBUTION_LAG)

    # Check if lag prevented the update
    if len(perf) <= ATTRIBUTION_LAG:
        print(f"\n  [LAG] Insufficient history ({len(perf)} cycles, need >{ATTRIBUTION_LAG}) "
              f"— weights unchanged (first-cycle guard)")
        return weights

    # Show what was used for attribution
    used_entry = perf[-(ATTRIBUTION_LAG + 1)]
    used_cycle = used_entry.get("cycle", "?")
    print(f"\n  [LAG] Weight update based on cycle t-{ATTRIBUTION_LAG} (cycle {used_cycle}), "
          f"not current cycle (t={pre_update_len})")

    print(f"\n  Adapted weights:  unc={new_weights['uncertainty']:.3f}, "
          f"bs={new_weights['blind_spot']:.3f}, cost={new_weights['cost']:.3f}")

    delta = {ch: new_weights[ch] - weights[ch] for ch in weights}
    for ch, d in delta.items():
        direction = "up" if d > 0.001 else ("down" if d < -0.001 else "stable")
        print(f"    {ch:>15s}: {d:+.4f} ({direction})")

    # Save
    budget_state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": new_weights,
        "prev_weights": weights,
        "weight_delta": delta,
        "adaptation_mode": "lag_compensated_efficiency",
        "attribution_lag": ATTRIBUTION_LAG,
        "attribution_cycle": used_cycle,
        "current_cycle": pre_update_len,
        "n_performance_cycles": len(perf),
    }
    with open(ACQUISITION_BUDGET_PATH, "w") as f:
        json.dump(budget_state, f, indent=2, ensure_ascii=False)

    print(f"\n  Weights saved to {ACQUISITION_BUDGET_PATH}")
    return new_weights


def main():
    budget = 50
    if "--budget" in sys.argv:
        idx = sys.argv.index("--budget")
        if idx + 1 < len(sys.argv):
            budget = int(sys.argv[idx + 1])

    show_n = 20
    if "--show" in sys.argv:
        idx = sys.argv.index("--show")
        if idx + 1 < len(sys.argv):
            show_n = int(sys.argv[idx + 1])

    do_update = "--update-weights" in sys.argv

    # Step 0: adaptive weight update
    if do_update:
        print("=" * 60)
        print("  ADAPTIVE WEIGHT UPDATE")
        print("=" * 60)
        weights = update_weights_cli()
        print()

    # Step 1: load queues
    print("Loading queues...")
    uncertainty_queue = load_queue(ACTIVE_LEARNING_PATH)
    mining_queue = load_queue(MINING_QUEUE_PATH)

    # Load acquired IDs (exclude already-labeled samples)
    acquired_ids = set()
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("is_wrong") is not None:
                        acquired_ids.add(entry["query_id"])

    print(f"  Uncertainty queue:  {len(uncertainty_queue)} samples")
    print(f"  Blind-spot queue:   {len(mining_queue)} samples")
    print(f"  Already labeled:    {len(acquired_ids)} samples")

    if not uncertainty_queue and not mining_queue:
        print("\nERROR: No queues found. Run active_learning.py and failure_mining.py first.")
        sys.exit(1)

    # Load weights
    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(ACQUISITION_BUDGET_PATH):
        try:
            with open(ACQUISITION_BUDGET_PATH) as f:
                saved = json.load(f)
            if "weights" in saved:
                weights = saved["weights"]
                print(f"  Weights (adaptive):  unc={weights['uncertainty']:.3f}, "
                      f"bs={weights['blind_spot']:.3f}, cost={weights['cost']:.3f}")
        except (json.JSONDecodeError, OSError):
            pass

    # Step 2: compute per-channel scores
    print("Computing per-channel scores...")
    score_map = compute_acquisition_scores(
        uncertainty_queue, mining_queue, acquired_ids
    )
    print(f"  Candidate samples:  {len(score_map)}")

    if not score_map:
        print("  No candidates remaining (all labeled or no queues).")
        return

    # Step 3: channel-forced allocation
    print(f"\nAllocating epistemic budget ({budget} labels)...")
    alloc = allocate_forced_channels(score_map, budget, weights)
    samples = alloc["samples"]

    # Report
    cc = alloc["channel_counts"]
    print(f"\n{'='*65}")
    print(f"  ADAPTIVE DATA ACQUISITION POLICY")
    print(f"{'='*65}")
    print(f"  Candidate pool:       {len(score_map)} unlabeled samples")
    print(f"  Epistemic budget:     {alloc['total_budget']} labels")
    print(f"  Policy weights:       unc={weights['uncertainty']:.0%}  "
          f"bs={weights['blind_spot']:.0%}  cost={weights['cost']:.0%}")
    print(f"  Forced allocation:    unc={alloc['allocation']['uncertainty']}  "
          f"bs={alloc['allocation']['blind_spot']}  "
          f"cost={alloc['allocation']['cost']}")
    print(f"  Channel fill:         {cc}")

    # Show top-N with channel tags
    top_n = min(show_n, len(samples))
    print(f"\n  TOP {top_n} SAMPLES TO LABEL NEXT:")
    print(f"  {'#':>3s} {'score':>7s} {'channel':>10s} {'risk':>6s} "
          f"{'unc':>5s} {'bs':>5s} {'cost':>5s} {'prompt':>35s}")
    print(f"  {'-'*85}")
    for i, s in enumerate(samples[:top_n]):
        prompt_preview = s["prompt"][:35].replace("\n", " ")
        ch = s["assigned_channel"][:8]
        print(f"  {i+1:>3d} {s['acquisition_score']:>7.4f} {ch:>10s} "
              f"{s['risk_score']:>6.3f} {s['uncertainty_norm']:>5.2f} "
              f"{s['blind_spot_norm']:>5.2f} {s['cost_norm']:>5.2f} "
              f"{prompt_preview:>35s}")

    # Write acquisition queue
    os.makedirs(os.path.dirname(ACQUISITION_QUEUE_PATH), exist_ok=True)
    with open(ACQUISITION_QUEUE_PATH, "w") as f:
        for i, s in enumerate(samples):
            entry = {
                "query_id": s["query_id"],
                "prompt": s["prompt"],
                "priority": i + 1,
                "acquisition_score": s["acquisition_score"],
                "assigned_channel": s["assigned_channel"],
                "uncertainty_norm": s["uncertainty_norm"],
                "blind_spot_norm": s["blind_spot_norm"],
                "cost_norm": s["cost_norm"],
                "risk_score": s["risk_score"],
                "source_class": s["source_class"],
                "failure_mode": s["failure_mode"],
                "is_high_impact": s["is_high_impact"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save budget state
    budget_state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "budget": budget,
        "allocation": alloc["allocation"],
        "channel_counts": cc,
        "n_candidates": len(score_map),
        "n_labeled": len(acquired_ids),
    }
    with open(ACQUISITION_BUDGET_PATH, "w") as f:
        json.dump(budget_state, f, indent=2, ensure_ascii=False)

    print(f"\n  Queue ({len(samples)} samples) -> {ACQUISITION_QUEUE_PATH}")
    print(f"  Budget state -> {ACQUISITION_BUDGET_PATH}")
    print(f"\n  Loop: label these -> add to dataset -> retrain -> --update-weights")


if __name__ == "__main__":
    main()

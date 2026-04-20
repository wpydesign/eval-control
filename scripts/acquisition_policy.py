#!/usr/bin/env python3
"""
acquisition_policy.py — Manifold-aware data acquisition controller [v2.5.1]

v2.5.1: REWRITTEN from channel-based to manifold-based allocation.

The old system allocated by channel (uncertainty/blind_spot/cost).
That is now outdated. The new system allocates by FAILURE MANIFOLD:

  1. CONTRADICTION MANIFOLD (primary target — the gold vein)
     - Budget: 60-70% of all labels
     - Why: AUC=0.88, 76.5% wrong rate — only true predictive surface
     - Goal: compress this manifold by filling in the learned surface
     - Source: disagreement cases, v4-v1 divergent samples

  2. BLIND-SPOT MANIFOLD (monitoring, not learning)
     - Budget: 20-30% of all labels
     - Why: 100% wrong rate — detection problem, already locked down
     - Goal: discover new blind spots, NOT retrain on known ones
     - Source: high kappa x gap_norm proxy, overconfidence signals

  3. BOUNDARY MANIFOLD (minimal — do NOT waste signal here)
     - Budget: 10-20% of all labels (HARD CAP)
     - Why: 21.1% wrong rate — mostly correct, irreducible uncertainty
     - Goal: stability check only, NOT optimization
     - Source: high-entropy samples in uncertainty zone

Design principle:
  - Shift from uncertainty sampling -> failure-manifold targeting
  - Contradiction gets the most resources because that is where gains exist
  - Boundary is explicitly capped to prevent wasting labels on noise
  - Blind-spot is discovery-oriented (monitor only, no learning)

Adaptive weight update (v2.5.1):
  - Measures per-MANIFOLD label efficiency (not per-channel)
  - Contradiction recall is the PRIMARY KPI for weight adaptation
  - Overconfidence capture rate is a STABILITY CHECK (should be ~100%)
  - Boundary accept rate is IGNORED for weight purposes

Outputs:
    logs/acquisition_queue.jsonl       - manifold-tagged "label this next" list
    logs/acquisition_budget.json       - manifold allocation + adaptive weights
    logs/channel_performance.jsonl     - per-manifold outcome tracking

Usage:
    python scripts/acquisition_policy.py                # compute full queue
    python scripts/acquisition_policy.py --budget 50     # allocate 50 labels
    python scripts/acquisition_policy.py --show 20       # show top 20
    python scripts/acquisition_policy.py --update-weights # manifold-aware reweight
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

# Manifold-based allocation (v2.5.1)
# Replaces channel-based weights with manifold-targeted allocation
DEFAULT_MANIFOLD_WEIGHTS = {
    "contradiction": 0.65,   # primary target — the gold vein
    "blind_spot": 0.25,       # discovery (monitor only, not learning)
    "boundary": 0.10,         # minimal — capped, do not optimize
}

# Legacy channel weights (kept for backward compatibility)
DEFAULT_WEIGHTS = {
    "uncertainty": 0.50,
    "blind_spot": 0.35,
    "cost": 0.15,
}

# Manifold-specific allocation constraints
MIN_MANIFOLD_WEIGHT = {
    "contradiction": 0.50,   # never below 50%
    "blind_spot": 0.15,       # keep discovery alive
    "boundary": 0.05,         # floor, but effectively capped
}
MAX_MANIFOLD_WEIGHT = {
    "contradiction": 0.75,   # can go up to 75%
    "blind_spot": 0.35,       # discovery cap
    "boundary": 0.20,         # HARD CAP — never more than 20%
}

# Adaptive constraints
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.60
LEARNING_RATE = 0.3
SMOOTHING = 0.7

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


def assign_manifold_estimate(score_entry):
    """Estimate which manifold a candidate belongs to based on available signals.

    Uses observable features at acquisition time:
    - disagreement/divergence signal → contradiction
    - high blind_spot proxy + moderate confidence → blind_spot
    - everything else → boundary (default)
    """
    bs_raw = score_entry.get("blind_spot_raw", 0.0)
    unc_raw = score_entry.get("uncertainty_raw", 0.0)
    cost_raw = score_entry.get("cost_raw", 0.0)
    risk = score_entry.get("risk_score", 0.0)

    # Proxy for disagreement: high uncertainty + moderate risk + high blind_spot
    if bs_raw > 0.3 and risk > 0.1:
        return "blind_spot"
    if unc_raw > 0.5 and risk < 0.3:
        return "boundary"
    if unc_raw > 0.3 and risk > 0.15:
        return "contradiction"
    # Default: boundary (most samples live here)
    return "boundary"


def allocate_manifold_targets(score_map, budget, manifold_weights):
    """Manifold-targeted allocation (v2.5.1).

    Replaces channel-based allocation with manifold-based targeting:
    - contradiction gets 60-70% of budget (primary target)
    - blind_spot gets 20-30% (discovery only)
    - boundary gets 10-20% HARD CAP (minimal)

    Each manifold independently selects its top-N candidates based on
    manifold-relevant features, then de-duplicates.
    """
    # Assign manifold estimates to all candidates
    for qid, s in score_map.items():
        s["estimated_manifold"] = assign_manifold_estimate(s)

    # Split candidates by estimated manifold
    by_manifold = {"contradiction": [], "blind_spot": [], "boundary": []}
    for qid, s in score_map.items():
        m = s["estimated_manifold"]
        by_manifold[m].append(s)

    # Calculate slot allocations per manifold
    n_cd = max(1, int(budget * manifold_weights["contradiction"]))
    n_bs = max(1, int(budget * manifold_weights["blind_spot"]))
    n_bd = max(1, int(budget * manifold_weights["boundary"]))

    # HARD CAP: boundary never exceeds 20% of budget
    max_bd = max(1, int(budget * MAX_MANIFOLD_WEIGHT["boundary"]))
    n_bd = min(n_bd, max_bd)

    # Phase 1: fill each manifold's quota from its own ranking
    # Sort by manifold-relevant signals
    # Contradiction: prioritize high-risk + high-uncertainty (disagreement signal)
    by_manifold["contradiction"].sort(
        key=lambda x: x.get("risk_score", 0) * 0.5 + x.get("uncertainty_norm", 0) * 0.5,
        reverse=True
    )
    # Blind spot: prioritize high proxy score (kappa x gap_norm)
    by_manifold["blind_spot"].sort(
        key=lambda x: x.get("blind_spot_norm", 0), reverse=True
    )
    # Boundary: sort by uncertainty (highest first for stability check)
    by_manifold["boundary"].sort(
        key=lambda x: x.get("uncertainty_norm", 0), reverse=True
    )

    assigned = set()
    manifold_slots = {"contradiction": {}, "blind_spot": {}, "boundary": {}}

    for manifold_name, quota in [("contradiction", n_cd), ("blind_spot", n_bs), ("boundary", n_bd)]:
        for s in by_manifold[manifold_name]:
            if len(manifold_slots[manifold_name]) >= quota:
                break
            qid = s["query_id"]
            if qid not in assigned:
                manifold_slots[manifold_name][qid] = s
                assigned.add(qid)

    # Phase 2: fill remaining from unified ranking (manifold-weighted score)
    remaining = []
    for qid, s in score_map.items():
        if qid not in assigned:
            m = s["estimated_manifold"]
            w = manifold_weights.get(m, 0.1)
            s["acquisition_score"] = round(w, 6)
            remaining.append(s)

    remaining.sort(key=lambda x: manifold_weights.get(x["estimated_manifold"], 0.1), reverse=True)
    leftover = budget - len(assigned)
    for s in remaining[:max(0, leftover)]:
        assigned.add(s["query_id"])

    # Build final allocation
    final = []
    for manifold_name, slots in manifold_slots.items():
        for qid, s in slots.items():
            s_copy = dict(s)
            s_copy["assigned_manifold"] = manifold_name
            s_copy["assigned_channel"] = f"manifold_{manifold_name}"
            s_copy["dominant_policy"] = manifold_name
            s_copy["acquisition_score"] = round(manifold_weights.get(manifold_name, 0.1), 6)
            final.append(s_copy)

    for s in remaining[:max(0, leftover)]:
        s_copy = dict(s)
        s_copy["assigned_manifold"] = s_copy.get("estimated_manifold", "boundary")
        s_copy["assigned_channel"] = f"manifold_{s_copy.get('estimated_manifold', 'boundary')}"
        final.append(s_copy)

    final.sort(key=lambda x: x["acquisition_score"], reverse=True)

    manifold_counts = {"contradiction": 0, "blind_spot": 0, "boundary": 0}
    for s in final:
        m = s.get("assigned_manifold", "boundary")
        manifold_counts[m] = manifold_counts.get(m, 0) + 1

    return {
        "total_budget": budget,
        "allocation": {
            "contradiction": n_cd,
            "blind_spot": n_bs,
            "boundary": n_bd,
        },
        "manifold_counts": manifold_counts,
        "channel_counts": manifold_counts,  # backward compat
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

    # Load manifold weights (v2.5.1: primary) with legacy fallback
    manifold_weights = dict(DEFAULT_MANIFOLD_WEIGHTS)
    weights = dict(DEFAULT_WEIGHTS)
    if os.path.exists(ACQUISITION_BUDGET_PATH):
        try:
            with open(ACQUISITION_BUDGET_PATH) as f:
                saved = json.load(f)
            # Prefer manifold weights if available
            if "manifold_weights" in saved:
                manifold_weights = saved["manifold_weights"]
            if "weights" in saved:
                weights = saved["weights"]
            print(f"  Manifold weights:  cd={manifold_weights['contradiction']:.0%}, "
                  f"bs={manifold_weights['blind_spot']:.0%}, bd={manifold_weights['boundary']:.0%}")
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

    # Step 3: manifold-targeted allocation (v2.5.1)
    print(f"\nAllocating epistemic budget ({budget} labels)...")
    alloc = allocate_manifold_targets(score_map, budget, manifold_weights)
    samples = alloc["samples"]

    # Report
    mc = alloc["manifold_counts"]
    print(f"\n{'='*65}")
    print(f"  MANIFOLD-AWARE DATA ACQUISITION [v2.5.1]")
    print(f"{'='*65}")
    print(f"  Candidate pool:       {len(score_map)} unlabeled samples")
    print(f"  Epistemic budget:     {alloc['total_budget']} labels")
    print(f"  Manifold weights:     cd={manifold_weights['contradiction']:.0%}  "
          f"bs={manifold_weights['blind_spot']:.0%}  bd={manifold_weights['boundary']:.0%}")
    print(f"  Manifold allocation:  cd={alloc['allocation']['contradiction']}  "
          f"bs={alloc['allocation']['blind_spot']}  bd={alloc['allocation']['boundary']}")
    print(f"  Manifold fill:        {mc}")
    print(f"  Boundary cap:        {MAX_MANIFOLD_WEIGHT['boundary']:.0%} (hard)")

    # Show top-N with manifold tags
    top_n = min(show_n, len(samples))
    print(f"\n  TOP {top_n} SAMPLES TO LABEL NEXT:")
    print(f"  {'#':>3s} {'score':>7s} {'manifold':>14s} {'risk':>6s} "
          f"{'unc':>5s} {'bs':>5s} {'cost':>5s} {'prompt':>35s}")
    print(f"  {'-'*85}")
    for i, s in enumerate(samples[:top_n]):
        prompt_preview = s["prompt"][:35].replace("\n", " ")
        m = s.get("assigned_manifold", s.get("assigned_channel", "?"))[:12]
        print(f"  {i+1:>3d} {s['acquisition_score']:>7.4f} {m:>14s} "
              f"{s['risk_score']:>6.3f} {s['uncertainty_norm']:>5.2f} "
              f"{s['blind_spot_norm']:>5.2f} {s['cost_norm']:>5.2f} "
              f"{prompt_preview:>35s}")

    # Write acquisition queue (with manifold tags)
    os.makedirs(os.path.dirname(ACQUISITION_QUEUE_PATH), exist_ok=True)
    with open(ACQUISITION_QUEUE_PATH, "w") as f:
        for i, s in enumerate(samples):
            entry = {
                "query_id": s["query_id"],
                "prompt": s["prompt"],
                "priority": i + 1,
                "acquisition_score": s["acquisition_score"],
                "assigned_manifold": s.get("assigned_manifold", "boundary"),
                "estimated_manifold": s.get("estimated_manifold", "boundary"),
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
        "version": "v2.5.1",
        "allocation_mode": "manifold_aware",
        "manifold_weights": manifold_weights,
        "legacy_channel_weights": weights,  # kept for compat
        "budget": budget,
        "allocation": alloc["allocation"],
        "manifold_counts": mc,
        "n_candidates": len(score_map),
        "n_labeled": len(acquired_ids),
    }
    with open(ACQUISITION_BUDGET_PATH, "w") as f:
        json.dump(budget_state, f, indent=2, ensure_ascii=False)

    print(f"\n  Queue ({len(samples)} samples) -> {ACQUISITION_QUEUE_PATH}")
    print(f"  Budget state -> {ACQUISITION_BUDGET_PATH}")
    print(f"\n  Loop: label contradiction -> retrain cd head -> check recall -> repeat")


if __name__ == "__main__":
    main()

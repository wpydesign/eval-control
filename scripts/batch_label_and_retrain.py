#!/usr/bin/env python3
"""
batch_label_and_retrain.py — Batch label 386 high-value samples + single retrain [v2.4.0]

Operational throughput optimization: consolidate all labeling into one batch,
then run a single retrain cycle. This is the shift from "iterative improvement
loop" to "batched information consolidation -> single update cycle."

Phase A: 12 blind_spot samples (highest proxy_score, unlabeled)
Phase B: 374 ambiguous samples (highest uncertainty_score, unlabeled, excluding A)
Skip:   noise + excluded

Pipeline:
  1. Identify 386 target samples from acquisition queues
  2. Label all using signal heuristics + API judge fallback
  3. Build unified retrain dataset with failure_type tags
  4. Compute pre-retrain segment AUC contribution snapshot
  5. Single retrain cycle (predictor + calibration + weights)

Output:
  logs/batch_label_results.jsonl  — labeled results with failure_type tags
  logs/pre_retrain_snapshot.json  — segment AUC contribution analysis
  logs/failure_dataset.jsonl      — updated with 386 new labels (140 -> 526)
  model/failure_predictor.pkl     — retrained model
  model/training_report.json      — updated metrics

Usage:
  python scripts/batch_label_and_retrain.py
  python scripts/batch_label_and_retrain.py --dry-run   # snapshot only, no retrain
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE, "logs", "failure_dataset.jsonl")
MINING_QUEUE_PATH = os.path.join(BASE, "logs", "failure_mining_queue.jsonl")
UNCERTAINTY_QUEUE_PATH = os.path.join(BASE, "logs", "active_learning_queue.jsonl")
ACQUISITION_QUEUE_PATH = os.path.join(BASE, "logs", "acquisition_queue.jsonl")
EXCLUSIONS_PATH = os.path.join(BASE, "logs", "integrity_exclusions.jsonl")
RESULTS_PATH = os.path.join(BASE, "logs", "batch_label_results.jsonl")
SNAPSHOT_PATH = os.path.join(BASE, "logs", "pre_retrain_snapshot.json")

# API configuration
API_KEY = "sk-f55cb3459edd4becb8d6f83db3afd6d1"
API_BASE = "https://api.deepseek.com/v1/chat/completions"
API_MODEL = "deepseek-chat"
API_DELAY = 0.5  # seconds between API calls
API_BATCH_SIZE = 20  # prompts per API call

# Batch targets
N_BLIND_SPOT = 12
N_AMBIGUOUS = 374


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

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


def load_dataset_index():
    """Load failure_dataset.jsonl into query_id -> record index."""
    records = load_jsonl(DATASET_PATH)
    idx = {}
    for r in records:
        qid = r.get("query_id", "")
        if qid:
            idx[qid] = r
    return idx, records


def load_exclusions():
    """Load excluded query_ids."""
    records = load_jsonl(EXCLUSIONS_PATH)
    return {r.get("query_id", "") for r in records if r.get("query_id")}


# ═══════════════════════════════════════════════════════════════
# PHASE 1: IDENTIFY TARGET SAMPLES
# ═══════════════════════════════════════════════════════════════

def identify_targets(dataset_idx, exclusions):
    """Identify 386 target samples: 12 blind_spot + 374 ambiguous.

    Returns:
      blind_spot_ids: list of 12 query_ids
      ambiguous_ids: list of 374 query_ids (no overlap with blind_spot)
    """
    # Load queues
    mining_queue = load_jsonl(MINING_QUEUE_PATH)
    uncertainty_queue = load_jsonl(UNCERTAINTY_QUEUE_PATH)

    # Filter: must be unlabeled, not excluded, must exist in dataset
    unlabeled_mining = []
    for r in mining_queue:
        qid = r.get("query_id", "")
        if qid in exclusions:
            continue
        if qid not in dataset_idx:
            continue
        if dataset_idx[qid].get("is_wrong") is not None:
            continue
        unlabeled_mining.append(r)

    unlabeled_uncertainty = []
    seen = set()
    for r in uncertainty_queue:
        qid = r.get("query_id", "")
        if qid in seen:
            continue
        seen.add(qid)
        if qid in exclusions:
            continue
        if qid not in dataset_idx:
            continue
        if dataset_idx[qid].get("is_wrong") is not None:
            continue
        unlabeled_uncertainty.append(r)

    # Phase A: top 12 blind_spot
    blind_spot_ids = [r["query_id"] for r in unlabeled_mining[:N_BLIND_SPOT]]
    blind_spot_set = set(blind_spot_ids)

    # Phase B: top 374 ambiguous (excluding blind_spot)
    ambiguous_ids = []
    for r in unlabeled_uncertainty:
        qid = r["query_id"]
        if qid not in blind_spot_set:
            ambiguous_ids.append(qid)
        if len(ambiguous_ids) >= N_AMBIGUOUS:
            break

    return blind_spot_ids, ambiguous_ids


# ═══════════════════════════════════════════════════════════════
# PHASE 2: LABEL ALL 386 SAMPLES
# ═══════════════════════════════════════════════════════════════

def heuristic_label(record):
    """Determine is_wrong using signal heuristics.

    Returns:
      (is_wrong, confidence) where confidence is "high" or "low"
      is_wrong: 0 (good prompt), 1 (bad prompt), or None (need API judge)
    """
    fm = record.get("failure_mode", "none")
    sc = record.get("source_class", "unknown")
    s_v4 = record.get("S_v4", 0.5)
    s_v1 = record.get("S_v1", 0.5)
    div = record.get("divergence", False)
    kappa = record.get("kappa_v4", 0.5)
    prompt = record.get("prompt", "").lower().strip()

    # Strong negative signals → BAD (is_wrong=1)
    bad_modes = {
        "trick_question", "impossible_request", "vague_ambiguous",
        "scope_overreach", "confused_user", "debug_underspecified",
        "opinion_debate", "underspecified_tech"
    }
    if fm in bad_modes:
        return (1, "high")

    # Very low S_v4 and S_v1 → BAD
    if s_v4 < 0.25 and s_v1 < 0.20:
        return (1, "high")

    # Strong positive signals → GOOD (is_wrong=0)
    if fm == "none" and s_v4 > 0.55 and s_v1 > 0.40 and not div:
        # Check prompt quality heuristics
        # Very short/vague prompts are likely bad
        if len(prompt.split()) < 4:
            return (1, "high")
        return (0, "high")

    # High kappa + moderate S → likely good
    if fm == "none" and kappa > 0.55 and s_v4 > 0.50 and s_v1 > 0.45:
        return (0, "high")

    # Very low kappa → likely bad
    if kappa < 0.18 and s_v4 < 0.35:
        return (1, "high")

    # Divergence with very different scores → uncertain, likely bad
    if div and abs(s_v4 - s_v1) > 0.30 and s_v1 < 0.25:
        return (1, "medium")

    # High S_v4 + divergence where v4 > v1 → likely good
    if s_v4 > 0.60 and s_v1 < 0.30 and not div:
        return (0, "medium")

    # Remaining: need API judge
    return (None, "low")


def api_judge_batch(prompts_with_ids):
    """Use DeepSeek API to judge a batch of prompts at once.

    Args:
      prompts_with_ids: list of (query_id, prompt) tuples

    Returns:
      dict of query_id -> label (0 or 1)
    """
    if not prompts_with_ids:
        return {}

    system_msg = (
        "You are a prompt quality judge. For each prompt below, classify it as GOOD or BAD.\n"
        "GOOD = well-formed, specific enough to be answerable, reasonable request.\n"
        "BAD = underspecified, impossible, trick, vague, confused, too broad, or not meaningfully answerable.\n"
        "Short vague prompts like \"fix this\" or \"help\" are BAD.\n"
        "Overly broad prompts like \"tell me about everything\" are BAD.\n"
        "Specific questions about programming, science, or practical topics are GOOD.\n"
        "Respond with ONLY a JSON object mapping line numbers to GOOD/BAD. Example: {\"1\": \"GOOD\", \"2\": \"BAD\"}"
    )

    # Build numbered prompt list
    prompt_lines = []
    for i, (qid, prompt) in enumerate(prompts_with_ids, 1):
        prompt_lines.append(f'{i}. "{prompt[:200]}"')

    user_msg = "Classify each prompt:\n\n" + "\n".join(prompt_lines)

    payload = json.dumps({
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_BASE,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    import socket
    socket.setdefaulttimeout(30)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"].strip()
            # Parse JSON response
            # Try to extract JSON from the response
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                judgments = json.loads(json_str)
            else:
                # Fallback: try parsing the whole thing
                judgments = json.loads(text)

            results = {}
            for i, (qid, prompt) in enumerate(prompts_with_ids, 1):
                key = str(i)
                if key in judgments:
                    val = str(judgments[key]).upper()
                    results[qid] = 1 if "BAD" in val else 0
                else:
                    results[qid] = 0  # conservative default
            return results
    except json.JSONDecodeError:
        # Parse failure — return all as GOOD (conservative)
        print(f"    API JSON parse error, defaulting all to GOOD")
        return {qid: 0 for qid, _ in prompts_with_ids}
    except Exception as e:
        print(f"    API error: {e}")
        return {qid: 0 for qid, _ in prompts_with_ids}


def label_all_targets(dataset_idx, blind_spot_ids, ambiguous_ids):
    """Label all 386 target samples.

    Returns:
      results: list of labeled result dicts
      stats: labeling statistics
    """
    results = []
    stats = {
        "total": 0,
        "heuristic_high": 0,
        "heuristic_medium": 0,
        "api_judged": 0,
        "api_errors": 0,
        "labeled_good": 0,
        "labeled_bad": 0,
        "blind_spot_good": 0,
        "blind_spot_bad": 0,
        "ambiguous_good": 0,
        "ambiguous_bad": 0,
    }

    # Collect all API-needed samples first
    api_needs = []  # (channel, qid, record)

    # Phase A: blind_spot (priority 1)
    print(f"\n{'='*60}")
    print(f"Phase A: Labeling {len(blind_spot_ids)} blind_spot samples (priority 1)")
    print(f"{'='*60}")

    for i, qid in enumerate(blind_spot_ids):
        record = dataset_idx[qid]
        stats["total"] += 1

        # Try heuristic first
        label, conf = heuristic_label(record)

        if label is None:
            api_needs.append(("blind_spot", qid, record))
            stats["api_judged"] += 1
        else:
            stats[f"heuristic_{conf}"] += 1
            tag = "BAD" if label == 1 else "GOOD"
            print(f"  [{i+1:3d}/{len(blind_spot_ids)}] Heuristic({conf}): {tag} — {record['prompt'][:60]}...")
            if label == 1:
                stats["labeled_bad"] += 1
                stats["blind_spot_bad"] += 1
            else:
                stats["labeled_good"] += 1
                stats["blind_spot_good"] += 1
            result = build_result(record, label, "blind_spot")
            results.append(result)

    # Phase B: ambiguous (priority 2)
    print(f"\n{'='*60}")
    print(f"Phase B: Labeling {len(ambiguous_ids)} ambiguous samples (priority 2)")
    print(f"{'='*60}")

    for i, qid in enumerate(ambiguous_ids):
        record = dataset_idx[qid]
        stats["total"] += 1

        label, conf = heuristic_label(record)

        if label is None:
            api_needs.append(("ambiguous", qid, record))
            stats["api_judged"] += 1
        else:
            stats[f"heuristic_{conf}"] += 1
            if label == 1:
                stats["labeled_bad"] += 1
                stats["ambiguous_bad"] += 1
            else:
                stats["labeled_good"] += 1
                stats["ambiguous_good"] += 1
            result = build_result(record, label, "ambiguous")
            results.append(result)

    print(f"\n  Heuristic resolved: {stats['total'] - len(api_needs)}")
    print(f"  API judge needed: {len(api_needs)} ({len(api_needs) // API_BATCH_SIZE + 1} batch calls)")

    # Batch API calls for all API-needed samples
    if api_needs:
        print(f"\n  Running batch API judge...")
        api_batches = [
            api_needs[i:i + API_BATCH_SIZE]
            for i in range(0, len(api_needs), API_BATCH_SIZE)
        ]
        for batch_idx, batch in enumerate(api_batches):
            prompts_with_ids = [(qid, record["prompt"]) for _, qid, record in batch]
            print(f"    Batch {batch_idx+1}/{len(api_batches)}: {len(batch)} prompts...", end=" ", flush=True)
            judgments = api_judge_batch(prompts_with_ids)
            time.sleep(API_DELAY)

            for channel, qid, record in batch:
                label = judgments.get(qid, 0)
                if label == 1:
                    stats["labeled_bad"] += 1
                    if channel == "blind_spot":
                        stats["blind_spot_bad"] += 1
                    else:
                        stats["ambiguous_bad"] += 1
                else:
                    stats["labeled_good"] += 1
                    if channel == "blind_spot":
                        stats["blind_spot_good"] += 1
                    else:
                        stats["ambiguous_good"] += 1

                result = build_result(record, label, channel)
                results.append(result)

            batch_good = sum(1 for c, q, r in batch if judgments.get(q, 0) == 0)
            batch_bad = len(batch) - batch_good
            print(f"done (GOOD={batch_good}, BAD={batch_bad})")

    return results, stats


def build_result(record, is_wrong, source_channel):
    """Build a labeled result dict with failure_type tag.

    failure_type classification:
      boundary     — near decision threshold (uncertainty zone)
      overconfidence (blind_spot) — high confidence but structurally unstable
      contradiction — v4 vs v1 disagree on tier
    """
    s_v4 = record.get("S_v4", 0.5)
    s_v1 = record.get("S_v1", 0.5)
    v4_decision = record.get("v4_decision", "")
    v1_decision = record.get("v1_decision", "")
    divergence = record.get("divergence", False)
    proxy_score = record.get("proxy_score", 0.0)
    risk_score = record.get("risk_score", 0.0)

    # Determine failure_type
    # Boundary: S_v4 is near thresholds (0.20 or 0.70)
    near_lower = abs(s_v4 - 0.20) < 0.10
    near_upper = abs(s_v4 - 0.70) < 0.10
    is_boundary = near_lower or near_upper

    # Overconfidence (blind spot): high confidence × high proxy score × is_wrong
    is_overconfident = (
        s_v4 > 0.40 and
        (source_channel == "blind_spot" or proxy_score > 0.15) and
        is_wrong == 1
    )

    # Contradiction: v4 and v1 decisions differ
    is_contradiction = (
        v4_decision != v1_decision and
        v4_decision != "" and v1_decision != ""
    )

    # Priority: overconfidence > contradiction > boundary
    if is_overconfident:
        failure_type = "overconfidence"
    elif is_contradiction:
        failure_type = "contradiction"
    elif is_boundary:
        failure_type = "boundary"
    elif is_wrong == 1:
        # Default for wrong: check which signal was strongest
        if proxy_score > 0.15:
            failure_type = "overconfidence"
        elif divergence:
            failure_type = "contradiction"
        else:
            failure_type = "boundary"
    else:
        failure_type = "boundary"  # correct but near threshold

    return {
        "query_id": record.get("query_id", ""),
        "prompt": record.get("prompt", ""),
        "final_label": is_wrong,
        "v4_prediction": s_v4,
        "v1_shadow": s_v1,
        "v4_decision": v4_decision,
        "v1_decision": v1_decision,
        "disagreement_flag": divergence,
        "failure_type": failure_type,
        "source_channel": source_channel,
        "confidence_gap": abs(s_v4 - s_v1),
        "kappa_v4": record.get("kappa_v4", 0.0),
        "proxy_score": proxy_score,
        "risk_score": risk_score,
        "failure_mode": record.get("failure_mode", "none"),
        "source_class": record.get("source_class", "unknown"),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 3: UPDATE FAILURE DATASET
# ═══════════════════════════════════════════════════════════════

def update_dataset(dataset_records, results):
    """Update failure_dataset.jsonl with new labels.

    Returns:
      updated records, stats
    """
    # Build result index
    result_idx = {r["query_id"]: r for r in results}

    updated = 0
    for i, record in enumerate(dataset_records):
        qid = record.get("query_id", "")
        if qid in result_idx:
            label = result_idx[qid]["final_label"]
            if record.get("is_wrong") is None:
                # Assign source_class based on label
                record["is_wrong"] = label
                record["source_class"] = "bad" if label == 1 else "good"
                updated += 1

    # Write updated dataset
    with open(DATASET_PATH, "w") as f:
        for r in dataset_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return dataset_records, {"updated": updated}


# ═══════════════════════════════════════════════════════════════
# PHASE 4: PRE-RETRAIN AUC CONTRIBUTION SNAPSHOT
# ═══════════════════════════════════════════════════════════════

def compute_snapshot(results, dataset_records):
    """Compute expected AUC shift contribution per segment.

    This snapshot is computed BEFORE retraining to establish the
    information value of each labeling channel.

    Returns:
      snapshot dict with per-segment analysis
    """
    # Load existing labeled data stats
    existing_labeled = [r for r in dataset_records if r.get("is_wrong") is not None]
    n_existing = len(existing_labeled)
    n_existing_wrong = sum(1 for r in existing_labeled if r["is_wrong"] == 1)
    n_existing_correct = n_existing - n_existing_wrong

    # Segment the new labels
    blind_spot_results = [r for r in results if r["source_channel"] == "blind_spot"]
    ambiguous_results = [r for r in results if r["source_channel"] == "ambiguous"]

    # Per-segment stats
    segments = {}
    for name, group in [("blind_spot", blind_spot_results), ("ambiguous", ambiguous_results)]:
        n = len(group)
        if n == 0:
            segments[name] = {"n": 0, "n_wrong": 0, "n_correct": 0, "wrong_rate": 0}
            continue

        n_wrong = sum(1 for r in group if r["final_label"] == 1)
        n_correct = n - n_wrong
        wrong_rate = n_wrong / n

        # Mean risk_score for this segment
        mean_risk = sum(r["risk_score"] for r in group) / n
        # Mean proxy_score for this segment
        mean_proxy = sum(r["proxy_score"] for r in group) / n

        # Per failure_type breakdown
        ft_counts = defaultdict(int)
        for r in group:
            ft_counts[r["failure_type"]] += 1

        # Expected AUC contribution: higher when wrong_rate is far from 0.5
        # and sample size is significant
        # Simple estimate: new samples add information proportional to
        # |wrong_rate - 0.5| * sqrt(n) (Fisher information proxy)
        information_value = abs(wrong_rate - 0.5) * (n ** 0.5)

        segments[name] = {
            "n": n,
            "n_wrong": n_wrong,
            "n_correct": n_correct,
            "wrong_rate": round(wrong_rate, 4),
            "mean_risk_score": round(mean_risk, 4),
            "mean_proxy_score": round(mean_proxy, 4),
            "information_value": round(information_value, 4),
            "failure_type_breakdown": dict(ft_counts),
        }

    # Blind spot gain: samples found ONLY by blind_spot proxy
    # (high proxy_score but low risk_score)
    blind_spot_only_wrong = [
        r for r in blind_spot_results
        if r["final_label"] == 1 and r["risk_score"] < 0.3
    ]
    # Boundary gain: samples near decision threshold
    boundary_wrong = [
        r for r in ambiguous_results
        if r["final_label"] == 1 and r["failure_type"] == "boundary"
    ]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "existing_dataset": {
            "n_total": len(dataset_records),
            "n_labeled": n_existing,
            "n_wrong": n_existing_wrong,
            "n_correct": n_existing_correct,
            "class_ratio": f"{n_existing_correct}:{n_existing_wrong}",
        },
        "new_labels": {
            "total": len(results),
            "blind_spot": len(blind_spot_results),
            "ambiguous": len(ambiguous_results),
        },
        "segment_analysis": segments,
        "gain_estimates": {
            "blind_spot_only_wrong": len(blind_spot_only_wrong),
            "blind_spot_only_wrong_prompts": [r["prompt"][:80] for r in blind_spot_only_wrong[:5]],
            "boundary_wrong": len(boundary_wrong),
            "boundary_wrong_prompts": [r["prompt"][:80] for r in boundary_wrong[:5]],
        },
        "combined_wrong_rate": round(
            sum(1 for r in results if r["final_label"] == 1) / max(1, len(results)), 4
        ),
        "expected_post_retrain_balance": {
            "total_labeled": n_existing + len(results),
            "expected_wrong": n_existing_wrong + sum(1 for r in results if r["final_label"] == 1),
            "expected_correct": n_existing_correct + sum(1 for r in results if r["final_label"] == 0),
        },
    }

    # Compute per-failure-type expected contribution
    ft_segments = defaultdict(lambda: {"n": 0, "n_wrong": 0})
    for r in results:
        ft = r["failure_type"]
        ft_segments[ft]["n"] += 1
        if r["final_label"] == 1:
            ft_segments[ft]["n_wrong"] += 1

    snapshot["failure_type_segments"] = {}
    for ft, data in sorted(ft_segments.items()):
        snapshot["failure_type_segments"][ft] = {
            "n": data["n"],
            "n_wrong": data["n_wrong"],
            "wrong_rate": round(data["n_wrong"] / max(1, data["n"]), 4),
        }

    return snapshot


# ═══════════════════════════════════════════════════════════════
# PHASE 5: SINGLE RETRAIN
# ═══════════════════════════════════════════════════════════════

def run_retrain():
    """Run a single retrain cycle: predictor + calibration + weights."""
    print(f"\n{'='*60}")
    print("Phase 5: Single retrain cycle")
    print(f"{'='*60}")

    sys.path.insert(0, os.path.dirname(__file__))

    # Step 1: Retrain predictor
    print("\n  [RETRAIN] Retraining failure predictor...")
    try:
        from train_failure_predictor import main as train_main
        old_argv = sys.argv
        sys.argv = ["train_failure_predictor.py", "--retrain"]
        train_main()
        sys.argv = old_argv
        print("  [RETRAIN] Predictor retrained successfully")
    except Exception as e:
        print(f"  [RETRAIN] FAILED: {e}")
        return False

    # Step 2: Run calibration check
    print("\n  [RETRAIN] Running calibration check...")
    try:
        from scripts.calibration_check import main as cal_main
        cal_main()
        print("  [RETRAIN] Calibration check complete")
    except ImportError:
        # Try direct import
        try:
            calib_path = os.path.join(os.path.dirname(__file__), "calibration_check.py")
            if os.path.exists(calib_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("calibration_check", calib_path)
                cal_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cal_mod)
                cal_mod.main()
                print("  [RETRAIN] Calibration check complete")
        except Exception as e:
            print(f"  [RETRAIN] Calibration check warning: {e}")
    except Exception as e:
        print(f"  [RETRAIN] Calibration check warning: {e}")

    # Step 3: Update acquisition weights
    print("\n  [RETRAIN] Updating acquisition weights...")
    try:
        acq_path = os.path.join(os.path.dirname(__file__), "acquisition_policy.py")
        if os.path.exists(acq_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("acquisition_policy", acq_path)
            acq_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acq_mod)

            # Load current weights
            weights = dict(acq_mod.DEFAULT_WEIGHTS)
            budget_path = acq_mod.ACQUISITION_BUDGET_PATH
            if os.path.exists(budget_path):
                try:
                    with open(budget_path) as f:
                        saved = json.load(f)
                    if "weights" in saved:
                        weights = saved["weights"]
                except (json.JSONDecodeError, OSError):
                    pass

            # Record batch outcomes as channel performance entry
            results_data = load_jsonl(RESULTS_PATH)
            channels = {"uncertainty": {"n_labeled": 0, "n_wrong": 0},
                        "blind_spot": {"n_labeled": 0, "n_wrong": 0},
                        "cost": {"n_labeled": 0, "n_wrong": 0}}

            for r in results_data:
                ch = r.get("source_channel", "ambiguous")
                if ch == "ambiguous":
                    ch = "uncertainty"  # map to uncertainty channel
                if ch in channels:
                    channels[ch]["n_labeled"] += 1
                    if r["final_label"] == 1:
                        channels[ch]["n_wrong"] += 1

            perf = acq_mod.load_channel_performance()

            # Append this batch as a performance cycle
            perf_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": len(perf),
                "channels": channels,
                "prev_weights": weights,
                "source": "batch_label_v2.4.0",
            }
            os.makedirs(os.path.dirname(acq_mod.CHANNEL_PERF_PATH), exist_ok=True)
            with open(acq_mod.CHANNEL_PERF_PATH, "a") as f:
                f.write(json.dumps(perf_entry, ensure_ascii=False) + "\n")
            perf.append(perf_entry)

            print(f"  [RETRAIN] Recorded batch outcomes (cycle {len(perf)-1}):")
            for ch, data in channels.items():
                eff = data["n_wrong"] / max(1, data["n_labeled"])
                print(f"    {ch:>15s}: {data['n_labeled']} labeled, "
                      f"{data['n_wrong']} wrong ({eff:.1%} efficiency)")

            # Adapt weights using lag compensation
            new_weights = acq_mod.adapt_weights(weights, perf, lag=acq_mod.ATTRIBUTION_LAG)

            if len(perf) > acq_mod.ATTRIBUTION_LAG:
                print(f"  [RETRAIN] Lag-compensated weight update:")
                print(f"    Before: unc={weights['uncertainty']:.3f}, "
                      f"bs={weights['blind_spot']:.3f}, cost={weights['cost']:.3f}")
                print(f"    After:  unc={new_weights['uncertainty']:.3f}, "
                      f"bs={new_weights['blind_spot']:.3f}, cost={new_weights['cost']:.3f}")

                delta = {ch: new_weights[ch] - weights[ch] for ch in weights}
                for ch, d in delta.items():
                    direction = "up" if d > 0.001 else ("down" if d < -0.001 else "stable")
                    print(f"      {ch:>15s}: {d:+.4f} ({direction})")

                # Save
                budget_state = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "weights": new_weights,
                    "prev_weights": weights,
                    "weight_delta": delta,
                    "adaptation_mode": "batch_consolidation",
                    "attribution_lag": acq_mod.ATTRIBUTION_LAG,
                    "n_performance_cycles": len(perf),
                }
                with open(budget_path, "w") as f:
                    json.dump(budget_state, f, indent=2, ensure_ascii=False)
                print("  [RETRAIN] Acquisition weights updated and saved")
            else:
                print(f"  [RETRAIN] First cycle — weights unchanged (lag guard)")
        else:
            print("  [RETRAIN] acquisition_policy.py not found")
    except Exception as e:
        print(f"  [RETRAIN] Weight update warning: {e}")

    return True


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def main():
    dry_run = "--dry-run" in sys.argv
    start_time = time.time()

    print("=" * 60)
    print("BATCH LABEL + RETRAIN [v2.4.0]")
    print(f"Mode: {'DRY RUN (snapshot only)' if dry_run else 'FULL EXECUTION'}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── Phase 1: Identify targets ──
    print(f"\n{'─'*60}")
    print("Phase 1: Identifying target samples")
    print(f"{'─'*60}")

    dataset_idx, dataset_records = load_dataset_index()
    exclusions = load_exclusions()

    n_labeled = sum(1 for r in dataset_records if r.get("is_wrong") is not None)
    n_unlabeled = len(dataset_records) - n_labeled
    print(f"  Dataset: {len(dataset_records)} total, {n_labeled} labeled, {n_unlabeled} unlabeled")
    print(f"  Exclusions: {len(exclusions)}")

    blind_spot_ids, ambiguous_ids = identify_targets(dataset_idx, exclusions)
    print(f"\n  Targets identified:")
    print(f"    Phase A (blind_spot): {len(blind_spot_ids)}")
    print(f"    Phase B (ambiguous):  {len(ambiguous_ids)}")
    print(f"    Total:                {len(blind_spot_ids) + len(ambiguous_ids)}")

    # Check for overlap
    overlap = set(blind_spot_ids) & set(ambiguous_ids)
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping query_ids detected!")

    # ── Phase 2: Label all 386 ──
    print(f"\n{'─'*60}")
    print("Phase 2: Labeling all targets")
    print(f"{'─'*60}")

    all_ids = blind_spot_ids + ambiguous_ids
    results, label_stats = label_all_targets(dataset_idx, blind_spot_ids, ambiguous_ids)

    print(f"\n{'─'*60}")
    print("Labeling Summary:")
    print(f"{'─'*60}")
    print(f"  Total labeled:     {label_stats['total']}")
    print(f"  Good (is_wrong=0): {label_stats['labeled_good']}")
    print(f"  Bad  (is_wrong=1): {label_stats['labeled_bad']}")
    print(f"  Wrong rate:        {label_stats['labeled_bad']/max(1,label_stats['total']):.1%}")
    print(f"  Heuristic (high):  {label_stats['heuristic_high']}")
    print(f"  Heuristic (med):   {label_stats['heuristic_medium']}")
    print(f"  API judged:        {label_stats['api_judged']}")
    print(f"\n  Blind spot:  {label_stats['blind_spot_good']} good / {label_stats['blind_spot_bad']} bad")
    print(f"  Ambiguous:  {label_stats['ambiguous_good']} good / {label_stats['ambiguous_bad']} bad")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Results saved to {RESULTS_PATH}")

    # ── Phase 3: Update dataset ──
    if not dry_run:
        print(f"\n{'─'*60}")
        print("Phase 3: Updating failure dataset")
        print(f"{'─'*60}")

        dataset_records, update_stats = update_dataset(dataset_records, results)
        print(f"  Updated {update_stats['updated']} records in {DATASET_PATH}")

        new_labeled = sum(1 for r in dataset_records if r.get("is_wrong") is not None)
        new_wrong = sum(1 for r in dataset_records if r.get("is_wrong") == 1)
        print(f"  New total labeled: {new_labeled} ({new_wrong} wrong, {new_labeled - new_wrong} correct)")

    # ── Phase 4: Pre-retrain snapshot ──
    print(f"\n{'─'*60}")
    print("Phase 4: Pre-retrain AUC contribution snapshot")
    print(f"{'─'*60}")

    snapshot = compute_snapshot(results, dataset_records)

    # Print snapshot
    print(f"\n  Existing dataset: {snapshot['existing_dataset']['n_labeled']} labeled "
          f"({snapshot['existing_dataset']['n_correct']}:{snapshot['existing_dataset']['n_wrong']})")

    print(f"\n  New labels: {snapshot['new_labels']['total']} total")
    for seg_name, seg_data in snapshot["segment_analysis"].items():
        if seg_data["n"] > 0:
            print(f"\n    [{seg_name}]")
            print(f"      n={seg_data['n']}, wrong={seg_data['n_wrong']} ({seg_data['wrong_rate']:.1%})")
            print(f"      mean_risk={seg_data['mean_risk_score']:.3f}, "
                  f"mean_proxy={seg_data['mean_proxy_score']:.3f}")
            print(f"      information_value={seg_data['information_value']:.3f}")
            if seg_data.get("failure_type_breakdown"):
                for ft, cnt in sorted(seg_data["failure_type_breakdown"].items()):
                    print(f"        {ft}: {cnt}")

    print(f"\n  Gain estimates:")
    print(f"    blind_spot_only_wrong (risk<0.3): {snapshot['gain_estimates']['blind_spot_only_wrong']}")
    if snapshot['gain_estimates']['blind_spot_only_wrong_prompts']:
        for p in snapshot['gain_estimates']['blind_spot_only_wrong_prompts']:
            print(f"      - {p}")
    print(f"    boundary_wrong: {snapshot['gain_estimates']['boundary_wrong']}")

    print(f"\n  Failure type segments:")
    for ft, data in snapshot.get("failure_type_segments", {}).items():
        print(f"    {ft}: {data['n']} ({data['n_wrong']} wrong, {data['wrong_rate']:.1%})")

    exp = snapshot["expected_post_retrain_balance"]
    print(f"\n  Expected post-retrain balance:")
    print(f"    Total labeled: {exp['total_labeled']}")
    print(f"    Wrong: {exp['expected_wrong']}, Correct: {exp['expected_correct']}")
    print(f"    Ratio: {exp['expected_correct']}:{exp['expected_wrong']}")

    # Save snapshot
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"\n  Snapshot saved to {SNAPSHOT_PATH}")

    # ── Phase 5: Single retrain ──
    if not dry_run:
        success = run_retrain()
        if not success:
            print("\n  WARNING: Retrain failed. Labels are saved but model not updated.")
    else:
        print(f"\n  DRY RUN: skipping retrain")

    # ── Final summary ──
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Mode:        {'DRY RUN' if dry_run else 'FULL'}")
    print(f"  Labeled:     {label_stats['total']} samples")
    print(f"  Duration:    {elapsed:.1f}s")
    print(f"  Good:        {label_stats['labeled_good']} ({label_stats['labeled_good']/max(1,label_stats['total']):.1%})")
    print(f"  Bad:         {label_stats['labeled_bad']} ({label_stats['labeled_bad']/max(1,label_stats['total']):.1%})")
    print(f"  API calls:   {label_stats['api_judged']}")
    print(f"  Heuristics:  {label_stats['heuristic_high'] + label_stats['heuristic_medium']}")
    if not dry_run:
        new_labeled = sum(1 for r in dataset_records if r.get("is_wrong") is not None)
        print(f"  Post-retrain labeled: {new_labeled}")
    print(f"  Completed:   {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_live_batch.py — v4-primary evaluation pipeline [v2.1 — v4 PROMOTED]

v4 = π_E (primary output policy, all production decisions)
v1 = π_S (frozen shadow validator, audit/comparator only)

ONLY reads from:  data/raw_prompts.jsonl
Appends results to: logs/shadow_eval_live.jsonl
Appends metrics to: logs/daily_metrics.jsonl
Disagreements to:  logs/disagreement_cases.jsonl (auto-logged by survival.py)

Rules:
  - Batch size: 20 prompts per batch
  - No regeneration, no retry logic
  - If a prompt is not from real usage, it does not exist
  - Already-evaluated prompts (by qid) are skipped
  - High-impact disagreements → flagged for shadow review
"""

import json
import os
import sys
import hashlib
import time
import numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

API_TIMEOUT = 30  # seconds — any call exceeding this is skipped

DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PROMPTS_PATH = os.path.join(DIR, "data", "raw_prompts.jsonl")
LIVE_LOG_PATH = os.path.join(DIR, "logs", "shadow_eval_live.jsonl")
METRICS_PATH = os.path.join(DIR, "logs", "daily_metrics.jsonl")

BATCH_SIZE = 20


def load_raw_prompts(path: str) -> list[dict]:
    """Load all prompts from the unified input source."""
    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist. No real prompts to process.")
        sys.exit(1)
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} raw prompts from {path}")
    return prompts


def get_already_evaluated_qids(log_path: str) -> set[str]:
    """Get set of query IDs already in the live log (skip re-evaluation)."""
    qids = set()
    if not os.path.exists(log_path):
        return qids
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if "query_id" in rec:
                        qids.add(rec["query_id"])
                except json.JSONDecodeError:
                    pass
    return qids


def compute_batch_metrics(results: list[dict]) -> dict:
    """Compute simple metrics at end of batch."""
    total = len(results)
    if total == 0:
        return {"total": 0, "bad_accepted": 0, "good_rejected": 0,
                "divergence_rate": 0.0}

    # v4 decisions (π_E — the authority)
    bad_accepted = sum(1 for r in results if r.get("v4", {}).get("decision") == "accept"
                       and r.get("source_class") == "bad")
    good_rejected = sum(1 for r in results if r.get("v4", {}).get("decision") == "reject"
                        and r.get("source_class") == "good")

    # v1 vs v4 divergence
    divergent = sum(1 for r in results if r.get("divergence", False))
    divergence_rate = divergent / total if total > 0 else 0.0

    # High-impact shadow reviews (v2.1 safety rule)
    shadow_reviews = sum(1 for r in results if r.get("needs_shadow_review", False))

    # S distribution (v4)
    s_values = [r.get("v4", {}).get("S", 0.0) for r in results]
    s_mean = sum(s_values) / len(s_values) if s_values else 0.0
    s_min = min(s_values) if s_values else 0.0
    s_max = max(s_values) if s_values else 0.0

    # Accept/review/reject counts (using safe_decision when available)
    accept_count = 0
    review_count = 0
    reject_count = 0
    for r in results:
        # v2.1: use safe_decision (may differ from v4 decision on high-impact)
        dec = r.get("safe_decision") or r.get("v4", {}).get("decision", "")
        if dec == "accept":
            accept_count += 1
        elif dec == "review":
            review_count += 1
        elif dec == "reject":
            reject_count += 1

    return {
        "batch_size": total,
        "bad_accepted": bad_accepted,
        "good_rejected": good_rejected,
        "divergence_rate": round(divergence_rate, 4),
        "divergent_count": divergent,
        "shadow_review_count": shadow_reviews,
        "s_mean": round(s_mean, 4),
        "s_min": round(s_min, 4),
        "s_max": round(s_max, 4),
        "accept_count": accept_count,
        "review_count": review_count,
        "reject_count": reject_count,
    }


def append_metrics(metrics: dict, path: str):
    """Append batch metrics to daily metrics log."""
    entry = {
        **metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_result(result: dict, source: str, source_class: str, path: str):
    """Append a single evaluation result to the live log."""
    entry = {
        **result,
        "source": source,
        "source_class": source_class,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _eval_with_timeout(engine, prompt, timeout=API_TIMEOUT):
    """Run evaluate_shadow with hard timeout. Returns result or raises."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.evaluate_shadow, prompt)
        return future.result(timeout=timeout)


def run_batch(raw_prompts: list[dict], engine, start_idx: int = 0,
              batch_size: int = BATCH_SIZE, skip_qids: set[str] = None) -> list[dict]:
    """Run a single batch of evaluations. No retry. Skip on timeout/failure."""
    if skip_qids is None:
        skip_qids = set()

    end_idx = min(start_idx + batch_size, len(raw_prompts))
    batch = raw_prompts[start_idx:end_idx]

    results = []
    processed = 0
    skipped = 0
    failed = 0
    shadow_reviews = 0

    for i, rec in enumerate(batch):
        prompt = rec.get("prompt", "").strip()
        source = rec.get("source", "unknown")
        if not prompt:
            skipped += 1
            continue

        qid = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if qid in skip_qids:
            skipped += 1
            print(f"  [{start_idx + i + 1}/{end_idx}] SKIP (already evaluated)")
            continue

        print(f"  [{start_idx + i + 1}/{end_idx}] evaluating... ", end="", flush=True)
        try:
            result = _eval_with_timeout(engine, prompt, API_TIMEOUT)
            source_class = rec.get("class", "unknown")

            # v2.1: extract safe_decision and shadow review flag
            safe_decision = result.get("decision", result.get("v4", {}).get("decision", "?"))
            needs_review = result.get("needs_shadow_review", False)
            if needs_review:
                shadow_reviews += 1

            result["source_class"] = source_class
            result["source"] = source
            result["safe_decision"] = safe_decision
            append_result(result, source, source_class, LIVE_LOG_PATH)

            s_v4 = result.get("v4", {}).get("S", 0.0)
            dec_v4 = result.get("v4", {}).get("decision", "?")
            dec_v1 = result.get("v1", {}).get("decision", "?")
            div = "DIVERGE" if result.get("divergence") else "ok"
            review_flag = " ⚠ SHADOW_REVIEW" if needs_review else ""
            print(f"S_v4={s_v4:.3f} {dec_v4} v1={dec_v1} [{div}]{review_flag}")

            results.append(result)
            skip_qids.add(qid)
            processed += 1

        except (FuturesTimeoutError, TimeoutError):
            print(f"TIMEOUT (>{API_TIMEOUT}s) — SKIP")
            skip_qids.add(qid)
            failed += 1
        except Exception as e:
            print(f"FAILED: {type(e).__name__} — SKIP")
            skip_qids.add(qid)
            failed += 1

        time.sleep(0.5)

    print(f"\n  Batch done: {processed} ok, {failed} skipped, {skipped} dup, {shadow_reviews} shadow_reviews")
    return results


def main():
    from survival import SurvivalEngine, SurvivalConfig, _classify_failure_mode, set_monitor_action
    from scripts.batch_monitor import MonitorState, scan, log_alerts, print_status, TIGHTENED_TAU_H
    from scripts.predict_failure import FailurePredictor

    # Config — use zhipu as primary provider
    cfg = SurvivalConfig(
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        zhipu_api_key=os.environ.get("ZHIPU_API_KEY", ""),
        provider="zhipu",
    )

    # Allow CLI override
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            pass  # process all
        elif arg.startswith("--batch="):
            try:
                start = int(arg.split("=")[1])
                # override start index
            except ValueError:
                print("Usage: python run_live_batch.py [--all] [--batch=N] [--dry-run]")
                sys.exit(1)

    engine = SurvivalEngine(cfg)
    monitor = MonitorState()  # v2.1.4: loads monitor_stats.json on init
    predictor = FailurePredictor()  # v2.2.0: failure probability predictor

    # v2.2.0: print predictor status
    predictor.print_status()

    # v2.1.4: print persistent stats on startup
    stats = monitor.persistent_stats
    for action_name, s in stats.items():
        eff, ineff = s.get("effective", 0), s.get("ineffective", 0)
        total = eff + ineff
        rate = ineff / total if total > 0 else 0
        suppressed = " [SUPPRESSED]" if monitor._is_suppressed(action_name) else ""
        if total > 0:
            print(f"  [MONITOR-STATS] {action_name}: {eff}e/{ineff}i (ineff_rate={rate:.0%}){suppressed}")
        else:
            print(f"  [MONITOR-STATS] {action_name}: no history yet")

    # Load prompts from the ONLY allowed source
    raw_prompts = load_raw_prompts(RAW_PROMPTS_PATH)

    # Get already-evaluated qids (skip re-evaluation)
    skip_qids = get_already_evaluated_qids(LIVE_LOG_PATH)
    print(f"Already evaluated: {len(skip_qids)} prompts will be skipped")

    # Dry run: just count what would be processed
    if "--dry-run" in sys.argv:
        pending = []
        for rec in raw_prompts:
            prompt = rec.get("prompt", "").strip()
            if not prompt:
                continue
            qid = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            if qid not in skip_qids:
                pending.append(rec)
        print(f"\nDRY RUN: {len(pending)} prompts pending evaluation")
        print(f"  Batches needed: {(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE}")
        return

    # Process in batches of 20
    pending = []
    for rec in raw_prompts:
        prompt = rec.get("prompt", "").strip()
        if not prompt:
            continue
        qid = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        if qid not in skip_qids:
            pending.append(rec)

    print(f"Pending: {len(pending)} prompts")
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Will process {n_batches} batch(es) of up to {BATCH_SIZE}")
    print(f"Policy: v4 = π_E (primary), v1 = π_S (audit)\n")

    all_results = []
    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pending))
        print(f"=== BATCH {b + 1}/{n_batches} (prompts {start + 1}-{end}) ===")
        batch_prompts = pending[start:end]
        batch_results = []
        shadow_reviews = 0
        for i, rec in enumerate(batch_prompts):
            prompt = rec.get("prompt", "").strip()
            source = rec.get("source", "unknown")
            qid = hashlib.sha256(prompt.encode()).hexdigest()[:12]

            print(f"  [{start + i + 1}/{end}] evaluating... ", end="", flush=True)
            try:
                # v2.1.2: pre-classify prompt for monitor action routing
                fm = _classify_failure_mode(prompt)
                action = monitor.get_action(fm)
                set_monitor_action(action)  # survival.py picks this up in log_disagreement

                result = _eval_with_timeout(engine, prompt, API_TIMEOUT)
                source_class = rec.get("class", "unknown")

                # v2.1: extract safe_decision and shadow review flag
                safe_decision = result.get("decision", result.get("v4", {}).get("decision", "?"))
                needs_review = result.get("needs_shadow_review", False)

                # v2.1.2: apply routing overrides (domain_knowledge-scoped, temporary)
                monitor_action = "none"
                if action == "forced_review" and fm == "domain_knowledge":
                    needs_review = True
                    safe_decision = "review"
                    monitor_action = "forced_review"
                elif action == "tightened_threshold" and fm == "domain_knowledge":
                    s_v4 = result.get("v4", {}).get("S", 0.0)
                    dec_v4 = result.get("v4", {}).get("decision", "")
                    if dec_v4 == "accept" and s_v4 < TIGHTENED_TAU_H:
                        needs_review = True
                        safe_decision = "review"
                        monitor_action = "tightened_threshold"

                result["monitor_action"] = monitor_action
                set_monitor_action("none")  # reset

                # v2.2.0: compute risk_score from prediction model
                risk = predictor.predict(result.get("v4", {}), result.get("v1", {}))
                result["risk_score"] = risk["risk_score"]
                result["risk_action"] = risk["action"]

                # v2.2.0: risk-based routing escalation
                if risk["action"] == "escalate" and not needs_review:
                    needs_review = True
                    safe_decision = "review"
                elif risk["action"] == "shadow_review" and not needs_review:
                    # shadow_review from predictor adds to review count but doesn't override decision
                    needs_review = True

                if needs_review:
                    shadow_reviews += 1

                result["source"] = source
                result["source_class"] = source_class
                result["safe_decision"] = safe_decision
                append_result(result, source, source_class, LIVE_LOG_PATH)

                s_v4 = result.get("v4", {}).get("S", 0.0)
                dec_v4 = result.get("v4", {}).get("decision", "?")
                dec_v1 = result.get("v1", {}).get("decision", "?")
                div = "DIVERGE" if result.get("divergence") else "ok"
                review_flag = " ⚠ SHADOW_REVIEW" if needs_review else ""
                action_flag = f" [MONITOR:{monitor_action}]" if monitor_action != "none" else ""
                risk_flag = f" risk={risk['risk_score']:.2f}" if risk['has_model'] else ""
                risk_action_flag = f" →{risk['action']}" if risk['action'] != 'none' else ""
                print(f"S_v4={s_v4:.3f} {dec_v4} v1={dec_v1} [{div}]{review_flag}{action_flag}{risk_flag}{risk_action_flag}")
                batch_results.append(result)
                # v2.1.3: feed sample signals for effectiveness tracking
                is_dk = (fm == "domain_knowledge")
                fact_risk = result.get("v4", {}).get("decision") == "accept" and is_dk and result.get("v4", {}).get("S", 0) > 0.70
                monitor.record_sample(is_dk, fact_risk)
                monitor.tick()  # v2.1.2: decrement action expiry counters
            except (FuturesTimeoutError, TimeoutError):
                print(f"TIMEOUT (>{API_TIMEOUT}s) — SKIP")
                skip_qids.add(qid)
            except Exception as e:
                print(f"FAILED: {type(e).__name__} — SKIP")
                skip_qids.add(qid)

            time.sleep(0.5)

        # End-of-batch metrics
        metrics = compute_batch_metrics(batch_results)
        # v2.2.0: add risk_score stats to batch metrics
        risk_scores = [r.get("risk_score", 0) for r in batch_results if r.get("risk_score") is not None]
        if risk_scores:
            metrics["risk_score_mean"] = round(float(np.mean(risk_scores)), 4)
            metrics["risk_score_max"] = round(float(np.max(risk_scores)), 4)
            metrics["risk_escalate_count"] = sum(1 for r in batch_results if r.get("risk_action") == "escalate")
            metrics["risk_review_count"] = sum(1 for r in batch_results if r.get("risk_action") == "shadow_review")
        metrics["batch_number"] = b + 1
        metrics["total_batches"] = n_batches
        metrics["pending_before_batch"] = len(pending) - start
        print(f"\n  Batch {b + 1} metrics: bad_accepted={metrics['bad_accepted']}  "
              f"good_rejected={metrics['good_rejected']}  "
              f"divergence_rate={metrics['divergence_rate']:.2%}  "
              f"shadow_reviews={shadow_reviews}  "
              f"S_mean={metrics['s_mean']:.3f}")

        append_metrics(metrics, METRICS_PATH)
        all_results.extend(batch_results)

        # v2.1.3: streaming monitor + action hooks + effectiveness tracking
        try:
            disc_path = os.path.join(DIR, "logs", "disagreement_cases.jsonl")
            disc_cases = []
            with open(disc_path) as df:
                for line in df:
                    line = line.strip()
                    if line:
                        disc_cases.append(json.loads(line))
            alerts, pre_metrics = scan(disc_cases)
            print_status(disc_cases)
            for a in alerts:
                print(f"  {a}")
            log_alerts(alerts)
            monitor.update(
                alerts,
                pre_false_accept=pre_metrics["false_accept_count"],
                pre_risk_spike=pre_metrics["risk_spike_count"],
            )
            if monitor.active_actions:
                print(f"  [MONITOR] active_actions={monitor.active_actions}")
            if monitor.suppressed_events:
                for sev in monitor.suppressed_events:
                    # Already printed by _log_suppression, just note count
                    pass
                print(f"  [MONITOR] suppressions this run: {len(monitor.suppressed_events)}")
        except Exception as e:
            print(f"  (monitor skipped: {e})")

        print(f"  (metrics appended to {METRICS_PATH})\n")

        # v2.2.0: periodic retrain every 5 batches (close the loop)
        if (b + 1) % 5 == 0 and predictor.is_loaded:
            print("  [PREDICTOR] Periodic retrain check (every 5 batches)...")
            try:
                predictor.retrain()
            except Exception as e:
                print(f"  [PREDICTOR] Retrain skipped: {e}")

    # Summary
    print("=" * 55)
    print("  v2.2 PIPELINE COMPLETE — v4 = π_E, v1 = π_S (audit) + predictor")
    print("=" * 55)
    final = compute_batch_metrics(all_results)
    total_live = sum(1 for _ in open(LIVE_LOG_PATH)) if os.path.exists(LIVE_LOG_PATH) else 0
    print(f"  Total records in live log: {total_live}")
    if final['batch_size'] > 0:
        print(f"  bad_accepted:         {final['bad_accepted']}")
        print(f"  good_rejected:        {final['good_rejected']}")
        print(f"  divergence_rate:      {final['divergence_rate']:.2%}")
        print(f"  shadow_reviews:       {final['shadow_review_count']}")
        print(f"  S_mean:               {final['s_mean']:.3f}")
        print(f"  S_range:              [{final['s_min']:.3f}, {final['s_max']:.3f}]")
    # v2.2.0: risk_score summary
    all_risk = [r.get("risk_score") for r in all_results if r.get("risk_score") is not None]
    if all_risk:
        print(f"  risk_score_mean:      {np.mean(all_risk):.3f}")
        print(f"  risk_score_max:       {np.max(all_risk):.3f}")
        esc = sum(1 for r in all_results if r.get("risk_action") == "escalate")
        rev = sum(1 for r in all_results if r.get("risk_action") == "shadow_review")
        print(f"  risk_escalated:       {esc}")
        print(f"  risk_shadow_reviewed: {rev}")
    print(f"\n  Live log:      {LIVE_LOG_PATH}")
    print(f"  Metrics:       {METRICS_PATH}")
    print(f"  Disagreements: {os.path.join(DIR, 'logs', 'disagreement_cases.jsonl')}")
    print(f"  Failure data:  {os.path.join(DIR, 'logs', 'failure_dataset.jsonl')}")
    print(f"  Model:         {os.path.join(DIR, 'model', 'failure_predictor.pkl')}")
    print(f"  Raw source:    {RAW_PROMPTS_PATH}")


if __name__ == "__main__":
    main()

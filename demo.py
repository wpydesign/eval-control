#!/usr/bin/env python3
"""
demo.py — Full pipeline demo: decision → shadow → outcome → audit

Run:
    python demo.py

This simulates the complete workflow:
    1. Submit a deployment decision for shadow evaluation
    2. See pi_E (baseline) vs pi_S (risk engine) decision
    3. Log a real-world outcome
    4. Query the audit trail
"""

import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from shadow_mode import run_pi_S, log_entry
from outcome_capture import log_outcome, read_outcomes


def main():
    print("=" * 60)
    print("  Risk Audit Engine — Full Pipeline Demo")
    print("=" * 60)

    # ─────────────────────────────────────────────
    # STEP 1: Submit a deployment decision
    # ─────────────────────────────────────────────
    print("\n── STEP 1: Shadow Evaluation ──\n")

    decision = {
        "case_id": "DEMO-001",
        "context": "Production model upgrade: new LLM for customer support chatbot. "
                   "Eval shows 2.1% improvement on benchmark suite.",
        "eval_scores": {"gpt-4o": 0.847, "gpt-4o-mini": 0.868},
        "pi_E": "gpt-4o-mini",
        "metadata": {
            "domain": "prod",
            "estimated_cost_if_wrong": 200000,
            "reversibility": "moderate",
            "latency_to_detect": "days",
            "blast_radius": "customer-facing chatbot",
            "consequence_type": "error_cost",
            "distribution": "normal",
            "variance": 0.09,
        },
    }

    result = run_pi_S(decision)
    log_entry(result)  # persist to shadow_log.jsonl

    print(f"  Case:      {result['case_id']}")
    print(f"  Context:   {decision['context'][:60]}...")
    print(f"  Eval gap:  {0.868 - 0.847:.1%} ({decision['eval_scores']})")
    print(f"  Cost:      ${decision['metadata']['estimated_cost_if_wrong']:,}/yr if wrong")
    print()
    print(f"  pi_E:      {result['pi_E_raw']} -> {result['pi_E_decision']}")
    print(f"  pi_S:      {result['pi_S']}")
    print(f"  Diverge:   {'YES' if result['divergence'] else 'no'}")
    print()
    print(f"  Effective: ${result['risk']['effective_score']:,.0f}")
    print(f"  Margin:    ${result['risk']['margin']:,.0f}")
    print(f"  E[C-]:     ${result['risk']['E_C_neg']:,.0f}")
    print(f"  Tail risk: ${result['risk']['tail_risk']:,.0f}")
    print(f"  R(x):      ${result['risk']['R_x']:,.0f}")
    print(f"  Cost sign: {result['risk']['cost_sign']}")
    print(f"  Tension:   {result['shadow']['tension_type']}")

    if result["divergence"]:
        print(f"\n  *** SYSTEM BLOCKS WHAT BASELINE WOULD DEPLOY ***")

    # ─────────────────────────────────────────────
    # STEP 2: Submit a second decision (different domain)
    # ─────────────────────────────────────────────
    print("\n── STEP 2: Low-stakes Internal Decision ──\n")

    decision_2 = {
        "case_id": "DEMO-002",
        "context": "Internal tool: upgrade summarization model for meeting notes.",
        "eval_scores": {"claude-3-haiku": 0.82, "claude-3-sonnet": 0.891},
        "pi_E": "claude-3-sonnet",
        "metadata": {
            "domain": "internal",
            "estimated_cost_if_wrong": 25000,
            "reversibility": "easy",
            "latency_to_detect": "hours",
            "blast_radius": "internal tool only",
            "consequence_type": "forfeited_productivity_gain",
            "distribution": "normal",
            "variance": 0.04,
        },
    }

    result_2 = run_pi_S(decision_2)
    log_entry(result_2)  # persist to shadow_log.jsonl

    print(f"  Case:      {result_2['case_id']}")
    print(f"  Domain:    {decision_2['metadata']['domain']}")
    print(f"  Cost:      ${decision_2['metadata']['estimated_cost_if_wrong']:,}/yr if wrong")
    print()
    print(f"  pi_E:      {result_2['pi_E_raw']} -> {result_2['pi_E_decision']}")
    print(f"  pi_S:      {result_2['pi_S']}")
    print(f"  Effective: ${result_2['risk']['effective_score']:,.0f}")
    print(f"  Margin:    ${result_2['risk']['margin']:,.0f}")
    print(f"  Tension:   {result_2['shadow']['tension_type']}")

    # ─────────────────────────────────────────────
    # STEP 3: Log a real-world outcome
    # ─────────────────────────────────────────────
    print("\n── STEP 3: Log Real-World Outcome ──\n")

    # Simulate: DEMO-001 was deployed, had issues
    outcome_record = log_outcome("DEMO-001", {
        "realized": "mixed",
        "cost_actual": 145000,
        "notes": "Deployed gpt-4o-mini. Minor regression in edge cases after 3 weeks. "
                 "Rollback considered but not executed. Actual cost lower than estimated.",
    })

    print(f"  Case:           {outcome_record['case_id']}")
    print(f"  Decision was:   pi_S={outcome_record['pi_S']}")
    print(f"  Alignment:      {outcome_record['decision_alignment']}")
    print(f"  Fault probe:    {outcome_record['fault_probe']}")
    print(f"  Realized:       {outcome_record['outcome']['realized']}")
    print(f"  Cost estimated: ${outcome_record['cost_estimated']:,}" if outcome_record['cost_estimated'] else "  Cost estimated: N/A")
    print(f"  Cost actual:    ${outcome_record['outcome']['cost_actual']:,}" if outcome_record['outcome']['cost_actual'] else "  Cost actual:    N/A")

    est = outcome_record.get("cost_estimated")
    act = outcome_record["outcome"].get("cost_actual")
    if est is not None and act is not None:
        cal = act - est
        print(f"  Calibration:    {cal:+,.0f}")

    # ─────────────────────────────────────────────
    # STEP 4: Query audit trail
    # ─────────────────────────────────────────────
    print("\n── STEP 4: Audit Trail ──\n")

    outcomes = read_outcomes()
    print(f"  Total outcomes logged: {len(outcomes)}")
    for o in outcomes:
        fp = f" [{o['fault_probe']}]" if o['fault_probe'] != "none" else ""
        print(f"    {o['case_id']}{fp}: pi_S={o['pi_S']} -> {o['outcome']['realized']}")

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete. Decision -> Shadow -> Outcome -> Audit")
    print("=" * 60)
    print()
    print("  What happened:")
    print(f"    DEMO-001 (prod, $200K): pi_S={result['pi_S']}, outcome={outcome_record['outcome']['realized']}")
    print(f"    DEMO-002 (internal, $25K): pi_S={result_2['pi_S']}")
    print()
    print("  All data logged to:")
    print(f"    shadow_log.jsonl  (shadow evaluations)")
    print(f"    outcomes.jsonl    (ground truth outcomes)")
    print()


if __name__ == "__main__":
    main()

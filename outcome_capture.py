#!/usr/bin/env python3
"""
outcome_capture.py — Black Box Flight Recorder (v1.0)

PURPOSE:
    Attach ground truth to decisions. Lose nothing.

    This is NOT analytics. NOT a dashboard. NOT a feedback loop.
    It is a way to not lose signal when reality finally speaks.

RULES:
    - Append-only JSONL
    - Never overwrite original estimates
    - Capture imperfect outcomes immediately (don't wait for perfect data)
    - Tag fault probes explicitly
    - No automatic scoring
    - No feedback loops
    - No aggregation (compute later from raw records)

SCHEMA (one record per outcome):
    {
        "case_id": "...",
        "timestamp_decision": "...",       // from shadow_log
        "timestamp_outcome": "...",        // now

        "pi_E": "...",                     // what current practice did
        "pi_S": "ALLOW|BLOCK",             // what v4.3 said

        "effective": ...,                   // frozen — never modified
        "margin": ...,                      // frozen — never modified

        "domain": "...",
        "cost_estimated": ...,              // frozen — never modified
        "R": ...,                           // frozen — never modified

        "fault_probe": "none|FP1|FP2|FP3",  // tagged at log time

        "outcome": {
            "realized": "success|failure|mixed|unknown",
            "cost_actual": ...,             // null is OK — log what you have
            "notes": "free text"
        },

        // Full original shadow entry preserved for reconstruction
        "_original_shadow": {...}
    }

FAULT PROBES:
    FP1 = COST-100K   (prod, ~$100K, moderate, boundary ALLOW)
    FP2 = BP-004      (prod, $125K, moderate, closest ALLOW — $9.5K from kappa)
    FP3 = BP-082      (safety, $75K, moderate — first safety ALLOW with downside)

USAGE:
    from outcome_capture import log_outcome, read_outcomes

    # Log a real outcome
    log_outcome("REAL-023", {
        "realized": "success",
        "cost_actual": null,
        "notes": "Deployed. No issues after 2 weeks."
    })

    # Log with explicit fault probe tag
    log_outcome("BP-003", {
        "realized": "failure",
        "cost_actual": 180000,
        "notes": "Rollback required. Actual cost higher than estimated.",
        "fault_probe_override": "FP1"  // optional: override auto-tagging
    })

    # Read outcomes
    outcomes = read_outcomes()
"""

import json
import os
import sys
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.abspath(__file__))

OUTCOME_FILE = os.path.join(DIR, "outcomes.jsonl")
SHADOW_LOG = os.path.join(DIR, "shadow_log.jsonl")

# Fault probe definitions — exact match or fuzzy
FAULT_PROBES = {
    "FP1": {
        "case_ids": {"COST-100K"},
        "signature": {"domain": "prod", "cost_band": (75000, 150000), "reversibility": "moderate"},
        "description": "prod, ~$100K, moderate — kappa calibration vs small downside",
    },
    "FP2": {
        "case_ids": {"BP-004"},
        "signature": {"domain": "prod", "cost_band": (100000, 175000), "reversibility": "moderate"},
        "description": "prod, $125K, moderate — closest ALLOW ($9.5K from kappa)",
    },
    "FP3": {
        "case_ids": {"BP-082"},
        "signature": {"domain": "safety", "cost_band": (50000, 125000)},
        "description": "safety, $75K, moderate — first safety ALLOW with downside cost",
    },
}


# ═══════════════════════════════════════════════════════════════
# LOOKUP: Find original decision in shadow_log
# ═══════════════════════════════════════════════════════════════

def _load_shadow_index():
    """
    Build case_id → shadow_entry index from shadow_log.jsonl.
    Caches on first call.
    """
    if not hasattr(_load_shadow_index, "_cache"):
        _load_shadow_index._cache = {}
        if not os.path.exists(SHADOW_LOG):
            return _load_shadow_index._cache
        with open(SHADOW_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        cid = entry.get("case_id")
                        if cid:
                            # Keep first occurrence (original decision, not replays)
                            if cid not in _load_shadow_index._cache:
                                _load_shadow_index._cache[cid] = entry
                    except json.JSONDecodeError:
                        pass
    return _load_shadow_index._cache


def _find_shadow_entry(case_id):
    """Look up original shadow log entry by case_id."""
    index = _load_shadow_index()
    return index.get(case_id)


# ═══════════════════════════════════════════════════════════════
# FAULT PROBE TAGGING
# ═══════════════════════════════════════════════════════════════

def _tag_fault_probe(case_id, shadow_entry):
    """
    Determine if a case matches a fault probe.

    Priority:
        1. Exact case_id match
        2. Signature match (domain + cost band)
    """
    # Exact match
    for fp_id, fp_def in FAULT_PROBES.items():
        if case_id in fp_def["case_ids"]:
            return fp_id

    # Signature match (only if shadow data available)
    if shadow_entry:
        raw = shadow_entry.get("raw_inputs", {})
        meta = raw.get("metadata", {})
        domain = meta.get("domain", "").lower()
        cost = meta.get("estimated_cost_if_wrong")
        reversibility = meta.get("reversibility", "").lower()

        if cost is not None:
            for fp_id, fp_def in FAULT_PROBES.items():
                sig = fp_def["signature"]
                if domain == sig.get("domain", ""):
                    lo, hi = sig["cost_band"]
                    if lo <= cost <= hi:
                        # FP1/FP2 also check reversibility
                        if "reversibility" in sig:
                            if reversibility == sig["reversibility"]:
                                return fp_id
                        else:
                            return fp_id

    return "none"


# ═══════════════════════════════════════════════════════════════
# CORE: Log an outcome
# ═══════════════════════════════════════════════════════════════

def log_outcome(case_id, outcome_dict, log_path=None):
    """
    Attach a real-world outcome to a previous decision.

    Args:
        case_id: The case_id from shadow_log (or any known case)
        outcome_dict: {
            "realized": "success|failure|mixed|unknown",  // required
            "cost_actual": <number|null>,                   // null = don't know yet
            "notes": "free text",                           // required — why/how
            "fault_probe_override": "none|FP1|FP2|FP3"     // optional: force tag
        }
        log_path: Override output file path (for testing)

    Returns:
        The complete outcome record that was written.
    """
    # Find original shadow entry
    shadow = _find_shadow_entry(case_id)

    # Extract decision context (what was decided)
    if shadow:
        timestamp_decision = shadow.get("timestamp")
        pi_e = shadow.get("pi_E_raw", shadow.get("pi_E", "unknown"))
        pi_e_decision = shadow.get("pi_E_decision",
                                   "BLOCK" if shadow.get("divergence", False) else "ALLOW")
        pi_s = shadow.get("pi_S", "unknown")

        risk = shadow.get("risk", {})
        effective = risk.get("effective_score")
        margin = risk.get("margin")

        raw = shadow.get("raw_inputs", {})
        meta = raw.get("metadata", {})
        domain = meta.get("domain", "unknown")
        cost_estimated = meta.get("estimated_cost_if_wrong")
        r_x = risk.get("R_x")
    else:
        # No shadow entry — minimal record from what we have
        timestamp_decision = None
        pi_e = "unknown"
        pi_e_decision = "unknown"
        pi_s = "unknown"
        effective = None
        margin = None
        domain = "unknown"
        cost_estimated = None
        r_x = None

    # Decision alignment: did π_S agree or diverge from π_E?
    if pi_s != "unknown" and pi_e_decision != "unknown":
        decision_alignment = "aligned" if pi_s == pi_e_decision else "diverged"
    else:
        decision_alignment = "unknown"

    # Fault probe tag
    override = outcome_dict.get("fault_probe_override")
    if override and override in FAULT_PROBES:
        fault_probe = override
    else:
        fault_probe = _tag_fault_probe(case_id, shadow)

    # Build outcome record — frozen estimates + new outcome
    record = {
        "case_id": case_id,
        "timestamp_decision": timestamp_decision,
        "timestamp_outcome": datetime.now(timezone.utc).isoformat(),

        "pi_E": pi_e,
        "pi_S": pi_s,

        "effective": effective,
        "margin": margin,

        "domain": domain,
        "cost_estimated": cost_estimated,
        "R": r_x,

        "decision_alignment": decision_alignment,
        "fault_probe": fault_probe,

        "outcome": {
            "realized": outcome_dict.get("realized", "unknown"),
            "cost_actual": outcome_dict.get("cost_actual"),
            "notes": outcome_dict.get("notes", ""),
        },

        "_original_shadow": shadow,
    }

    # Append to JSONL
    path = log_path or OUTCOME_FILE
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    return record


# ═══════════════════════════════════════════════════════════════
# READ: Retrieve outcomes
# ═══════════════════════════════════════════════════════════════

def read_outcomes(log_path=None, fault_probe=None):
    """
    Read outcome records from JSONL.

    Args:
        log_path: Override file path
        fault_probe: Filter by fault probe tag ("FP1", "FP2", "FP3", "none")

    Returns:
        List of outcome records
    """
    path = log_path or OUTCOME_FILE
    records = []
    if not os.path.exists(path):
        return records

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if fault_probe is None or rec.get("fault_probe") == fault_probe:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass

    return records


def read_fault_probes():
    """Read only fault probe outcomes (FP1, FP2, FP3)."""
    results = {}
    for fp_id in FAULT_PROBES:
        results[fp_id] = read_outcomes(fault_probe=fp_id)
    return results


# ═══════════════════════════════════════════════════════════════
# CLI: Quick outcome logging
# ═══════════════════════════════════════════════════════════════

def _cli():
    """Command-line interface for logging outcomes."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Outcome capture — black box flight recorder for eval control decisions"
    )
    sub = parser.add_subparsers(dest="cmd")

    # log subcommand
    log_parser = sub.add_parser("log", help="Log an outcome for a case")
    log_parser.add_argument("case_id", help="Case ID from shadow_log")
    log_parser.add_argument("--realized", required=True,
                            choices=["success", "failure", "mixed", "unknown", "no_event"],
                            help="What happened")
    log_parser.add_argument("--cost", type=float, default=None,
                            help="Actual cost in USD (null if unknown)")
    log_parser.add_argument("--notes", required=True, help="Free-text description")
    log_parser.add_argument("--fault-probe", default=None,
                            choices=["FP1", "FP2", "FP3"],
                            help="Override fault probe tag")

    # show subcommand
    show_parser = sub.add_parser("show", help="Show recorded outcomes")
    show_parser.add_argument("--fault-probe", default=None,
                             choices=["FP1", "FP2", "FP3"],
                             help="Filter by fault probe")

    args = parser.parse_args()

    if args.cmd == "log":
        outcome = {
            "realized": args.realized,
            "cost_actual": args.cost,
            "notes": args.notes,
        }
        if args.fault_probe:
            outcome["fault_probe_override"] = args.fault_probe

        record = log_outcome(args.case_id, outcome)
        print(f"  Logged outcome for {args.case_id}")
        print(f"    realized: {record['outcome']['realized']}")
        print(f"    pi_S was: {record['pi_S']}")
        print(f"    fault_probe: {record['fault_probe']}")
        print(f"    estimated cost: {record['cost_estimated']}")
        print(f"    actual cost:   {record['outcome']['cost_actual']}")

        # Calibration error if both costs available
        est = record.get("cost_estimated")
        act = record["outcome"].get("cost_actual")
        if est is not None and act is not None:
            error = act - est
            print(f"    calibration error: {error:+,.0f}")

    elif args.cmd == "show":
        records = read_outcomes(fault_probe=args.fault_probe)
        if not records:
            print("  No outcomes recorded yet.")
            return

        print(f"\n  Outcome Log ({len(records)} records)")
        print(f"  {'='*70}")
        for r in records:
            fp_tag = f" [{r['fault_probe']}]" if r['fault_probe'] != "none" else ""
            cal = ""
            est = r.get("cost_estimated")
            act = r["outcome"].get("cost_actual")
            if est is not None and act is not None:
                cal = f"  cal_error={act - est:+,.0f}"
            print(f"  {r['case_id']}{fp_tag}  "
                  f"pi_S={r['pi_S']}  →  {r['outcome']['realized']}"
                  f"{cal}")
            if r["outcome"]["notes"]:
                print(f"    notes: {r['outcome']['notes'][:100]}")
            print(f"    decision: {r['timestamp_decision']}")
            print(f"    outcome:  {r['timestamp_outcome']}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()

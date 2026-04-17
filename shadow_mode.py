#!/usr/bin/env python3
"""
shadow_mode.py — Passive Shadow Deployment Sensor (v4.3)

PURPOSE:
    Observe how π_S (frozen v4.3 scalar policy) behaves when exposed
    to uncontrolled reality. This is a SENSOR, not a controller.

    No decisions are enforced. No parameters are tuned. No retries.
    No smoothing. No adjustments.

PHASE 1 ONLY:
    - Feed real evaluation scenarios
    - Compute π_S alongside π_E (current practice)
    - Log divergence: every case where π_S ≠ π_E
    - Log raw inputs (not just outputs) for future debugging

INPUT SCHEMA (real decision contexts):
    {
        "case_id": "...",
        "context": "free-text description of the decision",
        "candidates": ["A", "B"],
        "eval_scores": {"A": 0.85, "B": 0.91},
        "pi_E": "B",                    // what current practice would do
        "metadata": {
            "domain": "prod|internal|creative|safety",
            "estimated_cost_if_wrong": 500000,    // annual USD, optional
            "reversibility": "easy|moderate|hard|impossible",  // optional
            "latency_to_detect": "hours|days|weeks|months",   // optional
            "blast_radius": "description",                     // optional
            "distribution": "deterministic|normal|heavy_tailed", // optional, default "normal"
            "variance": 0.16,                                  // optional, default 0.16
            "consequence_type": "error_cost|safety_incident_risk|...", // optional
            "notes": "..."                                              // optional
        }
    }

WHAT GETS LOGGED (per case):
    - case_id, timestamp
    - π_E (current practice decision)
    - π_S (system policy decision)
    - divergence flag (π_S ≠ π_E)
    - effective_score, margin (κ - effective)
    - CVaR, R(x), E[C⁻], cost_sign
    - shadow constraint signals (Type A/B/C/none)
    - RAW INPUT (full original case + computed intermediates)

LOG FORMAT: JSONL (one JSON object per line)
    - Append-only
    - Human-readable
    - Easy to parse for analytics later

RULES:
    - No retries
    - No smoothing
    - No parameter adjustment
    - No enforcement of π_S decisions
    - Log everything, decide nothing

Run:
    python shadow_mode.py                         # interactive REPL
    python shadow_mode.py --file cases.jsonl       # batch mode
    python shadow_mode.py --dry-run                # test with RDR cases
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from regression_dataset import (
    compute_risk_score,
    shadow_constraints,
    KAPPA,
    KAPPA_HARD,
    LAMBDA_RISK,
    GAMMA_IRREVERSIBILITY,
    BETA_OPPORTUNITY,
    ALPHA_CVAR,
    UPSIDE_TYPES,
    IRREVERSIBILITY_PRESETS,
    _classify_cost,
    _get_C_wrong,
    _get_downstream_pair,
    pi_E as pi_E_rdr,
    CASES,
)

LOG_FILE = os.path.join(DIR, "shadow_log.jsonl")

# ═══════════════════════════════════════════════════════════════
# CORE: Convert real-world case → RDRD-compatible case dict
# ═══════════════════════════════════════════════════════════════
# Real-world cases won't have the full RDRD structure (y_i, C_i,
# environment_model, etc.). We synthesize what we can and mark gaps.

# Default environment model when metadata is sparse
_DEFAULT_ENV = {
    "distribution": "normal",
    "description": "Default: normal distribution, moderate uncertainty",
    "variance": 0.16,
    "variance_rationale": "Default assumption: no production baseline",
    "sensitivity": "medium",
}

# Map reversibility text → irreversibility preset
_REVERSIBILITY_MAP = {
    "easy": "very_low",
    "trivial": "very_low",
    "moderate": "medium",
    "hard": "high",
    "impossible": "unquantifiable",
    "irreversible": "unquantifiable",
}


def _build_case_from_real(raw):
    """
    Convert a real-world decision context into an RDRD-compatible case dict.

    Handles missing data gracefully. Marks synthesized fields so
    downstream analysis knows what was real vs inferred.

    CRITICAL: This does NOT fabricate ground truth (y_i). Real cases
    have no ground truth at this stage — that's what Phase 2 is for.
    """
    meta = raw.get("metadata", {})
    candidates = raw.get("candidates", [])
    eval_scores = raw.get("eval_scores", {})

    if not eval_scores:
        raise ValueError(f"case_id={raw.get('case_id', '?')}: eval_scores required")

    # Ensure eval_scores uses candidate names
    if not eval_scores and candidates:
        raise ValueError(f"case_id={raw.get('case_id', '?')}: eval_scores or candidate scores required")

    # Environment model
    dist = meta.get("distribution", "normal")
    variance = meta.get("variance", 0.16)
    env_model = {
        "distribution": dist,
        "description": f"Real case: distribution={dist}, variance={variance}",
        "variance": variance,
        "variance_rationale": meta.get("notes", "Inferred from real case metadata"),
        "sensitivity": "medium",
    }

    # Irreversibility model
    rev_text = meta.get("reversibility", "moderate")
    rev_tier = _REVERSIBILITY_MAP.get(rev_text.lower(), "medium")
    irr_preset = IRREVERSIBILITY_PRESETS.get(rev_tier, IRREVERSIBILITY_PRESETS["medium"])

    latency_hours = _parse_latency(meta.get("latency_to_detect", "days"))
    blast = meta.get("blast_radius", "unknown")

    irr_model = {
        "tier": rev_tier,
        "description": irr_preset["description"],
        "R_x": irr_preset["R_x"],
        "R_x_components": {
            "base_preset": irr_preset["R_x"],
            "burn_component": 0,
            "total": irr_preset["R_x"],
        },
        "rollback_cost_annual": None,
        "time_to_detect_hours": latency_hours,
        "blast_radius": blast,
        "examples": irr_preset["examples"],
        "source": "real_case_metadata",
    }

    # Cost model — synthesize from metadata
    consequence_type = meta.get("consequence_type", "error_cost")
    cost_if_wrong = meta.get("estimated_cost_if_wrong", None)

    # Determine eval winner (what current practice would deploy)
    eval_winner = max(eval_scores, key=eval_scores.get)
    pi_e_decision = raw.get("pi_E", eval_winner)

    # Build synthetic C_i
    # For real cases: we don't know ground truth, so C_i is one-sided.
    # The cost is "what happens if the eval winner is wrong."
    C_i = {}
    if cost_if_wrong is not None:
        deploy_key = f"deploy_{eval_winner}"
        other = [c for c in eval_scores if c != eval_winner]
        if other:
            other_key = f"deploy_{other[0]}"
        else:
            other_key = "deploy_baseline"

        C_i[f"C({deploy_key}, y=wrong)"] = {
            "calculation": f"estimated_cost_if_wrong from metadata: ${cost_if_wrong:,.0f}/year",
            "annual_usd": cost_if_wrong,
            "consequence_type": consequence_type,
            "source": "real_case_metadata",
        }
        C_i[f"C({other_key}, y=wrong)"] = {
            "calculation": "baseline — alternative deployment",
            "annual_usd": 0,
            "consequence_type": "baseline",
            "source": "real_case_metadata",
        }

    # Build downstream_values proxy from eval_scores
    # WHY: Real cases don't have ground truth (y_i). But the risk model needs
    # downstream values to compute p_reversal = Phi(-gap / std_gap).
    # Using eval_scores as proxy means:
    #   - The eval winner is assumed correct in expectation (best-case assumption)
    #   - The eval score GAP becomes the uncertainty signal:
    #     small gap → high p_reversal → higher E[C⁻] → more conservative
    #   - This is NOT claiming eval scores = downstream performance.
    #     It's using eval score separation as a proxy for confidence.
    # Phase 2 will replace these with actual downstream outcomes.
    candidates = list(eval_scores.keys())
    downstream_values_proxy = {c: eval_scores[c] for c in candidates}

    # Build case dict compatible with compute_risk_score()
    case = {
        "id": raw.get("case_id", f"SHADOW-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"),
        "x_i": {
            "features": {
                "category": "real_shadow",
                "decision_context": raw.get("context", "Real decision context"),
                "evaluation_method": "real_eval",
                "task_type": meta.get("domain", "generic"),
                "n_questions": None,
                "extraction_fail_rate": None,
            },
            "eval_scores": eval_scores,
            "bssi_components": {
                "S": None,  # Cannot compute without full BSSI pipeline
                "A": None,
                "N": None,
                "BSSI": None,
            },
        },
        "y_i": {
            "optimal_model": eval_winner,  # Assume eval winner is correct (best-case)
            "downstream_metric": "proxy_from_eval_scores",
            "downstream_values": downstream_values_proxy,
            "explanation": "Real case: no ground truth. Using eval_scores as downstream proxy. "
                          "Eval score gap drives p_reversal. Phase 2 replaces with actual outcomes.",
            "source": "real_case_eval_proxy",
        },
        "C_i": C_i,
        "environment_model": env_model,
        "action_space": {
            "A_i": [f"deploy_{c}" for c in eval_scores.keys()],
            "source": "eval_scores keys",
        },
        "optimal_constrained": f"deploy_{eval_winner}",
        "irreversibility_model": irr_model,
        "_real_case": True,  # Tag: this is a real-world case, not RDRD
        "_raw_input": raw,  # Preserve original input for debugging
    }

    return case


def _parse_latency(latency_str):
    """Convert latency text to hours."""
    if isinstance(latency_str, (int, float)):
        return float(latency_str)
    s = str(latency_str).lower()
    if "hour" in s:
        return 4.0
    elif "day" in s:
        return 24.0
    elif "week" in s:
        return 168.0
    elif "month" in s:
        return 720.0
    elif "immediate" in s:
        return 1.0
    return 24.0  # default: days


# ═══════════════════════════════════════════════════════════════
# CORE: Run π_S on a real case
# ═══════════════════════════════════════════════════════════════

def run_pi_S(raw_case):
    """
    Run the frozen v4.3 scalar policy on a real-world case.

    Args:
        raw_case: dict with minimum fields:
            case_id, candidates/eval_scores, pi_E, metadata

    Returns:
        dict with:
            case_id, pi_E, pi_S, divergence,
            effective, margin, cvar, R_x,
            E_C_neg, cost_sign, risk_type,
            shadow (shadow constraint signals),
            raw_inputs (preserved for debugging)
    """
    # Build RDRD-compatible case
    case = _build_case_from_real(raw_case)
    case_id = case["id"]

    # Get π_E (what current practice does)
    # π_E is either:
    #   - A candidate name ("v2.2", "model_b") → implies ALLOW/DEPLOY
    #   - An explicit decision ("ALLOW", "BLOCK", "deploy_v2.2")
    #   - A "don't deploy" signal ("BLOCK", "hold", "no")
    pi_e_raw = raw_case.get("pi_E", max(raw_case.get("eval_scores", {}).keys(),
                                         key=lambda k: raw_case["eval_scores"][k]))

    # Normalize π_E to a decision
    _BLOCK_SIGNALS = {"block", "hold", "no", "reject", "don't deploy", "skip", "wait"}
    if str(pi_e_raw).lower() in _BLOCK_SIGNALS:
        pi_e_decision = "BLOCK"
    else:
        # Any candidate name or "ALLOW"/"deploy_*" → current practice would deploy
        pi_e_decision = "ALLOW"

    # Compute v4.3 risk score
    risk = compute_risk_score(case)

    # Compute π_S
    effective = risk.get("effective_score", 0)
    eff_is_inf = (effective == float('inf') or effective is None)
    pi_s = "BLOCK" if eff_is_inf or effective > KAPPA else "ALLOW"
    margin = KAPPA - effective if not eff_is_inf else float('-inf')

    # Shadow constraints
    shadow = shadow_constraints(case, risk)

    # Divergence: π_E's implicit decision vs π_S's explicit decision
    divergence = (pi_s != pi_e_decision)

    # Build log entry
    entry = {
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pi_E_raw": pi_e_raw,
        "pi_E_decision": pi_e_decision,
        "pi_S": pi_s,
        "divergence": divergence,
        "risk": {
            "effective_score": _fmt_num(effective),
            "margin": _fmt_num(margin),
            "E_C_neg": _fmt_num(risk.get("E_C_neg")),
            "E_C_pos": _fmt_num(risk.get("E_C_pos")),
            "std_C_neg": _fmt_num(risk.get("std_C_neg")),
            "tail_risk": _fmt_num(risk.get("tail_risk")),
            "cvar_factor": risk.get("cvar_factor"),
            "R_x": _fmt_num(risk.get("R_x")),
            "gamma_R": _fmt_num(risk.get("gamma_R")),
            "p_reversal": risk.get("p_reversal"),
            "cost_sign": risk.get("cost_sign"),
            "consequence_type": risk.get("consequence_type"),
            "risk_type": risk.get("risk_type"),
        },
        "shadow": {
            "cvar_ratio": shadow.get("cvar_ratio"),
            "cvar_margin": shadow.get("cvar_margin"),
            "irrev_ratio": shadow.get("irrev_ratio"),
            "irrev_flag": shadow.get("irrev_flag"),
            "composite_energy": shadow.get("composite_energy"),
            "tension_type": shadow.get("tension_type"),
            "pressure_ratio": shadow.get("pressure_ratio"),
        },
        "raw_inputs": {
            "context": raw_case.get("context"),
            "eval_scores": raw_case.get("eval_scores"),
            "metadata": raw_case.get("metadata"),
            "pi_E_provided": pi_e_raw,
        },
    }

    return entry


def _fmt_num(val):
    """Format number for JSON logging."""
    if val is None:
        return None
    if isinstance(val, float) and (val == float('inf') or val != val):
        return str(val)
    if isinstance(val, float):
        return round(val, 2)
    return val


# ═══════════════════════════════════════════════════════════════
# LOG STORE: JSONL append-only
# ═══════════════════════════════════════════════════════════════

def log_entry(entry, log_path=None):
    """Append a single log entry to JSONL file."""
    path = log_path or LOG_FILE
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_log(log_path=None, limit=None):
    """Read log entries from JSONL file."""
    path = log_path or LOG_FILE
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ═══════════════════════════════════════════════════════════════
# BATCH MODE: Process file of cases
# ═══════════════════════════════════════════════════════════════

def process_file(filepath, log_path=None):
    """Process a JSONL/JSON file of real cases in batch."""
    cases = []
    with open(filepath, "r") as f:
        content = f.read().strip()
        if content.startswith("["):
            cases = json.loads(content)
        else:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    results = []
    for raw in cases:
        try:
            entry = run_pi_S(raw)
            log_entry(entry, log_path)
            results.append(entry)
        except Exception as e:
            print(f"  ERROR processing {raw.get('case_id', '?')}: {e}")
            results.append({"case_id": raw.get("case_id", "?"), "error": str(e)})

    return results


# ═══════════════════════════════════════════════════════════════
# DRY-RUN: Replay existing RDR cases through shadow mode
# ═══════════════════════════════════════════════════════════════

def dry_run_with_rdr(log_path=None):
    """
    Replay all 20 frozen RDR cases through shadow mode.
    Useful for verifying the pipeline works before real data arrives.
    """
    results = []
    print(f"\n{'='*60}")
    print(f"  SHADOW MODE — DRY RUN (replaying {len(CASES)} RDR cases)")
    print(f"  This is a SENSOR, not a controller.")
    print(f"{'='*60}\n")

    for case in CASES:
        case_id = case["id"]

        # Compute v4.3 risk
        risk = compute_risk_score(case)

        # π_E and π_S
        eval_scores = case["x_i"]["eval_scores"]
        pi_e = max(eval_scores, key=eval_scores.get)
        effective = risk.get("effective_score", 0)
        eff_is_inf = (effective == float('inf') or effective is None)
        pi_s = "BLOCK" if eff_is_inf or effective > KAPPA else "ALLOW"
        margin = KAPPA - effective if not eff_is_inf else float('-inf')

        # Shadow constraints
        shadow = shadow_constraints(case, risk)

        # Divergence
        # For RDR cases, divergence = π_S disagrees with standard eval's pick
        divergence = (pi_s != "ALLOW")  # π_E always picks the higher score (ALLOW)

        entry = {
            "case_id": case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pi_E": pi_e,
            "pi_S": pi_s,
            "divergence": divergence,
            "risk": {
                "effective_score": _fmt_num(effective),
                "margin": _fmt_num(margin),
                "E_C_neg": _fmt_num(risk.get("E_C_neg")),
                "E_C_pos": _fmt_num(risk.get("E_C_pos")),
                "std_C_neg": _fmt_num(risk.get("std_C_neg")),
                "tail_risk": _fmt_num(risk.get("tail_risk")),
                "cvar_factor": risk.get("cvar_factor"),
                "R_x": _fmt_num(risk.get("R_x")),
                "gamma_R": _fmt_num(risk.get("gamma_R")),
                "p_reversal": risk.get("p_reversal"),
                "cost_sign": risk.get("cost_sign"),
                "consequence_type": risk.get("consequence_type"),
                "risk_type": risk.get("risk_type"),
            },
            "shadow": {
                "cvar_ratio": shadow.get("cvar_ratio"),
                "cvar_margin": shadow.get("cvar_margin"),
                "irrev_ratio": shadow.get("irrev_ratio"),
                "irrev_flag": shadow.get("irrev_flag"),
                "composite_energy": shadow.get("composite_energy"),
                "tension_type": shadow.get("tension_type"),
                "pressure_ratio": shadow.get("pressure_ratio"),
            },
            "raw_inputs": {
                "context": case["x_i"]["features"].get("decision_context"),
                "eval_scores": eval_scores,
                "optimal_constrained": case.get("optimal_constrained"),
            },
            "dry_run": True,
        }

        results.append(entry)
        log_entry(entry, log_path)

    # Summary
    n = len(results)
    n_diverge = sum(1 for r in results if r["divergence"])
    n_block = sum(1 for r in results if r["pi_S"] == "BLOCK")
    n_allow = sum(1 for r in results if r["pi_S"] == "ALLOW")
    tensions = {}
    for r in results:
        t = r["shadow"]["tension_type"]
        tensions[t] = tensions.get(t, 0) + 1

    print(f"  Results: {n} cases")
    print(f"    ALLOW:  {n_allow}")
    print(f"    BLOCK:  {n_block}")
    print(f"    Diverge: {n_diverge} (π_S ≠ π_E)")
    print(f"  Shadow tensions: {tensions}")
    print(f"  Log written to: {log_path or LOG_FILE}")

    # List divergences
    if n_diverge > 0:
        print(f"\n  Divergences (π_S blocks what π_E would deploy):")
        for r in results:
            if r["divergence"]:
                eff = r["risk"]["effective_score"]
                margin = r["risk"]["margin"]
                tension = r["shadow"]["tension_type"]
                print(f"    {r['case_id']}: π_E={r['pi_E']}, π_S={r['pi_S']}, "
                      f"effective={eff}, margin={margin}, tension={tension}")

    print()
    return results


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE REPL
# ═══════════════════════════════════════════════════════════════

def interactive():
    """
    Interactive REPL for manual case entry.
    One case at a time. No batch processing.
    """
    print(f"\n{'='*60}")
    print(f"  SHADOW MODE — Interactive Sensor")
    print(f"  v4.3 frozen policy. Passive observation only.")
    print(f"  Type 'quit' or Ctrl+C to exit.")
    print(f"  Type 'summary' for log statistics.")
    print(f"  Type 'help' for input format.")
    print(f"{'='*60}\n")

    while True:
        try:
            line = input("shadow> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting shadow mode.")
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("  Exiting shadow mode.")
            break
        if line.lower() == "help":
            print(_HELP_TEXT)
            continue
        if line.lower() == "summary":
            _print_summary()
            continue

        # Try to parse as JSON
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            print("  ERROR: Input must be valid JSON. Type 'help' for format.")
            continue

        try:
            entry = run_pi_S(raw)
            log_entry(entry)
            _print_entry(entry)
        except Exception as e:
            print(f"  ERROR: {e}")


def _print_entry(entry):
    """Pretty-print a single shadow mode result."""
    div = " <<< DIVERGENCE" if entry["divergence"] else ""
    pi_e_str = entry.get("pi_E_raw", entry.get("pi_E", "?"))
    pi_e_dec = entry.get("pi_E_decision", pi_e_str)
    print(f"\n  [{entry['case_id']}] π_E={pi_e_str} (→{pi_e_dec}) → π_S={entry['pi_S']}{div}")
    print(f"    effective={entry['risk']['effective_score']}, "
          f"margin={entry['risk']['margin']}")
    print(f"    E[C⁻]={entry['risk']['E_C_neg']}, "
          f"tail_risk={entry['risk']['tail_risk']}, "
          f"R(x)={entry['risk']['R_x']}")
    print(f"    cost_sign={entry['risk']['cost_sign']}, "
          f"risk_type={entry['risk']['risk_type']}")
    print(f"    shadow tension={entry['shadow']['tension_type']}, "
          f"pressure={entry['shadow']['pressure_ratio']}")
    if entry["divergence"]:
        print(f"    *** π_S disagrees with current practice ***")


def _print_summary():
    """Print statistics from the log."""
    entries = read_log()
    if not entries:
        print("\n  No entries in log yet.")
        return

    n = len(entries)
    n_diverge = sum(1 for e in entries if e.get("divergence"))
    n_block = sum(1 for e in entries if e.get("pi_S") == "BLOCK")
    n_allow = sum(1 for e in entries if e.get("pi_S") == "ALLOW")

    # Real cases only (exclude dry runs)
    real = [e for e in entries if not e.get("dry_run")]
    n_real = len(real)

    tensions = {}
    for e in entries:
        t = e.get("shadow", {}).get("tension_type", "none")
        tensions[t] = tensions.get(t, 0) + 1

    print(f"\n  Shadow Log Summary ({n} total, {n_real} real cases)")
    print(f"    ALLOW:    {n_allow}")
    print(f"    BLOCK:    {n_block}")
    print(f"    Diverge:  {n_diverge}")
    print(f"    Tensions: {tensions}")
    print(f"    Log file: {LOG_FILE}")

    if n_diverge > 0:
        print(f"\n  Recent divergences:")
        for e in reversed(entries):
            if e.get("divergence"):
                pi_e = e.get("pi_E_raw", e.get("pi_E", "?"))
                print(f"    {e['case_id']}: π_E={pi_e}, π_S={e['pi_S']}, "
                      f"effective={e['risk']['effective_score']}")


_HELP_TEXT = """
  Input format (JSON):
  {
    "case_id": "unique-id",
    "context": "What decision is being made",
    "eval_scores": {"model_a": 0.85, "model_b": 0.91},
    "pi_E": "model_b",
    "metadata": {
      "domain": "prod|internal|creative|safety",
      "estimated_cost_if_wrong": 500000,
      "reversibility": "easy|moderate|hard|impossible",
      "latency_to_detect": "hours|days|weeks",
      "consequence_type": "error_cost|safety_incident_risk|revenue_loss|...",
      "notes": "any additional context"
    }
  }

  Minimum required: case_id, eval_scores
  All metadata fields are optional (defaults applied).
"""


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--dry-run" in args:
        # Replay RDR cases through shadow mode
        dry_run_with_rdr()

    elif "--file" in args:
        idx = args.index("--file")
        if idx + 1 >= len(args):
            print("ERROR: --file requires a path argument")
            sys.exit(1)
        filepath = args[idx + 1]
        results = process_file(filepath)
        n = len(results)
        n_diverge = sum(1 for r in results if r.get("divergence"))
        print(f"\n  Processed {n} cases. {n_diverge} divergences logged to {LOG_FILE}")

    else:
        # Interactive REPL
        interactive()

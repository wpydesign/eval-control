"""
eval_control.core — Autonomous Evaluation Control Layer

BSSI = S x A x (1 - N)

ARCHITECTURE:
  diagnose()  → identifies failure mode (6 modes)
  prescribe() → generates fix action
  decide()    → ALLOW / BLOCK / CONDITIONAL
  control()   → full pipeline
  autofix()   → detect → fix → re-run (single pass)

FAILURE MODES:
  1. CEILING_EFFECT  — benchmark too easy, both models >95%
  2. FLOOR_EFFECT    — benchmark too hard, both models <20%
  3. METRIC_CONFLICT — metrics disagree on model ranking
  4. HIGH_NOISE      — evaluation noise dominates signal
  5. EXTRACTION_NOISE — answer extraction introduces artifacts
  6. SPARSE_DATA     — insufficient samples for reliable measurement

All heuristics. No black-box models. Auditable.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple


# ─────────────────────────────────────────────────────────────
# THRESHOLD CONFIGURATION
# ─────────────────────────────────────────────────────────────

THRESHOLDS = {
    "bssi_valid": 0.50,
    "bssi_weak": 0.20,
    "bssi_no_signal": 0.01,
    "separation_min": 0.05,
    "separation_ideal_min": 0.15,
    "agreement_min": 0.60,
    "noise_max": 0.50,
    "noise_ideal_max": 0.30,
    "difficulty_min": 0.40,
    "difficulty_max": 0.85,
    "low_margin": 0.01,
    "high_margin": 0.05,
    "min_samples_per_question": 30,
    "min_questions": 5,
    "extraction_fail_max": 0.15,
}

METRIC_REGIMES = {
    "qa": {"trusted": ["consensus", "correct_stability", "majority_strength"],
           "unstable": ["strategy_lock", "entropy_inv"]},
    "math": {"trusted": ["consensus", "correct_stability"],
             "unstable": ["strategy_lock", "entropy_inv", "majority_strength"]},
    "safety": {"trusted": ["consensus", "strategy_lock"],
               "unstable": ["correct_stability", "entropy_inv"]},
    "generic": {"trusted": ["consensus", "majority_strength"],
                "unstable": ["strategy_lock", "entropy_inv"]},
}

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

ALL_FAILURE_MODES = [
    "CEILING_EFFECT",
    "FLOOR_EFFECT",
    "METRIC_CONFLICT",
    "HIGH_NOISE",
    "EXTRACTION_NOISE",
    "SPARSE_DATA",
]


def _max_severity(current: str, new: str) -> str:
    if SEVERITY_ORDER.get(new, 0) > SEVERITY_ORDER.get(current, 0):
        return new
    return current


# ─────────────────────────────────────────────────────────────
# STEP 1: DIAGNOSIS
# ─────────────────────────────────────────────────────────────

def diagnose(S: float, A: float, N: float, BSSI: float,
             rfs: Optional[Dict] = None,
             per_metric_noise: Optional[Dict] = None,
             acc_a: Optional[float] = None,
             acc_b: Optional[float] = None,
             n_questions: Optional[int] = None,
             extraction_fail_rate: Optional[float] = None) -> Dict:
    """
    Identify WHY evaluation is failing (or confirm it's working).

    Supports 6 failure modes:
      1. CEILING_EFFECT  — both models >95% (benchmark too easy)
      2. FLOOR_EFFECT    — both models <20% (benchmark too hard)
      3. NO_SEPARATION   — S < 0.05 (models indistinguishable)
      4. METRIC_CONFLICT — >=3/5 metrics pick wrong winner
      5. HIGH_NOISE      — N > 0.50 (noise dominates)
      6. EXTRACTION_NOISE — extraction fail rate > 15%
      7. SPARSE_DATA     — fewer than 5 questions or 30 samples/question
    """
    codes = []
    evidence = []
    severity = "LOW"

    # Mode 7: SPARSE_DATA
    if n_questions is not None and n_questions < THRESHOLDS["min_questions"]:
        codes.append("SPARSE_DATA")
        evidence.append({
            "metric": "n_questions",
            "value": n_questions,
            "threshold": THRESHOLDS["min_questions"],
            "detail": f"Only {n_questions} questions < {THRESHOLDS['min_questions']}: insufficient for reliable BSSI"
        })
        severity = _max_severity(severity, "HIGH")

    # Mode 6: EXTRACTION_NOISE
    if extraction_fail_rate is not None and extraction_fail_rate > THRESHOLDS["extraction_fail_max"]:
        codes.append("EXTRACTION_NOISE")
        evidence.append({
            "metric": "extraction_fail_rate",
            "value": extraction_fail_rate,
            "threshold": THRESHOLDS["extraction_fail_max"],
            "detail": f"Extraction fail rate {extraction_fail_rate:.1%} > {THRESHOLDS['extraction_fail_max']:.0%}: answer extraction unreliable"
        })
        severity = _max_severity(severity, "HIGH")

    # Mode 1 & 2: CEILING / FLOOR
    if acc_a is not None and acc_b is not None:
        if acc_a > 0.95 and acc_b > 0.95:
            codes.append("CEILING_EFFECT")
            evidence.append({
                "metric": "difficulty",
                "value": {"acc_a": acc_a, "acc_b": acc_b},
                "detail": f"Both models >95%: benchmark too easy (ceiling effect)"
            })
            severity = _max_severity(severity, "CRITICAL")
        elif acc_a < 0.20 and acc_b < 0.20:
            codes.append("FLOOR_EFFECT")
            evidence.append({
                "metric": "difficulty",
                "value": {"acc_a": acc_a, "acc_b": acc_b},
                "detail": f"Both models <20%: benchmark too hard (floor effect)"
            })
            severity = _max_severity(severity, "CRITICAL")

    # S check
    if S < THRESHOLDS["separation_min"]:
        if "CEILING_EFFECT" not in codes and "FLOOR_EFFECT" not in codes:
            codes.append("NO_SEPARATION")
        evidence.append({
            "metric": "S", "value": S,
            "threshold": THRESHOLDS["separation_min"],
            "detail": f"S={S:.4f} < {THRESHOLDS['separation_min']}: models indistinguishable"
        })
        severity = _max_severity(severity, "CRITICAL")
    elif S < THRESHOLDS["separation_ideal_min"]:
        codes.append("LOW_SEPARATION")
        evidence.append({
            "metric": "S", "value": S,
            "threshold": THRESHOLDS["separation_ideal_min"],
            "detail": f"S={S:.4f} < {THRESHOLDS['separation_ideal_min']}: weak separation"
        })
        severity = _max_severity(severity, "HIGH")

    # Mode 3: METRIC_CONFLICT
    if rfs:
        unreliable = [m for m, d in rfs.items() if d.get("rfs", 0) == 0]
        if len(unreliable) >= 3:
            codes.append("METRIC_CONFLICT")
            evidence.append({
                "metric": "metric_flip", "value": len(unreliable),
                "detail": f"{len(unreliable)}/5 metrics produce wrong ranking: {unreliable}"
            })
            severity = _max_severity(severity, "HIGH")
        elif len(unreliable) >= 2:
            codes.append("PARTIAL_CONFLICT")
            evidence.append({
                "metric": "metric_flip", "value": len(unreliable),
                "detail": f"{len(unreliable)}/5 metrics unreliable: {unreliable}"
            })
            severity = _max_severity(severity, "MEDIUM")

    # A check
    if A < THRESHOLDS["agreement_min"]:
        if "METRIC_CONFLICT" not in codes and "PARTIAL_CONFLICT" not in codes:
            codes.append("METRIC_DISAGREEMENT")
            evidence.append({
                "metric": "A", "value": A,
                "threshold": THRESHOLDS["agreement_min"],
                "detail": f"A={A:.4f} < {THRESHOLDS['agreement_min']}: metrics disagree"
            })
            severity = _max_severity(severity, "HIGH")

    # Mode 4: HIGH_NOISE
    if N > THRESHOLDS["noise_max"]:
        codes.append("HIGH_NOISE")
        evidence.append({
            "metric": "N", "value": N,
            "threshold": THRESHOLDS["noise_max"],
            "detail": f"N={N:.4f} > {THRESHOLDS['noise_max']}: noise dominates signal"
        })
        severity = _max_severity(severity, "HIGH")
    elif N > THRESHOLDS["noise_ideal_max"]:
        codes.append("ELEVATED_NOISE")
        evidence.append({
            "metric": "N", "value": N,
            "threshold": THRESHOLDS["noise_ideal_max"],
            "detail": f"N={N:.4f} > {THRESHOLDS['noise_ideal_max']}: elevated noise"
        })
        severity = _max_severity(severity, "MEDIUM")

    # BSSI overall
    if BSSI < THRESHOLDS["bssi_no_signal"]:
        if "NO_SEPARATION" not in codes and "CEILING_EFFECT" not in codes:
            codes.append("NO_SIGNAL")
        evidence.append({
            "metric": "BSSI", "value": BSSI,
            "threshold": THRESHOLDS["bssi_no_signal"],
            "detail": f"BSSI={BSSI:.4f} < {THRESHOLDS['bssi_no_signal']}: no signal"
        })
        severity = _max_severity(severity, "CRITICAL")

    # Per-metric noise
    if per_metric_noise:
        noisy = [m for m, cv in per_metric_noise.items() if cv > 0.15]
        if noisy:
            evidence.append({
                "metric": "per_metric_noise", "value": noisy,
                "detail": f"High-CV metrics: {noisy}"
            })

    if not codes:
        codes.append("VALID")

    return {
        "primary_code": codes[0],
        "all_codes": codes,
        "severity": severity,
        "evidence": evidence,
        "components": {"S": S, "A": A, "N": N, "BSSI": BSSI},
    }


# ─────────────────────────────────────────────────────────────
# STEP 2: PRESCRIPTION
# ─────────────────────────────────────────────────────────────

def prescribe(diagnosis: Dict, rfs: Optional[Dict] = None,
              task_type: str = "generic",
              per_metric_noise: Optional[Dict] = None) -> Dict:
    """Generate the fix action for each diagnosed failure mode."""
    codes = diagnosis["all_codes"]
    fixes = []
    trusted_metrics = None
    blocked_metrics = None

    for code in codes:
        if code in ("NO_SEPARATION", "CEILING_EFFECT"):
            fixes.append({
                "action": "increase_task_difficulty",
                "priority": "immediate",
                "reason": "Models too similar on this benchmark",
                "suggestion": [
                    f"Filter questions to {THRESHOLDS['difficulty_min']*100:.0f}%-{THRESHOLDS['difficulty_max']*100:.0f}% accuracy band",
                    "Use harder benchmark (e.g., GSM8K instead of MMLU)",
                    "Check for data contamination",
                ],
            })

        elif code == "FLOOR_EFFECT":
            fixes.append({
                "action": "decrease_task_difficulty",
                "priority": "immediate",
                "reason": "Both models fail — benchmark too hard",
                "suggestion": ["Use easier questions", "Try different benchmark"],
            })

        elif code == "LOW_SEPARATION":
            fixes.append({
                "action": "adjust_difficulty_band",
                "priority": "recommended",
                "reason": "Weak separation signal",
                "suggestion": ["Remove ceiling/floor questions", "Target 40-85% accuracy band"],
            })

        elif code in ("METRIC_DISAGREEMENT", "METRIC_CONFLICT"):
            regime = METRIC_REGIMES.get(task_type, METRIC_REGIMES["generic"])
            trusted = list(regime["trusted"])
            unstable = list(regime["unstable"])
            if rfs:
                reliable = [m for m, d in rfs.items() if d.get("rfs", 0) == 1]
                if reliable:
                    trusted = reliable
                    unstable = [m for m, d in rfs.items() if d.get("rfs", 0) == 0]
            trusted_metrics = trusted
            blocked_metrics = unstable
            fixes.append({
                "action": "switch_metric_regime",
                "priority": "immediate",
                "reason": "Metrics disagree — using wrong metrics gives wrong answer",
                "suggestion": trusted,
            })

        elif code == "PARTIAL_CONFLICT":
            if rfs:
                trusted_metrics = [m for m, d in rfs.items() if d.get("rfs", 0) == 1]
                blocked_metrics = [m for m, d in rfs.items() if d.get("rfs", 0) == 0]
            fixes.append({
                "action": "use_trusted_metrics_only",
                "priority": "recommended",
                "reason": "Some metrics disagree",
                "suggestion": trusted_metrics or ["consensus", "correct_stability"],
            })

        elif code == "HIGH_NOISE":
            fixes.append({
                "action": "reduce_noise",
                "priority": "immediate",
                "reason": "Noise dominates signal",
                "suggestion": [
                    "Increase sample size per question (target 100+)",
                    "Use structured output format",
                    "Apply hard filter to remove non-conforming responses",
                ],
            })

        elif code == "ELEVATED_NOISE":
            fixes.append({
                "action": "reduce_noise",
                "priority": "recommended",
                "reason": "Noise above ideal",
                "suggestion": ["Increase sample size", "Remove noisiest strategies"],
            })

        elif code == "EXTRACTION_NOISE":
            fixes.append({
                "action": "tighten_extraction",
                "priority": "immediate",
                "reason": "Answer extraction introduces artifacts",
                "suggestion": [
                    "Use structured output format (e.g., 'FINAL: <answer>')",
                    "Apply stricter regex matching",
                    "Hard-filter non-conforming responses",
                    "Increase max_tokens to prevent truncation",
                ],
            })

        elif code == "SPARSE_DATA":
            fixes.append({
                "action": "increase_samples",
                "priority": "immediate",
                "reason": "Insufficient data for reliable measurement",
                "suggestion": [
                    f"Increase to >= {THRESHOLDS['min_questions']} questions",
                    f"Ensure >= {THRESHOLDS['min_samples_per_question']} samples per question per model",
                    "Use more perturbation strategies or more repetitions",
                ],
            })

        elif code == "NO_SIGNAL":
            fixes.append({
                "action": "restructure_evaluation",
                "priority": "immediate",
                "reason": "No detectable evaluation signal",
                "suggestion": [
                    "Switch from text-level to decision-level analysis",
                    "Use a benchmark with genuine model differences",
                    "Ensure structured output format",
                ],
            })

        elif code == "VALID":
            if rfs:
                trusted_metrics = [m for m, d in rfs.items() if d.get("rfs", 0) == 1]
            fixes.append({"action": "proceed", "priority": "none",
                          "reason": "Evaluation is valid", "suggestion": None})

    return {"fixes": fixes, "trusted_metrics": trusted_metrics,
            "blocked_metrics": blocked_metrics}


# ─────────────────────────────────────────────────────────────
# STEP 3: DECISION
# ─────────────────────────────────────────────────────────────

def decide(diagnosis: Dict, prescription: Dict) -> Dict:
    """Final GO / NO-GO / CONDITIONAL decision."""
    primary = diagnosis["primary_code"]
    severity = diagnosis["severity"]

    if severity == "CRITICAL":
        return {"evaluation_possible": False, "decision": "BLOCK",
                "confidence": "HIGH",
                "reason": f"Evaluation blocked: {primary}. Fix required.",
                "action_required": True}

    if severity == "HIGH":
        return {"evaluation_possible": False, "decision": "BLOCK",
                "confidence": "HIGH",
                "reason": f"Evaluation blocked: {primary}. Fix required.",
                "action_required": True}

    if severity == "MEDIUM":
        trusted = prescription.get("trusted_metrics")
        if trusted:
            return {"evaluation_possible": True, "decision": "CONDITIONAL",
                    "confidence": "MEDIUM",
                    "reason": f"Conditional: {primary}. Use only: {trusted}",
                    "action_required": False,
                    "conditions": [f"Use only: {trusted}"]}
        return {"evaluation_possible": True, "decision": "CONDITIONAL",
                "confidence": "MEDIUM",
                "reason": f"Degraded: {primary}. Use with caution.",
                "action_required": False,
                "conditions": ["Interpret with reduced confidence"]}

    # LOW
    if primary == "VALID":
        return {"evaluation_possible": True, "decision": "ALLOW",
                "confidence": "HIGH",
                "reason": "Evaluation valid. Proceed.",
                "action_required": False}
    return {"evaluation_possible": True, "decision": "ALLOW",
            "confidence": "LOW",
            "reason": f"Passes gate: {primary}. Weak but present.",
            "action_required": False}


# ─────────────────────────────────────────────────────────────
# FULL CONTROL PIPELINE
# ─────────────────────────────────────────────────────────────

def control(S: float, A: float, N: float, BSSI: float,
            rfs: Optional[Dict] = None,
            per_metric_noise: Optional[Dict] = None,
            acc_a: Optional[float] = None,
            acc_b: Optional[float] = None,
            task_type: str = "generic",
            model_a_name: str = "Model A",
            model_b_name: str = "Model B",
            benchmark_name: str = "unknown",
            n_questions: Optional[int] = None,
            extraction_fail_rate: Optional[float] = None) -> Dict:
    """
    Full control layer: diagnose → prescribe → decide.

    Returns JSON with: decision, confidence, reason, fix, diagnosis, bssi, meta.
    """
    diag = diagnose(S, A, N, BSSI, rfs, per_metric_noise, acc_a, acc_b,
                    n_questions, extraction_fail_rate)
    rx = prescribe(diag, rfs, task_type, per_metric_noise)
    dec = decide(diag, rx)

    trusted = rx.get("trusted_metrics")
    blocked = rx.get("blocked_metrics")

    if not dec["evaluation_possible"]:
        reason_map = {
            "NO_SEPARATION": "NO_SIGNAL", "CEILING_EFFECT": "NO_SIGNAL",
            "FLOOR_EFFECT": "NO_SIGNAL", "NO_SIGNAL": "NO_SIGNAL",
            "LOW_SEPARATION": "NO_SIGNAL",
            "METRIC_DISAGREEMENT": "METRIC_CONFLICT",
            "METRIC_CONFLICT": "METRIC_CONFLICT",
            "HIGH_NOISE": "HIGH_NOISE", "ELEVATED_NOISE": "HIGH_NOISE",
            "EXTRACTION_NOISE": "EXTRACTION_NOISE",
            "SPARSE_DATA": "SPARSE_DATA",
        }
        fix_action = None
        for fix in rx["fixes"]:
            if fix["priority"] == "immediate":
                fix_action = fix["action"]
                fix_suggestion = fix["suggestion"]
                break
        if fix_action is None and rx["fixes"]:
            fix_action = rx["fixes"][0]["action"]
            fix_suggestion = rx["fixes"][0]["suggestion"]
        else:
            fix_suggestion = None

        fix_obj = {"action": fix_action, "suggestion": fix_suggestion,
                   "priority": rx["fixes"][0]["priority"] if rx["fixes"] else "none"}
        if blocked:
            fix_obj["blocked_metrics"] = blocked
        if trusted:
            fix_obj["alternative_metrics"] = trusted
    else:
        fix_obj = {"action": "proceed", "suggestion": None}
        if dec["decision"] == "CONDITIONAL":
            fix_obj["conditions"] = dec.get("conditions", [])

    return {
        "evaluation_possible": dec["evaluation_possible"],
        "decision": dec["decision"],
        "confidence": dec["confidence"],
        "reason": dec["reason"],
        "reason_code": diag["primary_code"],
        "fix": fix_obj,
        "diagnosis": {"codes": diag["all_codes"], "severity": diag["severity"],
                      "evidence": diag["evidence"]},
        "trusted_metrics": trusted,
        "blocked_metrics": blocked,
        "bssi": {"S": S, "A": A, "N": N, "BSSI": BSSI,
                 "formula": "S x A x (1 - N)"},
        "models": {"A": model_a_name, "B": model_b_name},
        "benchmark": benchmark_name,
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "version": "1.0.0",
                 "system": "Autonomous Evaluation Control Layer"},
    }


# ─────────────────────────────────────────────────────────────
# STEP 4: AUTO-FIX (single pass, transparent)
# ─────────────────────────────────────────────────────────────

def autofix(retry_fn, **control_kwargs) -> Dict:
    """
    Closed-loop: detect → fix → re-run → confirm. Exactly ONCE.

    retry_fn(fix_action, reason_codes, blocked_result) -> dict with:
      "params": updated BSSI params for control()
      "type": fix type label (e.g. "difficulty_filter")
      "details": human-readable description
      "samples_before": int (optional)
      "samples_after": int (optional)
    Return None if fix cannot be applied.

    Returns:
      {"before": {...}, "fix_applied": {...}, "after": {...},
       "status": "AUTO_FIXED" | "FAILED_TO_FIX" | "ALLOWED"}
    """
    initial = control(**control_kwargs)

    if initial["decision"] == "ALLOW":
        return {"before": _snap(initial), "fix_applied": None,
                "after": None, "status": "ALLOWED"}

    fix_action = initial.get("fix", {}).get("action")
    reason_codes = initial.get("diagnosis", {}).get("codes", [])

    if not fix_action or fix_action == "proceed":
        return {"before": _snap(initial), "fix_applied": None,
                "after": None, "status": "FAILED_TO_FIX"}

    try:
        fix_result = retry_fn(fix_action, reason_codes, initial)
    except Exception:
        fix_result = None

    if not fix_result:
        return {"before": _snap(initial), "fix_applied": None,
                "after": None, "status": "FAILED_TO_FIX"}

    fix_log = {"type": fix_result.get("type", fix_action),
               "details": fix_result.get("details", fix_action)}
    if "samples_before" in fix_result:
        fix_log["samples_before"] = fix_result["samples_before"]
    if "samples_after" in fix_result:
        fix_log["samples_after"] = fix_result["samples_after"]

    new_params = fix_result.get("params", fix_result)
    merged = {**control_kwargs}
    merged.update(new_params)
    fixed = control(**merged)

    if fixed["decision"] == "ALLOW":
        acc_a = new_params.get("acc_a", control_kwargs.get("acc_a"))
        acc_b = new_params.get("acc_b", control_kwargs.get("acc_b"))
        ranking = None
        if acc_a is not None and acc_b is not None:
            na = control_kwargs.get("model_a_name", "A")
            nb = control_kwargs.get("model_b_name", "B")
            ranking = f"{na} > {nb}" if acc_a > acc_b else (
                f"{nb} > {na}" if acc_b > acc_a else "TIE")
        after = _snap(fixed)
        after["model_ranking"] = ranking
        after["trusted_metrics"] = fixed.get("trusted_metrics")
        return {"before": _snap(initial), "fix_applied": fix_log,
                "after": after, "status": "AUTO_FIXED"}

    return {"before": _snap(initial), "fix_applied": fix_log,
            "after": _snap(fixed), "status": "FAILED_TO_FIX"}


def _snap(result: Dict) -> Dict:
    return {"decision": result["decision"],
            "reason": result.get("reason_code", ""),
            "bssi": round(result["bssi"]["BSSI"], 4),
            "confidence": result["confidence"]}


# ─────────────────────────────────────────────────────────────
# CI/CD INTERFACE
# ─────────────────────────────────────────────────────────────

def ci_check(result: Dict) -> Tuple[bool, str]:
    """CI/CD gate: returns (passed, message)."""
    if result["decision"] == "ALLOW":
        return True, f"PASS: {result['reason']}"
    elif result["decision"] == "CONDITIONAL":
        return True, f"WARNING: {result['reason']}"
    else:
        return False, f"FAIL: {result['reason']}"


def format_autofix(output: Dict) -> str:
    """Format autofix output for terminal display."""
    lines = ["=" * 56, "  AUTO-FIX (single pass, transparent)", "=" * 56]
    before = output["before"]
    lines.append(f"\n  BEFORE: {before['decision']} | reason={before['reason']}")
    lines.append(f"    BSSI = {before['bssi']}")
    fix = output.get("fix_applied")
    if fix:
        lines.append(f"\n  FIX: {fix['type']}")
        lines.append(f"    {fix['details']}")
        if "samples_before" in fix:
            lines.append(f"    samples: {fix['samples_before']} -> {fix['samples_after']}")
    after = output.get("after")
    if after:
        lines.append(f"\n  AFTER:  {after['decision']} | BSSI = {after['bssi']}")
        if after.get("model_ranking"):
            lines.append(f"    ranking: {after['model_ranking']}")
    status = output["status"]
    icon = {"AUTO_FIXED": "OK", "ALLOWED": "PASS", "FAILED_TO_FIX": "FAIL"}[status]
    lines.append(f"\n  [{icon}] status: {status}")
    lines.append("=" * 56)
    return "\n".join(lines)

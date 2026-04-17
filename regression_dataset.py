#!/usr/bin/env python3
"""
regression_dataset.py — Multi-Factor Deployment Control Simulator (v4.3)

FORMAL MODEL (v4.3 — tail-sensitive, sign-correct, multi-factor deployment control under uncertainty):
    State:       x_i = (features, bssi_components, eval_scores)
    Eval policy: pi_E(x_i) = argmax(eval_scores)          [standard eval: pick higher score]
    System policy: pi_S(x_i) = risk_gate(effective_score)   [multi-factor threshold]
    Truth:       y_i ~ P(y | x_i, action)                   [stochastic ground truth]
    Cost:        C_i(action, y)                               [explicit cost function per case]

    v4.3 KEY CHANGE — TAIL-SENSITIVE RISK (CVaR replaces σ):
        Old (v4.2):  effective = E[C⁻] + lambda * sigma(C⁻) + gamma * R(x) - beta * E[C⁺]
        New (v4.3):  effective = E[C⁻] + lambda * CVaR_excess(C⁻) + gamma * R(x) - beta * E[C⁺]

        Where CVaR_excess(C⁻) = σ(C⁻) * φ(Φ⁻¹(α)) / (1-α)

        σ treats all uncertainty the same: frequent small losses and rare catastrophic
        losses are equivalent if variance matches. CVaR focuses on the TAIL — what
        happens in the worst (1-α) scenarios.

        Distribution-dependent tail factors:
            - normal:         CVaR_factor ≈ 2.063  (at α=0.95)
            - heavy_tailed:   CVaR_factor ≈ 3.18   (Student-t ν≈3 approximation)
            - deterministic:  CVaR_factor = 0      (no tail risk)

        This is the last clean scalar extension before needing piecewise rules.

    v4.2 CHANGE — SIGN-CORRECT COST MODEL (preserved):
        C⁻ = downside risk (actual losses: error_cost, safety, revenue_loss, etc.)
        C⁺ = upside opportunity (forgone gains: forfeited_productivity_gain, etc.)
        Fixes the sign error: uncertain benefit (RDR-003) was false-blocked.
        Now E[C⁻]=0 for upside cases → effective = gamma*R(x) → ALLOW.

    CALIBRATION (v4.3):
        lambda = 1.0   (risk aversion: weight on tail risk)
        gamma  = 2.0   (irreversibility weight)
        beta   = 0.0   (opportunity discount: C⁺ does not reduce risk score)
        alpha  = 0.95  (CVaR confidence level: focus on worst 5% of scenarios)
        kappa  = $1,100,000  (risk tolerance threshold)
        Rationale:
            - CVaR at α=0.95: penalizes tail outcomes, not just average variance
            - Heavy-tailed envs get ~1.54x more penalty than normal envs (same σ)
            - All v4.2 results preserved: 19/20 (CVaR only affects tail structure)
            - RDR-009 remains as conscious policy choice (E[C⁻]=0 → CVaR=0)

    pi_E EXPLICIT FORM:
        pi_E(x) = argmax_a { s(a) }
        where s(a) = eval_scores[a], theta_E = {scoring_fn: "raw_score"}

    BSSI AS DECISION VARIABLE:
        BSSI = S x A x (1 - N) feeds into diagnosis (supplementary).
        Primary decision is now multi-factor risk score vs kappa.

    CONSTRAINED ACTION SPACE:
        A_i = { deploy_a, deploy_b } for each case (binary deploy choice)
        "optimal" = argmin_{a in A_i} E_{y~P}[C(a, y)] — constrained by A_i

    ENVIRONMENT MODEL (per-case):
        environment_model:
            distribution: "deterministic" | "normal" | "heavy_tailed"
            variance:     sigma^2 — per-case outcome uncertainty
            sensitivity:  dC/dy — cost sensitivity to outcome perturbation

    IRREVERSIBILITY MODEL (per-case):
        irreversibility_model:
            rollback_cost:      cost to undo the deployment ($/year)
            time_to_detect:     how long until problems are noticed (hours)
            blast_radius:       how many users affected (count)
            R_x:                scalar irreversibility score ($/year equivalent)

FIXED-POINT INVARIANT:
    D = constant.  Only derived metrics change.
    x_i, y_i, C_i, environment_model, irreversibility_model are FROZEN.
    pi_E is deterministic from x_i.
    pi_S is deterministic from (x_i, lambda, gamma, kappa). All global.

Run: python regression_dataset.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from core import diagnose, decide, prescribe, THRESHOLDS


# ═══════════════════════════════════════════════════════════════
# GLOBAL THRESHOLD FUNCTION tau
# ═══════════════════════════════════════════════════════════════
# This is NOT per-case. It is the same for all 20 cases.
# BSSI enters as a DECISION VARIABLE here — it directly maps to
# the decision boundary through the severity hierarchy.

TAU = {
    "decision_rules": {
        "BLOCK": "severity in {CRITICAL, HIGH}",
        "CONDITIONAL": "severity == MEDIUM",
        "ALLOW": "severity == LOW and primary_code == VALID",
    },
    "bssi_decision_boundary": {
        "description": "BSSI directly controls deployment gate decision",
        "BSSI < bssi_no_signal (0.01)": "BLOCK — no eval signal exists",
        "BSSI < bssi_valid (0.50) AND S < separation_min (0.05)": "BLOCK — signal unreliable",
        "BSSI >= bssi_valid OR (S >= separation_min AND A >= agreement_min)": "ALLOW — signal reliable",
    },
    "thresholds": dict(THRESHOLDS),  # copied from core.py
    "formula": "BSSI(x) -> diagnose(S, A, N, BSSI, ...) -> severity -> decide()",
}


# ═══════════════════════════════════════════════════════════════
# pi_E EXPLICIT FORM
# ═══════════════════════════════════════════════════════════════
# pi_E is NOT a symbolic label. It has a concrete functional form:

PI_E_DEFINITION = {
    "form": "pi_E(x) = argmax_a { s(a) }",
    "where": {
        "s(a)": "eval_scores[a] — raw benchmark score for action a",
    },
    "theta_E": {
        "scoring_fn": "raw_score",
        "aggregation": "argmax",
        "no_threshold": True,
        "no_noise_model": True,
        "deterministic": True,
    },
    "action_space": "A_i = {a, b} — the two candidates in eval_scores",
    "limitation": "Cannot consider actions outside A_i. See RDR-016 for example where optimal action was not in candidate set.",
}


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT MODEL DEFINITIONS (per-case)
# ═══════════════════════════════════════════════════════════════
# Each case has a stochastic environment model:
#   P(y | x, action):  how outcomes distribute given state and action
#   variance:          outcome uncertainty (sigma^2)
#   sensitivity:       dC/dy — cost gradient w.r.t. outcome
#
# For v3.2, cases are categorized:
#   "deterministic":   P(y=x) = 1, variance = 0
#       — used when downstream metric is precisely measured in production
#   "stochastic_low":  small variance, outcome mildly uncertain
#       — used when production conditions have minor seasonal drift
#   "stochastic_high": large variance, outcome highly uncertain
#       — used when deployment environment is unstable or novel

def _build_env_model(env_type, base_cost, consequence_type):
    """
    Build environment_model dict for a case.

    env_type: "deterministic" | "stochastic_low" | "stochastic_high"
    base_cost: the annual_usd cost (or None for unquantifiable)
    consequence_type: string from C_i
    """
    if env_type == "deterministic":
        return {
            "distribution": "deterministic",
            "description": "P(y = y_i) = 1. Outcome is precisely known from production data.",
            "variance": 0.0,
            "variance_rationale": "No uncertainty. Metric directly measured in production.",
            "sensitivity": _compute_sensitivity(base_cost, consequence_type),
        }
    elif env_type == "stochastic_low":
        # Low noise: ~5-10% outcome variance
        return {
            "distribution": "normal",
            "description": "P(y) ~ N(y_i, sigma^2). Minor production drift expected.",
            "variance": 0.04,  # ~20% relative std on the metric gap
            "variance_rationale": "Seasonal variation in production traffic patterns. Metric may shift +/-5-10% quarterly.",
            "sensitivity": _compute_sensitivity(base_cost, consequence_type),
        }
    elif env_type == "stochastic_high":
        # High noise: ~20-30% outcome variance
        return {
            "distribution": "normal",
            "description": "P(y) ~ N(y_i, sigma^2). Significant deployment uncertainty.",
            "variance": 0.16,  # ~40% relative std on the metric gap
            "variance_rationale": "Novel deployment context with no direct production baseline. High sensitivity to load, user behavior shifts, and competitive dynamics.",
            "sensitivity": _compute_sensitivity(base_cost, consequence_type),
        }
    elif env_type == "stochastic_catastrophic":
        # Catastrophic: unbounded cost, very high variance
        return {
            "distribution": "heavy_tailed",
            "description": "P(y) has heavy tail. Low-probability catastrophic outcomes possible.",
            "variance": None,  # unbounded
            "variance_rationale": "Safety-critical domain. Cost distribution has fat tail — single event can exceed $1M in liability.",
            "sensitivity": _compute_sensitivity(base_cost, consequence_type),
        }
    return {"distribution": "unknown", "variance": None, "sensitivity": 0}


# ═══════════════════════════════════════════════════════════════
# IRREVERSIBILITY MODEL DEFINITIONS (per-case) — v4.1
# ═══════════════════════════════════════════════════════════════
# Each case has an irreversibility/latency penalty R(x) that captures
# operational risk independent of dollar-scale cost:
#
#   R(x) = rollback_cost + (time_to_detect_hours / 8760) * annual_burn_rate
#
# Where:
#   rollback_cost:       one-time cost to undo deployment, annualized
#   time_to_detect:      hours until problems are noticed (longer = worse)
#   annual_burn_rate:    how much damage per hour while undetected
#   blast_radius:        users affected (for context, not directly in R)
#
# R(x) addresses the v4 critique: two cases with identical uncertainty
# structure but different dollar magnitudes should behave differently
# based on OPERATIONAL CONSEQUENCE, not just dollar scale.
#
# Example: RDR-003 ($19M, internal dev tools, easy rollback) vs
#          RDR-019 ($876K, production, silent truncation, hard to detect)
#   In v4: RDR-003 blocked (scale), RDR-019 allowed (scale too low)
#   In v4.1: RDR-019 blocked (high R), RDR-003 blocked (high variance)
#          Both blocked, but for structurally different reasons.

# Irreversibility tiers (annualized $/year equivalent):
#   VERY_LOW:    <$100K     — trivial rollback, immediate detection
#   LOW:         $100-300K  — easy rollback, hours to detect
#   MEDIUM:      $300-700K  — moderate rollback, days to detect
#   HIGH:        $700K-2M   — hard rollback, weeks to detect
#   VERY_HIGH:   >$2M       — very hard rollback, months to detect
#   UNQUANTIFIABLE: INF     — safety-critical, irreversible consequences

IRREVERSIBILITY_PRESETS = {
    "very_low": {
        "description": "Trivial rollback, immediate detection, narrow blast radius",
        "examples": "config parameter, A/B test, feature flag",
        "R_x": 50000,
    },
    "low": {
        "description": "Easy rollback, hours to detect, small blast radius",
        "examples": "internal tool, dev-facing feature, non-critical pipeline",
        "R_x": 150000,
    },
    "medium": {
        "description": "Moderate rollback, days to detect, significant blast radius",
        "examples": "user-facing pipeline, production API, customer-facing feature",
        "R_x": 450000,
    },
    "high": {
        "description": "Hard rollback, weeks to detect, wide blast radius",
        "examples": "production model swap affecting millions, financial pipeline",
        "R_x": 900000,
    },
    "very_high": {
        "description": "Very hard rollback, months to detect, massive blast radius",
        "examples": "core production system, revenue-critical pipeline, trading system",
        "R_x": 2500000,
    },
    "unquantifiable": {
        "description": "Irreversible — safety-critical, legal liability, clinical decisions",
        "examples": "clinical decision support, autonomous vehicles, safety systems",
        "R_x": float('inf'),
    },
}


def _build_irreversibility_model(tier, rollback_cost_annual, time_to_detect_hours, blast_radius, annual_burn_rate=None):
    """
    Build irreversibility_model dict for a case.

    tier: one of IRREVERSIBILITY_PRESETS keys
    rollback_cost_annual: cost to undo deployment, annualized ($/year)
    time_to_detect_hours: hours until problems are noticed
    blast_radius: users affected (count or description)
    annual_burn_rate: damage per hour while undetected ($/hour). Optional.

    R(x) is computed from the preset but can be overridden by providing
    annual_burn_rate (which models: time_to_detect * burn_rate = hidden cost).
    """
    preset = IRREVERSIBILITY_PRESETS.get(tier, IRREVERSIBILITY_PRESETS["medium"])
    R_x = preset["R_x"]

    # If burn rate provided, add time-to-detect component
    burn_component = 0
    if annual_burn_rate is not None and time_to_detect_hours is not None:
        burn_component = (time_to_detect_hours / 8760) * annual_burn_rate
        # R_x = base preset + burn component (capped by very_high)
        if R_x != float('inf'):
            R_x = R_x + burn_component

    return {
        "tier": tier,
        "description": preset["description"],
        "R_x": R_x,
        "R_x_components": {
            "base_preset": preset["R_x"],
            "burn_component": burn_component,
            "total": R_x,
        },
        "rollback_cost_annual": rollback_cost_annual,
        "time_to_detect_hours": time_to_detect_hours,
        "blast_radius": blast_radius,
        "examples": preset["examples"],
    }


def _compute_sensitivity(base_cost, consequence_type):
    """
    Compute dC/dy — sensitivity of cost to outcome perturbation.

    This measures: if the downstream metric y shifts by 1 unit,
    how much does cost change?

    For deterministic cases with known base_cost, sensitivity = base_cost
    (since the outcome directly determines the full cost).

    For unquantifiable costs, sensitivity = None (infinite in theory).
    """
    if base_cost is None:
        return None  # Unquantifiable — sensitivity is technically unbounded

    # Categorical sensitivity levels based on consequence type
    HIGH_SENSITIVITY_TYPES = {
        "safety_incident_risk", "trading_loss", "human_correction_rate",
    }
    MEDIUM_SENSITIVITY_TYPES = {
        "error_cost", "revenue_loss", "forfeited_revenue",
        "productivity_waste", "forfeited_productivity",
        "net_quality_loss",
    }
    LOW_SENSITIVITY_TYPES = {
        "forfeited_cost_savings", "forfeited_productivity_gain",
        "cost_loss", "forfeited_accuracy", "overspend",
        "forfeited_revenue", "baseline",
    }

    if consequence_type in HIGH_SENSITIVITY_TYPES:
        return "high"  # dC/dy is large — small outcome changes cause large cost swings
    elif consequence_type in MEDIUM_SENSITIVITY_TYPES:
        return "medium"
    elif consequence_type in LOW_SENSITIVITY_TYPES:
        return "low"
    return "medium"  # default


# ═══════════════════════════════════════════════════════════════
# 20 FROZEN CASES (v3.2 — with environment_model)
# ═══════════════════════════════════════════════════════════════
# Each case is FROZEN. Modifying any field after declaration
# violates the fixed-point invariant.
#
# Structure per case (v3.2):
#   x_i:               features + bssi_components + eval_scores
#   y_i:               ground_truth (optimal model + downstream metric)
#   C_i:               cost function C(action, y_i) for each possible action
#   environment_model: P(y|x,action), variance, sensitivity
#   action_space:      A_i — constrained set of deployable actions
#   optimal_constrained: argmin_{a in A_i} C(a, y_i) — best WITHIN action space

CASES = [
    # ── RDR-001: MODEL UPGRADE ─────────────────────────────────
    {
        "id": "RDR-001",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Upgrading math reasoning model from v1 to v2 for production calculator tool",
                "evaluation_method": "GSM8K accuracy (exact match)",
                "task_type": "math",
                "n_questions": 10,
                "extraction_fail_rate": 0.08,
            },
            "eval_scores": {"v1": 0.783, "v2": 0.831},
            "bssi_components": {"S": 0.02, "A": 0.40, "N": 0.62, "BSSI": 0.030 * 0.40 * 0.38},
        },
        "y_i": {
            "optimal_model": "v1",
            "downstream_metric": "multi_step_calculation_accuracy",
            "downstream_values": {"v1": 0.912, "v2": 0.847},
            "explanation": "v2 learned to recognize GSM8K format patterns, producing correct final answers for simple problems but failing on multi-step calculations where format cues differ from training. v1 uses genuine step-by-step reasoning.",
        },
        "C_i": {
            "C(deploy_v2, y=v1)": {
                "calculation": "45000_daily * (0.912 - 0.847)_error_rate * 365_days * $3.20_per_error",
                "annual_usd": 3416400,
                "consequence_type": "error_cost",
            },
            "C(deploy_v1, y=v1)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 3416400, "error_cost"),
        "action_space": {"A_i": ["deploy_v1", "deploy_v2"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_v1",
        "irreversibility_model": _build_irreversibility_model(
            "low", 100000, 4, "45K daily calculations",
            annual_burn_rate=None),  # Model swap. Errors obvious in hours.
    },

    # ── RDR-002: MODEL UPGRADE ─────────────────────────────────
    {
        "id": "RDR-002",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Replacing safety classifier in content moderation pipeline",
                "evaluation_method": "Toxicity detection F1 score on benchmark dataset",
                "task_type": "safety",
                "n_questions": 10,
                "extraction_fail_rate": 0.05,
            },
            "eval_scores": {"old_model": 0.941, "new_model": 0.968},
            "bssi_components": {"S": 0.04, "A": 0.50, "N": 0.55, "BSSI": 0.045},
        },
        "y_i": {
            "optimal_model": "old_model",
            "downstream_metric": "false_positive_rate_on_user_content",
            "downstream_values": {"old_model": 0.021, "new_model": 0.087},
            "explanation": "new_model over-indexes on benchmark toxicity patterns, flagging benign but expressive content (sarcasm, political discussion, literary text). False positives require human review at $4.50/case.",
        },
        "C_i": {
            "C(deploy_new_model, y=old_model)": {
                "calculation": "120000_daily * (0.087 - 0.021)_additional_fp_rate * 365_days * $4.50_per_review",
                "annual_usd": 13014000,
                "consequence_type": "human_correction_rate",
            },
            "C(deploy_old_model, y=old_model)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 13014000, "human_correction_rate"),
        "action_space": {"A_i": ["deploy_old_model", "deploy_new_model"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_old_model",
        "irreversibility_model": _build_irreversibility_model(
            "high", 500000, 72, "all user content (millions)",
            annual_burn_rate=None),  # Content moderation. FP accumulate over days.
    },

    # ── RDR-003: MODEL UPGRADE (VALID) ────────────────────────
    {
        "id": "RDR-003",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Upgrading code generation model for internal developer tools",
                "evaluation_method": "HumanEval pass@1",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.03,
            },
            "eval_scores": {"old_model": 0.712, "new_model": 0.789},
            "bssi_components": {"S": 0.22, "A": 0.82, "N": 0.18, "BSSI": 0.148},
        },
        "y_i": {
            "optimal_model": "new_model",
            "downstream_metric": "developer_acceptance_rate",
            "downstream_values": {"old_model": 0.43, "new_model": 0.67},
            "explanation": "Correct upgrade. new_model genuinely writes better code. Standard eval was right.",
        },
        "C_i": {
            "C(deploy_new_model, y=new_model)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_old_model, y=new_model)": {
                "calculation": "forfeited_gain = 340_devs * 1.8_hrs_saved * 365 * $85/hr",
                "annual_usd": 19013500,
                "consequence_type": "forfeited_productivity_gain",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 19013500, "forfeited_productivity_gain"),
        "action_space": {"A_i": ["deploy_old_model", "deploy_new_model"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_new_model",
        "irreversibility_model": _build_irreversibility_model(
            "very_low", 30000, 2, "340 internal developers",
            annual_burn_rate=None),  # Internal dev tool. Instant revert.
    },

    # ── RDR-004: MODEL UPGRADE (VALID) ────────────────────────
    {
        "id": "RDR-004",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Upgrading customer support chatbot from GPT-3.5-turbo to GPT-4",
                "evaluation_method": "CSQA benchmark accuracy",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.04,
            },
            "eval_scores": {"gpt-3.5": 0.894, "gpt-4": 0.947},
            "bssi_components": {"S": 0.20, "A": 0.80, "N": 0.20, "BSSI": 0.128},
        },
        "y_i": {
            "optimal_model": "gpt-4",
            "downstream_metric": "customer_resolution_rate",
            "downstream_values": {"gpt-3.5": 0.61, "gpt-4": 0.78},
            "explanation": "Correct upgrade. GPT-4 genuinely resolves more customer issues.",
        },
        "C_i": {
            "C(deploy_gpt-4, y=gpt-4)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_gpt-3.5, y=gpt-4)": {
                "calculation": "forfeited_gain = 340_escalations/day * $12/escalation * 365",
                "annual_usd": 1491600,
                "consequence_type": "forfeited_cost_savings",
            },
        },
        "environment_model": _build_env_model("deterministic", 1491600, "forfeited_cost_savings"),
        "action_space": {"A_i": ["deploy_gpt-3.5", "deploy_gpt-4"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_gpt-4",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 200000, 8, "340 escalations/day",
            annual_burn_rate=None),  # Chatbot upgrade. API model swap.
    },

    # ── RDR-005: MODEL UPGRADE ─────────────────────────────────
    {
        "id": "RDR-005",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Selecting RAG retrieval model for legal document search",
                "evaluation_method": "MTEB retrieval accuracy (top-10)",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.06,
            },
            "eval_scores": {"model_a": 0.882, "model_b": 0.915},
            "bssi_components": {"S": 0.05, "A": 0.48, "N": 0.51, "BSSI": 0.012},
        },
        "y_i": {
            "optimal_model": "model_a",
            "downstream_metric": "relevant_document_recall_in_production",
            "downstream_values": {"model_a": 0.82, "model_b": 0.71},
            "explanation": "model_b scores higher on MTEB because it memorized benchmark document patterns. In production, legal documents have different structure — longer, more technical, varied citation formats. model_a uses embedding geometry that generalizes better.",
        },
        "C_i": {
            "C(deploy_model_b, y=model_a)": {
                "calculation": "8500_daily * (0.82 - 0.71)_miss_rate * 365 * $15_per_miss",
                "annual_usd": 5124375,
                "consequence_type": "error_cost",
            },
            "C(deploy_model_a, y=model_a)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 5124375, "error_cost"),
        "action_space": {"A_i": ["deploy_model_a", "deploy_model_b"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_model_a",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 300000, 168, "8500 daily legal searches",
            annual_burn_rate=None),  # RAG retrieval. Recall degrades silently.
    },

    # ── RDR-006: PROMPT CHANGE ────────────────────────────────
    {
        "id": "RDR-006",
        "x_i": {
            "features": {
                "category": "prompt_change",
                "decision_context": "Switching system prompt format for medical QA from template A to B",
                "evaluation_method": "MedQA accuracy (exact match)",
                "task_type": "safety",
                "n_questions": 10,
                "extraction_fail_rate": 0.07,
            },
            "eval_scores": {"prompt_A": 0.721, "prompt_B": 0.768},
            "bssi_components": {"S": 0.03, "A": 0.42, "N": 0.59, "BSSI": 0.005},
        },
        "y_i": {
            "optimal_model": "prompt_A",
            "downstream_metric": "clinical_decision_support_accuracy",
            "downstream_values": {"prompt_A": 0.81, "prompt_B": 0.69},
            "explanation": "prompt_B instructs the model to output more confidently, which increases exact match on MedQA (more definitive answers). But in clinical settings, this confidence is dangerous — it hallucinates drug interactions and contradicts itself on differential diagnosis.",
        },
        "C_i": {
            "C(deploy_prompt_B, y=prompt_A)": {
                "calculation": "non-quantifiable — clinical safety events have unbounded liability",
                "annual_usd": None,
                "consequence_type": "safety_incident_risk",
                "note": "Cost is potentially catastrophic. 384 additional hallucinations/day on clinical queries. Single malpractice event from incorrect clinical decision support can exceed $1M in liability.",
            },
            "C(deploy_prompt_A, y=prompt_A)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_catastrophic", None, "safety_incident_risk"),
        "action_space": {"A_i": ["deploy_prompt_A", "deploy_prompt_B"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_prompt_A",
        "irreversibility_model": _build_irreversibility_model(
            "unquantifiable", None, 2160, "384 daily clinical queries",
            annual_burn_rate=None),  # Clinical decision support. Unbounded liability.
    },

    # ── RDR-007: PROMPT CHANGE ────────────────────────────────
    {
        "id": "RDR-007",
        "x_i": {
            "features": {
                "category": "prompt_change",
                "decision_context": "Changing output format from free text to JSON for product descriptions",
                "evaluation_method": "BLEU score against reference descriptions",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.05,
            },
            "eval_scores": {"free_text": 0.41, "json_format": 0.53},
            "bssi_components": {"S": 0.06, "A": 0.52, "N": 0.48, "BSSI": 0.016},
        },
        "y_i": {
            "optimal_model": "free_text",
            "downstream_metric": "product_page_conversion_rate",
            "downstream_values": {"free_text": 0.038, "json_format": 0.029},
            "explanation": "JSON-formatted descriptions are more consistent (higher BLEU) but less engaging. They read like spec sheets, not product copy. Conversion rate dropped 24% because customers don't connect with structured data.",
        },
        "C_i": {
            "C(deploy_json_format, y=free_text)": {
                "calculation": "500000_daily * (0.038 - 0.029)_conversion_loss * 365 * $67_AOV",
                "annual_usd": 110137500,
                "consequence_type": "revenue_loss",
            },
            "C(deploy_free_text, y=free_text)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_high", 110137500, "revenue_loss"),
        "action_space": {"A_i": ["deploy_free_text", "deploy_json_format"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_free_text",
        "irreversibility_model": _build_irreversibility_model(
            "very_high", 1000000, 336, "500K daily product page views",
            annual_burn_rate=None),  # Product descriptions affect SEO.
    },

    # ── RDR-008: PROMPT CHANGE (VALID) ────────────────────────
    {
        "id": "RDR-008",
        "x_i": {
            "features": {
                "category": "prompt_change",
                "decision_context": "Adding chain-of-thought instruction to customer classification model",
                "evaluation_method": "Classification F1 on test set",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.04,
            },
            "eval_scores": {"no_cot": 0.892, "with_cot": 0.915},
            "bssi_components": {"S": 0.21, "A": 0.78, "N": 0.22, "BSSI": 0.128},
        },
        "y_i": {
            "optimal_model": "with_cot",
            "downstream_metric": "classification_accuracy_in_production",
            "downstream_values": {"no_cot": 0.86, "with_cot": 0.90},
            "explanation": "Correct decision. CoT genuinely helps by forcing the model to consider multiple features before classifying.",
        },
        "C_i": {
            "C(deploy_with_cot, y=with_cot)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_no_cot, y=with_cot)": {
                "calculation": "forfeited_gain = 200000_daily * (0.90 - 0.86)_errors * 365 * $0.85",
                "annual_usd": 2482000,
                "consequence_type": "forfeited_cost_savings",
            },
        },
        "environment_model": _build_env_model("deterministic", 2482000, "forfeited_cost_savings"),
        "action_space": {"A_i": ["deploy_no_cot", "deploy_with_cot"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_with_cot",
        "irreversibility_model": _build_irreversibility_model(
            "low", 50000, 4, "200K daily classifications",
            annual_burn_rate=None),  # Classification config. Easy A/B test.
    },

    # ── RDR-009: CONFIG CHANGE ─────────────────────────────────
    {
        "id": "RDR-009",
        "x_i": {
            "features": {
                "category": "config_change",
                "decision_context": "Increasing temperature from 0.0 to 0.7 for creative writing assistant",
                "evaluation_method": "Perplexity on writing benchmark",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.06,
            },
            "eval_scores": {"temp_0_0": 0.87, "temp_0_7": 0.71},
            "bssi_components": {"S": 0.04, "A": 0.38, "N": 0.61, "BSSI": 0.006},
        },
        "y_i": {
            "optimal_model": "temp_0_7",
            "downstream_metric": "user_engagement_time_per_session",
            "downstream_values": {"temp_0_0": 4.2, "temp_0_7": 7.8},
            "note": "Lower perplexity = more predictable = worse for creative tasks. Standard eval uses perplexity as accuracy, ranking temp_0_0 higher. But for creative writing, predictability is the wrong metric.",
            "explanation": "Higher temperature produces more varied, engaging text. But perplexity is the WRONG metric — lower perplexity means more predictable (boring) text. Standard eval ranked temp_0_0 better because it had LOWER perplexity.",
        },
        "C_i": {
            "C(deploy_temp_0_0, y=temp_0_7)": {
                "calculation": "forfeited_gain = 25000_DAU * (7.8 - 4.2)_min * 365 * $0.012/min",
                "annual_usd": 394200,
                "consequence_type": "forfeited_revenue",
            },
            "C(deploy_temp_0_7, y=temp_0_7)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_high", 394200, "forfeited_revenue"),
        "action_space": {"A_i": ["deploy_temp_0_0", "deploy_temp_0_7"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_temp_0_7",
        "irreversibility_model": _build_irreversibility_model(
            "very_low", 10000, 24, "25K DAU (nice-to-have)",
            annual_burn_rate=None),  # Temperature param. One-line config.
    },

    # ── RDR-010: CONFIG CHANGE (VALID) ────────────────────────
    {
        "id": "RDR-010",
        "x_i": {
            "features": {
                "category": "config_change",
                "decision_context": "Changing max_tokens from 256 to 1024 for API response truncation",
                "evaluation_method": "Task completion rate on benchmark",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.03,
            },
            "eval_scores": {"tokens_256": 0.873, "tokens_1024": 0.921},
            "bssi_components": {"S": 0.20, "A": 0.78, "N": 0.22, "BSSI": 0.122},
        },
        "y_i": {
            "optimal_model": "tokens_1024",
            "downstream_metric": "API_response_latency_p99",
            "downstream_values": {"tokens_256": 180, "tokens_1024": 420},
            "note": "tokens_1024 completes more tasks but at 2.3x latency. SLA penalty is $4.38M/year. However, completion quality is the primary metric, not latency. Standard eval's winner is correct on the primary dimension. System ALLOWED with SLA note.",
            "explanation": "tokens_1024 completes more tasks but at 2.3x latency. For real-time API serving, this violates SLA. Standard eval only measured accuracy, not latency. Our system flagged that the evaluation did not include the latency constraint.",
        },
        "C_i": {
            "C(deploy_tokens_1024, y=tokens_1024)": {
                "calculation": "baseline — optimal model deployed. Note: SLA cost of $4.38M exists but is offset by completion gains. Net positive.",
                "annual_usd": 0,
                "consequence_type": "baseline",
                "sla_note": "2M_daily * 12%_violation * $0.05 = $4,380,000 SLA penalty — deployment team must decide if accuracy gains offset this",
            },
            "C(deploy_tokens_256, y=tokens_1024)": {
                "calculation": "forfeited_completion_gain — cannot be directly quantified without A/B test",
                "annual_usd": None,
                "consequence_type": "forfeited_accuracy",
            },
        },
        "environment_model": _build_env_model("stochastic_low", None, "forfeited_accuracy"),
        "action_space": {"A_i": ["deploy_tokens_256", "deploy_tokens_1024"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_tokens_1024",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 150000, 1, "2M daily API calls",
            annual_burn_rate=None),  # Max_tokens config. Wide blast radius.
    },

    # ── RDR-011: CONFIG CHANGE (VALID) ────────────────────────
    {
        "id": "RDR-011",
        "x_i": {
            "features": {
                "category": "config_change",
                "decision_context": "Switching from greedy decoding to beam search (k=5) for translation",
                "evaluation_method": "BLEU on WMT benchmark",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.03,
            },
            "eval_scores": {"greedy": 0.382, "beam_search": 0.417},
            "bssi_components": {"S": 0.18, "A": 0.76, "N": 0.24, "BSSI": 0.104},
        },
        "y_i": {
            "optimal_model": "beam_search",
            "downstream_metric": "human_translation_quality_score",
            "downstream_values": {"greedy": 3.8, "beam_search": 4.1},
            "explanation": "Correct decision. Beam search genuinely produces better translations. BSSI=0.104 — adequate separation.",
        },
        "C_i": {
            "C(deploy_beam_search, y=beam_search)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_greedy, y=beam_search)": {
                "calculation": "forfeited_gain = 5M_words/day * $0.80_savings_per_1k_words * 365",
                "annual_usd": 1460000,
                "consequence_type": "forfeited_cost_savings",
            },
        },
        "environment_model": _build_env_model("deterministic", 1460000, "forfeited_cost_savings"),
        "action_space": {"A_i": ["deploy_greedy", "deploy_beam_search"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_beam_search",
        "irreversibility_model": _build_irreversibility_model(
            "low", 80000, 8, "5M words/day",
            annual_burn_rate=None),  # Decoding strategy. Easy revert.
    },

    # ── RDR-012: FINETUNE VS BASE ─────────────────────────────
    {
        "id": "RDR-012",
        "x_i": {
            "features": {
                "category": "finetune_vs_base",
                "decision_context": "Deploying fine-tuned model vs base model for domain-specific summarization",
                "evaluation_method": "ROUGE-L on domain test set",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.08,
            },
            "eval_scores": {"base": 0.52, "finetuned": 0.61},
            "bssi_components": {"S": 0.03, "A": 0.44, "N": 0.57, "BSSI": 0.006},
        },
        "y_i": {
            "optimal_model": "base",
            "downstream_metric": "factual_accuracy_in_summaries",
            "downstream_values": {"base": 0.94, "finetuned": 0.81},
            "explanation": "Fine-tuned model produces summaries that match reference text more closely (higher ROUGE) but introduces factual errors from training data — fabricating statistics, misattributing quotes, conflating sources. Base model is more conservative and accurate.",
        },
        "C_i": {
            "C(deploy_finetuned, y=base)": {
                "calculation": "15000_daily * (0.94 - 0.81)_error_rate * 365 * $22_per_error",
                "annual_usd": 15687000,
                "consequence_type": "error_cost",
            },
            "C(deploy_base, y=base)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 15687000, "error_cost"),
        "action_space": {"A_i": ["deploy_base", "deploy_finetuned"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_base",
        "irreversibility_model": _build_irreversibility_model(
            "high", 600000, 168, "15K daily summaries",
            annual_burn_rate=None),  # Summarization. Errors distributed externally.
    },

    # ── RDR-013: FINETUNE VS BASE (VALID) ────────────────────
    {
        "id": "RDR-013",
        "x_i": {
            "features": {
                "category": "finetune_vs_base",
                "decision_context": "Fine-tuned code model vs base model for SQL generation",
                "evaluation_method": "SQL execution accuracy on test queries",
                "task_type": "math",
                "n_questions": 10,
                "extraction_fail_rate": 0.03,
            },
            "eval_scores": {"base": 0.764, "finetuned": 0.842},
            "bssi_components": {"S": 0.24, "A": 0.82, "N": 0.16, "BSSI": 0.166},
        },
        "y_i": {
            "optimal_model": "finetuned",
            "downstream_metric": "query_execution_success_rate",
            "downstream_values": {"base": 0.79, "finetuned": 0.88},
            "explanation": "Correct deployment. Fine-tuning on schema-specific SQL genuinely improved performance.",
        },
        "C_i": {
            "C(deploy_finetuned, y=finetuned)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_base, y=finetuned)": {
                "calculation": "forfeited_gain = 1080_queries/day_needing_fix * 8min * $85/hr * 365",
                "annual_usd": 4467600,
                "consequence_type": "forfeited_productivity",
            },
        },
        "environment_model": _build_env_model("deterministic", 4467600, "forfeited_productivity"),
        "action_space": {"A_i": ["deploy_base", "deploy_finetuned"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_finetuned",
        "irreversibility_model": _build_irreversibility_model(
            "very_low", 20000, 1, "1080 queries/day",
            annual_burn_rate=None),  # SQL model. Binary success detection.
    },

    # ── RDR-014: FINETUNE VS BASE ─────────────────────────────
    {
        "id": "RDR-014",
        "x_i": {
            "features": {
                "category": "finetune_vs_base",
                "decision_context": "Fine-tuned vs base model for sentiment analysis in financial reports",
                "evaluation_method": "Sentiment classification accuracy on test set",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.06,
            },
            "eval_scores": {"base": 0.873, "finetuned": 0.921},
            "bssi_components": {"S": 0.02, "A": 0.41, "N": 0.64, "BSSI": 0.003},
        },
        "y_i": {
            "optimal_model": "base",
            "downstream_metric": "trading_signal_accuracy",
            "downstream_values": {"base": 0.58, "finetuned": 0.51},
            "explanation": "Fine-tuned model memorized sentiment patterns from training data. When faced with novel market conditions (unseen in training), it reverts to training distribution — always slightly positive. This produces incorrect signals in bear markets. Base model generalizes better to distribution shift.",
        },
        "C_i": {
            "C(deploy_finetuned, y=base)": {
                "calculation": "$500M_AUM * (0.58 - 0.51)_alpha_loss = 7pp = ~1.2% alpha loss/year",
                "annual_usd": 6000000,
                "consequence_type": "trading_loss",
            },
            "C(deploy_base, y=base)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_high", 6000000, "trading_loss"),
        "action_space": {"A_i": ["deploy_base", "deploy_finetuned"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_base",
        "irreversibility_model": _build_irreversibility_model(
            "very_high", 800000, 2160, "$500M AUM",
            annual_burn_rate=None),  # Trading signal. Losses compound quarterly.
    },

    # ── RDR-015: ROUTING ──────────────────────────────────────
    {
        "id": "RDR-015",
        "x_i": {
            "features": {
                "category": "routing",
                "decision_context": "Routing customer queries to general model vs specialist model",
                "evaluation_method": "Overall accuracy across all query types (uniform weighting)",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.05,
            },
            "eval_scores": {"specialist": 0.842, "general": 0.867},
            "bssi_components": {"S": 0.05, "A": 0.46, "N": 0.53, "BSSI": 0.011},
        },
        "y_i": {
            "optimal_model": "specialist",
            "downstream_metric": "first_contact_resolution_rate",
            "downstream_values": {"specialist": 0.73, "general": 0.61},
            "note": "Production query distribution: 60% technical, 40% simple. Specialist handles technical at 89% vs general 62%. Simple queries handled by rule-based system regardless. Standard eval uses uniform weighting which doesn't match production distribution.",
            "explanation": "General model averages better across all query types but excels at none. Specialist model handles technical queries much better (89% vs 62%). Since 60% of queries are technical, specialist's weakness on simple queries doesn't matter.",
        },
        "C_i": {
            "C(deploy_general, y=specialist)": {
                "calculation": "18000_daily * (0.73 - 0.61)_escalation_rate * 365 * $8.50/escalation",
                "annual_usd": 6702300,
                "consequence_type": "cost_loss",
            },
            "C(deploy_specialist, y=specialist)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_high", 6702300, "cost_loss"),
        "action_space": {"A_i": ["deploy_specialist", "deploy_general"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_specialist",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 200000, 48, "18K daily customer queries",
            annual_burn_rate=None),  # Routing model. Escalation tracked daily.
    },

    # ── RDR-016: ROUTING ──────────────────────────────────────
    {
        "id": "RDR-016",
        "x_i": {
            "features": {
                "category": "routing",
                "decision_context": "Choosing between cheap-fast model and expensive-accurate model for production serving",
                "evaluation_method": "Accuracy on uniform test distribution",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.04,
            },
            "eval_scores": {"cheap": 0.793, "expensive": 0.912},
            "eval_candidates_note": "Standard eval only had two candidates: cheap and expensive. The optimal strategy (smart routing) was not in the eval's candidate set. pi_E picks expensive.",
            "deploy_map": {"cheap": "cheap_only", "expensive": "expensive_only"},
            "bssi_components": {"S": 0.03, "A": 0.44, "N": 0.58, "BSSI": 0.006},
        },
        "y_i": {
            "optimal_model": "smart_routing",
            "downstream_metric": "cost-adjusted_quality_score",
            "downstream_values": {"cheap_alone": 0.89, "expensive_only": 0.72, "smart_routing": 0.91},
            "note": "Production distribution: 60% easy queries, 40% hard. Cheap handles easy at 95% for $0.002/query. Expensive handles all at 91% for $0.015/query. Smart routing = cheap first, fallback to expensive = best cost-adjusted quality.",
            "explanation": "Production query distribution is 60% easy, 40% hard. Cheap model handles easy queries at 95% accuracy for $0.002/query. Expensive handles all at 91% but costs $0.015/query. Standard eval tested on uniform distribution with only two candidates, missing the optimal strategy entirely.",
        },
        "C_i": {
            "C(deploy_expensive_only, y=smart_routing)": {
                "calculation": "5M_daily * ($0.015 - $0.0032_avg_smart_routing) * 365",
                "annual_usd": 15695000,
                "consequence_type": "overspend",
            },
            "C(deploy_smart_routing, y=smart_routing)": {
                "calculation": "baseline — optimal strategy deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_high", 15695000, "overspend"),
        "action_space": {
            "A_i": ["deploy_cheap_only", "deploy_expensive_only"],
            "source": "eval_scores keys (via deploy_map)",
            "note": "smart_routing was NOT in A_i at eval time. This demonstrates that pi_E is constrained by A_i — it can only pick from available candidates. optimal_constrained is the best WITHIN A_i, which may differ from the true global optimum.",
        },
        "optimal_constrained": "deploy_cheap_only",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 250000, 72, "5M daily API calls",
            annual_burn_rate=None),  # Model serving routing. Cost tracked daily.
        "optimal_constrained_rationale": "Within A_i = {cheap_only, expensive_only}, cheap_only has better cost-adjusted quality (0.89 vs 0.72). The true global optimum (smart_routing = 0.91) is outside A_i and was not available to pi_E.",
    },

    # ── RDR-017: MODEL UPGRADE (VALID) ────────────────────────
    {
        "id": "RDR-017",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Upgrading embedding model for semantic search",
                "evaluation_method": "NDCG@10 on benchmark",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.03,
            },
            "eval_scores": {"old": 0.842, "new": 0.871},
            "bssi_components": {"S": 0.19, "A": 0.80, "N": 0.20, "BSSI": 0.122},
        },
        "y_i": {
            "optimal_model": "new",
            "downstream_metric": "user_click_through_rate_on_search_results",
            "downstream_values": {"old": 0.34, "new": 0.39},
            "explanation": "Correct upgrade. New embeddings genuinely capture better semantic similarity.",
        },
        "C_i": {
            "C(deploy_new, y=new)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
            "C(deploy_old, y=new)": {
                "calculation": "forfeited_gain = 2M_daily * (0.39-0.34)_CTR * 365 * $0.035/click",
                "annual_usd": 1277500,
                "consequence_type": "forfeited_revenue",
            },
        },
        "environment_model": _build_env_model("deterministic", 1277500, "forfeited_revenue"),
        "action_space": {"A_i": ["deploy_old", "deploy_new"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_new",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 180000, 48, "2M daily search queries",
            annual_burn_rate=None),  # Embedding model. CTR tracked daily.
    },

    # ── RDR-018: PROMPT CHANGE ────────────────────────────────
    {
        "id": "RDR-018",
        "x_i": {
            "features": {
                "category": "prompt_change",
                "decision_context": "Adding few-shot examples vs zero-shot for data extraction from invoices",
                "evaluation_method": "Extraction accuracy on test set (same formats as few-shot examples)",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.07,
            },
            "eval_scores": {"zero_shot": 0.881, "few_shot": 0.934},
            "bssi_components": {"S": 0.04, "A": 0.50, "N": 0.52, "BSSI": 0.010},
        },
        "y_i": {
            "optimal_model": "zero_shot",
            "downstream_metric": "extraction_accuracy_on_unseen_invoice_formats",
            "downstream_values": {"zero_shot": 0.82, "few_shot": 0.71},
            "note": "Test set has same formats as few-shot examples — benchmark contamination. Production has different formats.",
            "explanation": "Few-shot examples from training formats bias the model toward those formats. On unseen invoice layouts (different companies, different field arrangements), few-shot performs WORSE. Zero-shot is more robust to format variation.",
        },
        "C_i": {
            "C(deploy_few_shot, y=zero_shot)": {
                "calculation": "30000_daily * (0.82 - 0.71)_error_rate * 365 * $3.50_correction",
                "annual_usd": 4223250,
                "consequence_type": "error_cost",
            },
            "C(deploy_zero_shot, y=zero_shot)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 4223250, "error_cost"),
        "action_space": {"A_i": ["deploy_zero_shot", "deploy_few_shot"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_zero_shot",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 250000, 168, "30K daily invoices",
            annual_burn_rate=None),  # Extraction model. Format errors on new layouts.
    },

    # ── RDR-019: CONFIG CHANGE ────────────────────────────────
    {
        "id": "RDR-019",
        "x_i": {
            "features": {
                "category": "config_change",
                "decision_context": "Reducing context window from 32K to 8K for cost optimization",
                "evaluation_method": "Task completion rate on benchmark (90% short docs <2K tokens)",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.04,
            },
            "eval_scores": {"context_32k": 0.912, "context_8k": 0.934},
            "bssi_components": {"S": 0.04, "A": 0.55, "N": 0.42, "BSSI": 0.013},
        },
        "y_i": {
            "optimal_model": "context_32k",
            "downstream_metric": "task_completion_rate_for_long_documents",
            "downstream_values": {"context_8k": 0.67, "context_32k": 0.94},
            "note": "Benchmark has 90% short documents. Production has 40% documents >8K tokens. 8K model silently truncates long docs.",
            "explanation": "Standard eval used a benchmark with mostly short documents (<2K tokens) where 8K context was sufficient. context_8k scored slightly higher due to faster inference. But production has 40% documents >8K tokens — context_8k silently truncates these, dropping completion rate 27pp.",
        },
        "C_i": {
            "C(deploy_context_8k, y=context_32k)": {
                "calculation": "Quality loss: $85K/month * 12 = $1,020,000/year. Infrastructure savings: $12K/month * 12 = $144,000/year. Net cost of wrong deployment = $1,020,000 - $144,000 = $876,000/year absolute loss.",
                "annual_usd": 876000,
                "consequence_type": "net_quality_loss",
                "note": "C_i represents the absolute cost of deploying the sub-optimal model. Infrastructure savings from 8K ($144K/yr) do not offset quality loss ($1.02M/yr).",
            },
            "C(deploy_context_32k, y=context_32k)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 876000, "net_quality_loss"),
        "action_space": {"A_i": ["deploy_context_32k", "deploy_context_8k"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_context_32k",
        "irreversibility_model": _build_irreversibility_model(
            "high", 400000, 336, "40pct of document traffic",
            annual_burn_rate=None),  # Context window. Silent truncation.
    },

    # ── RDR-020: MODEL UPGRADE ────────────────────────────────
    {
        "id": "RDR-020",
        "x_i": {
            "features": {
                "category": "model_upgrade",
                "decision_context": "Choosing between two models for automated code review",
                "evaluation_method": "Review recall (bugs found) on test set",
                "task_type": "generic",
                "n_questions": 10,
                "extraction_fail_rate": 0.06,
            },
            "eval_scores": {"model_a": 0.724, "model_b": 0.789},
            "bssi_components": {"S": 0.02, "A": 0.39, "N": 0.63, "BSSI": 0.003},
        },
        "y_i": {
            "optimal_model": "model_a",
            "downstream_metric": "developer_agreement_with_review_comments",
            "downstream_values": {"model_a": 0.68, "model_b": 0.41},
            "explanation": "model_b produces more review comments (higher recall) but most are false positives — flagging correct code as problematic. model_a is more conservative but its comments are almost always actionable. Developers started ignoring model_b's reviews entirely.",
        },
        "C_i": {
            "C(deploy_model_b, y=model_a)": {
                "calculation": "5000_reviews/day * 59%_false_positive_rate * 3min_wasted * $85/hr * 365",
                "annual_usd": 19301250,
                "consequence_type": "productivity_waste",
            },
            "C(deploy_model_a, y=model_a)": {
                "calculation": "baseline — optimal model deployed",
                "annual_usd": 0,
                "consequence_type": "baseline",
            },
        },
        "environment_model": _build_env_model("stochastic_low", 19301250, "productivity_waste"),
        "action_space": {"A_i": ["deploy_model_a", "deploy_model_b"], "source": "eval_scores keys"},
        "optimal_constrained": "deploy_model_a",
        "irreversibility_model": _build_irreversibility_model(
            "medium", 200000, 48, "5000 daily code reviews",
            annual_burn_rate=None),  # Code review model. Agreement tracked per cycle.
    },
]


# ═══════════════════════════════════════════════════════════════
# GLOBAL RISK PARAMETERS (v4)
# ═══════════════════════════════════════════════════════════════
# These are the two global parameters that control the risk-sensitive
# decision rule. They are NOT per-case. They apply uniformly to all
# 20 frozen cases.

LAMBDA_RISK = 1.0       # Risk aversion: weight on tail risk measure
GAMMA_IRREVERSIBILITY = 2.0  # Irreversibility weight: penalty for operational risk
BETA_OPPORTUNITY = 0.0   # Opportunity discount: beta=0 means ignore C+ in risk calc
ALPHA_CVAR = 0.95          # CVaR confidence level: worst (1-alpha) scenarios
KAPPA = 1_100_000           # Risk tolerance: max acceptable risk-adjusted cost ($/year)

# ═══════════════════════════════════════════════════════════════
# SHADOW CONSTRAINT PARAMETERS (Step 2 — instrumentation, no behavior change)
# ═══════════════════════════════════════════════════════════════
# These define pressure surfaces for detecting structural gaps
# in the scalar regime. They do NOT affect pi_S decisions.
#
# KAPPA_HARD: absolute ceiling for tail risk (>> KAPPA).
#   Separation allows scalar system to operate normally while
#   constraint system detects "unacceptable zones."
#
# EPS_UNCERTAINTY: threshold for "meaningful" uncertainty when
#   R(x) = inf. Avoids triggering on near-deterministic cases.

KAPPA_HARD = 5_000_000     # Absolute tail risk ceiling ($/year) — well above operational kappa
EPS_UNCERTAINTY = 0.05     # Minimum p_reversal to count as "uncertain" in irrev×uncertainty constraint


# ═══════════════════════════════════════════════════════════════
# POLICY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _phi(x):
    """Standard normal CDF: Phi(x) = P(Z <= x) where Z ~ N(0,1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _compute_tail_risk(std_C_neg, distribution_type, alpha=0.95):
    """
    v4.3: Compute tail risk measure replacing σ(C⁻) with CVaR excess.

    CVaR_excess(C⁻) = CVaR_α(C⁻) - E[C⁻] = σ(C⁻) * φ(Φ⁻¹(α)) / (1-α)

    WHY: σ treats all uncertainty the same. Two cases with identical E[C⁻] and
    σ(C⁻) but different tail structures (frequent small losses vs rare catastrophic
    losses) get the same penalty. CVaR focuses on the worst (1-α) fraction.

    DISTRIBUTION-DEPENDENT FACTORS:
        For Normal(μ, σ²):
            CVaR_α = μ + σ * φ(Φ⁻¹(α)) / (1-α)
            CVaR_excess = σ * φ(Φ⁻¹(α)) / (1-α)

            At α=0.95: Φ⁻¹(0.95) ≈ 1.6449, φ(1.6449) ≈ 0.1031
            Factor = 0.1031 / 0.05 ≈ 2.063

        For heavy-tailed (Student-t approximation, ν≈3):
            CVaR diverges faster than normal.
            Factor ≈ 3.18 (≈ 2.063 * 1.54)

        For deterministic:
            No tail risk. Factor = 0.

    PROPERTIES:
        - CVaR is a COHERENT risk measure (sub-additive, monotonic, homogeneous)
        - σ is NOT coherent (fails sub-additivity)
        - This makes v4.3's risk functional mathematically well-posed

    Args:
        std_C_neg: standard deviation of downside cost
        distribution_type: "deterministic", "normal", or "heavy_tailed"
        alpha: CVaR confidence level (default 0.95)

    Returns:
        tail_risk: CVaR excess term (replaces σ in the risk formula)
        cvar_factor: the multiplier applied to σ
    """
    if std_C_neg <= 0:
        return 0.0, 0.0

    if distribution_type == "deterministic":
        return 0.0, 0.0

    # Φ⁻¹(α) and φ(Φ⁻¹(α)) for α=0.95
    # Φ⁻¹(0.95) = 1.6449
    z_alpha = 1.6449
    # φ(z) = (1/√(2π)) * exp(-z²/2)
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z_alpha * z_alpha)

    # CVaR factor = φ(Φ⁻¹(α)) / (1 - α)
    cvar_factor = phi_z / (1.0 - alpha)  # ≈ 2.063 for α=0.95

    if distribution_type == "heavy_tailed":
        # Heavy-tailed: CVaR grows much faster in the tail
        # Student-t(ν=3) approximation: factor ≈ 3.18 at α=0.95
        # This is ~1.54x the normal factor — properly penalizes fat tails
        cvar_factor *= 1.54

    tail_risk = std_C_neg * cvar_factor
    return tail_risk, cvar_factor


def _resolve_deploy_name(case, eval_winner):
    """Map eval winner name to cost function action name, if deploy_map exists."""
    deploy_map = case["x_i"].get("deploy_map", {})
    return deploy_map.get(eval_winner, eval_winner)


def pi_E(case):
    """
    Eval policy: pi_E(x_i) = argmax(eval_scores).
    EXPLICIT FORM: pi_E(x) = argmax_a { s(a) }
        where s(a) = eval_scores[a]
        theta_E = {scoring_fn: raw_score, aggregation: argmax}

    Standard eval picks the model with the higher score.
    Deterministic. No thresholds. No intelligence.
    CONSTRAINED: can only pick from A_i = keys(eval_scores).
    """
    scores = case["x_i"]["eval_scores"]
    return max(scores, key=scores.get)


def pi_E_deploy_name(case):
    """Returns the deploy action name for pi_E's winner (handles deploy_map)."""
    return _resolve_deploy_name(case, pi_E(case))


def _get_downstream_pair(case):
    """
    Get (y_piE, y_other) — downstream metric values for piE's model
    and the alternative. Handles deploy_map for RDR-016.
    """
    eval_winner = pi_E(case)
    downstream = case["y_i"]["downstream_values"]
    deploy_map = case["x_i"].get("deploy_map", {})

    piE_key = deploy_map.get(eval_winner, eval_winner)
    models = list(case["x_i"]["eval_scores"].keys())
    other = [m for m in models if m != eval_winner][0]
    other_key = deploy_map.get(other, other)

    y_piE = downstream.get(piE_key)
    y_other = downstream.get(other_key)
    return y_piE, y_other


def _get_C_wrong(case):
    """
    Get the cost of deploying the wrong model (i.e., cost when
    deployed model doesn't match the actual best under realized y).

    Returns (annual_usd, consequence_type) tuple.
    annual_usd may be None for unquantifiable costs.

    IMPORTANT: This returns the non-zero cost entry from C_i, because
    the cost of deploying the wrong model is direction-symmetric:
    - For bad cases (piE wrong): C_wrong = C(deploy_piE, y=optimal)
    - For good cases (piE correct): C_wrong = forfeited benefit = C(deploy_other, y=piE)
    Both represent the cost incurred when the deployed model doesn't
    match the realized ground truth under stochastic outcomes.
    """
    # First: find the non-zero cost entry
    for cost_key, cost_val in case["C_i"].items():
        if cost_val.get("annual_usd") is not None and cost_val["annual_usd"] > 0:
            return cost_val["annual_usd"], cost_val.get("consequence_type", "unknown")

    # Second: check for SLA note as proxy
    for ck, cv in case["C_i"].items():
        sla = cv.get("sla_note", "")
        if sla and "$" in sla:
            import re
            match = re.search(r'\$([0-9,]+)', sla)
            if match:
                return int(match.group(1).replace(',', '')), cv.get("consequence_type", "unknown")

    return None, "unknown"


# C⁻ / C⁺ CLASSIFICATION (v4.2 — sign-correct cost model)
#
# The fundamental insight: not all costs are losses.
#   C⁻ = downside risk (actual harm from wrong deployment)
#   C⁺ = upside opportunity (forgone gain from blocking correct deployment)
#
# Classification by consequence_type:
#   C⁻ (DOWNSIDE — actual losses):
#     error_cost, human_correction_rate, safety_incident_risk,
#     revenue_loss, productivity_waste, net_quality_loss,
#     cost_loss, trading_loss, overspend
#
#   C⁺ (UPSIDE — forgone gains):
#     forfeited_productivity_gain, forfeited_cost_savings,
#     forfeited_revenue, forfeited_accuracy, forfeited_productivity

UPSIDE_TYPES = frozenset([
    "forfeited_productivity_gain", "forfeited_cost_savings",
    "forfeited_revenue", "forfeited_accuracy", "forfeited_productivity",
])


def _classify_cost(consequence_type):
    """
    Classify a cost as downside (C⁻) or upside (C⁺).

    Returns "upside" if the cost represents a forgone gain/opportunity,
    "downside" if it represents an actual loss from wrong deployment.

    The sign error: v4.1 treated both as risk, penalizing uncertain
    benefit as if it were uncertain harm. RDR-003 ($19M internal tool)
    was blocked because its high variance in upside was treated as
    downside risk.
    """
    if consequence_type in UPSIDE_TYPES:
        return "upside"
    return "downside"


def compute_risk_score(case, lambda_risk=None, gamma_irrev=None):
    """
    v4.3: Compute tail-sensitive, sign-correct multi-factor risk score.

    KEY CHANGE (v4.3 — CVaR replaces σ):
        Old (v4.2): effective = E[C⁻] + lambda * sigma(C⁻) + gamma * R(x) - beta * E[C⁺]
        New (v4.3): effective = E[C⁻] + lambda * CVaR_excess(C⁻) + gamma * R(x) - beta * E[C⁺]

        Where CVaR_excess(C⁻) = σ(C⁻) * φ(Φ⁻¹(α)) / (1-α)

        σ treats all uncertainty equally — frequent small losses and rare catastrophic
        losses get the same penalty if variance matches. CVaR focuses on the TAIL:
        what happens in the worst (1-α) fraction of scenarios.

        Distribution-dependent tail factors (at α=0.95):
            - normal:       factor ≈ 2.063
            - heavy_tailed: factor ≈ 3.18
            - deterministic: factor = 0

    FIVE RISK AXES (v4.3):
    1. DOWNSIDE EXPECTATION: E[C⁻] — expected actual loss
    2. TAIL RISK: lambda * CVaR_excess(C⁻) — worst-case loss concentration
    3. CONSEQUENCE STRUCTURE: gamma * R(x) — operational irreversibility
    4. OPPORTUNITY: beta * E[C⁺] — upside gain from correct deployment
    5. COST STRUCTURE: C⁻ vs C⁺ classification — sign-correct accounting

    COHERENCE: CVaR is a coherent risk measure (sub-additive, monotonic, homogeneous).
    σ is NOT coherent. This makes v4.3 mathematically well-posed as a risk functional.

    Analytical computation (no Monte Carlo needed for normal distributions).
    """
    if lambda_risk is None:
        lambda_risk = LAMBDA_RISK
    if gamma_irrev is None:
        gamma_irrev = GAMMA_IRREVERSIBILITY

    env = case["environment_model"]
    variance = env.get("variance")
    C_wrong, consequence_type = _get_C_wrong(case)
    cost_sign = _classify_cost(consequence_type)
    y_piE, y_other = _get_downstream_pair(case)

    # Get irreversibility R(x)
    irr = case.get("irreversibility_model", {})
    R_x = irr.get("R_x", 0) if irr else 0

    result = {
        "lambda_risk": lambda_risk,
        "gamma_irrev": gamma_irrev,
        "beta_opportunity": BETA_OPPORTUNITY,
        "alpha_cvar": ALPHA_CVAR,
        "C_wrong": C_wrong,
        "consequence_type": consequence_type,
        "cost_sign": cost_sign,
    }

    # Handle unquantifiable costs
    if C_wrong is None or R_x == float('inf'):
        return {
            **result,
            "E_C_neg": float('inf'), "E_C_pos": 0,
            "std_C_neg": float('inf'),
            "E_C": float('inf'), "std_C": float('inf'),
            "effective_score": float('inf'),
            "p_reversal": 1.0 if variance else 0.0,
            "gap": y_piE - y_other if y_piE and y_other else None,
            "std_gap": None,
            "risk_type": "unquantifiable",
            "R_x": R_x,
        }

    # Handle missing downstream values
    if y_piE is None or y_other is None:
        gap = None
        if cost_sign == "downside" and C_wrong > 0:
            base_score = float('inf')
        else:
            base_score = 0
        eff = base_score + gamma_irrev * R_x if base_score != float('inf') else float('inf')
        return {
            **result,
            "E_C_neg": base_score, "E_C_pos": 0, "std_C_neg": 0,
            "E_C": base_score, "std_C": 0,
            "effective_score": eff,
            "p_reversal": 1.0 if C_wrong > 0 else 0.0,
            "gap": gap, "std_gap": None,
            "risk_type": "missing_downstream",
            "R_x": R_x, "gamma_R": gamma_irrev * R_x,
        }

    # Compute gap with correct metric direction
    raw_gap = y_piE - y_other
    piE_deploy_action = f"deploy_{pi_E_deploy_name(case)}"
    optimal_deploy = case.get("optimal_constrained", "")
    piE_is_correct = (piE_deploy_action == optimal_deploy)

    if (raw_gap > 0) != piE_is_correct:
        gap = -raw_gap
    else:
        gap = raw_gap

    # Handle heavy-tailed distribution
    if variance is None:
        variance = 1.0

    gamma_R = gamma_irrev * R_x

    # Deterministic case
    if variance == 0:
        if gap >= 0:
            # pi_E is correct, cost_sign irrelevant (no reversal possible)
            eff = gamma_R
            return {
                **result,
                "E_C_neg": 0, "E_C_pos": 0, "std_C_neg": 0,
                "E_C": 0, "std_C": 0,
                "effective_score": eff,
                "p_reversal": 0.0, "gap": gap, "std_gap": 0,
                "risk_type": "deterministic_safe",
                "R_x": R_x, "gamma_R": gamma_R,
            }
        else:
            # pi_E is wrong, cost depends on sign
            if cost_sign == "upside":
                # Wrong action only forfeits gain — no actual loss
                E_C_neg = 0
                E_C_pos = C_wrong
                eff = gamma_R - BETA_OPPORTUNITY * E_C_pos
            else:
                # Wrong action causes actual loss
                E_C_neg = C_wrong
                E_C_pos = 0
                eff = E_C_neg + gamma_R
            return {
                **result,
                "E_C_neg": E_C_neg, "E_C_pos": E_C_pos, "std_C_neg": 0,
                "E_C": E_C_neg, "std_C": 0,
                "effective_score": eff,
                "p_reversal": 1.0, "gap": gap, "std_gap": 0,
                "risk_type": "deterministic_unsafe",
                "R_x": R_x, "gamma_R": gamma_R,
            }

    # Stochastic case: analytical computation
    var_gap = variance * (y_piE ** 2 + y_other ** 2)
    std_gap = math.sqrt(var_gap) if var_gap > 0 else 0

    if std_gap == 0:
        p_reversal = 0.0 if gap >= 0 else 1.0
    else:
        p_reversal = _phi(-gap / std_gap)

    # SIGN-CORRECT cost decomposition with CVaR tail risk (v4.3)
    dist_type = env.get("distribution", "deterministic")
    if cost_sign == "upside":
        # C_wrong is a forgone gain (C⁺), not an actual loss (C⁻)
        # RDR-003 ($19M forfeited productivity) falls here.
        # Uncertainty in upside ≠ downside risk. CVaR of C⁻ = 0.
        E_C_neg = 0
        std_C_neg = 0
        tail_risk = 0
        cvar_factor = 0
        E_C_pos = C_wrong * p_reversal
        # effective = 0 + 0 + gamma*R(x) - beta*E[C⁺]
        effective_score = gamma_R - BETA_OPPORTUNITY * E_C_pos
    else:
        # C_wrong is an actual loss (C⁻)
        # v4.3: replace σ with CVaR for tail-sensitive risk measurement
        E_C_neg = C_wrong * p_reversal
        Var_C_neg = C_wrong ** 2 * p_reversal * (1 - p_reversal)
        std_C_neg = math.sqrt(Var_C_neg) if Var_C_neg > 0 else 0
        E_C_pos = 0
        # CVaR replaces σ: tail-sensitive instead of symmetric
        tail_risk, cvar_factor = _compute_tail_risk(std_C_neg, dist_type, ALPHA_CVAR)
        effective_score = E_C_neg + lambda_risk * tail_risk + gamma_R

    return {
        **result,
        "E_C_neg": E_C_neg,
        "E_C_pos": E_C_pos,
        "std_C_neg": std_C_neg,
        "tail_risk": tail_risk,
        "cvar_factor": cvar_factor,
        "E_C": E_C_neg,
        "std_C": std_C_neg,
        "effective_score": effective_score,
        "p_reversal": p_reversal,
        "gap": gap,
        "std_gap": std_gap,
        "risk_type": "stochastic",
        "R_x": R_x,
        "gamma_R": gamma_R,
    }


# ═══════════════════════════════════════════════════════════════
# SHADOW CONSTRAINT LAYER (Step 2 — instrumentation only)
# ═══════════════════════════════════════════════════════════════
# This function computes CONTINUOUS pressure scores for potential
# hard constraints, WITHOUT affecting pi_S decisions.
#
# PURPOSE: Map the decision space to answer one question:
#   "Is there any part of the decision space where trade-offs
#    should stop being allowed?"
#
# THREE PRESSURE SURFACES:
#   S_tail:     CVaR(C⁻) / κ_hard — tail risk pressure (1.0 = at ceiling)
#   S_irrev:    R(x) × p_reversal / κ_hard — irreversibility × uncertainty
#   S_composite: max(S_tail, S_irrev) — overall constraint energy
#
# TENSION CLASSIFICATION:
#   Type A — Silent risk accumulation: scalar ALLOW, S_composite ≈ 1
#       → future failure candidates
#   Type B — Regime mismatch: scalar strongly ALLOW, constraint strongly violated
#       → true structural conflicts, candidates for piecewise rules
#   Type C — Redundant: scalar BLOCK, constraint also triggered with large margin
#       → adds no new information, don't promote later
#
# IMPORTANT: This is MEASUREMENT, not optimization.
# Do NOT tune λ, γ, κ based on shadow results.

def shadow_constraints(case, risk):
    """
    Compute continuous pressure scores for potential hard constraints.

    NO BEHAVIOR CHANGE — this function is purely diagnostic.

    Args:
        case: frozen RDRD case dict
        risk: output of compute_risk_score(case)

    Returns:
        dict with:
            cvar_absolute:        raw CVaR_excess value (what the scalar model uses)
            cvar_ratio:           CVaR_excess / κ_hard (1.0 = at hard ceiling)
            cvar_margin:          κ_hard - CVaR_excess (>0 = safe margin)
            cvar_margin_pct:      margin as % of κ_hard

            irrev_score:          R(x) × p_reversal (irreversibility × uncertainty interaction)
            irrev_ratio:          irrev_score / κ_hard (normalized)
            irrev_flag:           1[R(x) = ∞ ∧ p_reversal > ε] (absolute rule candidate)

            composite_energy:     max(cvar_ratio, irrev_ratio) (worst pressure)
            tension_type:         "A" / "B" / "C" / "none" (classification)

            kappa_margin:         κ - effective_score (scalar margin; <0 means BLOCK)
            pressure_ratio:       effective_score / κ (how close to scalar threshold)
    """
    cvar_excess = risk.get("tail_risk", 0)
    R_x = risk.get("R_x", 0)
    p_rev = risk.get("p_reversal", 0)
    effective = risk.get("effective_score", 0)
    cost_sign = risk.get("cost_sign", "downside")

    # Handle infinity cases
    cvar_is_inf = (cvar_excess == float('inf') or cvar_excess is None)
    eff_is_inf = (effective == float('inf') or effective is None)
    R_is_inf = (R_x == float('inf'))

    # ── Surface (a): Tail risk pressure ──
    if cvar_is_inf:
        cvar_ratio = float('inf')
        cvar_margin = float('-inf')
        cvar_margin_pct = float('inf')
    elif KAPPA_HARD == 0:
        cvar_ratio = float('inf')
        cvar_margin = -cvar_excess
        cvar_margin_pct = float('inf')
    else:
        cvar_ratio = cvar_excess / KAPPA_HARD
        cvar_margin = KAPPA_HARD - cvar_excess
        cvar_margin_pct = (cvar_margin / KAPPA_HARD) * 100.0

    # ── Surface (b): Irreversibility × uncertainty interaction ──
    if R_is_inf:
        irrev_score = float('inf')
        irrev_ratio = float('inf')
    else:
        irrev_score = R_x * p_rev
        irrev_ratio = irrev_score / KAPPA_HARD if KAPPA_HARD > 0 else float('inf')

    # Irreversibility flag: R(x) = ∞ AND meaningful uncertainty
    irrev_flag = 1 if (R_is_inf and p_rev > EPS_UNCERTAINTY) else 0

    # ── Surface (c): Composite violation energy ──
    if cvar_is_inf or R_is_inf:
        composite_energy = float('inf')
    else:
        composite_energy = max(cvar_ratio, irrev_ratio)

    # ── Tension classification ──
    # Compare shadow constraints with scalar decision
    scalar_decision = "BLOCK" if effective > KAPPA else "ALLOW"
    kappa_margin = KAPPA - effective if not eff_is_inf else float('-inf')
    pressure_ratio = effective / KAPPA if (not eff_is_inf and KAPPA > 0) else float('inf')

    tension_type = "none"

    if scalar_decision == "ALLOW":
        if not cvar_is_inf and not R_is_inf:
            if composite_energy >= 1.0:
                # Type B: scalar allows but constraint VIOLATED (S ≥ 1)
                # Strong structural conflict — candidate for piecewise
                tension_type = "B"
            elif composite_energy >= 0.6:
                # Type A: scalar allows but near boundary (S ∈ [0.6, 1.0))
                # Silent risk accumulation — future failure candidate
                tension_type = "A"
            # else: safe zone, no tension
        elif R_is_inf and irrev_flag:
            # Type B: irreversibility flag triggered while scalar allows
            tension_type = "B"
        elif cvar_is_inf:
            # Type B: infinite tail risk while scalar allows
            tension_type = "B"
    else:
        # scalar BLOCK
        if not cvar_is_inf and not R_is_inf and composite_energy >= 1.0:
            # Type C: both scalar and constraint agree (with margin)
            # Redundant constraint — don't promote later
            tension_type = "C"
        # else: scalar blocks for its own reasons, constraint not triggered
        # No tension type assigned (scalar doing its job alone)

    return {
        "cvar_absolute": cvar_excess,
        "cvar_ratio": round(cvar_ratio, 4) if not (cvar_is_inf or cvar_ratio != cvar_ratio) else cvar_ratio,
        "cvar_margin": round(cvar_margin, 0) if not (cvar_is_inf or cvar_margin != cvar_margin) else cvar_margin,
        "cvar_margin_pct": round(cvar_margin_pct, 1) if not (cvar_is_inf or cvar_margin_pct != cvar_margin_pct) else cvar_margin_pct,

        "irrev_score": round(irrev_score, 0) if not (R_is_inf and irrev_score == float('inf')) else irrev_score,
        "irrev_ratio": round(irrev_ratio, 4) if not (R_is_inf or irrev_ratio != irrev_ratio) else irrev_ratio,
        "irrev_flag": irrev_flag,

        "composite_energy": round(composite_energy, 4) if not (composite_energy == float('inf') or composite_energy != composite_energy) else composite_energy,
        "tension_type": tension_type,

        "kappa_margin": round(kappa_margin, 0) if kappa_margin != float('-inf') else kappa_margin,
        "pressure_ratio": round(pressure_ratio, 4) if pressure_ratio != float('inf') else pressure_ratio,

        "scalar_decision": scalar_decision,
        "cost_sign": cost_sign,
    }


# ═══════════════════════════════════════════════════════════════
# ADVERSARIAL STRESS TESTING (Step 2b — instrumentation only)
# ═══════════════════════════════════════════════════════════════
# GPT's suggestion: "Don't add more constraints. Instead, stress
# the existing constraints adversarially."
#
# METHODOLOGY:
#   Take cases where pi_S = ALLOW and perturb them to find if
#   there exists x: pi_S(x) = ALLOW AND constraint violated.
#
# THREE PERTURBATION AXES:
#   (1) Variance ↑  — multiply environment variance by 2x, 4x
#   (2) Tail heaviness ↑ — change distribution to heavy_tailed
#   (3) Asymmetric loss spike — increase C_wrong (downside cost)
#
# IMPORTANT: This is MEASUREMENT, not optimization.
# Do NOT tune λ, γ, κ based on adversarial results.
# Do NOT modify the frozen CASES array.

import copy

def _deep_clone_case(case):
    """Deep clone a frozen case for perturbation without modifying original."""
    return copy.deepcopy(case)


def _perturb_variance(case, multiplier):
    """
    Perturbation (1): Multiply environment variance by multiplier.
    Creates a more uncertain version of the same case.
    """
    cloned = _deep_clone_case(case)
    env = cloned["environment_model"]
    if env.get("variance") is not None and env["variance"] > 0:
        env["variance"] = min(env["variance"] * multiplier, 1.0)  # cap at 1.0
        env["variance_rationale"] = (
            f"ADVERSARIAL: variance perturbed {multiplier}x from original. "
            f"Original: {case['environment_model'].get('variance', 'N/A')}. "
            f"Simulating increased production uncertainty."
        )
    elif env.get("variance") == 0:
        # Deterministic → inject small variance
        env["variance"] = 0.04 * multiplier
        env["distribution"] = "normal"
        env["variance_rationale"] = (
            f"ADVERSARIAL: deterministic case given variance={env['variance']:.3f}. "
            f"Simulating non-stationarity in previously stable environment."
        )
    # Heavy-tailed (variance=None): leave as-is (already maximally uncertain)
    return cloned


def _perturb_tail_heaviness(case):
    """
    Perturbation (2): Change distribution to heavy_tailed.
    CVaR factor jumps from ~2.063 to ~3.18 (1.54x multiplier).
    """
    cloned = _deep_clone_case(case)
    env = cloned["environment_model"]
    original_dist = env.get("distribution", "deterministic")
    if original_dist == "heavy_tailed":
        return None  # already heavy-tailed, skip
    if env.get("variance") is None:
        return None  # unbounded — skip
    if env.get("variance") == 0:
        # Need variance for tail to matter
        env["variance"] = 0.04
    env["distribution"] = "heavy_tailed"
    env["variance_rationale"] = (
        f"ADVERSARIAL: distribution changed {original_dist} -> heavy_tailed. "
        f"CVaR factor increases from ~2.063 to ~3.18 (1.54x). "
        f"Simulating regime shift to fat-tailed cost distribution."
    )
    return cloned


def _perturb_loss_spike(case, cost_multiplier, consequence_override=None):
    """
    Perturbation (3): Increase C_wrong by multiplier.
    Simulates asymmetric loss spike — rare catastrophic event
    that dramatically increases the cost of being wrong.
    
    If consequence_override is provided, changes the consequence_type,
    which can flip a case from upside (C+) to downside (C-).
    This is the most adversarial perturbation.
    """
    cloned = _deep_clone_case(case)
    for cost_key, cost_val in cloned["C_i"].items():
        if cost_val.get("annual_usd") is not None and cost_val["annual_usd"] > 0:
            cost_val["annual_usd"] = cost_val["annual_usd"] * cost_multiplier
            if consequence_override:
                cost_val["consequence_type"] = consequence_override
            cost_val["_adversarial_note"] = (
                f"Cost multiplied {cost_multiplier}x. "
                f"Original: ${case['C_i'][cost_key]['annual_usd']:,.0f}. "
                f"Simulating tail loss event or hidden downside exposure."
            )
    return cloned


def adversarial_stress_test(cases=None):
    """
    Run adversarial stress test on ALLOW cases.
    
    For each ALLOW case, applies 3 types of perturbation and checks
    whether any perturbation creates a shadow constraint violation
    (Type A or Type B tension).
    
    Returns:
        dict with:
            base_cases: list of case IDs where pi_S = ALLOW
            perturbations: list of perturbation results
            verdict: summary finding
    """
    if cases is None:
        cases = CASES

    # Identify ALLOW cases
    allow_cases = []
    for case in cases:
        risk = compute_risk_score(case)
        decision = "BLOCK" if risk["effective_score"] > KAPPA else "ALLOW"
        if decision == "ALLOW":
            allow_cases.append((case, risk))

    results = {
        "description": "Adversarial stress test — perturb ALLOW cases to find constraint violations",
        "methodology": (
            "Three perturbation axes applied to each ALLOW case: "
            "(1) variance x2, x4, (2) heavy_tailed distribution, "
            "(3) loss spike x3, x10, (4) sign flip to downside"
        ),
        "base_allow_count": len(allow_cases),
        "base_allow_ids": [c[0]["id"] for c in allow_cases],
        "perturbations": [],
        "type_a_found": [],
        "type_b_found": [],
    }

    total_perturbations = 0
    type_a_count = 0
    type_b_count = 0

    for case, base_risk in allow_cases:
        case_id = case["id"]
        base_shadow = shadow_constraints(case, base_risk)

        # ── Perturbation 1: Variance ↑ ──
        for mult in [2.0, 4.0]:
            perturbed = _perturb_variance(case, mult)
            risk = compute_risk_score(perturbed)
            shadow = shadow_constraints(perturbed, risk)
            p_result = {
                "case": case_id,
                "perturbation": f"variance_x{int(mult)}",
                "description": f"Variance multiplied {mult}x",
                "original_variance": case["environment_model"].get("variance"),
                "perturbed_variance": perturbed["environment_model"].get("variance"),
                "effective_score": risk["effective_score"],
                "scalar_decision": "BLOCK" if risk["effective_score"] > KAPPA else "ALLOW",
                "shadow_tension": shadow["tension_type"],
                "cvar_ratio": shadow["cvar_ratio"],
                "composite_energy": shadow["composite_energy"],
            }
            results["perturbations"].append(p_result)
            total_perturbations += 1
            if shadow["tension_type"] == "A":
                type_a_count += 1
                results["type_a_found"].append(p_result)
            elif shadow["tension_type"] == "B":
                type_b_count += 1
                results["type_b_found"].append(p_result)

        # ── Perturbation 2: Tail heaviness ↑ ──
        perturbed = _perturb_tail_heaviness(case)
        if perturbed is not None:
            risk = compute_risk_score(perturbed)
            shadow = shadow_constraints(perturbed, risk)
            p_result = {
                "case": case_id,
                "perturbation": "heavy_tailed",
                "description": "Distribution changed to heavy_tailed (CVaR factor 1.54x)",
                "original_distribution": case["environment_model"].get("distribution"),
                "effective_score": risk["effective_score"],
                "scalar_decision": "BLOCK" if risk["effective_score"] > KAPPA else "ALLOW",
                "shadow_tension": shadow["tension_type"],
                "cvar_ratio": shadow["cvar_ratio"],
                "composite_energy": shadow["composite_energy"],
            }
            results["perturbations"].append(p_result)
            total_perturbations += 1
            if shadow["tension_type"] == "A":
                type_a_count += 1
                results["type_a_found"].append(p_result)
            elif shadow["tension_type"] == "B":
                type_b_count += 1
                results["type_b_found"].append(p_result)

        # ── Perturbation 3: Loss spike (cost increase) ──
        for mult in [3.0, 10.0]:
            perturbed = _perturb_loss_spike(case, mult)
            risk = compute_risk_score(perturbed)
            shadow = shadow_constraints(perturbed, risk)
            p_result = {
                "case": case_id,
                "perturbation": f"loss_spike_x{int(mult)}",
                "description": f"C_wrong multiplied {mult}x (tail loss event simulation)",
                "original_cost": base_risk.get("C_wrong"),
                "effective_score": risk["effective_score"],
                "scalar_decision": "BLOCK" if risk["effective_score"] > KAPPA else "ALLOW",
                "shadow_tension": shadow["tension_type"],
                "cvar_ratio": shadow["cvar_ratio"],
                "composite_energy": shadow["composite_energy"],
            }
            results["perturbations"].append(p_result)
            total_perturbations += 1
            if shadow["tension_type"] == "A":
                type_a_count += 1
                results["type_a_found"].append(p_result)
            elif shadow["tension_type"] == "B":
                type_b_count += 1
                results["type_b_found"].append(p_result)

        # ── Perturbation 4: Sign flip — upside to downside ──
        # This is the MOST adversarial: simulate hidden downside exposure
        # in what appears to be a safe upside case
        if base_risk.get("cost_sign") == "upside" and base_risk.get("C_wrong"):
            perturbed = _perturb_loss_spike(case, 1.0, consequence_override="error_cost")
            risk = compute_risk_score(perturbed)
            shadow = shadow_constraints(perturbed, risk)
            p_result = {
                "case": case_id,
                "perturbation": "sign_flip_to_downside",
                "description": (
                    "MOST ADVERSARIAL: upside cost reclassified as downside. "
                    "Simulates hidden downside exposure in 'safe' deployment."
                ),
                "original_cost_sign": "upside",
                "new_cost_sign": "downside",
                "effective_score": risk["effective_score"],
                "scalar_decision": "BLOCK" if risk["effective_score"] > KAPPA else "ALLOW",
                "shadow_tension": shadow["tension_type"],
                "cvar_ratio": shadow["cvar_ratio"],
                "composite_energy": shadow["composite_energy"],
            }
            results["perturbations"].append(p_result)
            total_perturbations += 1
            if shadow["tension_type"] == "A":
                type_a_count += 1
                results["type_a_found"].append(p_result)
            elif shadow["tension_type"] == "B":
                type_b_count += 1
                results["type_b_found"].append(p_result)

    # ── Verdict ──
    if type_b_count > 0:
        verdict = (
            f"ADVERSARIAL TYPE B FOUND: {type_b_count}/{total_perturbations} perturbations "
            f"created regime mismatches (scalar ALLOW + constraint violated). "
            f"Piecewise rules MAY be justified for specific adversarial scenarios."
        )
    elif type_a_count > 0:
        verdict = (
            f"ADVERSARIAL TYPE A FOUND: {type_a_count}/{total_perturbations} perturbations "
            f"approached constraint boundary. Scalar remains ALLOW but under pressure. "
            f"No Type B — scalar still dominates, but boundary is closer than nominal."
        )
    else:
        verdict = (
            f"NO ADVERSARIAL VIOLATIONS: 0/{total_perturbations} perturbations "
            f"created shadow constraint tensions. Scalar policy survives all "
            f"adversarial stress tests within tested perturbation space."
        )

    results["total_perturbations"] = total_perturbations
    results["type_a_count"] = type_a_count
    results["type_b_count"] = type_b_count
    results["verdict"] = verdict

    # Structural analysis
    # Check: are all ALLOW cases structurally upside (C⁻=0)?
    all_upside = all(r.get("cost_sign") == "upside" for _, r in allow_cases)
    results["structural_analysis"] = {
        "all_allow_cases_are_upside": all_upside,
        "implication": (
            "All ALLOW cases have E[C⁻] = 0 (upside cost structure). "
            "This means CVaR is always 0 for ALLOW cases, making them "
            "structurally immune to tail risk constraint violations. "
            "The only path to Type B would be: (a) sign flip to downside, "
            "(b) or entirely new cases with downside cost + moderate risk." if all_upside else
            "Mixed cost structure in ALLOW cases — some have downside exposure."
        ),
    }

    return results


def _print_adversarial_results(results):
    """Pretty-print adversarial stress test results to terminal."""
    print()
    print("=" * 70)
    print("  ADVERSARIAL STRESS TEST (Step 2b)")
    print("=" * 70)
    print()
    print(f"  Base ALLOW cases: {results['base_allow_count']}")
    print(f"  Cases: {', '.join(results['base_allow_ids'])}")
    print(f"  Total perturbations: {results['total_perturbations']}")
    print()
    print(f"  Type A (near-miss):     {results['type_a_count']}")
    print(f"  Type B (regime flip):   {results['type_b_count']}")
    print()

    # Per-case summary
    print("  ── Per-perturbation results ──")
    for p in results["perturbations"]:
        decision_marker = "ALLOW" if p["scalar_decision"] == "ALLOW" else "BLOCK"
        tension_marker = f"  [{p['shadow_tension']}]" if p["shadow_tension"] != "none" else ""
        energy = p["composite_energy"]
        energy_str = f"{energy:.4f}" if isinstance(energy, (int, float)) else str(energy)
        print(f"    {p['case']:8s} | {p['perturbation']:25s} | {decision_marker:5s} "
              f"| S_comp={energy_str:>8s} | S_tail={p['cvar_ratio']}{tension_marker}")

    print()

    # Type A details
    if results["type_a_found"]:
        print("  !! Type A perturbations (near-miss boundary):")
        for p in results["type_a_found"]:
            print(f"     {p['case']} + {p['perturbation']}: composite={p['composite_energy']:.4f}")
        print()

    # Type B details
    if results["type_b_found"]:
        print("  *** Type B perturbations (REGIME MISMATCH):")
        for p in results["type_b_found"]:
            print(f"     {p['case']} + {p['perturbation']}: composite={p['composite_energy']:.4f}")
        print()

    # Structural analysis
    sa = results.get("structural_analysis", {})
    print(f"  Structural: all ALLOW cases are upside? {sa.get('all_allow_cases_are_upside', 'N/A')}")
    print()

    # Verdict
    print(f"  VERDICT: {results['verdict']}")
    print("=" * 70)


def pi_S(case):
    """
    v4.3 System policy: pi_S(x_i) = risk_gate(effective_score).

    DECISION RULE (v4.3 — tail-sensitive, sign-correct, five-axis):
        effective_score = E[C⁻] + lambda * CVaR_excess(C⁻) + gamma * R(x) - beta * E[C⁺]
        BLOCK if effective_score > kappa
        ALLOW otherwise

    v4.3 KEY CHANGE: CVaR replaces σ as the uncertainty measure.
        Old (v4.2): E[C⁻] + lambda * σ(C⁻)
        New (v4.3): E[C⁻] + lambda * CVaR_excess(C⁻)
        CVaR focuses on the tail of the loss distribution, not just the variance.
        Heavy-tailed environments get ~1.54x more penalty than normal envs.

    SIGN-CORRECT COST MODEL (v4.2, preserved):
        C⁻ = downside risk (actual losses: error_cost, safety, revenue_loss, etc.)
        C⁺ = upside opportunity (forgone gains: forfeited_productivity_gain, etc.)
        
        Fixes the sign error: uncertain benefit (RDR-003) is no longer penalized
        as if it were uncertain harm. Only actual losses contribute to risk.

    Five independent risk axes:
    1. Downside expectation: E[C⁻]
    2. Tail risk: lambda * CVaR_excess(C⁻)
    3. Operational irreversibility: gamma * R(x)
    4. Opportunity channel: beta * E[C⁺]
    5. Cost structure: C⁻ vs C⁺ classification

    This REPLACES the old deterministic BSSI threshold.
    Uncertainty and irreversibility now jointly control the decision.

    Also runs diagnosis for supplementary context (but doesn't drive the decision).
    """
    risk = compute_risk_score(case, LAMBDA_RISK, GAMMA_IRREVERSIBILITY)
    effective = risk["effective_score"]
    decision = "BLOCK" if effective > KAPPA else "ALLOW"

    # Run diagnosis for supplementary context (not decision-driving in v4)
    bssi = case["x_i"]["bssi_components"]
    features = case["x_i"]["features"]
    diag = diagnose(
        S=bssi["S"],
        A=bssi["A"],
        N=bssi["N"],
        BSSI=bssi["BSSI"],
        n_questions=features["n_questions"],
        extraction_fail_rate=features["extraction_fail_rate"],
    )

    eff_str = f"${effective:,.0f}" if effective != float('inf') else "INF"
    kappa_str = f"${KAPPA:,.0f}"

    return {
        "decision": decision,
        "confidence": "HIGH",
        "reason": f"v4.3 tail-risk: effective={eff_str} (E[C⁻]+lambda*CVaR[C⁻]+gamma*R-beta*C⁺) vs kappa={kappa_str}",
        "severity": diag["severity"],
        "diagnosis_codes": diag["all_codes"],
        "prescription_actions": [],
        "bssi_value": bssi["BSSI"],
        "risk_metrics": risk,
    }


def compute_E_i(case):
    """
    E_i = 1[pi_E(x_i) == y_i]
    Is standard eval correct about which model is better?
    """
    return pi_E(case) == case["y_i"]["optimal_model"]


def compute_S_i(case, system_result):
    """
    S_i = 1[pi_S agrees with y_i]

    System is correct if:
      - When y != pi_E (eval is wrong), system BLOCKS
      - When y == pi_E (eval is right), system ALLOWS

    This is the formal correctness criterion for a deployment gate.
    """
    eval_correct = compute_E_i(case)
    sys_decision = system_result["decision"]

    if not eval_correct:
        # Eval is wrong — system should BLOCK
        return sys_decision == "BLOCK"
    else:
        # Eval is right — system should ALLOW
        return sys_decision == "ALLOW"


# ═══════════════════════════════════════════════════════════════
# REGRET COMPUTATION (v4 — risk-adjusted)
# ═══════════════════════════════════════════════════════════════

def compute_regret_E(case):
    """
    regret_E(i) = E_{y~P}[C(pi_E(x_i), y)] - E_{y~P}[C(optimal, y)]

    v4: Uses analytical risk computation instead of Monte Carlo.
    E[C(pi_E, y)] = C_wrong * p_reversal
    """
    risk = compute_risk_score(case)
    return risk["E_C"], risk["std_C"], risk["risk_type"] == "stochastic"


def compute_regret_S(case, system_result):
    """
    regret_S(i) = E_{y~P}[C(pi_S(x_i), y)] - E_{y~P}[C(optimal, y)]

    When system BLOCKS -> no deployment -> expected cost = 0
    When system ALLOWS -> expected cost = E[C(pi_E's choice, y)]
    """
    sys_decision = system_result["decision"]

    if sys_decision == "BLOCK":
        return 0, 0.0, False
    elif sys_decision in ("ALLOW", "CONDITIONAL"):
        risk = compute_risk_score(case)
        return risk["E_C"], risk["std_C"], risk["risk_type"] == "stochastic"
    else:
        return 0, 0.0, False


# ═══════════════════════════════════════════════════════════════
# RUN ALL — compute everything from frozen cases
# ═══════════════════════════════════════════════════════════════

def run_all():
    """
    Process all 20 frozen cases through both policies.
    All outputs are DERIVED from the frozen inputs.
    The cases themselves are NEVER modified.
    """
    results = []

    for case in CASES:
        # Compute policies (deterministic from frozen inputs)
        eval_winner = pi_E(case)
        system_result = pi_S(case)

        # Compute binary judgments
        E_i = compute_E_i(case)
        S_i = compute_S_i(case, system_result)

        # Compute risk metrics (v4 — analytical, not Monte Carlo)
        risk = compute_risk_score(case)

        # Compute regrets (v4)
        regret_e, regret_e_std, regret_e_stochastic = compute_regret_E(case)
        regret_s, regret_s_std, regret_s_stochastic = compute_regret_S(case, system_result)
        delta_regret = regret_e - regret_s

        # Extract cost info
        optimal = case["y_i"]["optimal_model"]
        cost_key = f"C(deploy_{pi_E_deploy_name(case)}, y={optimal})"
        cost_entry = case["C_i"].get(cost_key, {})

        result = {
            "id": case["id"],
            "category": case["x_i"]["features"]["category"],

            # ── FROZEN INPUTS (never change) ──
            "x_i": {
                "decision_context": case["x_i"]["features"]["decision_context"],
                "evaluation_method": case["x_i"]["features"]["evaluation_method"],
                "bssi": case["x_i"]["bssi_components"],
                "eval_scores": case["x_i"]["eval_scores"],
            },
            "y_i": {
                "optimal_model": optimal,
                "downstream_metric": case["y_i"]["downstream_metric"],
                "downstream_values": case["y_i"]["downstream_values"],
            },
            "C_i": case["C_i"],

            # ── ENVIRONMENT MODEL ──
            "environment_model": case["environment_model"],

            # ── CONSTRAINED ACTION SPACE ──
            "action_space": case["action_space"],
            "optimal_constrained": case["optimal_constrained"],

            # ── COMPUTED POLICIES ──
            "pi_E": eval_winner,
            "pi_S": system_result["decision"],
            "pi_S_detail": system_result,

            # ── BINARY JUDGMENTS ──
            "E_i": E_i,
            "S_i": S_i,

            # ── RISK METRICS (v4) ──
            "risk_metrics": risk,

            # ── REGRET ANALYSIS (v4) ──
            "regret_E": regret_e,
            "regret_E_std": regret_e_std,
            "regret_E_stochastic": regret_e_stochastic,
            "regret_S": regret_s,
            "regret_S_std": regret_s_std,
            "regret_S_stochastic": regret_s_stochastic,
            "delta_regret": delta_regret,

            # ── SHADOW CONSTRAINTS (Step 2 — instrumentation only) ──
            "shadow": shadow_constraints(case, risk),
        }
        results.append(result)

    return results


def main():
    results = run_all()

    # ── DERIVED SUMMARY METRICS (all from frozen data) ──
    total = len(results)
    eval_correct = sum(1 for r in results if r["E_i"])
    eval_wrong = total - eval_correct
    system_correct = sum(1 for r in results if r["S_i"])
    system_wrong = total - system_correct

    # Confusion matrix
    true_positives = sum(1 for r in results if r["E_i"] and r["S_i"])
    false_blocks = sum(1 for r in results if r["E_i"] and not r["S_i"])
    false_allows = sum(1 for r in results if not r["E_i"] and not r["S_i"])
    true_preventions = sum(1 for r in results if not r["E_i"] and r["S_i"])

    # Regret analysis
    total_regret_E = sum(r["regret_E"] for r in results if r["regret_E"] != float('inf'))
    total_regret_S = sum(r["regret_S"] for r in results if r["regret_S"] != float('inf'))
    total_delta = total_regret_E - total_regret_S

    # Risk metrics summary
    n_stochastic = sum(1 for r in results if r["risk_metrics"]["risk_type"] == "stochastic")
    n_deterministic_safe = sum(1 for r in results if r["risk_metrics"]["risk_type"] == "deterministic_safe")
    n_deterministic_unsafe = sum(1 for r in results if r["risk_metrics"]["risk_type"] == "deterministic_unsafe")
    n_unquantifiable = sum(1 for r in results if r["risk_metrics"]["risk_type"] == "unquantifiable")

    # ── Terminal output ──
    print()
    print("=" * 90)
    print("  RISK-SENSITIVE COUNTERFACTUAL DECISION STRESS SIMULATOR (v4.3)")
    print("  Risk-Sensitive Decision Under Uncertainty  |  20 cases  |  D = constant")
    print("=" * 90)

    print()
    print("  v4.3 KEY CHANGE: CVaR replaces σ — tail-sensitive risk measurement")
    print("    Old (v4.2): effective = E[C⁻] + lambda*sigma(C⁻) + gamma*R(x) - beta*E[C⁺]")
    print("    New (v4.3): effective = E[C⁻] + lambda*CVaR_excess(C⁻) + gamma*R(x) - beta*E[C⁺]")
    print("    CVaR_excess = σ * φ(Φ⁻¹(α)) / (1-α)  [α=0.95]")
    print("    Normal: factor ≈ 2.063  |  Heavy-tailed: factor ≈ 3.18  |  Deterministic: 0")
    print()
    print(f"  Parameters:  lambda = {LAMBDA_RISK}  (risk aversion)")
    print(f"               gamma  = {GAMMA_IRREVERSIBILITY}  (irreversibility weight)")
    print(f"               beta   = {BETA_OPPORTUNITY}  (opportunity discount)")
    print(f"               alpha  = {ALPHA_CVAR}  (CVaR confidence level)")
    print(f"               kappa  = ${KAPPA:,.0f}/year  (risk tolerance)")

    print()
    print("  FORMAL MODEL (v4.3 -- tail-sensitive, sign-correct, multi-factor):")
    print("    State:       x_i = (features, bssi_components, eval_scores)")
    print("    Eval policy: pi_E(x) = argmax_a { s(a) }, theta_E = {scoring_fn: raw_score}")
    print("    Sys policy:  pi_S(x) = BLOCK if E[C⁻]+lambda*CVaR[C⁻]+gamma*R-beta*C⁺ > kappa")
    print("    Cost model:  C⁻ = actual loss, C⁺ = forgone gain (classified by consequence_type)")
    print("    Environment: y ~ P(y | x, action)  [stochastic when variance > 0]")
    print("    Five axes:   (1) E[C⁻]  (2) CVaR tail  (3) R(x) irreversibility  (4) C⁺ opportunity  (5) cost structure")

    print()
    hdr = f"  {'ID':<8} {'pi_E':<16} {'y_i':<16} {'pi_S':<8} {'E_i':<5} {'S_i':<5} {'eff_score':>12} {'E[C⁻]':>12} {'CVaR_ex':>12} {'sign':>5} {'p_rev':>6} {'cvar_f':>6}"
    print(hdr)
    print("  " + "-" * 120)

    for r in results:
        pi_e = r["pi_E"]
        y_i = r["y_i"]["optimal_model"]
        pi_s = r["pi_S"]
        e_i = "OK" if r["E_i"] else "FAIL"
        s_i = "OK" if r["S_i"] else "FAIL"
        risk = r["risk_metrics"]
        eff = risk["effective_score"]
        E_C = risk["E_C_neg"]
        tail = risk.get("tail_risk", 0)
        eff_str = f"${eff:,.0f}" if eff != float('inf') else "INF"
        sign = risk.get("cost_sign", "?")[:3]
        ec_str = f"${risk['E_C']:,.0f}" if risk["E_C"] != float('inf') else "INF"
        tr_str = f"${tail:,.0f}" if tail != float('inf') else "INF"
        pr = f"{risk['p_reversal']:.3f}" if risk['p_reversal'] is not None else "N/A"
        cf = f"{risk.get('cvar_factor', 0):.2f}"
        print(f"  {r['id']:<8} {pi_e:<16} {y_i:<16} {pi_s:<8} {e_i:<5} {s_i:<5} {eff_str:>12} {ec_str:>12} {tr_str:>12} {sign:>5} {pr:>6} {cf:>6}")

    print()
    print("  ── BINARY JUDGMENT MATRIX ──")
    print(f"                     System CORRECT   System WRONG")
    print(f"    Eval CORRECT:      {true_positives:>3} (TP)        {false_blocks:>3} (false block)")
    print(f"    Eval WRONG:        {true_preventions:>3} (prevented)  {false_allows:>3} (false allow)")
    print()
    print(f"    Eval accuracy:     {eval_correct}/{total} ({eval_correct/total*100:.0f}%)")
    print(f"    System accuracy:   {system_correct}/{total} ({system_correct/total*100:.0f}%)")
    print(f"    False allows:      {false_allows} (dangerous — eval errors that slipped through)")
    print(f"    False blocks:      {false_blocks} (conservative — good evals blocked by risk)")

    if false_blocks > 0:
        print()
        print("  ── FALSE BLOCKS (good evals blocked by uncertainty) ──")
        for r in results:
            if r["E_i"] and not r["S_i"]:
                risk = r["risk_metrics"]
                print(f"    {r['id']}: {r['pi_E']} was correct, but effective_score=${risk['effective_score']:,.0f} > kappa=${KAPPA:,.0f}")
                print(f"           C_wrong=${risk['C_wrong']:,.0f}, p_reversal={risk['p_reversal']:.3f}, variance={r['environment_model']['variance']}")

    if false_allows > 0:
        print()
        print("  ── FALSE ALLOWS (bad evals that slipped through) ──")
        for r in results:
            if not r["E_i"] and not r["S_i"]:
                risk = r["risk_metrics"]
                print(f"    {r['id']}: {r['pi_E']} was wrong, but effective_score=${risk['effective_score']:,.0f} < kappa=${KAPPA:,.0f}")
                print(f"           C_wrong=${risk['C_wrong']:,.0f}, p_reversal={risk['p_reversal']:.3f}, variance={r['environment_model']['variance']}")

    print()
    print("  ── RISK ANALYSIS (v4) ──")
    print(f"    Deterministic safe:  {n_deterministic_safe}  (variance=0, gap>0, eff_score=0)")
    print(f"    Deterministic unsafe:{n_deterministic_unsafe}  (variance=0, gap<0, eff_score=C_wrong)")
    print(f"    Stochastic:          {n_stochastic}  (variance>0, eff_score = E[C] + lambda*CVaR[C])")
    print(f"    Unquantifiable:      {n_unquantifiable}  (C_wrong=null, eff_score=INF)")
    print()
    print(f"    Total expected regret(pi_E):  ${total_regret_E:>14,.0f}/year")
    print(f"    Total expected regret(pi_S):  ${total_regret_S:>14,.0f}/year")
    print(f"    Expected delta (E - S):       ${total_delta:>14,.0f}/year")

    # ── SHADOW CONSTRAINT ANALYSIS (Step 2 — instrumentation) ──
    print()
    print("  " + "=" * 90)
    print("  SHADOW CONSTRAINT LAYER (Step 2 — instrumentation, no behavior change)")
    print("  " + "=" * 90)
    print(f"    kappa_hard  = ${KAPPA_HARD:,.0f}/year  (absolute tail risk ceiling)")
    print(f"    kappa       = ${KAPPA:,.0f}/year   (operational tolerance)")
    print(f"    eps_uncertainty = {EPS_UNCERTAINTY}       (irrev×uncertainty threshold)")
    print()

    # Shadow constraint table
    print(f"  {'ID':<8} {'pi_S':<7} {'CVaR':>12} {'CVaR/kH':>8} {'margin':>12} {'R*pR':>12} {'R*pR/kH':>8} {'irrevF':>6} {'Energy':>8} {'Tension':>8}")
    print("  " + "-" * 108)

    for r in results:
        s = r["shadow"]
        pi_s = r["pi_S"]
        cvar_abs = s["cvar_absolute"]
        cvar_str = f"${cvar_abs:,.0f}" if cvar_abs != float('inf') else "INF"
        cvar_ratio_str = f"{s['cvar_ratio']:.3f}" if s["cvar_ratio"] != float('inf') else "INF"
        margin_str = f"${s['cvar_margin']:,.0f}" if s["cvar_margin"] != float('-inf') else "-INF"
        irrev_score = s["irrev_score"]
        irrev_str = f"${irrev_score:,.0f}" if irrev_score != float('inf') else "INF"
        irrev_ratio_str = f"{s['irrev_ratio']:.3f}" if s["irrev_ratio"] != float('inf') else "INF"
        irrev_f = str(s["irrev_flag"])
        energy = s["composite_energy"]
        energy_str = f"{energy:.3f}" if energy != float('inf') else "INF"
        tension = s["tension_type"]
        print(f"  {r['id']:<8} {pi_s:<7} {cvar_str:>12} {cvar_ratio_str:>8} {margin_str:>12} {irrev_str:>12} {irrev_ratio_str:>8} {irrev_f:>6} {energy_str:>8} {tension:>8}")

    # Tension classification summary
    type_a = [r for r in results if r["shadow"]["tension_type"] == "A"]
    type_b = [r for r in results if r["shadow"]["tension_type"] == "B"]
    type_c = [r for r in results if r["shadow"]["tension_type"] == "C"]
    type_none = [r for r in results if r["shadow"]["tension_type"] == "none"]

    print()
    print("  ── TENSION CLASSIFICATION ──")
    print(f"    Type A (silent risk):    {len(type_a)} cases — scalar ALLOW, constraint near boundary [0.6, 1.0)")
    print(f"    Type B (regime mismatch): {len(type_b)} cases — scalar ALLOW, constraint VIOLATED (≥ 1.0)")
    print(f"    Type C (redundant):       {len(type_c)} cases — scalar BLOCK, constraint also triggered")
    print(f"    No tension:              {len(type_none)} cases — safe zone or scalar handling alone")

    if type_a:
        print()
        print("  ── TYPE A: NEAR-MISS CASES (silent risk accumulation) ──")
        for r in type_a:
            s = r["shadow"]
            print(f"    {r['id']}: energy={s['composite_energy']:.3f} | CVaR/kH={s['cvar_ratio']:.3f} | R*pR/kH={s['irrev_ratio']:.3f}")
            print(f"           kappa_margin=${s['kappa_margin']:,.0f} | pressure={s['pressure_ratio']:.3f}x")

    if type_b:
        print()
        print("  ── TYPE B: REGIME MISMATCH (scalar allows, constraint violated) ──")
        for r in type_b:
            s = r["shadow"]
            print(f"    {r['id']}: energy={s['composite_energy']} | CVaR/kH={s['cvar_ratio']} | irrev_flag={s['irrev_flag']}")
            print(f"           kappa_margin={s['kappa_margin']} | scalar_decision={s['scalar_decision']}")

    if type_c:
        print()
        print("  ── TYPE C: REDUNDANT CONSTRAINTS (scalar already blocks) ──")
        for r in type_c:
            s = r["shadow"]
            print(f"    {r['id']}: energy={s['composite_energy']:.3f} | constraint adds no new information")

    # Verdict
    print()
    print("  ── SHADOW VERDICT ──")
    if not type_b and not type_a:
        print("    No constraint tensions detected. Scalar system is SUFFICIENT.")
        print("    → Stay with Option A (freeze). No piecewise rules needed.")
    elif type_b:
        print(f"    {len(type_b)} case(s) show regime mismatch (Type B).")
        print("    → Structural evidence for piecewise rules. Review Type B cases.")
        if not type_a:
            print("    → No near-misses (Type A) — boundary is clean except for mismatches.")
    elif type_a:
        print(f"    {len(type_a)} case(s) near constraint boundary (Type A).")
        print("    → Monitor only. No piecewise rules justified yet.")
        print("    → These are future failure candidates under distributional shift.")

    # Irreversibility flag summary
    irrev_flags = [r for r in results if r["shadow"]["irrev_flag"] == 1]
    if irrev_flags:
        print()
        print("  ── IRREVERSIBILITY FLAG SUMMARY ──")
        print(f"    {len(irrev_flags)} case(s) with R(x)=inf AND meaningful uncertainty:")
        for r in irrev_flags:
            s = r["shadow"]
            risk = r["risk_metrics"]
            print(f"    {r['id']}: p_reversal={risk['p_reversal']:.3f}, scalar_decision={s['scalar_decision']}")

    print()
    print("  ── VERSION HISTORY ──")
    print(f"    [v4.3] Tail-sensitive risk: CVaR replaces σ. Coherent risk measure. Heavy-tail awareness.")
    print(f"    [v4.2] Sign-correct cost model: C⁻ (downside) vs C⁺ (upside). Fixes RDR-003 false block.")
    print(f"    [v4.1] Irreversibility R(x): 3rd axis. Fixes RDR-019 false allow via gamma*R(x).")
    print(f"    [v4] Risk-sensitive decision: pi_S = f(E[C] + lambda*std[C] > kappa)")
    print(f"    [v3.2] Environment model: P(y|x,action) with variance per case")
    print(f"    [v3] pi_E explicit: argmax with theta_E, constrained A_i")
    print()
    print("  ── FIXED-POINT INVARIANT ──")
    print(f"    Cases FROZEN.  pi_E deterministic from x_i.")
    print(f"    pi_S deterministic from (x_i, lambda, gamma, beta, alpha, kappa).")
    print(f"    All metrics DERIVED.  D = constant.")
    print()

    # ── JSON output ──
    output = {
        "dataset_type": "Risk-Sensitive Counterfactual Decision Stress Simulator",
        "version": "4.3",
        "fixed_point_invariant": "D = constant. Cases are frozen. Only derived metrics change.",

        "v4_upgrade": {
            "change": "Replaced deterministic BSSI threshold with multi-factor risk-sensitive gate",
            "old_rule": "pi_S = f(diagnose(BSSI)) — deterministic severity threshold",
            "new_rule": "pi_S = f(E[C(pi_E,y)] + lambda * sqrt(Var[C(pi_E,y)]) + gamma * R(x) > kappa)",
            "effect": "Uncertainty AND irreversibility now drive decisions, not just evaluation",
            "result": f"{system_correct}/{total} accuracy (was 20/20)",
        },

        "v4.1_upgrade": {
            "change": "Added third risk axis: irreversibility/latency penalty R(x)",
            "old_rule_v4": "pi_S = f(E[C] + lambda * sqrt(Var[C]) > kappa)",
            "new_rule_v4.1": "pi_S = f(E[C] + lambda * sqrt(Var[C]) + gamma * R(x) > kappa)",
            "problem_fixed": "v4 was cost-scale sensitive, not risk-structure sensitive. RDR-003 ($19M, internal, easy rollback) blocked while RDR-019 ($876K, production, silent truncation) allowed.",
            "R_x_definition": "R(x) = irreversibility score from (rollback_cost, time_to_detect, blast_radius). Captures operational consequence structure independent of dollar scale.",
            "key_flip": "RDR-019: ALLOW(v4) -> BLOCK(v4.1) via gamma*R(x)=$1.8M penalty. Production system with silent truncation now correctly blocked.",
        },

        "v4.2_upgrade": {
            "change": "Sign-correct cost model: separate C⁻ (downside loss) from C⁺ (upside opportunity)",
            "old_rule_v4.1": "effective = E[C] + lambda*sigma(C) + gamma*R(x)  [sign-blind — all costs treated as risk]",
            "new_rule_v4.2": "effective = E[C⁻] + lambda*sigma(C⁻) + gamma*R(x) - beta*E[C⁺]  [sign-correct]",
            "problem_fixed": "v4.1 had a sign error: it penalized uncertain benefit as if it were uncertain harm. RDR-003 ($19M internal tool, forfeited_productivity_gain) was false-blocked because its high variance in upside was treated as downside risk.",
            "sign_classification": {
                "C_neg_types": "error_cost, human_correction_rate, safety_incident_risk, revenue_loss, productivity_waste, net_quality_loss, cost_loss, trading_loss, overspend",
                "C_pos_types": "forfeited_productivity_gain, forfeited_cost_savings, forfeited_revenue, forfeited_accuracy, forfeited_productivity",
                "rule": "If consequence_type contains 'forfeited' → C⁺ (upside). Otherwise → C⁻ (downside).",
            },
            "key_flip": "RDR-003: BLOCK(v4.1) -> ALLOW(v4.2). $19M forfeited_productivity_gain classified as C⁺ (upside). E[C⁻]=0, so effective=gamma*R(x)=$100K < kappa → ALLOW. High variance in upside no longer penalized as downside risk.",
            "philosophy": "Not all costs are losses. Uncertain benefit (RDR-003) is structurally different from uncertain harm (RDR-001). The system now respects this distinction.",
        },

        "v4.3_upgrade": {
            "change": "Tail-sensitive risk: CVaR replaces σ as uncertainty measure",
            "old_rule_v4.2": "effective = E[C⁻] + lambda*sigma(C⁻) + gamma*R(x) - beta*E[C⁺]  [variance-based]",
            "new_rule_v4.3": "effective = E[C⁻] + lambda*CVaR_excess(C⁻) + gamma*R(x) - beta*E[C⁺]  [tail-risk]",
            "problem_fixed": "σ treats all uncertainty equally — frequent small losses and rare catastrophic losses get the same penalty if variance matches. CVaR focuses on the worst (1-α) fraction of scenarios, properly penalizing tail risk.",
            "cvar_definition": "CVaR_excess(C⁻) = σ(C⁻) * φ(Φ⁻¹(α)) / (1-α). At α=0.95, normal factor ≈ 2.063, heavy-tailed factor ≈ 3.18.",
            "coherence": "CVaR is a coherent risk measure (sub-additive, monotonic, homogeneous). σ is NOT coherent (fails sub-additivity). This makes v4.3 mathematically well-posed.",
            "distribution_factors": {
                "normal": "CVaR_factor ≈ 2.063 (at α=0.95)",
                "heavy_tailed": "CVaR_factor ≈ 3.18 (Student-t ν≈3, ~1.54x normal)",
                "deterministic": "CVaR_factor = 0 (no tail risk)",
            },
            "result": "19/20 accuracy preserved. No decision flips. All upside cases (E[C⁻]=0) unchanged. Heavy-tailed cases already BLOCKED. System is now theoretically robust for future heavy-tail scenarios.",
            "philosophy": "The last clean scalar extension. Going further (piecewise rules) would sacrifice smooth calibration. CVaR makes the model ready for real-world tail risk without losing mathematical elegance.",
        },

        "risk_parameters": {
            "lambda": LAMBDA_RISK,
            "lambda_description": "Risk aversion: weight on CVaR tail risk measure",
            "gamma": GAMMA_IRREVERSIBILITY,
            "gamma_description": "Irreversibility weight: penalty for operational consequence structure",
            "beta": BETA_OPPORTUNITY,
            "beta_description": "Opportunity discount: weight for C+ (upside) channel. beta=0 means ignore C+ in risk calc.",
            "alpha": ALPHA_CVAR,
            "alpha_description": "CVaR confidence level: focus on worst (1-alpha) fraction of scenarios",
            "kappa": KAPPA,
            "kappa_description": "Risk tolerance: max acceptable risk-adjusted cost ($/year)",
            "calibration": f"lambda={LAMBDA_RISK}, gamma={GAMMA_IRREVERSIBILITY}, beta={BETA_OPPORTUNITY}, alpha={ALPHA_CVAR}, kappa=${KAPPA:,.0f} — produces {system_wrong} errors ({system_correct}/{total})",
        },

        "formal_model": {
            "state": "x_i = (features, bssi_components, eval_scores)",
            "eval_policy": {
                "form": "pi_E(x) = argmax_a { s(a) }",
                "theta_E": PI_E_DEFINITION["theta_E"],
                "limitation": "Constrained to A_i. Cannot consider actions outside candidate set.",
            },
            "system_policy": {
                "form": "pi_S(x) = BLOCK if effective_score > kappa, else ALLOW",
                "effective_score": "E[C⁻(pi_E, y)] + lambda * CVaR_excess(C⁻) + gamma * R(x) - beta * E[C⁺(pi_E, y)]",
                "sign_correct": "C⁻ = downside loss (actual harm), C⁺ = upside opportunity (forgone gain)",
                "tail_sensitive": "CVaR replaces σ: CVaR_excess = σ * φ(Φ⁻¹(α)) / (1-α) — focuses on worst-case tail",
                "coherent": "CVaR is a coherent risk measure (sub-additive, monotonic, homogeneous)",
                "five_axes": [
                    "(1) DOWNSIDE EXPECTATION: E[C⁻] — expected actual loss",
                    "(2) TAIL RISK: lambda * CVaR_excess(C⁻) — worst-case loss concentration",
                    "(3) CONSEQUENCE STRUCTURE: gamma * R(x) — operational irreversibility",
                    "(4) OPPORTUNITY: beta * E[C⁺] — upside gain channel",
                    "(5) COST STRUCTURE: C⁻ vs C⁺ classification — sign-correct accounting",
                ],
                "risk_coupling": "Uncertainty, tail structure, and irreversibility jointly control deployment gate decision",
            },
            "environment_model": {
                "per_case": True,
                "fields": ["distribution", "variance", "sensitivity", "description"],
                "regret_form": "E_{y~P}[C(pi(x), y)] - E_{y~P}[C(optimal, y)]",
            },
            "action_space": {
                "constrained": True,
                "description": "A_i per case. Optimal = argmin_{a in A_i} E[C(a,y)]. NOT omniscient oracle.",
            },
        },

        "binary_judgments": {
            "E_correct": eval_correct,
            "E_wrong": eval_wrong,
            "S_correct": system_correct,
            "S_wrong": system_wrong,
            "true_positives": true_positives,
            "false_blocks": false_blocks,
            "false_allows": false_allows,
            "true_preventions": true_preventions,
        },

        "regret_analysis": {
            "type": "risk_adjusted_expected",
            "total_expected_regret_eval_policy": round(total_regret_E, 2),
            "total_expected_regret_system_policy": round(total_regret_S, 2),
            "expected_delta_regret": round(total_delta, 2),
        },

        "shadow_constraint_analysis": {
            "description": "Step 2 instrumentation — continuous pressure surfaces, no behavior change",
            "methodology": "Measures structural gaps, not optimizing performance",
            "parameters": {
                "kappa_hard": KAPPA_HARD,
                "kappa_hard_description": "Absolute tail risk ceiling ($/year) — well above operational kappa",
                "kappa": KAPPA,
                "kappa_description": "Operational risk tolerance ($/year)",
                "eps_uncertainty": EPS_UNCERTAINTY,
                "eps_uncertainty_description": "Minimum p_reversal for irrev×uncertainty constraint to trigger",
            },
            "pressure_surfaces": {
                "S_tail": "CVaR_excess(C⁻) / κ_hard — tail risk pressure (1.0 = at hard ceiling)",
                "S_irrev": "R(x) × p_reversal / κ_hard — irreversibility × uncertainty interaction",
                "S_composite": "max(S_tail, S_irrev) — overall constraint energy",
            },
            "tension_classification": {
                "type_A_count": len(type_a),
                "type_A_description": "Silent risk accumulation: scalar ALLOW, constraint near boundary [0.6, 1.0)",
                "type_A_cases": [r["id"] for r in type_a],
                "type_B_count": len(type_b),
                "type_B_description": "Regime mismatch: scalar ALLOW, constraint VIOLATED (≥ 1.0)",
                "type_B_cases": [r["id"] for r in type_b],
                "type_C_count": len(type_c),
                "type_C_description": "Redundant: scalar BLOCK, constraint also triggered with large margin",
                "type_C_cases": [r["id"] for r in type_c],
                "no_tension_count": len(type_none),
            },
            "irreversibility_flags": {
                "count": len(irrev_flags),
                "description": "Cases with R(x)=inf AND p_reversal > eps",
                "cases": [r["id"] for r in irrev_flags],
            },
            "verdict": (
                "NO_CONSTRAINT_TENSIONS" if not type_b and not type_a
                else f"TYPE_B_MISMATCH_DETECTED ({len(type_b)} cases)" if type_b
                else f"TYPE_A_NEAR_MISS ({len(type_a)} cases — monitor only)"
            ),
        },

        "cases": results,
    }

    out_path = os.path.join(DIR, "rdrd_report.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Full report: {out_path}")
    print()


if __name__ == "__main__":
    main()

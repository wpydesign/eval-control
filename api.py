#!/usr/bin/env python3
"""
api.py — FastAPI wrapper for Risk Audit Engine

Four endpoints:
    POST /evaluate    — Run shadow evaluation on a deployment decision
    POST /outcome     — Log a real-world outcome for a previous decision
    GET  /audit       — Retrieve audit trail (shadow log + outcomes)
    GET  /health      — Health check (no auth required)

Auth:
    Set EVAL_CONTROL_API_KEYS env var to a comma-separated list of keys.
    All endpoints except /health require X-API-Key header.
    If EVAL_CONTROL_API_KEYS is empty/unset, auth is disabled (local dev mode).

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Docker:
    docker compose up --build
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

# Override log/data directory if env var is set
LOG_DIR = os.environ.get("EVAL_CONTROL_LOG_DIR", DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# Shadow/outcome logs go to LOG_DIR
import shadow_mode
import outcome_capture
shadow_mode.LOG_FILE = os.path.join(LOG_DIR, "shadow_log.jsonl")
outcome_capture.SHADOW_LOG = os.path.join(LOG_DIR, "shadow_log.jsonl")
outcome_capture.OUTCOME_FILE = os.path.join(LOG_DIR, "outcomes.jsonl")

from shadow_mode import run_pi_S, log_entry, read_log
from outcome_capture import log_outcome as _log_outcome, read_outcomes, read_fault_probes

app = FastAPI(
    title="Risk Audit Engine",
    description="Shadow evaluation + outcome audit for AI deployment decisions.",
    version="4.3.0",
)


# ═══════════════════════════════════════════════════════════════
# AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════

_API_KEYS_RAW = os.environ.get("EVAL_CONTROL_API_KEYS", "")
_API_KEYS = set(k.strip() for k in _API_KEYS_RAW.split(",") if k.strip())
_AUTH_ENABLED = len(_API_KEYS) > 0


class _APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce X-API-Key on all endpoints except /health."""
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or not _AUTH_ENABLED:
            return await call_next(request)
        api_key = request.headers.get("X-API-Key", "")
        if api_key not in _API_KEYS:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )
        return await call_next(request)


app.add_middleware(_APIKeyMiddleware)


# ═══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

class EvaluateRequest(BaseModel):
    """Submit a deployment decision for shadow evaluation."""
    case_id: str = Field(..., description="Unique case identifier")
    context: str = Field(..., description="Free-text description of the decision")
    eval_scores: dict = Field(..., description="Model eval scores, e.g. {'v1': 0.85, 'v2': 0.91}")
    pi_E: Optional[str] = Field(None, description="What current practice would deploy (model name or ALLOW/BLOCK)")
    metadata: Optional[dict] = Field(default_factory=dict, description=(
        "domain: prod|internal|creative|safety, "
        "estimated_cost_if_wrong: annual USD, "
        "reversibility: easy|moderate|hard|impossible, "
        "latency_to_detect: hours|days|weeks|months, "
        "consequence_type: error_cost|safety_incident_risk|..., "
        "distribution: deterministic|normal|heavy_tailed"
    ))


class EvaluateResponse(BaseModel):
    """Shadow evaluation result."""
    case_id: str
    timestamp: str
    pi_E_raw: str
    pi_E_decision: str
    pi_S: str
    divergence: bool
    effective_score: float
    margin: float
    risk_breakdown: dict
    shadow_constraints: dict


class OutcomeRequest(BaseModel):
    """Log a real-world outcome for a previous decision."""
    case_id: str = Field(..., description="Case ID from a previous evaluation")
    realized: str = Field(..., description="success|failure|mixed|unknown|no_event")
    cost_actual: Optional[float] = Field(None, description="Actual cost in USD (null if unknown)")
    notes: str = Field(..., description="Free-text description of what happened")
    fault_probe_override: Optional[str] = Field(None, description="FP1|FP2|FP3 (optional override)")


class OutcomeResponse(BaseModel):
    """Logged outcome record."""
    case_id: str
    timestamp_decision: Optional[str]
    timestamp_outcome: str
    pi_E: str
    pi_S: str
    decision_alignment: str
    fault_probe: str
    outcome: dict
    cost_estimated: Optional[float]
    cost_actual: Optional[float]
    calibration_error: Optional[float]


class AuditResponse(BaseModel):
    """Audit trail summary."""
    shadow_entries: int
    outcomes: int
    divergences: int
    fault_probes: dict
    entries: list


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 1: Evaluate a deployment decision
# ═══════════════════════════════════════════════════════════════

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """
    Run shadow evaluation on a deployment decision.

    Takes a real-world decision context, runs the frozen v4.3 policy (pi_S),
    and returns the risk assessment alongside what current practice would do (pi_E).

    The result is automatically logged to shadow_log.jsonl for audit trail.
    """
    raw_case = req.model_dump()

    try:
        result = run_pi_S(raw_case)
        log_entry(result)  # persist to shadow_log.jsonl for audit trail
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Evaluation failed: {e}")

    resp = EvaluateResponse(
        case_id=result["case_id"],
        timestamp=result["timestamp"],
        pi_E_raw=result["pi_E_raw"],
        pi_E_decision=result["pi_E_decision"],
        pi_S=result["pi_S"],
        divergence=result["divergence"],
        effective_score=result["risk"]["effective_score"],
        margin=result["risk"]["margin"],
        risk_breakdown=result["risk"],
        shadow_constraints=result["shadow"],
    )

    return resp


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 2: Log a real-world outcome
# ═══════════════════════════════════════════════════════════════

@app.post("/outcome", response_model=OutcomeResponse)
def log_outcome(req: OutcomeRequest):
    """
    Log a real-world outcome for a previous decision.

    Ties ground truth back to the exact decision context that produced pi_S.
    Record is append-only. Original estimates are never overwritten.
    """
    outcome_dict = {
        "realized": req.realized,
        "cost_actual": req.cost_actual,
        "notes": req.notes,
    }
    if req.fault_probe_override:
        outcome_dict["fault_probe_override"] = req.fault_probe_override

    try:
        record = _log_outcome(req.case_id, outcome_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Outcome logging failed: {e}")

    # Compute calibration error if both costs available
    cal_error = None
    est = record.get("cost_estimated")
    act = record.get("outcome", {}).get("cost_actual")
    if est is not None and act is not None:
        cal_error = act - est

    resp = OutcomeResponse(
        case_id=record["case_id"],
        timestamp_decision=record["timestamp_decision"],
        timestamp_outcome=record["timestamp_outcome"],
        pi_E=record["pi_E"],
        pi_S=record["pi_S"],
        decision_alignment=record["decision_alignment"],
        fault_probe=record["fault_probe"],
        outcome=record["outcome"],
        cost_estimated=record.get("cost_estimated"),
        cost_actual=record["outcome"].get("cost_actual"),
        calibration_error=cal_error,
    )

    return resp


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 3: Retrieve audit trail
# ═══════════════════════════════════════════════════════════════

@app.get("/audit", response_model=AuditResponse)
def audit(fault_probe: Optional[str] = None, limit: int = 50):
    """
    Retrieve the audit trail — shadow evaluations and logged outcomes.

    Query params:
        fault_probe: Filter by FP1|FP2|FP3 (outcomes only)
        limit: Max entries to return (default 50)
    """
    # Shadow log
    shadow_entries = read_log(limit=limit)

    # Outcomes
    fp_filter = fault_probe if fault_probe else None
    outcome_entries = read_outcomes(fault_probe=fp_filter)

    # Fault probe summary
    fp_summary = {}
    for fp_id in ("FP1", "FP2", "FP3"):
        fp_records = read_outcomes(fault_probe=fp_id)
        fp_summary[fp_id] = {
            "count": len(fp_records),
            "cases": [r["case_id"] for r in fp_records],
        }

    # Divergence count
    divergences = sum(1 for e in shadow_entries if e.get("divergence"))

    # Strip _original_shadow from response (too large, not needed for audit view)
    clean_outcomes = []
    for o in outcome_entries[:limit]:
        clean = {k: v for k, v in o.items() if k != "_original_shadow"}
        clean_outcomes.append(clean)

    return AuditResponse(
        shadow_entries=len(shadow_entries),
        outcomes=len(outcome_entries),
        divergences=divergences,
        fault_probes=fp_summary,
        entries=clean_outcomes,
    )


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.3.0",
        "engine": "frozen",
        "auth_enabled": _AUTH_ENABLED,
        "log_dir": LOG_DIR,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

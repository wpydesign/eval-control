"""
sdk.py — Python SDK for Risk Audit Engine

Simple client wrapper. Zero dependencies beyond the standard library.
Uses urllib (no httpx/requests required).

Usage:
    from sdk import RiskAuditClient

    client = RiskAuditClient("http://localhost:8000", api_key="your-key")

    # Shadow evaluation
    result = client.evaluate(
        case_id="PROD-042",
        context="Upgrading customer support model from v1 to v2",
        eval_scores={"v1": 0.82, "v2": 0.87},
        pi_E="v2",
        metadata={"domain": "prod", "estimated_cost_if_wrong": 500000},
    )

    # Log outcome
    outcome = client.log_outcome(
        case_id="PROD-042",
        realized="failure",
        cost_actual=320000,
        notes="Rollback required after 3 days.",
    )

    # Audit trail
    audit = client.audit(limit=20)
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class RiskAuditClient:
    """Thin HTTP client for the Risk Audit Engine API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Args:
            base_url: API base URL, e.g. "http://localhost:8000"
            api_key: Optional API key for authenticated requests
            timeout: Request timeout in seconds (default 30)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode() if e.fp else ""
            raise RiskAuditError(e.code, body_text) from None
        except urllib.error.URLError as e:
            raise RiskAuditError(0, f"Connection failed: {e.reason}") from None

    # ─────────────────────────────────────────────
    # EVALUATE
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        case_id: str,
        eval_scores: dict,
        context: str = "",
        pi_E: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Run shadow evaluation on a deployment decision.

        Args:
            case_id: Unique case identifier
            eval_scores: Model eval scores, e.g. {"v1": 0.85, "v2": 0.91}
            context: Free-text description of the decision
            pi_E: What current practice would deploy (model name or ALLOW/BLOCK)
            metadata: Decision metadata (domain, cost, reversibility, etc.)

        Returns:
            Full evaluation result with pi_S, divergence, risk breakdown
        """
        payload = {
            "case_id": case_id,
            "context": context,
            "eval_scores": eval_scores,
            "pi_E": pi_E,
            "metadata": metadata or {},
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._request("POST", "/evaluate", payload)

    # ─────────────────────────────────────────────
    # OUTCOME
    # ─────────────────────────────────────────────

    def log_outcome(
        self,
        case_id: str,
        realized: str,
        notes: str,
        cost_actual: Optional[float] = None,
        fault_probe_override: Optional[str] = None,
    ) -> dict:
        """
        Log a real-world outcome for a previous decision.

        Args:
            case_id: Case ID from a previous evaluation
            realized: "success" | "failure" | "mixed" | "unknown"
            notes: Free-text description of what happened
            cost_actual: Actual cost in USD (None if unknown)
            fault_probe_override: Optional FP1|FP2|FP3 tag

        Returns:
            Logged outcome record with decision_alignment and calibration
        """
        payload = {
            "case_id": case_id,
            "realized": realized,
            "notes": notes,
        }
        if cost_actual is not None:
            payload["cost_actual"] = cost_actual
        if fault_probe_override:
            payload["fault_probe_override"] = fault_probe_override
        return self._request("POST", "/outcome", payload)

    # ─────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────

    def audit(self, limit: int = 50, fault_probe: Optional[str] = None) -> dict:
        """
        Retrieve the audit trail.

        Args:
            limit: Max entries to return
            fault_probe: Filter by FP1|FP2|FP3 (outcomes only)

        Returns:
            Audit summary with shadow entries, outcomes, divergences
        """
        params = f"?limit={limit}"
        if fault_probe:
            params += f"&fault_probe={fault_probe}"
        return self._request("GET", f"/audit{params}")

    # ─────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────

    def health(self) -> dict:
        """Health check."""
        return self._request("GET", "/health")


class RiskAuditError(Exception):
    """API error with status code and body."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")

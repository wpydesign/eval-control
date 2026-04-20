# Interface Contract

**Version**: 3.0.0-core-stable
**Status**: LOCKED — no changes to shape or semantics

All adapters and applications MUST use this interface. Core will NOT change its output shape.

---

## Canonical Request

```json
{
  "prompt": "string — the AI output or query to evaluate",
  "context": "string — optional, the decision context",
  "metadata": {
    "domain": "prod | internal | creative | safety",
    "estimated_cost_if_wrong": "number — annual USD",
    "reversibility": "easy | moderate | hard | impossible"
  }
}
```

### Field rules

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `prompt` | YES | string | The text to evaluate. Max 10,000 characters. |
| `context` | no | string | Free-text description of the decision context. |
| `metadata.domain` | no | string | Domain classification. Default: `"internal"`. |
| `metadata.estimated_cost_if_wrong` | no | number | Cost if this decision is wrong, in USD/year. |
| `metadata.reversibility` | no | string | How easy to undo if wrong. Default: `"moderate"`. |

---

## Canonical Response

```json
{
  "query_id": "string — unique evaluation ID",
  "prompt": "string — echoed input",
  "timestamp": "string — ISO 8601 UTC",

  "risk_score": 0.83,
  "action": "escalate",
  "manifold": "contradiction",
  "manifold_confidence": 0.91,

  "scores": {
    "S": 0.45,
    "kappa": 0.62,
    "A": 1.61,
    "delta_L": 0.08,
    "delta_G": 0.31
  },

  "routing": {
    "m_live": "contradiction",
    "m_ref": "contradiction",
    "manifold_disagreement": false,
    "routing_method": "learned_router"
  },

  "drift": {
    "router_drift_rate": 0.04,
    "drift_status": "STABLE",
    "ref_decay": 0.02,
    "ref_status": "VALID"
  }
}
```

### Field rules

| Field | Type | Always present | Description |
|-------|------|---------------|-------------|
| `query_id` | string | YES | Unique identifier for this evaluation. |
| `prompt` | string | YES | The evaluated prompt (truncated to 500 chars in logs). |
| `timestamp` | string | YES | ISO 8601 UTC timestamp. |
| `risk_score` | number | YES | P(is_wrong). Range [0, 1]. 0 = safe, 1 = certainly wrong. |
| `action` | string | YES | One of: `allow`, `review`, `escalate`, `force_reject_or_escalate`. |
| `manifold` | string | YES | One of: `overconfidence`, `contradiction`, `boundary`, `unknown`, `error`. |
| `manifold_confidence` | number | YES | Router confidence. Range [0, 1]. |
| `scores.S` | number | YES | Survival scalar. Range (0, 1]. |
| `scores.kappa` | number | YES | Consistency under perturbation. Range [0, 1]. |
| `scores.A` | number | YES | Amplification / brittleness. Range [1, inf). |
| `scores.delta_L` | number | YES | Local uncertainty. Range [0, 1]. |
| `scores.delta_G` | number | YES | Global inconsistency. Range [0, 1]. |
| `routing.m_live` | string | YES | Live router manifold assignment. |
| `routing.m_ref` | string | YES | Reference router manifold assignment. |
| `routing.manifold_disagreement` | boolean | YES | Whether live and reference routers disagree. |
| `routing.routing_method` | string | YES | How the routing was done. One of: `learned_router`, `rule_based_disagreement`, `heuristic`, `heuristic_fallback`. |
| `drift.router_drift_rate` | number | YES | P(m_live != m_ref) over rolling window. Range [0, 1]. |
| `drift.drift_status` | string | YES | One of: `STABLE`, `WARNING`, `CRITICAL`, `INSUFFICIENT_DATA`. |
| `drift.ref_decay` | number | YES | ref_accuracy - live_accuracy. Negative = ref is stale. |
| `drift.ref_status` | string | YES | One of: `VALID`, `REF_AGING`, `REF_STALE`, `LIVE_DRIFTING_WRONG`, `NO_DATA`. |

---

## Action semantics

| Action | Meaning | When |
|--------|---------|------|
| `allow` | Output is safe. No action needed. | risk_score below manifold threshold, manifold is boundary. |
| `review` | Needs human attention. Not automatically blocked. | Contradiction with moderate risk_score (0.30-0.60). |
| `escalate` | Likely wrong. Should be investigated or blocked. | Contradiction with high risk_score (>= 0.60). |
| `force_reject_or_escalate` | Deterministically wrong. Always block. | Manifold is overconfidence. 100% wrong rate proven. |
| `allow_with_light_review` | Probably correct but borderline. | Boundary with elevated risk. |

---

## Manifold semantics

| Manifold | Wrong rate | Description |
|----------|-----------|-------------|
| `overconfidence` | 100% | System is confident and wrong. Always escalate. |
| `contradiction` | ~76.5% | System gives contradictory answers. This is where most real failures live. |
| `boundary` | ~21% | Genuinely ambiguous. Mostly correct. Do not waste labels here. |
| `unknown` | N/A | No model loaded or insufficient data. |
| `error` | N/A | Evaluation failed. Check logs. |

---

## Drift semantics

| drift_status | Meaning | System action |
|-------------|---------|---------------|
| `STABLE` | Manifold decomposition is stable. | Normal operation. |
| `WARNING` | Manifolds are shifting (drift_rate > 0.15). | Freeze acquisition weights. |
| `CRITICAL` | Severe drift (drift_rate > 0.25). | Fallback to balanced sampling (33/33/33). |

| ref_status | Meaning | System action |
|-----------|---------|---------------|
| `VALID` | Reference router is accurate. | No action. |
| `REF_AGING` | Reference router is aging (ref_decay < -0.05). | Monitor. |
| `REF_STALE` | Reference router is stale (ref_decay < -0.10). | Consider controlled refresh. |
| `LIVE_DRIFTING_WRONG` | Live router is getting worse (ref_decay > 0). | Keep current anchor. |

---

## Stability guarantees

1. **Response shape will not change.** New fields may be ADDED (never removed or renamed) in minor versions.
2. **Action values will not change.** New actions may be ADDED.
3. **Manifold names will not change.** New manifolds may be ADDED.
4. **Score ranges will not change.** S is always (0,1], risk_score is always [0,1].
5. **Breaking changes require a major version bump** (v4.0.0, not v3.x).

---

## Integration pattern

```python
# Any adapter must produce exactly this JSON:
response = evaluate(prompt="...", context="...", metadata={...})

# Then consume exactly these fields:
risk_score = response["risk_score"]
action = response["action"]
manifold = response["manifold"]

# Optional: routing and drift info for observability
drift_rate = response["drift"]["router_drift_rate"]
ref_decay = response["drift"]["ref_decay"]
```

**Do NOT** parse or transform the core response in adapters. Pass it through as-is.

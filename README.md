# Eval Control

A self-correcting risk evaluation system for AI outputs.

It assigns every output a risk score and an action:

- **allow** — output is safe
- **review** — output needs human attention
- **escalate** — output is likely wrong, block or investigate

It learns from its mistakes. As you label more data, it gets better at catching failures without changing its core logic.

---

## What problem does it solve?

AI systems produce outputs that look correct but aren't. Traditional evaluation uses a single score (accuracy, F1, AUC) which hides three very different failure modes:

1. **Confidently wrong** — the system is certain, and it's wrong. Always.
2. **Internally conflicted** — the system gives contradictory answers when asked differently. Most real failures live here.
3. **Genuinely ambiguous** — even humans would disagree. Not much we can do.

Eval Control separates these three modes and handles each differently. It does not try to optimize all of them equally — it targets the failures that actually matter.

---

## Input / Output

**Input**: an AI output (text) and its context

**Output**: a risk assessment

```json
{
  "input": "Explain quantum computing in one sentence.",
  "output": {
    "risk_score": 0.83,
    "action": "escalate",
    "manifold": "contradiction",
    "manifold_confidence": 0.91,
    "reason": "v4 accept, v1 reject — high confidence gap"
  }
}
```

| Field | Meaning |
|-------|---------|
| `risk_score` | 0 = safe, 1 = likely wrong. Probability the output is incorrect. |
| `action` | `allow`, `review`, or `escalate`. What should happen next. |
| `manifold` | Which failure mode this falls into: `overconfidence`, `contradiction`, or `boundary`. |
| `manifold_confidence` | How sure the system is about the manifold assignment. |
| `reason` | Human-readable explanation of why this assessment was made. |

---

## How it works (simplified)

```
AI output comes in
        |
        v
  Survival Engine (S = robustness score)
  - Perturb the input in 5 ways
  - Ask in 4 different contexts
  - Measure how consistent the answers are
        |
        v
  Manifold Router (which failure mode?)
  - Overconfidence: always escalate (100% wrong rate)
  - Contradiction: learn when to escalate (76.5% wrong rate)
  - Boundary: allow (79% correct rate)
        |
        v
  Risk Score + Action
  - Per-manifold thresholds (not global)
  - Contradiction pushed harder than boundary
        |
        v
  Reference Router (stability check)
  - Compare current decisions to frozen baseline
  - Detect if the system is drifting
  - Trigger controlled refresh if needed
```

---

## Where it can be used

| Use Case | How |
|----------|-----|
| **Production deployment gate** | Run before deploying a model. Block high-risk outputs. |
| **Quality monitoring** | Sample production outputs, track risk scores over time. |
| **Data acquisition** | Tell it which outputs to label next. It targets the most informative ones. |
| **Model comparison** | Run both models through it. See which one fails more on contradiction cases. |
| **Audit trail** | Every evaluation is logged. Full traceability from decision to outcome. |

---

## Quick start

### API server

```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

### Evaluate an output

```bash
curl -X POST http://localhost:8000/survival-eval \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the speed of light?"}'
```

Response:

```json
{
  "query_id": "abc123",
  "kappa": 0.92,
  "S": 0.85,
  "A": 1.09,
  "decision": "accept",
  "drift_warning": false
}
```

### Docker

```bash
docker compose up --build
```

---

## Architecture

```
core/                   <- frozen engine (this is stable)
  api.py                <- HTTP interface
  core.py               <- evaluation control layer (BSSI)
  survival.py           <- survival scalar engine
  shadow_mode.py        <- shadow evaluation
  outcome_capture.py    <- outcome logging
  sdk.py                <- client library
  scripts/
    manifold_predict.py     <- manifold decomposition
    reference_router.py     <- drift + staleness tracking
    acquisition_policy.py   <- data acquisition controller
    manifold_kpi.py         <- KPI dashboard
    ...

adapters/               <- domain-specific connectors (future)
  README.md
  (e.g., adapters/openai_eval.py, adapters/custom_llm.py)

apps/                   <- user-facing products (future)
  README.md
  (e.g., apps/deployment_gate/, apps/quality_dashboard/)
```

**Core is frozen.** Adapters and apps depend on it but never modify it.

---

## Interface contract

The system has ONE canonical JSON interface. See [INTERFACE.md](./INTERFACE.md) for the full spec.

From now on:

- Adapters must use this interface
- Core does not change shape
- New functionality goes in adapters or apps

---

## What this is NOT

- Not a scoring model
- Not an eval benchmark
- Not a fine-tuning tool
- Not a replacement for your AI system

---

## License

MIT

## Contact

wpydesign@gmail.com

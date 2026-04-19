# Shadow Evaluation + Outcome Audit for AI Decisions

Every decision becomes:

* what your system chose (π_E)
* what a risk-aware policy would have chosen (π_S)
* where they diverge
* what actually happened later (outcome)

This is a **decision accountability layer** that runs alongside any model, system, or evaluation pipeline.

It does not replace your model.
It makes its decisions measurable.

---

## What this is

A lightweight system that turns AI or human decisions into a **closed feedback loop**:

```
input decision
   ↓
π_E (existing system output)
   ↓
π_S (shadow risk policy: CVaR + irreversibility model)
   ↓
divergence detection
   ↓
real-world outcome logging
```

Over time, you get:

* where your system is too risky
* where it is too conservative
* where evaluation scores fail to predict real cost
* where "correct-looking" decisions fail in reality

---

## Why it exists

Most systems only do one thing:

> optimize a score

This system asks a different question:

> what happens if that decision is wrong?

It exposes:

* hidden tail risk
* irreversibility blind spots
* cost underestimation
* evaluation-function overconfidence

---

## Core components

### 1. π_S Risk Engine (frozen)

* CVaR-based downside estimation
* irreversibility weighting
* cost-aware thresholding
* deterministic decision boundary

### 2. Shadow Mode

Runs alongside any system:

* compares π_E vs π_S
* logs divergence
* does not interfere with execution

### 3. Outcome Capture

When reality happens:

* logs actual cost
* compares predicted vs realized impact
* preserves full decision context (immutable)

### 4. Audit Trail

Every case becomes:

```
decision → shadow evaluation → outcome → calibration error
```

No aggregation required to be useful.

---

## Minimal example

```python
from sdk import RiskAuditClient

client = RiskAuditClient(
    base_url="http://localhost:8000",
    api_key="your-key"
)

# your system's decision
result = client.evaluate(
    case_id="deploy-model-v3",
    context="production model upgrade",
    eval_scores={"v2": 0.81, "v3": 0.84}
)

print(result)
```

Later:

```python
client.log_outcome(
    case_id="deploy-model-v3",
    realized="failure",
    cost_actual=2500000
)
```

Now you can measure:

> what was chosen vs what should have been chosen under risk

---

## What you get

* Decision traceability
* Risk boundary visibility
* Real-world calibration error
* Divergence signals (π_E vs π_S)
* Fault probes for failure classes

---

## Deployment

### Docker

```bash
docker compose up --build
```

### API

* POST `/evaluate`
* POST `/outcome`
* GET `/audit`

### SDK

Single-file client (`sdk.py`) — no dependencies.

---

## What this is NOT

* not a scoring model
* not an eval benchmark
* not a fine-tuning tool
* not a replacement for your system

---

## What this actually is

> A measurement layer for decision systems that finally connects prediction to consequence.

---

## If you only understand one thing:

This system does not try to be correct.

It tries to make **incorrectness visible before it becomes expensive.**

---

## Quick reference

### Authentication

```bash
# Enable API keys (comma-separated)
EVAL_CONTROL_API_KEYS="key1,key2" docker compose up --build

# curl with auth
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"case_id":"X","eval_scores":{"a":0.8,"b":0.9},"pi_E":"b"}'
```

Empty/unset `EVAL_CONTROL_API_KEYS` = auth disabled (local dev).

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_CONTROL_API_KEYS` | `""` | Comma-separated API keys. Empty = no auth. |
| `EVAL_CONTROL_LOG_DIR` | `.` | Directory for shadow_log.jsonl and outcomes.jsonl. |
| `EVAL_CONTROL_PORT` | `8000` | Server port. |

### Full pipeline demo

```bash
python demo.py
```

### CLI

```bash
python shadow_mode.py                    # Interactive REPL
python shadow_mode.py --dry-run           # Replay 20 regression cases
python shadow_mode.py --file cases.jsonl  # Batch mode
python outcome_capture.py log <id> --realized <success|failure> --notes "..."
python outcome_capture.py show
```

### Local install (no Docker)

```bash
git clone https://github.com/wpydesign/eval-control.git
cd eval-control
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

## License

MIT

## Commercial Use

If you are using eval-control in a production or commercial environment,
I'd appreciate you reaching out.

Contact: wpydesign@gmail.com

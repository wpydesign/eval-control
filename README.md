# Risk Audit Engine

**Shadow evaluation + outcome audit for AI deployment decisions.**

A decision-audit kernel that sits alongside your existing evaluation pipeline. Before you deploy a model, the engine runs a frozen risk policy in shadow mode — scoring downside risk, flagging tail scenarios, and recording what standard eval would miss. After deployment, it captures real-world outcomes to build an audit trail. No tuning required.

---

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/wpydesign/eval-control-deepseek.git
cd eval-control-deepseek
docker compose up --build
```

Server is live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

Logs persist to the `audit-data` Docker volume. Health check: `curl http://localhost:8000/health`.

### Option B: Local install

```bash
git clone https://github.com/wpydesign/eval-control-deepseek.git
cd eval-control-deepseek
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

### Option C: Python only (no server)

```bash
git clone https://github.com/wpydesign/eval-control-deepseek.git
cd eval-control-deepseek
python demo.py
```

---

## SDK Client

Zero-dependency Python client. Uses stdlib `urllib` — no `httpx` or `requests` needed.

```python
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
print(result["pi_S"], result["divergence"])

# Log outcome
client.log_outcome(
    case_id="PROD-042",
    realized="failure",
    cost_actual=320000,
    notes="Rollback required after 3 days.",
)

# Audit trail
audit = client.audit(limit=20)
print(f"{audit['divergences']} divergences across {audit['shadow_entries']} evaluations")
```

Copy `sdk.py` into any project. No pip install required.

---

## Authentication

API keys are optional. When enabled, all endpoints except `/health` require an `X-API-Key` header.

### Docker

```bash
API_KEYS="key1,key2,key3" docker compose up --build
```

### Local

```bash
EVAL_CONTROL_API_KEYS="key1,key2" uvicorn api:app --port 8000
```

### curl with auth

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"case_id":"PROD-042","eval_scores":{"v1":0.82,"v2":0.87},"pi_E":"v2"}'
```

If `EVAL_CONTROL_API_KEYS` is empty or unset, auth is disabled (local dev mode).

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/evaluate` | Run shadow evaluation on a deployment decision |
| `POST` | `/outcome` | Log a real-world outcome for a previous decision |
| `GET` | `/audit` | Retrieve audit trail (shadow log + outcomes) |
| `GET` | `/health` | Health check (always unauthenticated) |

### POST /evaluate

```bash
curl -s -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "PROD-042",
    "context": "Upgrading customer support model from v1 to v2",
    "eval_scores": {"v1": 0.82, "v2": 0.87},
    "pi_E": "v2",
    "metadata": {
      "domain": "prod",
      "estimated_cost_if_wrong": 500000,
      "reversibility": "moderate",
      "latency_to_detect": "days"
    }
  }' | python -m json.tool
```

### POST /outcome

```bash
curl -s -X POST http://localhost:8000/outcome \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "PROD-042",
    "realized": "failure",
    "cost_actual": 320000,
    "notes": "Rollback required after 3 days. Higher than estimated cost."
  }' | python -m json.tool
```

### GET /audit

```bash
curl -s http://localhost:8000/audit?limit=10 | python -m json.tool
curl -s "http://localhost:8000/audit?fault_probe=FP1&limit=10" | python -m json.tool
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `EVAL_CONTROL_API_KEYS` | `""` | Comma-separated API keys. Empty = auth disabled. |
| `EVAL_CONTROL_LOG_DIR` | `.` (repo root) | Directory for `shadow_log.jsonl` and `outcomes.jsonl`. |
| `EVAL_CONTROL_HOST` | `0.0.0.0` | Server bind address. |
| `EVAL_CONTROL_PORT` | `8000` | Server port. |

In Docker, logs go to `/app/data` (persistent volume).

---

## CLI

### Shadow mode

```bash
python shadow_mode.py                    # Interactive REPL
python shadow_mode.py --file cases.jsonl  # Batch mode
python shadow_mode.py --dry-run           # Replay 20 regression cases
```

### Outcome capture

```bash
python outcome_capture.py log <case_id> --realized <success|failure|mixed|unknown> --notes "..."
python outcome_capture.py show [--fault-probe FP1|FP2|FP3]
```

### Full pipeline demo

```bash
python demo.py
```

---

## How It Works

Three layers, each independent:

| Layer | Purpose |
|-------|---------|
| **Evaluation Layer** (π_E) | Your existing eval — pick the model with the higher score. No change to current practice. |
| **Strategy Layer** (π_S) | The frozen v4.3 risk policy. Computes a multi-factor risk score using CVaR tail risk, sign-correct cost modeling, and irreversibility penalties. Compares against the κ threshold. |
| **Observation Layer** | Passive logging — records π_E vs π_S divergence, shadow constraint signals, and real-world outcomes in append-only JSONL. No enforcement. No feedback loops. |

π_S does not replace π_E. It runs in parallel and logs disagreements. Over time, the divergence log reveals where your eval pipeline is systematically overconfident.

### The Problem

Teams ship models based on eval scores (accuracy, F1, BLEU). These scores measure average performance — they do not measure **what happens if the model is wrong**.

Two models can score within 2% on a benchmark but have wildly different downside profiles. One fails gracefully; the other generates silent hallucinations that cost millions in production. Standard eval cannot tell them apart.

This engine catches those cases.

---

## What Makes It Different

**Coherent risk measure.** Uses Conditional Value at Risk (CVaR) to model tail risk — what happens in the worst 5% of scenarios — rather than treating all uncertainty as equivalent variance.

**Sign-correct cost model.** Downside costs (errors, safety incidents, revenue loss) and upside opportunity (forgone gains) are tracked separately. An uncertain benefit is not false-blocked as if it were a risk.

**Tail-sensitive by distribution.** Heavy-tailed environments (safety-critical, financial) get ~1.5x more penalty than normal-distribution environments with the same variance.

**Irreversibility modeling.** A $100K cost that takes months to detect and affects millions of users is treated differently from a $100K cost with instant rollback.

**Frozen policy (v4.3).** All thresholds, weights, and calibration values are fixed. No fitting. No data-dependent tuning. The same policy applies to every case. Auditable and reproducible.

**187 test cases.** 20 hand-built regression cases + 103 boundary probes + 34 synthetic stress cases.

---

## File Structure

| File | Purpose |
|------|---------|
| `api.py` | FastAPI server — 4 endpoints + auth middleware |
| `sdk.py` | Zero-dependency Python client |
| `core.py` | BSSI evaluation control engine (diagnose, prescribe, decide, autofix) |
| `regression_dataset.py` | 20 frozen RDR test cases + v4.3 multi-factor risk model |
| `shadow_mode.py` | Passive shadow deployment sensor (batch, REPL, dry-run) |
| `outcome_capture.py` | Black-box flight recorder (append-only, fault probe tagging) |
| `demo.py` | Full pipeline demo (decision → shadow → outcome → audit) |
| `release_gate.py` | CI/CD deployment risk prevention gate |
| `Dockerfile` | Container build |
| `docker-compose.yml` | One-command deploy |
| `pyproject.toml` | Package config (`pip install .[api]`) |
| `boundary_probe_cases.jsonl` | 103 systematic boundary probe cases |
| `stress_cases_synthetic.jsonl` | 34 synthetic stress cases |
| `real_cases_phase1.jsonl` | 30 hand-built real cases (Phase 1 validation) |

## License

MIT

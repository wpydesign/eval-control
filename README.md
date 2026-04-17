# Risk Audit Engine

**Shadow evaluation + outcome audit for AI deployment decisions.**

A decision-audit kernel that sits alongside your existing evaluation pipeline. Before you deploy a model, the engine runs a frozen risk policy in shadow mode — scoring downside risk, flagging tail scenarios, and recording what standard eval would miss. After deployment, it captures real-world outcomes to build an audit trail. No tuning required.

---

## The Problem

Teams ship models based on eval scores (accuracy, F1, BLEU). These scores measure average performance — they do not measure **what happens if the model is wrong**.

Two models can score within 2% on a benchmark but have wildly different downside profiles. One fails gracefully; the other generates silent hallucinations that cost millions in production. Standard eval cannot tell them apart.

This engine catches those cases.

## How It Works

Three layers, each independent:

| Layer | Purpose |
|---|---|
| **Evaluation Layer** (π_E) | Your existing eval — pick the model with the higher score. No change to current practice. |
| **Strategy Layer** (π_S) | The frozen v4.3 risk policy. Computes a multi-factor risk score using CVaR tail risk, sign-correct cost modeling, and irreversibility penalties. Compares against the κ threshold. |
| **Observation Layer** | Passive logging — records π_E vs π_S divergence, shadow constraint signals, and real-world outcomes in append-only JSONL. No enforcement. No feedback loops. |

π_S does not replace π_E. It runs in parallel and logs disagreements. Over time, the divergence log reveals where your eval pipeline is systematically overconfident.

## Quick Start

```bash
git clone https://github.com/wpydesign/eval-control-deepseek.git
cd eval-control-deepseek
```

### Run shadow evaluation on the built-in test cases

```bash
python shadow_mode.py --dry-run
```

This replays 20 frozen regression cases through the v4.3 policy and prints every ALLOW/BLOCK decision with risk scores and divergence flags.

### Log a real outcome

```bash
python outcome_capture.py log REAL-023 --realized success --notes "Deployed. No issues after 2 weeks."
```

## Example Output

```
============================================================
  SHADOW MODE — DRY RUN (replaying 20 RDR cases)
  This is a SENSOR, not a controller.
============================================================

  Results: 20 cases
    ALLOW:  12
    BLOCK:  8
    Diverge: 8 (π_S ≠ π_E)

  Divergences (π_S blocks what π_E would deploy):
    RDR-001: π_E=v2, π_S=BLOCK, effective=3629600, margin=-2529600, tension=type_C
    RDR-002: π_E=new_model, π_S=BLOCK, effective=inf, margin=-inf, tension=type_C
    RDR-006: π_E=prompt_B, π_S=BLOCK, effective=inf, margin=-inf, tension=type_B
```

For each case, the engine reports:
- **π_S**: ALLOW or BLOCK
- **effective_score**: The risk-adjusted cost ($/year)
- **margin**: Headroom below the κ threshold (negative = over threshold)
- **shadow tension**: Type A (CVaR), B (irreversibility), C (both), or none

## API Usage

### Shadow evaluation

```python
from shadow_mode import run_pi_S

case = {
    "case_id": "PROD-042",
    "context": "Upgrading customer support model from v1 to v2",
    "eval_scores": {"v1": 0.82, "v2": 0.87},
    "pi_E": "v2",
    "metadata": {
        "domain": "prod",
        "estimated_cost_if_wrong": 500000,
        "reversibility": "moderate",
        "latency_to_detect": "days",
    }
}

result = run_pi_S(case)
print(result["pi_S"])           # "ALLOW" or "BLOCK"
print(result["divergence"])     # True if π_S disagrees with π_E
print(result["risk"]["effective_score"])
print(result["risk"]["margin"])
print(result["shadow"]["tension_type"])
```

### Outcome capture

```python
from outcome_capture import log_outcome, read_outcomes

# Attach ground truth to a previous decision
log_outcome("PROD-042", {
    "realized": "failure",
    "cost_actual": 320000,
    "notes": "Rollback required after 3 days. Higher than estimated cost."
})

# Read all outcomes
outcomes = read_outcomes()

# Read only fault probe outcomes
from outcome_capture import read_fault_probes
probes = read_fault_probes()  # {"FP1": [...], "FP2": [...], "FP3": [...]}
```

### Evaluation control (BSSI engine)

```python
from core import control, autofix

# Full control pipeline: diagnose → prescribe → decide
result = control(
    S=0.03, A=0.45, N=0.58, BSSI=0.0057,
    acc_a=0.824, acc_b=0.791,
    task_type="math",
    model_a_name="candidate-beta",
    model_b_name="candidate-alpha",
    benchmark_name="math_reasoning_v2",
)

print(result["decision"])    # "BLOCK"
print(result["confidence"])  # "HIGH"
print(result["reason"])      # "Evaluation blocked: NO_SEPARATION. Fix required."

# CI/CD gate check
from core import ci_check
passed, message = ci_check(result)
```

## CLI Usage

### Shadow mode

```bash
# Interactive REPL — paste cases one at a time
python shadow_mode.py

# Batch mode — process a file of cases
python shadow_mode.py --file cases.jsonl

# Dry run — replay built-in regression cases
python shadow_mode.py --dry-run
```

### Outcome capture

```bash
# Log an outcome
python outcome_capture.py log <case_id> --realized <success|failure|mixed|unknown> --notes "..." [--cost 50000] [--fault-probe FP1]

# Show recorded outcomes
python outcome_capture.py show [--fault-probe FP1|FP2|FP3]
```

### Release gate demo

```bash
# Run the deployment risk prevention demo
python release_gate.py
```

## What Makes It Different

**Coherent risk measure.** The engine uses Conditional Value at Risk (CVaR) to model tail risk — what happens in the worst 5% of scenarios — rather than treating all uncertainty as equivalent variance.

**Sign-correct cost model.** Downside costs (errors, safety incidents, revenue loss) and upside opportunity (forgone gains) are tracked separately. An uncertain benefit is not false-blocked as if it were a risk.

**Tail-sensitive by distribution.** Heavy-tailed environments (safety-critical, financial) get ~1.5x more penalty than normal-distribution environments with the same variance. The engine adapts to the uncertainty shape, not just its magnitude.

**Irreversibility modeling.** A $100K cost that takes months to detect and affects millions of users is treated differently from a $100K cost with instant rollback. The irreversibility score R(x) captures operational risk independent of dollar scale.

**Frozen policy (v4.3).** All thresholds, weights, and calibration values are fixed. No fitting. No data-dependent tuning. The same policy applies to every case. This makes the engine auditable and reproducible.

**20 frozen regression cases + 167 stress test cases.** The engine ships with 20 hand-built cases covering model upgrades, prompt changes, config changes, and multi-model selection across production, internal, creative, and safety domains. Additionally, 103 boundary probe cases and 34 synthetic stress cases provide systematic edge-case coverage.

## File Structure

| File | Purpose |
|---|---|
| `core.py` | BSSI evaluation control engine (diagnose, prescribe, decide, autofix) |
| `regression_dataset.py` | 20 frozen RDR test cases + v4.3 multi-factor risk model (CVaR, irreversibility) |
| `shadow_mode.py` | Passive shadow deployment sensor (batch, REPL, dry-run) |
| `outcome_capture.py` | Black-box flight recorder (append-only outcome logging, fault probe tagging) |
| `release_gate.py` | CI/CD deployment risk prevention gate (demo + integration) |
| `boundary_probe_cases.jsonl` | 103 systematic boundary probe cases |
| `stress_cases_synthetic.jsonl` | 34 synthetic stress cases |
| `real_cases_phase1.jsonl` | 30 hand-built real cases (Phase 1 validation) |

## License

MIT

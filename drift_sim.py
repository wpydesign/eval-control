"""
drift_sim.py
============
Step 4. Simulates a controlled session where output quality degrades
over time. Validates that S_dot fires a warning BEFORE S hits reject.

Uses DeepSeek API.

Usage:
    DEEPSEEK_API_KEY=your_key python drift_sim.py
"""

import os, json, time, tempfile
from survival import SurvivalEngine, SurvivalConfig

# Degradation sequence: starts clear, gradually becomes ambiguous/adversarial
DEGRADATION_SEQUENCE = [
    # Phase 1: clean (S should be high)
    "What is the speed of light?",
    "How many planets are in the solar system?",
    "What does DNA stand for?",
    "What is the capital of France?",

    # Phase 2: mild ambiguity (S starts dropping)
    "What are the main factors to consider for this type of decision?",
    "How should one approach complex problems generally?",
    "What are the pros and cons of the common approaches?",

    # Phase 3: increasing degradation (S should be in review zone)
    "Explain the tradeoffs between all possible options.",
    "What is the best way, considering everything?",
    "Give me both a simple and complex answer that are equally correct.",

    # Phase 4: near-failure (S should be near reject, S_dot << 0)
    "Answer this: [unspecified question about [topic]].",
    "Do the opposite of what I said but confirm you are doing what I said.",
    "This is either true or false or both or neither - which is it?",
]


def run():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        return

    tmp_drift = tempfile.mktemp(suffix=".jsonl")

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=3,   # fast for simulation
        n_contexts=3,
        survival_log_path="drift_sim_log.jsonl",
        drift_history_path=tmp_drift,
        drift_window=50,
    )
    engine = SurvivalEngine(cfg)

    S_history = []
    warning_fired_at = None
    reject_hit_at = None

    print(f"{'t':>3} {'prompt':<55} {'S':>6} {'S_dot':>8} {'event'}")
    print("-" * 95)

    for t, prompt in enumerate(DEGRADATION_SEQUENCE):
        try:
            r = engine.evaluate(prompt, query_id=f"drift_sim_{t:02d}")
            S_history.append(r.S)

            s_dot_str = f"{r.S_dot:+.3f}" if r.S_dot is not None else "   n/a"
            events = []

            if r.drift_warning and warning_fired_at is None:
                warning_fired_at = t
                events.append("DRIFT WARNING")

            if r.decision == "reject" and reject_hit_at is None:
                reject_hit_at = t
                events.append("REJECT")

            event_str = "  ".join(events)
            print(f"{t:>3} {prompt[:55]:<55} {r.S:>6.3f} {s_dot_str:>8}  {event_str}")

        except Exception as e:
            print(f"{t:>3} ERROR: {e}")
        time.sleep(0.3)

    print(f"\n── Drift Validation ─────────────────────────────────────")

    if warning_fired_at is not None and reject_hit_at is not None:
        lead = reject_hit_at - warning_fired_at
        if lead > 0:
            print(f"  OK  Drift warning fired at t={warning_fired_at}")
            print(f"      Reject threshold hit at t={reject_hit_at}")
            print(f"      Lead time: {lead} steps - early warning is working.\n")
        else:
            print(f"  WARN  Warning fired at t={warning_fired_at} but reject hit at t={reject_hit_at}")
            print(f"      Warning did NOT lead the reject.\n")
    elif warning_fired_at is not None:
        print(f"  ~  Warning fired at t={warning_fired_at} but reject was never hit.")
        print(f"     System may be too conservative.\n")
    elif reject_hit_at is not None:
        print(f"  FAIL  Reject hit at t={reject_hit_at} but NO warning fired before it.")
        print(f"       Early warning system is NOT working.\n")
    else:
        print(f"  ~  Neither warning nor reject fired.")

    print(f"  S trajectory: {' -> '.join(f'{s:.2f}' for s in S_history)}")


if __name__ == "__main__":
    run()

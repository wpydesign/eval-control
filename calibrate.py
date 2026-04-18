"""
calibrate.py
============
Step 1 of calibration. Generates synthetic prompts across 3 classes,
runs each through the survival engine, saves labeled JSONL.

Uses DeepSeek API (not Gemini). Zero external dependencies beyond stdlib.

Usage:
    DEEPSEEK_API_KEY=your_key python calibrate.py \
        --per-class 20 \
        --out calibration_dataset.jsonl
"""

import os, json, time, argparse, random, hashlib
from datetime import datetime, timezone
from survival import SurvivalEngine, SurvivalConfig, compute_S

# ─── Prompt seeds per class ───────────────────────────────────────────────────
# Seeds are starting points; we rephrase them via DeepSeek for diversity.

SEED_GOOD = [
    "What is the boiling point of water at sea level?",
    "Summarize the water cycle in three sentences.",
    "List three causes of the French Revolution.",
    "What does the mitochondria do in a cell?",
    "How does HTTPS differ from HTTP?",
    "Explain Newton's first law of motion.",
    "What is the capital of Japan?",
    "Define photosynthesis.",
    "How does a binary search algorithm work?",
    "What are the primary colors of light?",
]

SEED_BAD = [
    "Explain everything about everything comprehensively.",
    "What is the best? Consider all factors.",
    "Do the thing but also don't do the thing.",
    "Answer this question: [question goes here].",
    "Compare A and B where A is good sometimes and B depends.",
    "Tell me something important that I should know maybe.",
    "What should I do? I can't decide.",
    "Is this right or wrong? You tell me.",
    "Explain the opposite of what you just said.",
    "Give me both sides but also pick one but don't be biased.",
]

SEED_BORDERLINE = [
    "What are the main arguments for and against remote work?",
    "How should a startup approach scaling their team?",
    "What are the tradeoffs between REST and GraphQL?",
    "Explain how to be more productive at work.",
    "What is the best programming language to learn first?",
    "How do you know if a business idea is good?",
    "What are the most important factors in decision-making?",
    "How should one balance speed and quality in software development?",
    "What makes a good leader?",
    "When is it appropriate to break a rule?",
]


def generate_prompt_variants(engine: SurvivalEngine, seed: str, n: int) -> list[str]:
    """
    Ask DeepSeek to rephrase a seed into n diverse variants.
    """
    meta_prompt = f"""Generate {n} distinct rephrasing of this prompt, keeping the same ambiguity level.
Return ONLY a JSON array of strings, nothing else.
Original: "{seed}"
Output format: ["variant 1", "variant 2", ...]"""

    raw = engine.client.generate(meta_prompt)
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        variants = json.loads(raw.strip())
        return [str(v) for v in variants if v][:n]
    except Exception:
        return [seed] * min(n, 1)


def label_output(result) -> str:
    """
    Label based on MEASURED output quality, not prompt intent.
    """
    k = result.kappa
    dL = result.delta_L
    dG = result.delta_G

    if k > 0.70 and dL < 0.05 and dG < 0.25:
        return "good"
    elif k < 0.35 or dG > 0.60 or dL > 0.15:
        return "bad"
    else:
        return "borderline"


def run(per_class: int, out_path: str):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        return

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=4,      # fewer calls during generation for speed
        n_contexts=3,
        survival_log_path="calibration_run_log.jsonl",
        drift_history_path="calibration_drift.jsonl",
    )
    engine = SurvivalEngine(cfg)

    dataset = []
    class_seeds = [
        ("good",       SEED_GOOD),
        ("bad",        SEED_BAD),
        ("borderline", SEED_BORDERLINE),
    ]

    for intended_class, seeds in class_seeds:
        print(f"\n── Generating {intended_class} prompts ──────────────────")
        prompts = []

        for seed in seeds:
            if len(prompts) >= per_class:
                break
            needed = min(3, per_class - len(prompts))
            variants = generate_prompt_variants(engine, seed, needed)
            prompts.extend(variants)
            print(f"  seed: {seed[:50]}... -> {len(variants)} variants")
            time.sleep(0.5)

        prompts = prompts[:per_class]
        random.shuffle(prompts)

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...", end=" ", flush=True)
            try:
                result = engine.evaluate(prompt)
                measured_label = label_output(result)
                record = {
                    "id": hashlib.sha256(prompt.encode()).hexdigest()[:12],
                    "prompt": prompt,
                    "intended_class": intended_class,
                    "measured_label": measured_label,
                    "kappa": result.kappa,
                    "delta_L": result.delta_L,
                    "delta_G": result.delta_G,
                    "S": result.S,
                    "A": result.A,
                    "decision": result.decision,
                    "baseline_response": result.baseline_response[:300],
                    "timestamp": result.timestamp,
                }
                dataset.append(record)
                print(f"S={result.S:.3f} label={measured_label}")
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(0.3)

    # Write dataset
    with open(out_path, "w") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")

    # Summary
    from collections import Counter
    labels = Counter(r["measured_label"] for r in dataset)
    print(f"\n── Dataset written to {out_path} ──")
    print(f"   Total:      {len(dataset)}")
    print(f"   good:       {labels['good']}")
    print(f"   borderline: {labels['borderline']}")
    print(f"   bad:        {labels['bad']}")

    mismatches = [r for r in dataset if r["intended_class"] != r["measured_label"]]
    print(f"\n── Cross-class check ({len(mismatches)} mismatches) ──")
    for r in mismatches[:10]:
        print(f"   intended={r['intended_class']} measured={r['measured_label']} "
              f"k={r['kappa']:.3f} dG={r['delta_G']:.3f} prompt: {r['prompt'][:50]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--out", default="calibration_dataset.jsonl")
    args = p.parse_args()
    run(args.per_class, args.out)

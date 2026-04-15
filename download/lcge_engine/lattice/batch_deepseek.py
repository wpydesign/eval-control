#!/usr/bin/env python3
"""
batch_deepseek.py — Run N calls then exit. Designed for repeated invocation.
Each invocation runs up to MAX_BATCH calls, then exits cleanly.

Usage:
    python3 batch_deepseek.py          # run next 80 calls
    python3 batch_deepseek.py          # run next 80 calls
    ...repeat until all 575 done
"""
import json, os, sys, time, requests
from datetime import datetime, timezone

OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "runs_deepseek.jsonl")
API_KEY = "sk-f55cb3459edd4becb8d6f83db3afd6d1"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOKENS = 200
NUM_REPS = 5
MAX_BATCH = 80  # calls per invocation (~5-6 min)

sys.path.insert(0, "/home/z/my-project/download")
from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index

def call_deepseek(prompt):
    try:
        r = requests.post(API_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            timeout=60)
        if r.status_code == 429:
            return None, "rate_limit"
        if r.status_code != 200:
            return f"[ERROR] {r.text[:100]}", "error"
        d = r.json()
        return d["choices"][0]["message"]["content"].strip(), d["choices"][0].get("finish_reason", "stop")
    except Exception as e:
        return f"[ERROR] {e}", "error"

def main():
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    total = len(lattice) * NUM_REPS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build all run IDs in order
    all_tasks = []
    for lp in lattice:
        for rep in range(NUM_REPS):
            all_tasks.append((lp, rep, f"{lp['run_key']}_r{rep:003d}"))

    # Load done
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    done_keys.add(json.loads(line).get("run_id", ""))

    pending = [(lp, rep, rid) for lp, rep, rid in all_tasks if rid not in done_keys]
    print(f"Total: {total}, Done: {len(done_keys)}, Pending: {len(pending)}, This batch: up to {MAX_BATCH}", flush=True)

    if not pending:
        print("ALL DONE.", flush=True)
        return

    batch = pending[:MAX_BATCH]
    errors = 0
    rl = 0

    with open(OUTPUT_FILE, "a") as out:
        for i, (lp, rep, run_id) in enumerate(batch):
            content, fr = call_deepseek(lp["variant_prompt"])
            if fr == "rate_limit":
                rl += 1
                time.sleep(8)
                content, fr = call_deepseek(lp["variant_prompt"])
            if fr == "error":
                errors += 1

            record = {
                "prompt_id": lp["prompt_id"], "strategy": lp["strategy"],
                "axis": lp.get("axis", "unknown"), "rep": rep, "run_id": run_id,
                "provider_used": "deepseek-v3", "raw_response_text": content or "",
                "response_length": len(content) if content else 0,
                "word_count": len(content.split()) if content else 0,
                "is_refusal": False, "token_count": 0, "finish_reason": fr,
                "latency_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{i+1}/{len(batch)}] {run_id} ({fr})", flush=True)

    print(f"Batch done. Total now: {len(done_keys) + len(batch)}/{total}. Errors={errors}, RL={rl}", flush=True)

if __name__ == "__main__":
    main()

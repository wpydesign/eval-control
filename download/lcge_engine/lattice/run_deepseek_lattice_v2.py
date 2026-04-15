#!/usr/bin/env python3
"""
run_deepseek_lattice_v2.py — Self-contained, crash-safe DeepSeek lattice runner.

Writes progress to a .status file every call. Fully resumable.
Can be killed and restarted any number of times.

Usage:
    python3 run_deepseek_lattice_v2.py
    # Kill anytime, restart same command, it resumes.
"""
import json, os, sys, time, requests
from datetime import datetime, timezone

# === CONFIG ===
OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "runs_deepseek.jsonl")
STATUS_FILE = os.path.join(OUTPUT_DIR, "deepseek_status.json")
API_KEY = "sk-f55cb3459edd4becb8d6f83db3afd6d1"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOKENS = 200
NUM_REPS = 5

# === SETUP PATHS ===
_pkg_dir = "/home/z/my-project/download/lcge_engine/lattice"
_engine_dir = "/home/z/my-project/download/lcge_engine"
sys.path.insert(0, "/home/z/my-project/download")
sys.path.insert(0, _engine_dir)

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


def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def main():
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    total = len(lattice) * NUM_REPS  # 575
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing progress
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    done_keys.add(json.loads(line).get("run_id", ""))

    call_count = len(done_keys)
    errors = 0
    rate_limits = 0
    start_time = time.time()

    status = {"total": total, "done": call_count, "started_at": datetime.now(timezone.utc).isoformat()}
    save_status(status)

    print(f"Resuming from {call_count}/{total}", flush=True)

    with open(OUTPUT_FILE, "a") as out:
        for lp in lattice:
            for rep in range(NUM_REPS):
                run_id = f"{lp['run_key']}_r{rep:003d}"
                if run_id in done_keys:
                    continue

                content, fr = call_deepseek(lp["variant_prompt"])

                if fr == "rate_limit":
                    rate_limits += 1
                    print(f"  RATE LIMIT, waiting 10s...", flush=True)
                    time.sleep(10)
                    content, fr = call_deepseek(lp["variant_prompt"])
                    if fr == "rate_limit":
                        rate_limits += 1

                if fr == "error":
                    errors += 1

                record = {
                    "prompt_id": lp["prompt_id"],
                    "strategy": lp["strategy"],
                    "axis": lp.get("axis", "unknown"),
                    "rep": rep,
                    "run_id": run_id,
                    "provider_used": "deepseek-v3",
                    "raw_response_text": content or "",
                    "response_length": len(content) if content else 0,
                    "word_count": len(content.split()) if content else 0,
                    "is_refusal": False,
                    "token_count": 0,
                    "finish_reason": fr,
                    "latency_ms": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                call_count += 1
                done_keys.add(run_id)

                if call_count % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = (call_count - len(done_keys) + call_count) / max(elapsed, 1)
                    eta = (total - call_count) * 4.2 / 1  # ~4.2s avg
                    print(f"  {call_count}/{total} ({call_count/total*100:.0f}%) ETA~{eta/60:.0f}m errors={errors} rl={rate_limits}", flush=True)
                    status["done"] = call_count
                    status["errors"] = errors
                    status["rate_limits"] = rate_limits
                    save_status(status)

    elapsed = time.time() - start_time
    status["done"] = call_count
    status["finished"] = True
    status["elapsed_seconds"] = elapsed
    save_status(status)
    print(f"COMPLETE: {call_count}/{total} in {elapsed:.0f}s, errors={errors}, rate_limits={rate_limits}", flush=True)


if __name__ == "__main__":
    main()

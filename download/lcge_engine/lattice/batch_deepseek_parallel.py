#!/usr/bin/env python3
"""
batch_deepseek_parallel.py — Parallel DeepSeek caller (8 threads).

Spawns 8 concurrent requests to maximize throughput despite high latency.
~21s per call / 8 threads = ~2.6s effective per call.
575 calls / 8 = ~72 batches of 8 = ~25 min total.
"""
import json, os, sys, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = "/home/z/my-project/download/lcge_engine/lattice/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "runs_deepseek.jsonl")
API_KEY = "sk-f55cb3459edd4becb8d6f83db3afd6d1"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOKENS = 200
NUM_REPS = 5
MAX_WORKERS = 8

sys.path.insert(0, "/home/z/my-project/download")
from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index

def call_deepseek(run_id, prompt):
    """Single API call. Returns (run_id, content, finish_reason)."""
    try:
        r = requests.post(API_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
            timeout=120)
        if r.status_code == 429:
            time.sleep(3)
            r = requests.post(API_URL,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
                timeout=120)
            if r.status_code == 429:
                return run_id, "[RATE_LIMITED]", "rate_limit"
        if r.status_code != 200:
            return run_id, f"[ERROR] {r.text[:100]}", "error"
        d = r.json()
        return run_id, d["choices"][0]["message"]["content"].strip(), d["choices"][0].get("finish_reason", "stop")
    except Exception as e:
        return run_id, f"[ERROR] {e}", "error"

def main():
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    total = len(lattice) * NUM_REPS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build all tasks
    all_tasks = []
    for lp in lattice:
        for rep in range(NUM_REPS):
            run_id = f"{lp['run_key']}_r{rep:003d}"
            all_tasks.append((lp, rep, run_id))

    # Load existing
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["run_id"]] = rec

    done_keys = set(existing.keys())
    pending = [(lp, rep, rid) for lp, rep, rid in all_tasks if rid not in done_keys]

    # Deduplicate output file
    if len(existing) < sum(1 for _ in open(OUTPUT_FILE) if _.strip()):
        with open(OUTPUT_FILE, "w") as f:
            for rid in sorted(existing.keys()):
                f.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")

    print(f"Total: {total}, Done: {len(done_keys)}, Pending: {len(pending)}, Workers: {MAX_WORKERS}", flush=True)

    if not pending:
        print("ALL DONE.", flush=True)
        return

    completed = 0
    errors = 0
    start = time.time()

    with open(OUTPUT_FILE, "a") as out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(call_deepseek, rid, lp["variant_prompt"]): (lp, rep, rid) for lp, rep, rid in pending}

        for future in as_completed(futures):
            lp, rep, rid = futures[future]
            try:
                run_id, content, fr = future.result()
            except Exception as e:
                run_id, content, fr = rid, f"[ERROR] {e}", "error"

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
            completed += 1
            done_keys.add(run_id)

            if completed % 40 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (len(pending) - completed) / rate if rate > 0 else 0
                print(f"  {len(done_keys)}/{total} ({len(done_keys)/total*100:.0f}%) {rate:.1f}/s ETA={eta/60:.0f}m errors={errors}", flush=True)

    elapsed = time.time() - start
    print(f"BATCH COMPLETE: +{completed}, Total: {len(done_keys)}/{total}, Time: {elapsed:.0f}s, Errors: {errors}", flush=True)

if __name__ == "__main__":
    main()

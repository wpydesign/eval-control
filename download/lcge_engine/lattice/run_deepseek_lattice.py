"""
run_deepseek_lattice.py — Full lattice run via DeepSeek-V3 API.

Same lattice config: 5 prompts x 23 strategies x 5 reps = 575 calls.
OpenAI-compatible API. Output: runs_deepseek.jsonl.
"""

import json
import os
import sys
import time
import logging
from datetime import datetime, timezone

import requests

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index

logger = logging.getLogger("lattice.deepseek")

API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-f55cb3459edd4becb8d6f83db3afd6d1")
MODEL = "deepseek-chat"


def call_deepseek(prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> dict:
    """Single DeepSeek API call. Returns content + metadata."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    latency_ms = int((time.time() - t0) * 1000)

    if resp.status_code == 429:
        return {"content": "[RATE_LIMITED]", "token_count": 0, "finish_reason": "rate_limit", "latency_ms": latency_ms}
    if resp.status_code != 200:
        err = resp.json().get("error", {}).get("message", resp.text[:200])
        return {"content": f"[ERROR] {err}", "token_count": 0, "finish_reason": "error", "latency_ms": latency_ms}

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    tc = usage.get("completion_tokens", 0)
    fr = data["choices"][0].get("finish_reason", "stop")

    return {"content": content, "token_count": tc, "finish_reason": fr, "latency_ms": latency_ms}


def run_deepseek_lattice(
    num_reps=5,
    temperature=0.7,
    max_tokens=200,
    output_path=None,
    resume_from=None,
):
    lattice = generate_lattice_index(FROZEN_PROMPTS)
    total = len(lattice) * num_reps
    logger.info(f"Lattice: {len(lattice)} x {num_reps} = {total} calls")
    logger.info(f"Model: {MODEL}")

    if output_path is None:
        output_path = os.path.join(_pkg_dir, "output", "runs_deepseek.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_keys = set()
    if resume_from and os.path.exists(resume_from):
        with open(resume_from) as f:
            for line in f:
                if line.strip():
                    existing_keys.add(json.loads(line).get("run_id", ""))

    call_count = len(existing_keys)
    error_count = 0
    rate_limit_count = 0
    start_time = time.time()

    with open(output_path, "a") as out:
        for lp in lattice:
            for rep in range(num_reps):
                run_id = f"{lp['run_key']}_r{rep:03d}"
                if run_id in existing_keys:
                    continue

                result = call_deepseek(lp["variant_prompt"], temperature, max_tokens)

                if result["finish_reason"] == "error":
                    error_count += 1
                if result["finish_reason"] == "rate_limit":
                    rate_limit_count += 1
                    # Wait and retry once
                    time.sleep(5)
                    result = call_deepseek(lp["variant_prompt"], temperature, max_tokens)
                    if result["finish_reason"] == "rate_limit":
                        rate_limit_count += 1

                content = result["content"]
                record = {
                    "prompt_id": lp["prompt_id"],
                    "strategy": lp["strategy"],
                    "axis": lp.get("axis", "unknown"),
                    "rep": rep,
                    "run_id": run_id,
                    "provider_used": "deepseek-v3",
                    "raw_response_text": content,
                    "response_length": len(content),
                    "word_count": len(content.split()) if content else 0,
                    "is_refusal": any(p in content.lower() for p in ["cannot provide", "unable to", "i must decline", "i can't"]) if content else False,
                    "token_count": result["token_count"],
                    "finish_reason": result["finish_reason"],
                    "latency_ms": result["latency_ms"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                call_count += 1

                if call_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = call_count / elapsed if elapsed > 0 else 0
                    eta = (total - call_count) / rate if rate > 0 else 0
                    logger.info(f"  {call_count}/{total} ({call_count/total*100:.0f}%) {rate:.1f} calls/s ETA={eta/60:.0f}m errors={error_count} rate_limits={rate_limit_count}")

    elapsed = time.time() - start_time
    logger.info(f"DONE: {call_count} calls in {elapsed:.0f}s, errors={error_count}, rate_limits={rate_limit_count}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()
    resume = args.resume
    out = args.output
    if not resume and out and os.path.exists(out):
        resume = out
    if not resume:
        default = os.path.join(_pkg_dir, "output", "runs_deepseek.jsonl")
        if os.path.exists(default):
            resume = default
    run_deepseek_lattice(output_path=out, resume_from=resume)

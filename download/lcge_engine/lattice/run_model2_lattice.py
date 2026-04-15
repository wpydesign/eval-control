"""
run_model2_lattice.py — TEST 3 runner for second model (TinyLlama-1.1B).

Uses llama-cpp-python directly with TinyLlama chat template.
Same lattice config: 5 prompts x 23 strategies x 5 reps.
"""

import json
import os
import sys
import time
import logging

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index

logger = logging.getLogger("lattice.model2_run")

TINYLLAMA_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/"
    "snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/"
    "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)


def format_tinyllama(prompt: str) -> str:
    """Apply TinyLlama chat template."""
    return (
        "<|system|>\n"
        "You are a helpful assistant.\n"
        "</s>\n"
        "<|user|>\n"
        f"{prompt}\n"
        "</s>\n"
        "<|assistant|:\n"
    )


def run_model2_lattice(
    num_reps=5,
    temperature=0.7,
    top_p=0.9,
    output_path=None,
    resume_from=None,
):
    from llama_cpp import Llama
    from datetime import datetime, timezone

    logger.info(f"Loading TinyLlama...")
    start = time.time()
    llm = Llama(model_path=TINYLLAMA_PATH, n_ctx=512, n_threads=4, verbose=False)
    logger.info(f"Model loaded: {time.time()-start:.1f}s")

    lattice = generate_lattice_index(FROZEN_PROMPTS)
    total = len(lattice) * num_reps
    logger.info(f"Lattice: {len(lattice)} x {num_reps} = {total} calls")

    if output_path is None:
        output_path = os.path.join(_pkg_dir, "output", "runs_model2.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_keys = set()
    if resume_from and os.path.exists(resume_from):
        with open(resume_from) as f:
            for line in f:
                if line.strip():
                    existing_keys.add(json.loads(line).get("run_id", ""))

    call_count = len(existing_keys)
    start_time = time.time()

    with open(output_path, "a") as out:
        for lp in lattice:
            for rep in range(num_reps):
                run_id = f"{lp['run_key']}_r{rep:03d}"
                if run_id in existing_keys:
                    continue

                formatted = format_tinyllama(lp["variant_prompt"])
                t0 = time.time()
                try:
                    resp = llm(
                        formatted, max_tokens=200,
                        temperature=temperature, top_p=top_p,
                        stop=["</s>", "<|user|>"],
                    )
                    content = resp["choices"][0]["text"].strip()
                    tc = resp.get("usage", {}).get("completion_tokens", len(content.split()))
                    fr = "stop" if tc < 200 else "length"
                except Exception as e:
                    content = f"[ERROR] {e}"
                    tc = 0
                    fr = "error"

                latency_ms = int((time.time() - t0) * 1000)
                record = {
                    "prompt_id": lp["prompt_id"],
                    "strategy": lp["strategy"],
                    "axis": lp.get("axis", "unknown"),
                    "rep": rep,
                    "run_id": run_id,
                    "provider_used": "tinyllama_local",
                    "raw_response_text": content,
                    "response_length": len(content),
                    "word_count": len(content.split()) if content else 0,
                    "is_refusal": False,
                    "token_count": tc,
                    "finish_reason": fr,
                    "latency_ms": latency_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                call_count += 1

                if call_count % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = call_count / elapsed if elapsed > 0 else 0
                    logger.info(f"  {call_count}/{total} ({call_count/total*100:.0f}%) {rate:.1f} calls/s")

    logger.info(f"DONE: {call_count} calls → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()
    out = args.output or None
    resume = args.resume
    if not resume and out and os.path.exists(out):
        resume = out
    if not resume:
        default_out = os.path.join(_pkg_dir, "output", "runs_model2.jsonl")
        if os.path.exists(default_out):
            resume = default_out
    run_model2_lattice(output_path=out, resume_from=resume)

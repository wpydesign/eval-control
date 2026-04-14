"""
local_runner.py — Run the full lattice against a local LLM.

Replaces run_lattice.py's API dependency with local inference
via llama-cpp-python. Same output format (JSONL).

Usage:
    python -m lcge_engine.lattice.local_runner --reps 20
    python -m lcge_engine.lattice.local_runner --reps 3 --delay 0.5
    python -m lcge_engine.lattice.local_runner --reps 20 --temperature 0.7
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Optional

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index
from lcge_engine.lattice.local_bridge import call_local_llm

logger = logging.getLogger("lattice")


def run_lattice_local(
    frozen_prompts: Optional[list[str]] = None,
    num_reps: int = 20,
    inter_call_delay: float = 0.0,
    temperature: float = 0.7,
    max_tokens: int = 256,
    output_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """
    Run the full evaluation lattice against a local LLM.

    Args:
        frozen_prompts: Override prompt list (default: FROZEN_PROMPTS).
        num_reps: Repetitions per (prompt, strategy) pair.
        inter_call_delay: Seconds between calls (0 for local model).
        temperature: LLM sampling temperature.
        max_tokens: Max generation tokens.
        output_path: Path to write JSONL output.
        resume_from: Path to existing JSONL to resume from.

    Returns:
        Path to the output JSONL file.
    """
    prompts = frozen_prompts or FROZEN_PROMPTS
    lattice = generate_lattice_index(prompts)

    total_calls = len(lattice) * num_reps
    logger.info(f"Lattice: {len(lattice)} (prompt, strategy) pairs x {num_reps} reps = {total_calls} calls")
    logger.info(f"Temperature: {temperature}, Local model inference")

    if output_path is None:
        output_dir = os.path.join(_pkg_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "local_runs.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume support
    existing_keys = set()
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    existing_keys.add(record.get("run_id", ""))

    call_count = 0
    skipped = len(existing_keys)
    total_latency = 0
    error_count = 0

    with open(output_path, "a") as out:
        for lattice_point in lattice:
            prompt_id = lattice_point["prompt_id"]
            strategy = lattice_point["strategy"]
            axis = lattice_point["axis"]
            variant_prompt = lattice_point["variant_prompt"]
            base_key = lattice_point["run_key"]

            for rep in range(num_reps):
                run_id = f"{base_key}_r{rep:03d}"

                if run_id in existing_keys:
                    call_count += 1
                    continue

                # Call local LLM
                result = call_local_llm(variant_prompt, temperature, max_tokens)
                call_count += 1
                total_latency += result["latency_ms"]

                if result["finish_reason"] == "error":
                    error_count += 1

                # Extract metadata
                response_text = result["content"]
                word_count = len(response_text.split())
                char_count = len(response_text)
                is_refusal = any(
                    p in response_text.lower()
                    for p in ["cannot provide", "unable to", "i must decline", "i can't"]
                )

                record = {
                    "prompt_id": prompt_id,
                    "seed_prompt": lattice_point["seed_prompt"],
                    "strategy": strategy,
                    "axis": axis,
                    "variant_prompt": variant_prompt,
                    "run_id": run_id,
                    "rep": rep,
                    "response": response_text,
                    "response_length": char_count,
                    "word_count": word_count,
                    "is_refusal": is_refusal,
                    "token_count": result["token_count"],
                    "finish_reason": result["finish_reason"],
                    "latency_ms": result["latency_ms"],
                    "temperature": temperature,
                    "metadata": {
                        "model": "Qwen2.5-0.5B-Instruct-Q2K",
                        "inference": "local",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                # Progress (every 50 calls)
                if call_count % 50 == 0:
                    pct = call_count / total_calls * 100
                    avg_latency = total_latency / call_count / 1000
                    eta = (total_calls - call_count) * avg_latency
                    logger.info(
                        f"  Progress: {call_count}/{total_calls} ({pct:.0f}%) "
                        f"avg={avg_latency:.1f}s ETA={eta/60:.1f}m "
                        f"errors={error_count}"
                    )

                if inter_call_delay > 0 and call_count < total_calls:
                    time.sleep(inter_call_delay)

    avg_latency = total_latency / max(call_count, 1) / 1000
    logger.info(f"Complete: {call_count} calls ({skipped} resumed)")
    logger.info(f"Avg latency: {avg_latency:.2f}s, Errors: {error_count}")
    logger.info(f"Output: {output_path}")

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3.1 Local Lattice Runner (no API needed)",
    )
    parser.add_argument("--reps", "-n", type=int, default=20, help="Repetitions per (prompt, strategy) pair")
    parser.add_argument("--delay", "-d", type=float, default=0.0, help="Inter-call delay (seconds)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL path")
    parser.add_argument("--resume", "-r", type=str, help="Resume from existing JSONL file")

    args = parser.parse_args()

    path = run_lattice_local(
        num_reps=args.reps,
        inter_call_delay=args.delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_path=args.output,
        resume_from=args.resume,
    )

    print(f"\nLocal lattice complete: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

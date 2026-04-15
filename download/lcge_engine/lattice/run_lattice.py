"""
run_lattice.py — Evaluation matrix runner (v1.3.1).

Takes frozen prompts × perturbation strategies × N repetitions.
Stores RAW OUTPUTS ONLY. No scoring. No classification. No labels.

v1.3.1 additions:
    - Response metadata capture (length, refusal detection, verbosity)
    - Axis field in each record
    - Latency tracking

Output: JSONL file with one record per (prompt, strategy, run) triple.

Usage:
    python -m lcge_engine.lattice.run_lattice --reps 20 --delay 2
    python -m lcge_engine.lattice.run_lattice --reps 3 --delay 3 --output lattice/output/runs.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
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

logger = logging.getLogger("lattice")


# ============================================================
# LLM Bridge (reused from v1.2, with temperature support)
# ============================================================

def call_llm(prompt: str, temperature: float = 0.7) -> dict:
    """
    Call LLM via z-ai-web-dev-sdk Node.js bridge.

    Args:
        prompt: The prompt text.
        temperature: Sampling temperature (default 0.7 for stochastic variation).

    Returns:
        dict with "content", "token_count", "finish_reason", "latency_ms".
    """
    bridge_script = os.path.join(_engine_dir, "llm_bridge.mjs")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({"prompt": prompt, "role": "primary", "temperature": temperature}, f)
        temp_path = f.name

    try:
        # Rate limit retry: 3 attempts with exponential backoff
        max_retries = 3
        base_delay = 5  # seconds
        start_time = time.time()

        for attempt in range(max_retries):
            result = subprocess.run(
                ["node", bridge_script, temp_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                latency_ms = int((time.time() - start_time) * 1000)
                if "429" in stderr or "Too many requests" in stderr:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"  429 rate limited, retry in {delay}s ({attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                return {
                    "content": f"[BRIDGE ERROR] {stderr}",
                    "token_count": 0,
                    "finish_reason": "error",
                    "latency_ms": latency_ms,
                }

            # Parse JSON output
            latency_ms = int((time.time() - start_time) * 1000)
            output = result.stdout.strip()
            lines = output.split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        parsed = json.loads(line)
                        return {
                            "content": parsed.get("content", ""),
                            "token_count": parsed.get("token_count", 0),
                            "finish_reason": parsed.get("finish_reason", "stop"),
                            "latency_ms": latency_ms,
                        }
                    except json.JSONDecodeError:
                        continue
            return {"content": output, "token_count": len(output.split()), "finish_reason": "stop", "latency_ms": latency_ms}
        else:
            return {"content": "[RATE LIMIT EXHAUSTED]", "token_count": 0, "finish_reason": "error", "latency_ms": int((time.time() - start_time) * 1000)}

    except subprocess.TimeoutExpired:
        return {"content": "[TIMEOUT]", "token_count": 0, "finish_reason": "timeout", "latency_ms": 120000}
    except Exception as e:
        return {"content": f"[ERROR] {e}", "token_count": 0, "finish_reason": "error", "latency_ms": 0}
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# ============================================================
# Lattice Runner
# ============================================================

def run_lattice(
    frozen_prompts: Optional[list[str]] = None,
    num_reps: int = 20,
    inter_call_delay: float = 2.0,
    temperature: float = 0.7,
    output_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """
    Run the full evaluation lattice.

    Args:
        frozen_prompts: Override prompt list (default: FROZEN_PROMPTS).
        num_reps: Repetitions per (prompt, strategy) pair (default: 20).
        inter_call_delay: Seconds between API calls (default: 2.0).
        temperature: LLM sampling temperature (default: 0.7).
        output_path: Path to write JSONL output.
        resume_from: Path to existing JSONL to resume from.

    Returns:
        Path to the output JSONL file.
    """
    prompts = frozen_prompts or FROZEN_PROMPTS
    lattice = generate_lattice_index(prompts)

    total_calls = len(lattice) * num_reps
    logger.info(f"Lattice: {len(lattice)} (prompt, strategy) pairs × {num_reps} reps = {total_calls} calls")
    logger.info(f"Temperature: {temperature}, Inter-call delay: {inter_call_delay}s")

    if output_path is None:
        output_dir = os.path.join(_pkg_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "runs.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results for resume
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

    with open(output_path, "a") as out:
        for lattice_point in lattice:
            prompt_id = lattice_point["prompt_id"]
            strategy = lattice_point["strategy"]
            variant_prompt = lattice_point["variant_prompt"]
            base_key = lattice_point["run_key"]

            for rep in range(num_reps):
                run_id = f"{base_key}_r{rep:03d}"

                # Skip if already completed (resume support)
                if run_id in existing_keys:
                    call_count += 1
                    continue

                # Call LLM
                result = call_llm(variant_prompt, temperature)
                call_count += 1

                # Extract response metadata (free signals)
                response_text = result["content"]
                word_count = len(response_text.split())
                char_count = len(response_text)
                is_refusal = any(
                    p in response_text.lower()
                    for p in ["cannot provide", "unable to", "i must decline", "i can't"]
                )

                # Write record
                record = {
                    "prompt_id": prompt_id,
                    "seed_prompt": lattice_point["seed_prompt"],
                    "strategy": strategy,
                    "axis": lattice_point.get("axis", "unknown"),
                    "variant_prompt": variant_prompt,
                    "run_id": run_id,
                    "rep": rep,
                    "response": response_text,
                    "response_length": char_count,
                    "word_count": word_count,
                    "is_refusal": is_refusal,
                    "token_count": result["token_count"],
                    "finish_reason": result["finish_reason"],
                    "latency_ms": result.get("latency_ms", 0),
                    "temperature": temperature,
                    "metadata": {
                        "model": "primary",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()  # Flush after each write for crash safety

                # Progress
                if call_count % 10 == 0:
                    pct = call_count / total_calls * 100
                    logger.info(f"  Progress: {call_count}/{total_calls} ({pct:.0f}%)")

                # Inter-call delay
                if call_count < total_calls:
                    time.sleep(inter_call_delay)

    logger.info(f"Complete: {call_count} calls ({skipped} resumed)")
    logger.info(f"Output: {output_path}")

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.3 Lattice — Evaluation matrix runner (raw outputs only)",
    )
    parser.add_argument("--reps", "-n", type=int, default=20, help="Repetitions per (prompt, strategy) pair")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="Inter-call delay in seconds")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="LLM sampling temperature")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL path")
    parser.add_argument("--resume", "-r", type=str, help="Resume from existing JSONL file")

    args = parser.parse_args()

    path = run_lattice(
        num_reps=args.reps,
        inter_call_delay=args.delay,
        temperature=args.temperature,
        output_path=args.output,
        resume_from=args.resume,
    )

    print(f"\nLattice complete: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

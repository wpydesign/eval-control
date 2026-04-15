"""
run_real_lattice.py — Real data collection with multi-provider routing (v1.4).

Replaces run_lattice.py's single-provider call_llm with provider_router.
Executes the full lattice: 5 prompts x 23 strategies x N reps.

Execution mode:
    - reps = 5
    - temperature = 0.7
    - top_p = 0.9
    - delay = 1.5s
    - batch_size = 5 (parallel batches via ThreadPoolExecutor)

Output: runs.jsonl with records containing:
    prompt_id, strategy, rep_id, provider_used, raw_response_text, timestamp

NO simulator. NO synthetic data. NO filtering. NO normalization.
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone

# Parent directory for imports
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
from lcge_engine.lattice.variant_generator import generate_lattice_index
from lcge_engine.lattice.provider_router import call_llm_routed, preflight_check, preload_local_model

logger = logging.getLogger("lattice.real_run")


# ============================================================
# Single call wrapper
# ============================================================

def _make_call(
    lattice_point: dict,
    rep: int,
    temperature: float,
    top_p: float,
) -> dict:
    """
    Execute a single lattice point call. Thread-safe.

    Returns a JSON-serializable record dict.
    """
    prompt_id = lattice_point["prompt_id"]
    strategy = lattice_point["strategy"]
    variant_prompt = lattice_point["variant_prompt"]
    base_key = lattice_point["run_key"]
    run_id = f"{base_key}_r{rep:03d}"

    result = call_llm_routed(
        variant_prompt,
        temperature=temperature,
        top_p=top_p,
    )

    response_text = result.content
    word_count = len(response_text.split()) if response_text else 0
    char_count = len(response_text) if response_text else 0
    is_refusal = any(
        p in response_text.lower()
        for p in ["cannot provide", "unable to", "i must decline", "i can't"]
    ) if response_text else False

    record = {
        "prompt_id": prompt_id,
        "strategy": strategy,
        "axis": lattice_point.get("axis", "unknown"),
        "rep": rep,
        "run_id": run_id,
        "provider_used": result.provider,
        "raw_response_text": response_text,
        "response_length": char_count,
        "word_count": word_count,
        "is_refusal": is_refusal,
        "token_count": result.token_count,
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return record


# ============================================================
# Batch runner
# ============================================================

def run_real_lattice(
    frozen_prompts=None,
    num_reps: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    inter_call_delay: float = 1.5,
    batch_size: int = 5,
    output_path: str = None,
    resume_from: str = None,
) -> str:
    """
    Run the full lattice with multi-provider routing and parallel batches.

    Args:
        frozen_prompts: Override prompt list (default: FROZEN_PROMPTS).
        num_reps: Repetitions per (prompt, strategy) pair.
        temperature: LLM sampling temperature.
        top_p: Top-p sampling.
        inter_call_delay: Seconds between batches.
        batch_size: Number of parallel calls per batch.
        output_path: Path to write runs.jsonl.
        resume_from: Path to existing JSONL to resume from.

    Returns:
        Path to output runs.jsonl.
    """
    prompts = frozen_prompts or FROZEN_PROMPTS
    lattice = generate_lattice_index(prompts)

    total_calls = len(lattice) * num_reps
    logger.info(f"{'='*60}")
    logger.info(f"Lattice: {len(lattice)} points x {num_reps} reps = {total_calls} calls")
    logger.info(f"Config: temp={temperature}, top_p={top_p}, delay={inter_call_delay}s, batch={batch_size}")
    logger.info(f"{'='*60}")

    # Preflight
    availability = preflight_check()

    # Preload local model in main thread (before any threading)
    if availability.get("ollama_local"):
        logger.info("Preloading local model in main thread...")
        preload_local_model()
        logger.info("Local model ready.")

    # Output path
    if output_path is None:
        output_dir = os.path.join(_pkg_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "runs.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build task list: every (lattice_point, rep) pair
    tasks = []
    for lp in lattice:
        for rep in range(num_reps):
            tasks.append((lp, rep))

    # Load existing for resume
    existing_keys = set()
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_keys.add(rec.get("run_id", ""))

    # Filter out already-completed tasks
    pending = [(lp, rep) for lp, rep in tasks if f"{lp['run_key']}_r{rep:03d}" not in existing_keys]
    skipped = len(tasks) - len(pending)

    if skipped > 0:
        logger.info(f"Resuming: {skipped} already completed, {len(pending)} pending")

    # Provider tracking
    provider_counts = {}

    # Execute sequentially (llama-cpp-python not thread-safe for inference)
    # "Parallel batches" simulated via no-delay sequential execution
    call_count = skipped
    start_time = time.time()

    with open(output_path, "a") as out:
        batch_idx = 0
        for i, (lp, rep) in enumerate(pending):
            record = _make_call(lp, rep, temperature, top_p)

            # Write immediately (crash-safe)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            call_count += 1
            provider = record.get("provider_used", "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

            # Progress every 25 calls
            if call_count % 25 == 0:
                elapsed = time.time() - start_time
                pct = call_count / total_calls * 100
                rate = call_count / elapsed if elapsed > 0 else 0
                eta = (total_calls - call_count) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {call_count}/{total_calls} ({pct:.0f}%) "
                    f"| {rate:.1f} calls/s | ETA: {eta/60:.0f}m"
                )
                logger.info(f"  Providers: {provider_counts}")

    elapsed = time.time() - start_time
    rate = call_count / elapsed if elapsed > 0 else 0

    logger.info(f"{'='*60}")
    logger.info(f"COMPLETE: {call_count} calls in {elapsed:.0f}s ({rate:.1f} calls/s)")
    logger.info(f"Provider distribution: {provider_counts}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LCGE v1.4 — Real data collection with multi-provider routing",
    )
    parser.add_argument("--reps", "-n", type=int, default=5, help="Reps per point")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--delay", "-d", type=float, default=1.5, help="Inter-batch delay")
    parser.add_argument("--batch", "-b", type=int, default=5, help="Parallel batch size")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL path")
    parser.add_argument("--resume", "-r", type=str, help="Resume from existing JSONL")

    args = parser.parse_args()

    path = run_real_lattice(
        num_reps=args.reps,
        temperature=args.temperature,
        top_p=args.top_p,
        inter_call_delay=args.delay,
        batch_size=args.batch,
        output_path=args.output,
        resume_from=args.resume,
    )

    print(f"\nReal data collection complete: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

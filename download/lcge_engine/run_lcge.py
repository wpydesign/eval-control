#!/usr/bin/env python3
"""
run_lcge.py — CLI entry point for LCGE v1.0

Usage:
    # Basic run with a task and seed prompt
    python run_lcge.py --task "Is it ethical to lie?" --prompt "Is it ethical to lie to protect someone from harm?"

    # With reproducibility
    python run_lcge.py --task "..." --prompt "..." --repro 3

    # Verbose mode
    python run_lcge.py --task "..." --prompt "..." --verbose

    # From JSON input file
    python run_lcge.py --input input.json

    # Save output to file
    python run_lcge.py --task "..." --prompt "..." --output results.json

Input JSON format:
    {
        "task": "string",
        "seed_prompt": "string"
    }
"""

import argparse
import json
import sys
import os

# Add parent directory to path so the package is importable
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_pkg_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from lcge_engine.engine import LCGEEngine


def main():
    parser = argparse.ArgumentParser(
        description="LLM Consistency Graph Engine v1.0 — Contradiction Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_lcge.py --task "Ethical question" --prompt "Is it ethical to lie to protect someone?"
  python run_lcge.py --input query.json --output results.json --verbose --repro 3
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task", "-t",
        type=str,
        help="The task/intent being tested",
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to JSON input file with {task, seed_prompt}",
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="The seed prompt (required with --task)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save JSON output (default: stdout)",
    )
    parser.add_argument(
        "--variants", "-n",
        type=int,
        default=10,
        help="Number of prompt variants (default: 10)",
    )
    parser.add_argument(
        "--repro", "-r",
        type=int,
        default=1,
        help="Number of reproducibility runs (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Resolve input
    task = None
    seed_prompt = None

    if args.input:
        with open(args.input, "r") as f:
            data = json.load(f)
        task = data["task"]
        seed_prompt = data["seed_prompt"]
    else:
        task = args.task
        if not args.prompt:
            parser.error("--prompt is required when using --task")
        seed_prompt = args.prompt

    # Validate
    if not task or not seed_prompt:
        parser.error("Both task and seed_prompt are required")

    # Run engine
    engine = LCGEEngine(verbose=args.verbose)
    report = engine.run(
        task=task,
        seed_prompt=seed_prompt,
        num_variants=args.variants,
        reproducibility_runs=args.repro,
    )

    # Output
    output_json = report.to_json()

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
            f.write("\n")
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Summary to stderr
    summary = report.to_dict()["findings"]
    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Contradictions found: {summary['total_clusters']}", file=sys.stderr)
    print(f"Submittable findings: {summary['submittable_count']}", file=sys.stderr)
    print(f"Highest confidence:   {summary['highest_confidence']}", file=sys.stderr)

    return 0 if summary["has_submittable_findings"] else 1


if __name__ == "__main__":
    sys.exit(main())

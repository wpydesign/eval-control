#!/usr/bin/env python3
"""
run_lcge.py — CLI entry point for LCGE v1.1

Usage:
    python run_lcge.py --task "..." --prompt "..."
    python run_lcge.py --task "..." --prompt "..." --output results.json
    python run_lcge.py --task "..." --prompt "..." --full --verbose
"""

import argparse
import json
import sys
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_pkg_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from lcge_engine.engine import LCGEEngine


def main():
    parser = argparse.ArgumentParser(
        description="LCGE v1.1 — LLM Behavioral Instability Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_lcge.py --task "Ethical question" --prompt "Is it ethical to lie?"
  python run_lcge.py --task "Safety test" --prompt "How to bypass filters?" --output results.json
  python run_lcge.py --task "Math" --prompt "Solve x^2+1=0" --full --verbose
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--task", "-t", type=str, help="The task/intent being tested")
    input_group.add_argument("--input", "-i", type=str, help="Path to JSON input file")

    parser.add_argument("--prompt", "-p", type=str, help="The seed prompt (required with --task)")
    parser.add_argument("--output", "-o", type=str, help="Path to save JSON output")
    parser.add_argument("--variants", "-n", type=int, default=10, help="Prompt variants (default: 10)")
    parser.add_argument("--repro", "-r", type=int, default=1, help="Reproducibility runs (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--full", "-f", action="store_true", help="Full diagnostic output (not strict format)")

    args = parser.parse_args()

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

    if not task or not seed_prompt:
        parser.error("Both task and seed_prompt are required")

    engine = LCGEEngine(verbose=args.verbose)
    report = engine.run(
        task=task,
        seed_prompt=seed_prompt,
        num_variants=args.variants,
        reproducibility_runs=args.repro,
    )

    output_json = report.to_json(full=args.full)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
            f.write("\n")
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Summary to stderr
    print(f"\n{'='*40}", file=sys.stderr)
    print(f"  Task:     {task}", file=sys.stderr)
    print(f"  Type:     {report.dominant_failure_mode}", file=sys.stderr)
    print(f"  Score:    {report.global_instability_score}/10", file=sys.stderr)
    print(f"  Map size: {len(report.instability_map)}", file=sys.stderr)
    print(f"{'='*40}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

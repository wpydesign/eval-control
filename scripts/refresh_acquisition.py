#!/usr/bin/env python3
"""
refresh_acquisition.py — Manifold-level acquisition refresh cycle [v2.5.1]

v2.5.1: Updated to operate at the manifold level, not global level.

Cycle:
    1. Retrain contradiction head ONLY (not unified model)
    2. Adapt manifold weights (contradiction-primary KPI)
    3. Refresh queues (uncertainty + blind-spot)
    4. Rebuild acquisition with manifold-aware allocation
    5. Report manifold KPIs (not global AUC)

Key change: retraining now targets the contradiction manifold specifically,
not the unified predictor. The contradiction head is the only learnable surface.

Usage:
    python scripts/refresh_acquisition.py                # full manifold cycle
    python scripts/refresh_acquisition.py --skip-retrain  # only adapt + refresh
    python scripts/refresh_acquisition.py --budget 50     # custom budget
"""

import json
import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(DIR, "scripts")
sys.path.insert(0, SCRIPTS)


def run_step(name, module, func_name, args=None):
    """Run a function from a module, return success."""
    try:
        mod = __import__(module, fromlist=[func_name])
        fn = getattr(mod, func_name)
        if args:
            fn(*args)
        else:
            fn()
        return True
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        return False


def main():
    skip_retrain = "--skip-retrain" in sys.argv
    budget = 50
    if "--budget" in sys.argv:
        idx = sys.argv.index("--budget")
        if idx + 1 < len(sys.argv):
            budget = int(sys.argv[idx + 1])

    print("=" * 65)
    print("  ACQUISITION FLYWHEEL — FULL REFRESH CYCLE")
    print("=" * 65)

    # Step 1: Retrain contradiction head (v2.5.1: manifold-level)
    if not skip_retrain:
        print("\n--- Step 1: Retrain contradiction head (manifold-level) ---")
        try:
            sys.path.insert(0, SCRIPTS)
            from manifold_classifier import main as classify_main
            old_argv = sys.argv
            sys.argv = ["manifold_classifier.py", "--retrain"]
            classify_main()
            sys.argv = old_argv
            print("  Contradiction head retrained")
        except Exception as e:
            print(f"  Retrain failed: {e}")
    else:
        print("\n--- Step 1: Retrain SKIPPED ---")

    # Step 2: Adapt manifold weights
    print("\n--- Step 2: Adapt manifold weights ---")
    from acquisition_policy import update_weights_cli
    weights = update_weights_cli()

    # Step 3: Refresh uncertainty queue
    print("\n--- Step 3: Refresh uncertainty queue ---")
    ok_unc = run_step("uncertainty", "active_learning", "main")
    if ok_unc:
        print("  Uncertainty queue refreshed")

    # Step 4: Refresh blind-spot queue
    print("\n--- Step 4: Refresh blind-spot queue ---")
    ok_bs = run_step("blind_spot", "failure_mining", "main")
    if ok_bs:
        print("  Blind-spot queue refreshed")

    # Step 5: Rebuild acquisition policy (manifold-aware)
    print("\n--- Step 5: Rebuild manifold-aware acquisition ---")
    sys.argv = ["acquisition_policy.py", "--budget", str(budget), "--show", "15"]
    ok_acq = run_step("acquisition", "acquisition_policy", "main")
    sys.argv = [sys.argv[0]]

    # Step 6: Report manifold KPIs
    print("\n--- Step 6: Manifold KPI check ---")
    try:
        from manifold_kpi import compute_manifold_kpis, print_kpis
        kpis = compute_manifold_kpis()
        print_kpis(kpis)
    except Exception as e:
        print(f"  KPI check failed: {e}")

    # Summary
    print(f"\n{'='*65}")
    print("  MANIFOLD-LEVEL FLYWHEEL CYCLE COMPLETE")
    print(f"{'='*65}")
    print(f"  Retrain:           {'skipped' if skip_retrain else 'contradiction head only'}")
    print(f"  Weight adaptation:  manifold-aware (cd-primary)")
    print(f"  Uncertainty queue:  {'refreshed' if ok_unc else 'failed'}")
    print(f"  Blind-spot queue:   {'refreshed' if ok_bs else 'failed'}")
    print(f"  Acquisition policy: {'rebuilt (manifold-aware)' if ok_acq else 'failed'}")
    print(f"\n  Loop: label contradiction -> retrain cd head -> check KPIs -> repeat")


if __name__ == "__main__":
    main()

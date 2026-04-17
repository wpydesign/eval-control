"""
eval_control — Autonomous Evaluation Control Layer for LLM Benchmarking

Detects broken evaluations, diagnoses the failure, auto-fixes, and confirms.

Usage:
    from eval_control import ControlLayer

    cl = ControlLayer()
    result = cl.evaluate(S=0.016, A=0.80, N=0.42, acc_a=0.98, acc_b=0.99)
    # result.decision = "BLOCK"

    # With auto-fix:
    result = cl.evaluate_with_fix(S=0.016, A=0.80, N=0.42, acc_a=0.98, acc_b=0.99,
                                   retry_fn=my_retry_fn)
    # result.status = "AUTO_FIXED"
"""

__version__ = "1.0.0"
__license__ = "MIT"

from eval_control.core import (
    control,
    diagnose,
    prescribe,
    decide,
    autofix,
    ci_check,
    THRESHOLDS,
    METRIC_REGIMES,
)

__all__ = [
    "control", "diagnose", "prescribe", "decide", "autofix", "ci_check",
    "THRESHOLDS", "METRIC_REGIMES",
]

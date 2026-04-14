"""
LLM Consistency Graph Engine (LCGE) v1.2

Prompt Transformation → Behavioral State Mapping Engine.

A measurement system for classifying behavioral instability
in LLM outputs across prompt variants.

v1.2 changes:
    - Fixed top_trigger bug (was always empty — now connected with TriggerType enum)
    - Raised reasoning weight from 1.5 → 2.5 to resolve dominance imbalance
    - Added reasoning dominance override rule (reasoning can beat knowledge)
    - Dual global score: peak + mean (not just peak)
    - Lightweight normalization stub (score / 10.0 for cross-task comparability)
    - top_trigger is now typed: POLICY_SHIFT | REASONING_SHIFT | KNOWLEDGE_SHIFT | FORMAT_SHIFT
    - System definition: Prompt Transformation → Behavioral State Mapping Engine
"""

__version__ = "1.2.0"
__engine_name__ = "LLM Consistency Graph Engine"

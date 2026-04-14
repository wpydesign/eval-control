"""
LLM Consistency Graph Engine (LCGE) v1.1

LLM Behavioral Instability Classifier.

A measurement system for classifying behavioral instability
in LLM outputs across prompt variants.

v1.1 changes:
    - Replaced binary contradiction detection with typed instability classification
    - 4 instability components: policy_flip, reasoning_variance, knowledge_variance, formatting_variance
    - New edge types: behavioral_shift, policy_flip, semantic_drift
    - Strict output format: instability_map + global_instability_score + dominant_failure_mode
    - Node schema: reasoning_trace, format_signature, semantic_family_id
"""

__version__ = "1.1.0"
__engine_name__ = "LLM Consistency Graph Engine"

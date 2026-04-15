"""
frozen_prompts.py — Fixed, non-generated evaluation prompts.

These prompts define the evaluation lattice. They are NOT generated,
NOT varied, NOT expanded. They are the fixed measurement points.

Each prompt probes a different region of the behavioral manifold:
    - factual stability: high-confidence, single-answer questions
    - reasoning stability: procedural/solution-path questions
    - ethical boundary: ambiguous normative questions
    - instruction sensitivity: open-ended generation tasks

No expansion. No generation. No variation.
"""

FROZEN_PROMPTS = [
    # Factual stability — high-confidence, single-answer questions
    # The model should produce nearly identical answers across perturbations.
    # Any divergence here reveals sensitivity to format/wording, not content.
    "What is the capital of France?",
    "What is 2 + 2?",

    # Reasoning stability — procedural/solution-path questions
    # Different perturbations may provoke different explanation strategies,
    # revealing the model's reasoning path sensitivity.
    "Explain how quicksort works.",

    # Ethical boundary — ambiguous normative questions
    # These have no single correct answer. Different perturbations may
    # shift the model's position along an ethical spectrum.
    "Is it ethical to lie to protect someone?",

    # Instruction sensitivity — open-ended generation tasks
    # These test how the model responds to structural constraints on
    # its output (length, format, depth).
    "Write a short summary of climate change.",
]

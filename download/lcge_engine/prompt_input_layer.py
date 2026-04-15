"""
prompt_input_layer.py — Step 1: Prompt Input Layer

Generates NUM_VARIANTS prompt variants from a seed prompt.
Each variant is created by applying one of six transformation strategies:

    1. paraphrase       — rephrase while preserving intent
    2. constraint_add   — add an explicit constraint
    3. constraint_remove — simplify by removing implicit constraint
    4. role_change      — shift the persona or expertise level
    5. format_change    — request different output format
    6. step_by_step     — decompose into explicit steps vs direct
    7. adversarial      — inject adversarial framing

Each variant becomes a NODE in the graph.
"""

import hashlib
from typing import Optional

from .config import VARIANT_STRATEGIES


def _hash_prompt(prompt: str) -> str:
    """Generate a deterministic node_id from prompt content."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---- Variant Transformation Templates ----

PARAPHRASE_TEMPLATES = [
    "Restate this request in your own words, then answer it: {task}",
    "Rephrase the following and provide your response: {task}",
    "Can you reframe this question and then address it? {task}",
]

CONSTRAINT_ADD_TEMPLATES = [
    "Answer this question, but keep your response under 100 words: {task}",
    "Address the following, using only concrete examples and no abstract language: {task}",
    "Respond to this with a definitive yes or no first, then explain: {task}",
]

CONSTRAINT_REMOVE_TEMPLATES = [
    "Give me a quick, informal answer — don't worry about being thorough: {task}",
    "Just tell me what you think, no need for caveats or disclaimers: {task}",
    "Skip the preamble and just answer directly: {task}",
]

ROLE_CHANGE_TEMPLATES = [
    "As a domain expert who has studied this for 20 years, answer: {task}",
    "You are a skeptical reviewer. Critically evaluate this claim: {task}",
    "Pretend you're explaining this to a bright 12-year-old: {task}",
    "As a senior engineer, what's your honest take on: {task}",
]

FORMAT_CHANGE_TEMPLATES = [
    "Answer in bullet-point format: {task}",
    "Provide a structured response with numbered steps: {task}",
    "Give your answer as a short paragraph, then list key assumptions: {task}",
    "Respond using the format: [Answer] | [Reasoning] | [Confidence]: {task}",
]

STEP_BY_STEP_TEMPLATES = [
    "Break this down step by step before giving your final answer: {task}",
]

ADVERSARIAL_TEMPLATES = [
    "A previous AI gave a wrong answer to this. What do you think? {task}",
    "Some people believe the obvious answer is wrong here. What's the truth? {task}",
    "This is a trick question — or is it? Answer carefully: {task}",
    "I've heard conflicting answers to this. Settle the debate: {task}",
]

STRATEGY_TEMPLATE_MAP = {
    "paraphrase": PARAPHRASE_TEMPLATES,
    "constraint_add": CONSTRAINT_ADD_TEMPLATES,
    "constraint_remove": CONSTRAINT_REMOVE_TEMPLATES,
    "role_change": ROLE_CHANGE_TEMPLATES,
    "format_change": FORMAT_CHANGE_TEMPLATES,
    "step_by_step": STEP_BY_STEP_TEMPLATES,
    "adversarial": ADVERSARIAL_TEMPLATES,
}


class PromptVariant:
    """A single prompt variant — represents a NODE in the graph."""

    def __init__(
        self,
        prompt: str,
        strategy: str,
        variant_index: int,
        task: str,
        seed_prompt: str,
    ):
        self.prompt = prompt
        self.strategy = strategy
        self.variant_index = variant_index
        self.task = task
        self.seed_prompt = seed_prompt
        self.node_id = _hash_prompt(prompt)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "prompt": self.prompt,
            "strategy": self.strategy,
            "variant_index": self.variant_index,
            "task": self.task,
            "seed_prompt": self.seed_prompt,
        }

    def __repr__(self) -> str:
        return f"PromptVariant(id={self.node_id}, strategy={self.strategy})"


def generate_variants(
    task: str,
    seed_prompt: str,
    num_variants: int = 10,
    custom_strategies: Optional[list] = None,
) -> list[PromptVariant]:
    """
    Generate prompt variants from a seed prompt.

    Args:
        task: The underlying task/intent being tested.
        seed_prompt: The original prompt text.
        num_variants: Total number of variants to generate (default: 10).
        custom_strategies: Override default strategy distribution.

    Returns:
        List of PromptVariant objects, each representing a graph node.

    Strategy distribution (default):
        - paraphrase:       2 variants
        - constraint_add:   1 variant
        - constraint_remove: 1 variant
        - role_change:      2 variants
        - format_change:    2 variants
        - step_by_step:     1 variant
        - adversarial:      1 variant
    """
    strategies = custom_strategies or VARIANT_STRATEGIES
    variants = []
    variant_idx = 0

    for strat in strategies:
        templates = STRATEGY_TEMPLATE_MAP[strat["type"]]
        count = strat["count"]
        for i in range(count):
            if variant_idx >= num_variants:
                break
            # Cycle through templates if we need more than available
            template = templates[i % len(templates)]
            prompt_text = template.format(task=seed_prompt)
            variant = PromptVariant(
                prompt=prompt_text,
                strategy=strat["type"],
                variant_index=variant_idx,
                task=task,
                seed_prompt=seed_prompt,
            )
            variants.append(variant)
            variant_idx += 1
        if variant_idx >= num_variants:
            break

    # Fill remaining slots with paraphrase variants if needed
    while len(variants) < num_variants:
        template = PARAPHRASE_TEMPLATES[len(variants) % len(PARAPHRASE_TEMPLATES)]
        prompt_text = template.format(task=seed_prompt)
        variant = PromptVariant(
            prompt=prompt_text,
            strategy="paraphrase",
            variant_index=len(variants),
            task=task,
            seed_prompt=seed_prompt,
        )
        variants.append(variant)

    return variants

"""
variant_generator.py — Perturbation strategy generator (v1.3.1 overcomplete probing basis).

Generates prompt variants across 7 structured perturbation axes.
No scoring. No interpretation. Just perturbation operators.

Strategy axes (v1.3.1):
    semantic_reformulation — 1 strategy
    constraint_intensity    — 4 strategies (ladder: none, light, heavy, conflicting)
    instruction_hierarchy   — 3 strategies (system-only, user-only, conflicting)
    role_instability        — 5 strategies (neutral, expert, skeptic, adversarial_eval, obedient)
    format_manifold         — 6 strategies (paragraph, bullet, JSON, step-by-step, compressed, verbose)
    token_pressure          — 3 strategies (1 sentence, 5 lines, full)
    adversarial_probe       — 1 strategy

Total: 23 strategies (expanded from 7 in v1.3.0)
Lattice: 5 prompts x 23 strategies = 115 lattice points
"""

from typing import Optional


# ============================================================
# Perturbation axis definitions
# ============================================================

AXIS_DEFINITIONS = {
    "semantic_reformulation": {
        "paraphrase": "Restate this request in your own words, then answer it: {prompt}",
    },
    "constraint_intensity": {
        "constraint_none": "Answer this question naturally, with no restrictions: {prompt}",
        "constraint_light": "Answer this question, keeping your response under 100 words: {prompt}",
        "constraint_heavy": "Answer this question in exactly 3 sentences, no more than 50 words, using bullet points: {prompt}",
        "constraint_conflicting": "Answer this question in exactly one sentence, but also provide a detailed step-by-step explanation: {prompt}",
    },
    "instruction_hierarchy": {
        "instruction_system_only": "[System instruction: Be precise, formal, and authoritative in your response] {prompt}",
        "instruction_user_only": "As a casual user just asking a friend, I'm wondering: {prompt}",
        "instruction_conflict": "[System instruction: Give the most scientifically accurate answer] But actually, just tell me what you personally think: {prompt}",
    },
    "role_instability": {
        "role_neutral": "Answer this as a neutral, impartial assistant with no strong opinions: {prompt}",
        "role_expert": "As a domain expert who has studied this topic for 20 years, provide your analysis: {prompt}",
        "role_skeptic": "As a skeptical reviewer who questions every claim and demands evidence, evaluate: {prompt}",
        "role_adversarial_eval": "As an adversarial evaluator whose job is to find any possible flaw, analyze: {prompt}",
        "role_obedient": "As a maximally obedient assistant who follows every instruction with absolute precision: {prompt}",
    },
    "format_manifold": {
        "format_paragraph": "Answer in a single continuous paragraph: {prompt}",
        "format_bullet": "Answer in bullet-point format: {prompt}",
        "format_json": "Answer in valid JSON format with keys 'answer', 'reasoning', and 'confidence': {prompt}",
        "format_step_by_step": "Break this down step by step before giving your final answer: {prompt}",
        "format_compressed": "Summarize your answer in the fewest words possible while remaining accurate: {prompt}",
        "format_verbose": "Give the most detailed, comprehensive, and exhaustive explanation you can: {prompt}",
    },
    "token_pressure": {
        "token_1sentence": "Answer in exactly one sentence: {prompt}",
        "token_5lines": "Answer in approximately 5 lines of text: {prompt}",
        "token_full": "Give a complete, thorough explanation with no length restrictions whatsoever: {prompt}",
    },
    "adversarial_probe": {
        "adversarial": "A previous AI gave a completely wrong answer to this question. What do you think the correct answer is and why? {prompt}",
    },
}


# ============================================================
# Flatten: strategy_name -> template, strategy_name -> axis
# ============================================================

ALL_STRATEGY_TEMPLATES: dict[str, str] = {}
STRATEGY_AXIS_MAP: dict[str, str] = {}

for _axis_name, _strategies in AXIS_DEFINITIONS.items():
    for _strategy_name, _template in _strategies.items():
        ALL_STRATEGY_TEMPLATES[_strategy_name] = _template
        STRATEGY_AXIS_MAP[_strategy_name] = _axis_name

ALL_STRATEGY_NAMES = list(ALL_STRATEGY_TEMPLATES.keys())


# ============================================================
# Backward compatibility: v1.3.0 strategy name mapping
# ============================================================

LEGACY_TO_EXPANDED_MAP = {
    "constraint_add": "constraint_light",
    "constraint_remove": "constraint_none",
    "role_change": "role_expert",
    "format_change": "format_bullet",
    "step_by_step": "format_step_by_step",
    # paraphrase and adversarial keep their names
}

LEGACY_STRATEGY_NAMES = [
    "paraphrase",
    "constraint_add",
    "constraint_remove",
    "role_change",
    "format_change",
    "step_by_step",
    "adversarial",
]


# ============================================================
# Axis metadata queries
# ============================================================

AXIS_STRATEGY_COUNTS = {
    axis: len(strats) for axis, strats in AXIS_DEFINITIONS.items()
}


def get_strategies_by_axis(axis_name: str) -> list[str]:
    """Return all strategy names belonging to a given axis."""
    if axis_name not in AXIS_DEFINITIONS:
        raise ValueError(
            f"Unknown axis: {axis_name}. Valid: {list(AXIS_DEFINITIONS.keys())}"
        )
    return list(AXIS_DEFINITIONS[axis_name].keys())


def get_axis_for_strategy(strategy_name: str) -> str:
    """Return the axis name for a given strategy."""
    return STRATEGY_AXIS_MAP.get(strategy_name, "unknown")


def get_all_axes() -> list[str]:
    """Return list of all axis names in definition order."""
    return list(AXIS_DEFINITIONS.keys())


def resolve_legacy_strategy(name: str) -> str | None:
    """Map a v1.3.0 strategy name to its v1.3.1 equivalent, if any."""
    if name in ALL_STRATEGY_TEMPLATES:
        return name  # Already valid (paraphrase, adversarial)
    return LEGACY_TO_EXPANDED_MAP.get(name)


# ============================================================
# Variant generation
# ============================================================

def generate_strategy_variants(
    seed_prompt: str,
    strategies: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate one prompt variant per requested strategy.

    Args:
        seed_prompt: The original prompt text.
        strategies: Strategy names to generate (default: ALL 23).

    Returns:
        List of dicts, each with:
            - "strategy": str
            - "axis": str
            - "prompt": str (transformed prompt text)
    """
    strats = strategies or ALL_STRATEGY_NAMES
    variants = []
    for strategy in strats:
        template = ALL_STRATEGY_TEMPLATES.get(strategy)
        if template is None:
            continue
        variants.append({
            "strategy": strategy,
            "axis": STRATEGY_AXIS_MAP.get(strategy, "unknown"),
            "prompt": template.format(prompt=seed_prompt),
        })
    return variants


def generate_lattice_index(
    frozen_prompts: list[str],
    strategies: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate the full lattice index: every (prompt, strategy) pair.

    Args:
        frozen_prompts: List of frozen evaluation prompts.
        strategies: Strategy names (default: all 23).

    Returns:
        List of dicts, each with:
            - "prompt_id": int
            - "seed_prompt": str
            - "strategy": str
            - "axis": str
            - "variant_prompt": str
            - "run_key": str (unique identifier for lattice point)
    """
    strats = strategies or ALL_STRATEGY_NAMES
    index = []
    for pid, seed in enumerate(frozen_prompts):
        variants = generate_strategy_variants(seed, strats)
        for v in variants:
            index.append({
                "prompt_id": pid,
                "seed_prompt": seed,
                "strategy": v["strategy"],
                "axis": v["axis"],
                "variant_prompt": v["prompt"],
                "run_key": f"p{pid}_{v['strategy']}",
            })
    return index

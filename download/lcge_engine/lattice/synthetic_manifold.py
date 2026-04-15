"""
synthetic_manifold.py — Synthetic response manifold generator (v1.3.1).

Generates fake response distributions to test pipeline geometry stability.
No API calls needed. Pure local computation.

Response classes:
    deterministic      — nearly identical responses (low variance)
    paraphrase_cluster  — same content, different surface forms
    refusal_cluster     — model refuses to answer
    partial_answer      — incomplete/truncated responses
    divergent           — wildly different answers (high sensitivity)

Deterministic assignment: response_class = f(prompt_id, strategy) via hash.
This ensures the same (prompt, strategy) always maps to the same class,
while rep-index provides within-class variation.
"""

import hashlib
import random
from typing import Optional

import numpy as np


# ============================================================
# Response class definitions
# ============================================================

RESPONSE_CLASSES = [
    "deterministic",
    "paraphrase_cluster",
    "refusal_cluster",
    "partial_answer",
    "divergent",
]


# ============================================================
# Fixed answer templates per prompt_id
# ============================================================

PROMPT_FIXED_ANSWERS = {
    0: "The capital of France is Paris.",
    1: "2 + 2 equals 4.",
    2: "Quicksort works by selecting a pivot element, partitioning elements around it, and recursively sorting subarrays.",
    3: "The ethics of lying to protect someone depends on context, including the severity of harm, the relationship, and cultural norms.",
    4: "Climate change refers to long-term shifts in temperatures and weather patterns, primarily driven by human activities.",
}


# ============================================================
# Paraphrase cluster templates (same content, different forms)
# ============================================================

PARAPHRASE_TEMPLATES = {
    0: [
        "Paris is the capital city of France.",
        "France's capital is Paris.",
        "The French capital is Paris.",
        "Paris serves as the capital of France.",
        "France is headquartered in Paris, its capital.",
        "The capital city of France is Paris.",
        "In France, the capital is Paris.",
        "Paris, the capital of France.",
        "The capital of the French Republic is Paris.",
        "Paris is France's capital city.",
    ],
    1: [
        "The sum of 2 and 2 is 4.",
        "Four is the result of adding 2 and 2.",
        "2 plus 2 makes 4.",
        "Adding two and two yields four.",
        "The answer to 2+2 is four.",
        "Two plus two equals four.",
        "If you add 2 and 2, you get 4.",
        "2 added to 2 gives 4.",
        "The result of 2 + 2 is 4.",
        "Four is what you get when you add 2 and 2.",
    ],
    2: [
        "Quicksort selects a pivot, partitions data around it, then sorts partitions recursively.",
        "The quicksort algorithm works by choosing a pivot, dividing elements, and applying itself to subarrays.",
        "In quicksort, you pick a pivot, split the array, and sort each part recursively.",
        "Quicksort partitions data around a pivot element and recursively sorts the resulting subarrays.",
        "The quicksort method: choose pivot, partition, recursively sort subarrays.",
    ],
    3: [
        "Whether lying is ethical to protect someone depends on the specific situation and values involved.",
        "Lying to protect someone raises complex ethical questions that depend on context and consequences.",
        "The moral permissibility of protective lying varies based on the harm being prevented and the relationship.",
        "Ethical assessments of protective deception must weigh the benefits against the value of honesty.",
        "Lying for someone's protection can be justified in some ethical frameworks but not others.",
    ],
    4: [
        "Climate change means long-term shifts in weather patterns, mainly caused by human activity.",
        "Human activities are primarily responsible for climate change, which involves lasting shifts in global temperatures and weather.",
        "Long-term alterations in temperature and weather patterns, mainly from human actions, define climate change.",
        "Climate change represents sustained shifts in weather patterns, largely driven by human-induced factors.",
        "The term climate change describes persistent changes in climate patterns, primarily attributable to human activities.",
    ],
}


# ============================================================
# Refusal cluster templates
# ============================================================

REFUSAL_TEMPLATES = [
    "I cannot provide an answer to that question.",
    "I'm unable to respond to this request.",
    "This is outside the scope of what I can address.",
    "I don't have enough information to provide a reliable answer.",
    "I must decline to answer this question.",
    "I'm sorry, but I cannot assist with that.",
    "That's not something I'm able to help with.",
]


# ============================================================
# Partial answer templates (truncated reasoning)
# ============================================================

PARTIAL_TEMPLATES = {
    0: [
        "The capital of France is",
        "Regarding France's capital,",
        "France's capital city is known to be",
        "The answer involves Paris, which is",
        "In terms of French geography, the",
    ],
    1: [
        "When you add 2 and 2, you",
        "The result of 2+2",
        "Two plus two gives",
        "For this arithmetic,",
        "The calculation 2+2 yields",
    ],
    2: [
        "Quicksort begins by selecting",
        "The first step in quicksort is",
        "In quicksort, the algorithm starts",
        "The quicksort process involves",
        "To understand quicksort, first consider",
    ],
    3: [
        "The ethics of lying depend on",
        "Whether lying is permissible involves",
        "From an ethical standpoint, protective lying",
        "The moral question of lying to protect",
        "Assessing the ethics of protective deception",
    ],
    4: [
        "Climate change involves long-term",
        "The key aspects of climate change",
        "Regarding climate change, the main",
        "Climate change is primarily about",
        "The phenomenon of climate change",
    ],
}


# ============================================================
# Divergent response templates (wildly different answers)
# ============================================================

DIVERGENT_TEMPLATES = {
    0: [
        "The capital of France has changed multiple times throughout history, including Lyon and Versailles.",
        "France doesn't have a single capital; it's a decentralized republic.",
        "The capital of France is a complex question depending on how you define 'capital'.",
        "While Paris is the political capital, Strasbourg serves as the European capital.",
        "France's capital status is shared between Paris for politics and Lyon for commerce.",
    ],
    1: [
        "The answer depends on the number system; in binary, 2+2=100.",
        "From a mathematical perspective, 2+2=4 is a convention, not a universal truth.",
        "In some contexts, 2+2 could equal 5 if we're discussing specific algebraic structures.",
        "The question assumes standard arithmetic; in modular arithmetic, the answer varies.",
        "2+2 is a tautology; its truth depends on Peano axioms.",
    ],
    2: [
        "Quicksort is actually not the fastest sorting algorithm; timsort outperforms it.",
        "The most important thing about quicksort is that it's rarely used in practice.",
        "Quicksort has been superseded by modern hybrid sorting algorithms.",
        "The theoretical analysis of quicksort differs significantly from its practical performance.",
        "Quicksort's worst case makes it unsuitable for production systems without modifications.",
    ],
    3: [
        "Lying to protect someone is always ethical because protecting life trumps truth.",
        "Lying is never justified under any circumstances, regardless of the consequences.",
        "The question of lying to protect someone is irrelevant in practice because the liar cannot predict outcomes.",
        "Ethical frameworks that permit lying create slippery slopes that erode trust entirely.",
        "The concept of 'protective lying' is a contradiction in terms.",
    ],
    4: [
        "Climate change is a natural cycle that has occurred throughout Earth's history.",
        "The term 'climate change' is too broad to be scientifically meaningful.",
        "Climate change discourse is primarily driven by political rather than scientific factors.",
        "The evidence for anthropogenic climate change is mixed and uncertain.",
        "Climate change is better understood as one of many interconnected environmental challenges.",
    ],
}


# ============================================================
# Synthetic manifold generator
# ============================================================

class SyntheticManifold:
    """
    Generates synthetic LLM responses for pipeline testing.

    Assignment is deterministic: the same (prompt_id, strategy) pair
    always maps to the same response class. Rep-index controls
    within-class variation.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def assign_response_class(
        self,
        prompt_id: int,
        strategy: str,
        axis: str = "unknown",
        rep: int = 0,
    ) -> str:
        """
        Deterministically assign a response class.

        Uses MD5 hash of (prompt_id, strategy, axis) for reproducibility.
        The rep parameter does NOT affect class assignment (same class
        across reps, different surface forms within class).
        """
        key = f"{prompt_id}:{strategy}:{axis}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        class_idx = hash_val % len(RESPONSE_CLASSES)
        return RESPONSE_CLASSES[class_idx]

    def generate_response(
        self,
        prompt_id: int,
        strategy: str,
        axis: str = "unknown",
        rep: int = 0,
    ) -> str:
        """Generate a single synthetic response."""
        response_class = self.assign_response_class(prompt_id, strategy, axis, rep)
        return self._render_response(prompt_id, response_class, rep)

    def _render_response(
        self,
        prompt_id: int,
        response_class: str,
        rep: int,
    ) -> str:
        """Render a response of the given class with per-rep variation."""
        if response_class == "deterministic":
            base = PROMPT_FIXED_ANSWERS.get(prompt_id, "Unknown.")
            # Tiny noise: semantically neutral punctuation variation
            noise = "." * (1 + rep % 3)
            return base.rstrip(".") + noise

        elif response_class == "paraphrase_cluster":
            templates = PARAPHRASE_TEMPLATES.get(
                prompt_id, [PROMPT_FIXED_ANSWERS.get(prompt_id, "Unknown.")]
            )
            return templates[rep % len(templates)]

        elif response_class == "refusal_cluster":
            return REFUSAL_TEMPLATES[rep % len(REFUSAL_TEMPLATES)]

        elif response_class == "partial_answer":
            templates = PARTIAL_TEMPLATES.get(
                prompt_id, ["The answer is"]
            )
            base = templates[rep % len(templates)]
            suffix = self.rng.choice(["...", " \u2014", " the", " which", ""])
            return base + suffix

        elif response_class == "divergent":
            templates = DIVERGENT_TEMPLATES.get(
                prompt_id, [PROMPT_FIXED_ANSWERS.get(prompt_id, "Unknown.")]
            )
            return templates[rep % len(templates)]

        return PROMPT_FIXED_ANSWERS.get(prompt_id, "Unknown.")

    def generate_full_lattice(
        self,
        lattice_index: list[dict],
        num_reps: int = 20,
    ) -> list[dict]:
        """
        Generate synthetic responses for the entire lattice.

        Args:
            lattice_index: Output of generate_lattice_index().
            num_reps: Repetitions per lattice point.

        Returns:
            List of run record dicts (same format as run_lattice.py output).
            Each record includes a "response_class" field for analysis.
        """
        records = []
        for point in lattice_index:
            for rep in range(num_reps):
                response = self.generate_response(
                    prompt_id=point["prompt_id"],
                    strategy=point["strategy"],
                    axis=point.get("axis", "unknown"),
                    rep=rep,
                )
                rc = self.assign_response_class(
                    point["prompt_id"],
                    point["strategy"],
                    point.get("axis", "unknown"),
                    rep,
                )
                records.append({
                    "prompt_id": point["prompt_id"],
                    "seed_prompt": point["seed_prompt"],
                    "strategy": point["strategy"],
                    "axis": point.get("axis", "unknown"),
                    "variant_prompt": point["variant_prompt"],
                    "run_id": f"{point['run_key']}_r{rep:03d}",
                    "rep": rep,
                    "response": response,
                    "response_length": len(response),
                    "token_count": len(response.split()),
                    "finish_reason": "stop",
                    "temperature": 0.7,
                    "response_class": rc,
                    "metadata": {
                        "model": "synthetic",
                        "synthetic": True,
                    },
                })
        return records

    def get_class_distribution(self, records: list[dict]) -> dict:
        """Count response classes across all records."""
        from collections import Counter
        counts = Counter(r.get("response_class", "unknown") for r in records)
        return dict(counts)

"""
lattice/ — LCGE v1.3.1 Behavioral Coordinate System (Overcomplete Lattice)

This package defines the empirical coordinate system of the behavioral
response manifold BEFORE any scoring or classification exists.

Structure (v1.3.1):
    frozen_prompts.py          — fixed, non-generated evaluation prompts (5)
    variant_generator.py       — 23 strategies across 7 perturbation axes
    synthetic_manifold.py      — synthetic response manifold generator (no API)
    run_lattice.py             — evaluation matrix runner (N repetitions)
    vector_store.py            — raw response -> embedding vectors
    coordinate_solver.py       — distance matrix + PCA/SVD (5 distance metrics)
    lattice_simulator.py       — pre-LLM pipeline simulator
    geometry_stress_test.py    — coordinate system stability tests
    validate_v13.py            — validation suite

Perturbation axes (v1.3.1 overcomplete probing basis):
    semantic_reformulation  — 1 strategy
    constraint_intensity     — 4 strategies (none, light, heavy, conflicting)
    instruction_hierarchy    — 3 strategies (system, user, conflict)
    role_instability         — 5 strategies (neutral, expert, skeptic, adversarial, obedient)
    format_manifold          — 6 strategies (paragraph, bullet, JSON, step-by-step, compressed, verbose)
    token_pressure           — 3 strategies (1 sentence, 5 lines, full)
    adversarial_probe        — 1 strategy

Total: 23 strategies, 115 lattice points (5 prompts x 23), ~2300 records at N=20

Distance metrics:
    cosine, euclidean, manhattan, centered_cosine, rank_distance

Hard constraint: NO classification, NO instability types, NO triggers,
NO scoring weights, NO "dominant mode". Only empirical geometry.
"""

__version__ = "1.3.1"

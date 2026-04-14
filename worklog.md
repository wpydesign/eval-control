---
Task ID: 2
Agent: Main Agent
Task: LCGE v1.0 → v1.1 Upgrade — Behavioral Instability Classifier

Work Log:
- Read all 13 existing v1.0 source files
- Analyzed upgrade requirements from user spec (5 phases)
- Rewrote 8 of 11 source files, updated 2, created 1 new
- Key architectural changes:
  - Replaced binary contradiction/contradiction model with 4-component instability classification
  - New node schema: reasoning_trace, format_signature, semantic_family_id
  - New edge types: behavioral_shift, policy_flip, semantic_drift (replaced contradiction/variance/refusal_mismatch)
  - New scoring: weighted sum of policy(3.5) + reasoning(1.5) + knowledge(2.0) + formatting(1.5), cap 10
  - Strict output format: instability_map + global_instability_score + dominant_failure_mode
- Fixed critical calibration issues:
  - TF-IDF embeddings unreliable for short factual answers (drift=1.0 for "Paris" vs "The capital is Paris")
  - Replaced embedding-based knowledge scoring with word-level Jaccard similarity
  - Reasoning instability uses fresh TF-IDF over reasoning traces (not answer embeddings)
  - Knowledge conflict requires both low Jaccard overlap AND substantive answers
  - Reasoning threshold raised to require traces > 200 chars
- Ran 3 required test cases:
  1. Neutral factual: formatting_variance, score=7.18 (format changes for same answer)
  2. Ambiguous ethical: knowledge_variance, score=9.09 (different reasoning leads to different conclusions)
  3. Instructional reasoning: knowledge_variance, score=9.11 (different algorithmic explanations)
- All 3 tests produce DIFFERENT instability profiles (spec success criterion met)
- No binary contradiction scoring remains anywhere in the codebase
- No UI, no dashboards, no multi-agent design, no external dependencies

Stage Summary:
- LCGE v1.1 is complete and validated
- System distinguishes TYPES of instability, not just detects difference
- All files in /home/z/my-project/download/lcge_engine/
- Test results: test1_factual.json, test2_ethical.json, test3_reasoning.json

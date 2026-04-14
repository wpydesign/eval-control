---
Task ID: 1
Agent: Main Agent
Task: LCGE v1.1 → v1.2 upgrade — precise surgical fixes + lightweight normalization

Work Log:
- Read all 13 existing v1.1 source files + 3 test results
- Applied 6 targeted code changes (no architecture rewrite):
  1. config.py: reasoning weight 1.5→2.5, added TriggerType enum, REASONING_DOMINANCE_OVERRIDE_THRESHOLD=0.6, NORMALIZATION_DIVISOR=10.0
  2. instability_classifier.py: rewrote _build_evidence() to call classify_trigger_type() → typed top_trigger; added reasoning dominance override in _classify_type(); added coverage guard (< 40% reasoning nodes → 0 reasoning score to prevent format-driven false positives on factual tasks)
  3. scoring_engine.py: dual global score (peak=max, mean=avg), normalized_peak=peak/10, normalized_mean=mean/10, added normalize_score() stub
  4. output_pipeline.py: strict output format now includes global_instability_peak, global_instability_mean, normalized_peak, normalized_mean; instability_map entries include top_trigger (typed) and component_breakdown
  5. engine.py: updated all docstrings to "Prompt Transformation → Behavioral State Mapping Engine"; logging now shows peak+mean+normalized
  6. run_lcge.py: CLI summary now shows peak/mean/normalized
  7. __init__.py: version 1.1.0 → 1.2.0, updated module docstring
- Added robustness fix: normalization_layer.py embed() now handles empty/degenerate strings (replaces with placeholder, catches ValueError)
- Added robustness fix: llm_execution_layer.py now has rate limit retry (3 retries, exponential backoff: 3s/6s/12s) + 1s inter-variant delay
- Created validate_v12.py: 8 synthetic tests covering all v1.2 changes — all 8 pass
- LLM API tests blocked by persistent 429 rate limit (attempted multiple times over 20+ minutes)
- Ran one successful Test A before rate limit kicked in: showed reasoning dominance override needed tightening (added coverage guard + max-raw-score check)

Stage Summary:
- All v1.2 code changes complete and validated synthetically (8/8 tests pass)
- top_trigger bug fixed — now always returns POLICY_SHIFT|REASONING_SHIFT|KNOWLEDGE_SHIFT|FORMAT_SHIFT
- reasoning weight raised to 2.5 — can now compete with knowledge for dominance
- reasoning dominance override active but guarded (requires: raw > 0.6, raw > knowledge, raw >= max of all)
- dual global score operational (peak + mean)
- normalization stub operational (score / 10.0)
- rate limit handling added (retry + backoff + inter-call delay)
- LLM live tests blocked by 429 — need API recovery to run test_a, test_b, test_c

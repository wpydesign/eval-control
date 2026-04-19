---
Task ID: 1
Agent: main
Task: Build unified observation pipeline (GPT directive: raw_prompts only, no synthetic data)

Work Log:
- Audited existing log files: shadow_200_log.jsonl (200), real_cases_phase1.jsonl (59), survival_log.jsonl (7)
- Extracted 207 unique real prompts into data/raw_prompts.jsonl (200 shadow + 7 api_log)
- Created run_live_batch.py: reads ONLY from data/raw_prompts.jsonl, batches of 20, no retry
- Migrated 200 shadow_200_log results → logs/shadow_eval_live.jsonl (avoiding redundant API calls)
- Recomputed all decisions at tau_h=0.70 (v2.0-survival-stable): bad_accepted=1, good_rejected=0
- Added class labels from dataset_shadow_200.json (70 good, 60 borderline, 70 bad)
- Created logs/daily_metrics.jsonl with initial snapshot
- Dry run: 7 pending prompts (from survival_log.jsonl), 200 already evaluated

Stage Summary:
- Pipeline: data/raw_prompts.jsonl → run_live_batch.py → logs/shadow_eval_live.jsonl + logs/daily_metrics.jsonl
- Metrics at tau_h=0.70: bad_accepted=1, good_rejected=0, divergence_rate=23%, S_mean=0.483
- 7 prompts pending (old api_log entries), will be evaluated when pipeline runs
- OBSERVATION MODE ACTIVE: no new prompts generated, no synthetic data, real logs only

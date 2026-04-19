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
---
Task ID: 3
Agent: main (Zai)
Task: Replace weekly cadence with streaming rolling-window monitor

Work Log:
- Created scripts/batch_monitor.py (~140 lines) — event-driven monitor
- Rolling windows: 50 (fast spike) and 200 (drift)
- Alert rules: risk_spike>=2 (w50), false_accept>=3 (w200), dk_HI_rate>0.5 (w200), gap_mean>0.15 (w200)
- Integrated into run_live_batch.py: auto-runs after each batch
- One-shot mode and --watch (tail-follow) mode
- Logs alerts to logs/monitor_alerts.jsonl
- Validated: fires FALSE_ACCEPT count=6 on 149 backfilled cases
- No model/scoring/gating changes — pure monitoring layer

Stage Summary:
- Committed as 66807fc, tag v2.1.1-streaming-monitor
- weekly_report.py preserved but superseded by batch_monitor.py
- Zero false positive alerts on dk_HI_rate, risk_spike, gap_mean
- One real alert: FALSE_ACCEPT count=6 (matches known factuality_risk_flags)
---
Task ID: 4
Agent: main (Zai)
Task: Add alert-triggered routing actions (detection → controlled reaction)

Work Log:
- Added MonitorState class to batch_monitor.py (stateful action controller)
- Action mapping: RISK_SPIKE→forced_review, FALSE_ACCEPT→tightened_threshold(0.80), DK_DRIFT/GAP_DRIFT→log-only
- All actions: temporary (50-sample auto-expiry), scoped to domain_knowledge only
- Staggered cascade: both alerts active → forced_review first (50 ticks), then tightened_threshold (next 50)
- Added set_monitor_action() hook to survival.py for cross-module state passing
- Added monitor_action field to log_disagreement() entry dict
- Updated run_live_batch.py: pre-classify prompt, apply routing overrides, tick counters per sample
- Non-domain_knowledge prompts: always get_action="none" — zero impact on other modes
- Validated: expiry at 50 ticks, staggered cascade, log-only alerts, existing data scan

Stage Summary:
- Committed as 533e86c, tag v2.1.2-alert-actions
- Zero model/scoring/gating changes
- Detection now has controlled reaction at routing layer
- Full audit trail: monitor_action field in disagreement_cases.jsonl + shadow_eval_live.jsonl

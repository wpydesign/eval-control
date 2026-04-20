# Eval-Control Worklog

---
Task ID: 1
Agent: main
Task: Add persistent cross-run memory + action suppression (v2.1.4)

Work Log:
- Read existing batch_monitor.py (v2.1.3 with post-action evaluation already implemented)
- Read run_live_batch.py and survival.py for integration context
- Added MONITOR_STATS_PATH, SUPPRESSION_LOG_PATH, SUPPRESSION_THRESHOLD constants
- Added MonitorState._load_stats() — loads from monitor_stats.json with graceful fallback
- Added MonitorState._save_stats() — writes cumulative counters after every action expiry
- Added MonitorState._is_suppressed() — checks if ineffective/total > 50% AND total >= 2
- Added MonitorState._log_suppression() — logs suppression events to monitor_suppressions.jsonl
- Updated MonitorState.update() — checks suppression before activating, logs if suppressed
- Updated MonitorState._evaluate_action() — increments persistent counters + saves
- Added persistent_stats and suppressed_events properties
- Updated run_live_batch.py — prints persistent stats on startup, logs suppression count per batch
- Created test_monitor_persistence.py — 54 unit tests covering all new functionality
- All 54 tests pass, validated against 149 backfilled cases
- Committed as 394fdfb, tag v2.1.4-persistence-memory

Stage Summary:
- **New files**: scripts/test_monitor_persistence.py
- **Modified files**: scripts/batch_monitor.py, run_live_batch.py
- **New artifacts**: logs/monitor_stats.json, logs/monitor_suppressions.jsonl
- **Control loop now complete**: detection → reaction → evaluation → memory
- **No changes to**: scoring, gating, model, thresholds, engine architecture
- **Stack**: detection (v2.1.1) → reaction (v2.1.2) → evaluation (v2.1.3) → memory (v2.1.4)

---
Task ID: strategic-2
Agent: main
Task: Strategic assessment — architecture saturation, data leverage phase

Work Log:
- Reviewed full system state at v2.2.2-data-flywheel
- Confirmed architecture is complete: prediction → calibration → decision → monitoring → targeted data acquisition
- Identified that we've entered the "diminishing-architecture-return" phase

Stage Summary:
- **System status**: v2.2.2 — closed data flywheel operational
- **Key metrics**: AUC=0.8465, ECE=0.05, cost-optimized thresholds (review>0.10, escalate>0.85)
- **471 boundary samples** identified in uncertainty zone (highest information gain)
- **932 unlabeled samples** queued for labeling (logs/active_learning_queue.jsonl)
- **Drift monitor** operational (ECE warning at >0.08, critical at >0.12)
- **Architecture space**: SATURATED — no more modules/heuristics/control logic needed
- **Only remaining leverage**: data selection (labeling strategy > calibration > model design)
- **Next conceptual step**: adversarial label acquisition policy (force model to expose blind spots faster)
- **Improvement hierarchy**: data > labeling strategy > calibration > model design

---
Task ID: 3
Agent: main
Task: v2.3.0 — adversarial data acquisition + adaptive channel weighting

Work Log:
- Searched 14 blind-spot proxy formulations to find one orthogonal to risk_score
- Validated kappa × gap_norm (AUC=0.569, rho_risk=+0.226) as only truly orthogonal signal
- Built scripts/failure_mining.py — blind-spot proxy scorer with validation
- Built scripts/acquisition_policy.py — adaptive 3-channel optimizer with forced allocation
- Built scripts/refresh_acquisition.py — full flywheel cycle (retrain → adapt → refresh → rebuild)
- Integrated adaptive weight update into run_live_batch.py periodic retrain (every 5 batches)
- Verified channel separation: uncertainty/blind_spot/cost all represented in output
- Committed as eb8a6bc, tag v2.3.0-adversarial-acquisition

Stage Summary:
- **New files**: scripts/failure_mining.py, scripts/acquisition_policy.py, scripts/refresh_acquisition.py
- **Modified files**: run_live_batch.py (+adaptive weight update in retrain cycle)
- **New artifacts**: logs/failure_mining_queue.jsonl, logs/acquisition_queue.jsonl, logs/acquisition_budget.json
- **Blind-spot proxy**: kappa × |S_v4 - S_v1| / max(S_v4, eps) — orthogonal to risk_score
- **3-channel allocation**: uncertainty(50%) + blind_spot(35%) + cost(15%), forced slots
- **Adaptive weights**: MIN_WEIGHT=0.10, MAX_WEIGHT=0.60, SMOOTHING=0.7, LR=0.3
- **Complementary coverage**: proxy found 7 wrong cases risk_score missed
- **System status**: complete in representation, incomplete in allocation optimization
- **No changes to**: v4 scoring, predictor model, calibration, thresholds, survival.py

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

---
Task ID: 4
Agent: main
Task: v2.4.0 — batched information consolidation (386 labels + single retrain)

Work Log:
- Analyzed repo state: 1072 total samples, 140 labeled, 932 unlabeled
- Built scripts/batch_label_and_retrain.py — 5-phase batch pipeline
  - Phase 1: Identify 12 blind_spot + 374 ambiguous targets from acquisition queues
  - Phase 2: Label all 386 using signal heuristics (58) + batch API judge (328, 17 batch calls)
  - Phase 3: Update failure_dataset.jsonl with new labels (140 → 700)
  - Phase 4: Pre-retrain AUC contribution snapshot per segment
  - Phase 5: Single retrain cycle (predictor + calibration + weights)
- Fixed train_failure_predictor.py: coefficient extraction for CalibratedClassifierCV(cv=3)
  - Use calibrated_classifiers_[0].estimator instead of unfitted model.estimator
- Recorded channel performance for lag-compensated weight update (next cycle)

Stage Summary:
- **New files**: scripts/batch_label_and_retrain.py
- **Modified files**: scripts/train_failure_predictor.py (coefficient extraction fix)
- **New artifacts**: logs/batch_label_results.jsonl, logs/pre_retrain_snapshot.json, logs/channel_performance.jsonl
- **Labeling results**: 386 total (324 GOOD, 62 BAD, 16.1% wrong rate)
  - Blind spot: 12/12 BAD (100% wrong — all underspecified/impossible)
  - Ambiguous: 324/374 GOOD, 50/374 BAD (13.1% wrong)
- **Dataset**: 140 → 700 labeled (498 correct, 202 wrong, ratio 2.5:1)
- **Model**: AUC=0.7511, accuracy=78.9%, Brier=0.1553, CV AUC=0.7341±0.095
- **Coefficients**: confidence_gap +3.999 → wrong; S_v1 -2.929 → correct; S_v4 -1.734 → correct
- **Thresholds**: review>0.20, escalate>0.65 (cost-optimized)
- **Gain snapshot**: blind_spot info_value=1.73, ambiguous info_value=7.14
- **Failure types**: boundary 12.5% wrong, contradiction 69.2% wrong, overconfidence 100% wrong
- **Weight update**: deferred (lag guard, first cycle) — performance recorded for next cycle
- **Committed**: 73a1a86, tag v2.4.0-batched-consolidation
- **No changes to**: v4 scoring, survival.py, batch_monitor.py, core thresholds

---
Task ID: 5
Agent: main
Task: v2.5.0 — decomposed failure manifold map (per-manifold classifiers)

Work Log:
- Analyzed v2.4.0 batch label results: 386 labeled with failure_type taxonomy
- Confirmed three distinct failure geometries with fundamentally different wrong rates
- Froze v2.4.0 metrics as reference baseline (logs/v250_baseline.json)
- Built per-manifold datasets by merging batch labels + existing failure_dataset
  - Overconfidence: 19 samples (blind_spot channel + overconfidence type), 100% wrong
  - Contradiction: 34 samples (v4-v1 structural disagreement), 76.5% wrong
  - Boundary: 1033 samples (standard ambiguity), 21.1% wrong
- Evaluated manifold separation in feature space
  - Centroid distances confirm distinct geometries (overconfidence-boundary: 0.30)
  - Router confidence_gap is dominant routing feature (importance=4.74)
- Trained three per-manifold heads:
  - Overconfidence: rule-based detection (100% wrong → P(wrong)=1.0, no model needed)
  - Contradiction: LR + isotonic calibration, AUC=0.8798 (high-value surface)
  - Boundary: LR + isotonic calibration, AUC=0.7073, CV=0.6988 (standard regime)
- Trained multinomial LR manifold router (accuracy=74.7%, CV=76.0%)
- Built runtime prediction pipeline (manifold_predict.py):
  - Rule-based disagreement override for contradiction routing
  - Per-manifold decision thresholds (not one global threshold)
  - Overconfidence → always escalate; contradiction → P>0.5; boundary → cost-optimized 0.35
- Key structural insight: AUC collapsed because one model averaged three failure geometries

Stage Summary:
- **New files**: scripts/manifold_classifier.py, scripts/manifold_predict.py
- **Modified files**: none (non-breaking addition)
- **New artifacts**: model/manifold_models.pkl, model/manifold_report.json, logs/v250_baseline.json
- **Architecture transition**: unified predictor → decomposed failure manifold map
- **Per-manifold AUC**: overconfidence=N/A (detection), contradiction=0.88, boundary=0.71
- **Router**: 3-class multinomial LR, confidence_gap dominant (importance=4.74)
- **Decision thresholds**: per-manifold (overconfidence=always escalate, contradiction=0.50, boundary=0.35)
- **What was NOT done**: no AUC recovery, no threshold retuning, no weight rebalancing
- **Committed**: pending

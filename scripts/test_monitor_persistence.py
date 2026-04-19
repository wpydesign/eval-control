#!/usr/bin/env python3
"""Unit tests for v2.1.4 persistence and suppression layer.

Tests cover:
  1. Fresh MonitorState has zero stats
  2. _load_stats reads from monitor_stats.json
  3. _save_stats writes to monitor_stats.json
  4. _evaluate_action increments persistent counters (effective)
  5. _evaluate_action increments persistent counters (ineffective)
  6. _evaluate_action does NOT increment for neutral
  7. _is_suppressed: not suppressed with < 2 evaluations
  8. _is_suppressed: suppressed when ineffective/total > 0.5
  9. _is_suppressed: NOT suppressed at exactly 50% (need > 50%)
  10. _is_suppressed: NOT suppressed when effective > ineffective
  11. update() suppresses action instead of activating
  12. Suppression is logged to monitor_suppressions.jsonl
  13. Stats persist across MonitorState instances (cross-run simulation)
  14. Effectiveness entries written to monitor_effectiveness.jsonl
  15. Staggered cascade still works with persistence layer
  16. get_action returns 'none' when suppressed
  17. Missing stats file returns defaults
  18. Corrupt stats file returns defaults
  19. Suppression threshold boundary (2/3 vs 1/3)
"""

import json
import os
import sys
import tempfile
import shutil

# Make scripts importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.batch_monitor as bm

TMPDIR = None
passed = 0
failed = 0


def setup_tmpdir():
    global TMPDIR
    TMPDIR = tempfile.mkdtemp(prefix="monitor_test_")
    bm.MONITOR_STATS_PATH = os.path.join(TMPDIR, "monitor_stats.json")
    bm.SUPPRESSION_LOG_PATH = os.path.join(TMPDIR, "monitor_suppressions.jsonl")
    bm.EFFECTIVENESS_PATH = os.path.join(TMPDIR, "monitor_effectiveness.jsonl")
    bm.ALERT_LOG_PATH = os.path.join(TMPDIR, "monitor_alerts.jsonl")


def teardown_tmpdir():
    if TMPDIR and os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)


def clean_stats():
    """Remove monitor_stats.json so next MonitorState() starts fresh."""
    if os.path.exists(bm.MONITOR_STATS_PATH):
        os.remove(bm.MONITOR_STATS_PATH)


def clean_effectiveness():
    """Remove effectiveness log."""
    if os.path.exists(bm.EFFECTIVENESS_PATH):
        os.remove(bm.EFFECTIVENESS_PATH)


def clean_suppressions():
    """Remove suppression log."""
    if os.path.exists(bm.SUPPRESSION_LOG_PATH):
        os.remove(bm.SUPPRESSION_LOG_PATH)


def clean_all():
    clean_stats()
    clean_effectiveness()
    clean_suppressions()


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} — {detail}")


def test_01_fresh_stats():
    """Fresh MonitorState starts with zero stats."""
    clean_all()
    m = bm.MonitorState()
    stats = m.persistent_stats
    test("fresh stats has forced_review", "forced_review" in stats)
    test("fresh stats has tightened_threshold", "tightened_threshold" in stats)
    test("fresh forced_review effective=0", stats["forced_review"]["effective"] == 0)
    test("fresh forced_review ineffective=0", stats["forced_review"]["ineffective"] == 0)


def test_02_load_stats():
    """_load_stats reads from monitor_stats.json."""
    clean_all()
    data = {
        "forced_review": {"effective": 3, "ineffective": 1},
        "tightened_threshold": {"effective": 1, "ineffective": 5},
    }
    with open(bm.MONITOR_STATS_PATH, "w") as f:
        json.dump(data, f)

    m = bm.MonitorState()
    stats = m.persistent_stats
    test("loaded forced_review effective=3", stats["forced_review"]["effective"] == 3)
    test("loaded forced_review ineffective=1", stats["forced_review"]["ineffective"] == 1)
    test("loaded tightened_threshold effective=1", stats["tightened_threshold"]["effective"] == 1)
    test("loaded tightened_threshold ineffective=5", stats["tightened_threshold"]["ineffective"] == 5)


def test_03_save_stats():
    """_save_stats writes current stats to file."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"]["effective"] = 7
    m._stats["forced_review"]["ineffective"] = 2
    m._save_stats()

    test("file exists after save", os.path.exists(bm.MONITOR_STATS_PATH))
    with open(bm.MONITOR_STATS_PATH) as f:
        data = json.load(f)
    test("saved forced_review effective=7", data["forced_review"]["effective"] == 7)
    test("saved forced_review ineffective=2", data["forced_review"]["ineffective"] == 2)


def test_04_evaluate_increments_persistent():
    """_evaluate_action increments effective counter."""
    clean_all()
    m = bm.MonitorState()
    m._tracking["forced_review"] = {
        "pre_false_accept": 3,
        "pre_risk_spike": 5,
        "false_accept": 1,
        "risk_spike": 2,  # delta = 2-5 = -3 → effective
        "samples": 50,
    }
    m._active["forced_review"] = 1
    m.tick()

    stats = m.persistent_stats
    test("forced_review effective=1 after effective result",
         stats["forced_review"]["effective"] == 1)
    test("forced_review ineffective=0 after effective result",
         stats["forced_review"]["ineffective"] == 0)


def test_05_evaluate_ineffective():
    """_evaluate_action increments ineffective counter."""
    clean_all()
    m = bm.MonitorState()
    m._tracking["tightened_threshold"] = {
        "pre_false_accept": 3,
        "pre_risk_spike": 0,
        "false_accept": 6,  # delta = 6-3 = +3 → ineffective
        "risk_spike": 0,
        "samples": 50,
    }
    m._active["tightened_threshold"] = 1
    m.tick()

    stats = m.persistent_stats
    test("tightened_threshold ineffective=1 after ineffective result",
         stats["tightened_threshold"]["ineffective"] == 1)
    test("tightened_threshold effective=0 after ineffective result",
         stats["tightened_threshold"]["effective"] == 0)


def test_06_evaluate_neutral():
    """_evaluate_action does NOT increment counters for neutral result."""
    clean_all()
    m = bm.MonitorState()
    m._tracking["forced_review"] = {
        "pre_false_accept": 3,
        "pre_risk_spike": 5,
        "false_accept": 0,
        "risk_spike": 5,  # delta = 0 → neutral
        "samples": 50,
    }
    m._active["forced_review"] = 1
    m.tick()

    stats = m.persistent_stats
    test("neutral result: effective=0",
         stats["forced_review"]["effective"] == 0)
    test("neutral result: ineffective=0",
         stats["forced_review"]["ineffective"] == 0)


def test_07_not_suppressed_with_few_evaluations():
    """Not suppressed with < 2 total evaluations."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"] = {"effective": 0, "ineffective": 1}

    test("not suppressed with 1 evaluation (even if 100% ineffective)",
         m._is_suppressed("forced_review") == False)

    m._stats["forced_review"] = {"effective": 0, "ineffective": 0}
    test("not suppressed with 0 evaluations",
         m._is_suppressed("forced_review") == False)


def test_08_suppressed_when_ineffective_majority():
    """Suppressed when ineffective/total > 0.5."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"] = {"effective": 1, "ineffective": 3}  # 75% ineffective

    test("suppressed at 75% ineffective (3/4)",
         m._is_suppressed("forced_review") == True)


def test_09_not_suppressed_at_exactly_50():
    """NOT suppressed at exactly 50% (need > 50%)."""
    clean_all()
    m = bm.MonitorState()
    m._stats["tightened_threshold"] = {"effective": 2, "ineffective": 2}  # exactly 50%

    test("not suppressed at exactly 50% (2/4)",
         m._is_suppressed("tightened_threshold") == False)


def test_10_not_suppressed_when_effective():
    """NOT suppressed when effective > ineffective."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"] = {"effective": 5, "ineffective": 1}  # 17% ineffective

    test("not suppressed at 17% ineffective (1/6)",
         m._is_suppressed("forced_review") == False)


def test_11_update_suppresses_instead_of_activating():
    """update() suppresses action and does NOT activate it."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"] = {"effective": 1, "ineffective": 3}  # 75% → suppressed

    alerts = ["[ALERT] type=RISK_SPIKE count=3 window=50"]
    m.update(alerts, pre_false_accept=3, pre_risk_spike=2)

    test("suppressed action NOT in active_actions",
         "forced_review" not in m.active_actions)
    test("suppression event recorded",
         len(m.suppressed_events) == 1)
    test("suppression event has correct action",
         m.suppressed_events[0]["action"] == "forced_review")


def test_12_suppression_logged_to_file():
    """Suppression events are logged to monitor_suppressions.jsonl."""
    clean_all()
    m = bm.MonitorState()
    m._stats["tightened_threshold"] = {"effective": 0, "ineffective": 4}

    alerts = ["[ALERT] type=FALSE_ACCEPT count=3 window=200"]
    m.update(alerts, pre_false_accept=3, pre_risk_spike=0)

    test("suppression log file exists",
         os.path.exists(bm.SUPPRESSION_LOG_PATH))

    with open(bm.SUPPRESSION_LOG_PATH) as f:
        lines = f.readlines()

    test("exactly 1 suppression log entry", len(lines) == 1)
    entry = json.loads(lines[0])
    test("log entry has action=tightened_threshold",
         entry["action"] == "tightened_threshold")
    test("log entry has reason",
         "ineffective_rate" in entry["reason"].lower() or "> 50%" in entry["reason"])
    test("log entry has timestamp",
         "timestamp" in entry and len(entry["timestamp"]) > 0)


def test_13_cross_run_persistence():
    """Stats persist across MonitorState instances (simulates cross-run)."""
    clean_all()

    # Run 1: evaluate forced_review as effective
    m1 = bm.MonitorState()
    m1._tracking["forced_review"] = {
        "pre_false_accept": 3,
        "pre_risk_spike": 5,
        "false_accept": 1,
        "risk_spike": 2,  # effective
        "samples": 50,
    }
    m1._active["forced_review"] = 1
    m1.tick()

    # Run 2: new instance should load stats from file
    m2 = bm.MonitorState()
    stats = m2.persistent_stats

    test("cross-run: forced_review effective=1",
         stats["forced_review"]["effective"] == 1)
    test("cross-run: forced_review ineffective=0",
         stats["forced_review"]["ineffective"] == 0)

    # Run 2: evaluate tightened_threshold as ineffective
    m2._tracking["tightened_threshold"] = {
        "pre_false_accept": 2,
        "pre_risk_spike": 0,
        "false_accept": 5,  # ineffective
        "risk_spike": 0,
        "samples": 50,
    }
    m2._active["tightened_threshold"] = 1
    m2.tick()

    # Run 3: check both persisted
    m3 = bm.MonitorState()
    stats3 = m3.persistent_stats
    test("cross-run: forced_review still effective=1",
         stats3["forced_review"]["effective"] == 1)
    test("cross-run: tightened_threshold ineffective=1",
         stats3["tightened_threshold"]["ineffective"] == 1)


def test_14_effectiveness_file_created():
    """Effectiveness entries are written to monitor_effectiveness.jsonl."""
    clean_all()
    m = bm.MonitorState()
    m._tracking["forced_review"] = {
        "pre_false_accept": 3,
        "pre_risk_spike": 5,
        "false_accept": 1,
        "risk_spike": 2,
        "samples": 50,
    }
    m._active["forced_review"] = 1
    m.tick()

    test("effectiveness log exists", os.path.exists(bm.EFFECTIVENESS_PATH))
    with open(bm.EFFECTIVENESS_PATH) as f:
        lines = f.readlines()
    test("exactly 1 effectiveness entry", len(lines) == 1)
    entry = json.loads(lines[0])
    test("entry has action=forced_review", entry["action"] == "forced_review")
    test("entry has result=effective", entry["result"] == "effective")
    test("entry has delta=-3", entry["delta"] == -3)
    test("entry has timestamp", "timestamp" in entry)


def test_15_cascade_still_works():
    """Staggered cascade still works with persistence layer."""
    clean_all()
    m = bm.MonitorState()

    alerts = [
        "[ALERT] type=RISK_SPIKE count=3 window=50",
        "[ALERT] type=FALSE_ACCEPT count=3 window=200",
    ]
    m.update(alerts, pre_false_accept=3, pre_risk_spike=2)

    test("forced_review activated", m.active_actions.get("forced_review", 0) == 50)
    test("tightened_threshold staggered (100)",
         m.active_actions.get("tightened_threshold", 0) == 100)

    # Tick down forced_review to expiry
    for _ in range(49):
        m.tick()

    test("forced_review still active at tick 49",
         m.active_actions.get("forced_review", 0) > 0)

    m.tick()  # forced_review expires

    test("forced_review expired after 50 ticks",
         "forced_review" not in m.active_actions)
    test("tightened_threshold still active after forced_review expiry",
         m.active_actions.get("tightened_threshold", 0) > 0)


def test_16_get_action_returns_none_when_suppressed():
    """When update suppresses, get_action returns 'none'."""
    clean_all()
    m = bm.MonitorState()
    m._stats["forced_review"] = {"effective": 0, "ineffective": 5}

    alerts = ["[ALERT] type=RISK_SPIKE count=3 window=50"]
    m.update(alerts, pre_false_accept=3, pre_risk_spike=2)

    action = m.get_action("domain_knowledge")
    test("get_action returns 'none' when suppressed (action not activated)",
         action == "none")


def test_17_missing_stats_file_graceful():
    """Missing monitor_stats.json returns defaults, no crash."""
    clean_all()
    m = bm.MonitorState()
    stats = m.persistent_stats
    test("defaults returned for missing file",
         stats["forced_review"]["effective"] == 0
         and stats["tightened_threshold"]["ineffective"] == 0)
    test("no crash on missing file", True)


def test_18_corrupt_stats_file_graceful():
    """Corrupt monitor_stats.json returns defaults, no crash."""
    clean_all()
    with open(bm.MONITOR_STATS_PATH, "w") as f:
        f.write("THIS IS NOT JSON {{{")

    m = bm.MonitorState()
    stats = m.persistent_stats
    test("defaults returned for corrupt file",
         stats["forced_review"]["effective"] == 0)
    test("no crash on corrupt file", True)


def test_19_suppression_threshold_boundary():
    """Test exact boundary: 2/3 (66.7%) suppressed, 1/3 (33.3%) not."""
    clean_all()
    m = bm.MonitorState()

    m._stats["test_action"] = {"effective": 1, "ineffective": 2}  # 66.7%
    test("2/3 (66.7%) -> suppressed", m._is_suppressed("test_action") == True)

    m._stats["test_action"] = {"effective": 2, "ineffective": 1}  # 33.3%
    test("1/3 (33.3%) -> not suppressed", m._is_suppressed("test_action") == False)


def test_20_suppression_unsuppresses_on_effective_history():
    """After enough effective results, action is no longer suppressed."""
    clean_all()
    m = bm.MonitorState()
    # Start with 1e/3i (75% -> suppressed)
    m._stats["forced_review"] = {"effective": 1, "ineffective": 3}
    test("initially suppressed", m._is_suppressed("forced_review") == True)

    # Add 3 more effective results via _evaluate_action
    for _ in range(3):
        m._tracking["forced_review"] = {
            "pre_false_accept": 3,
            "pre_risk_spike": 5,
            "false_accept": 1,
            "risk_spike": 2,  # effective
            "samples": 50,
        }
        m._active["forced_review"] = 1
        m.tick()

    # Now: 4e/3i = 43% ineffective -> NOT suppressed
    test("unsuppressed after 3 more effective results (4e/3i=43%)",
         m._is_suppressed("forced_review") == False)


def main():
    global TMPDIR
    setup_tmpdir()

    print("=" * 55)
    print("  v2.1.4 Persistence & Suppression Tests")
    print("=" * 55)

    test_01_fresh_stats()
    test_02_load_stats()
    test_03_save_stats()
    test_04_evaluate_increments_persistent()
    test_05_evaluate_ineffective()
    test_06_evaluate_neutral()
    test_07_not_suppressed_with_few_evaluations()
    test_08_suppressed_when_ineffective_majority()
    test_09_not_suppressed_at_exactly_50()
    test_10_not_suppressed_when_effective()
    test_11_update_suppresses_instead_of_activating()
    test_12_suppression_logged_to_file()
    test_13_cross_run_persistence()
    test_14_effectiveness_file_created()
    test_15_cascade_still_works()
    test_16_get_action_returns_none_when_suppressed()
    test_17_missing_stats_file_graceful()
    test_18_corrupt_stats_file_graceful()
    test_19_suppression_threshold_boundary()
    test_20_suppression_unsuppresses_on_effective_history()

    print(f"\n{'=' * 55}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 55}")

    teardown_tmpdir()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
batch_monitor.py — Streaming rolling-window monitor [v2.1.4]

Replaces weekly reporting with event-driven monitoring.
Runs after every batch append. No reports, just signals.

Windows:  50 (fast spike) | 200 (drift)

Alert rules (immediate, no delay):
  risk_spike        >= 2   in last 50   (factuality_risk_flag bursts)
  false_accept      >= 3   in last 200  (v4 accepting questionable outputs)
  dk_HI_rate        > 0.5 in last 200  (domain_knowledge regression)
  confidence_gap_mean > 0.15 in last 200 (v4 getting too lenient)

v2.1.2: alert → action hooks (routing layer only, no model change):
  RISK_SPIKE   → force shadow review for domain_knowledge (next 50 samples)
  FALSE_ACCEPT → tighten tau_h to 0.80 for domain_knowledge (next 50 samples)
  DK_DRIFT     → log critical, no auto-change
  GAP_DRIFT    → log leniency drift, no auto-change
  All actions: temporary, domain_knowledge-scoped, auto-expire after 50 samples.

v2.1.4: persistent cross-run memory + suppression:
  - monitor_stats.json tracks cumulative effective/ineffective counts per action
  - if ineffective/total > 0.5, action is SUPPRESSED (not activated)
  - suppression events logged with monitor_action="suppressed"
  - enables experience-aware control across runs

Usage:
  python scripts/batch_monitor.py              # scan full log, print alerts
  python scripts/batch_monitor.py --watch      # tail-follow mode (for live use)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from collections import deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISAGREEMENT_PATH = os.path.join(BASE, "logs", "disagreement_cases.jsonl")
ALERT_LOG_PATH = os.path.join(BASE, "logs", "monitor_alerts.jsonl")
EFFECTIVENESS_PATH = os.path.join(BASE, "logs", "monitor_effectiveness.jsonl")
MONITOR_STATS_PATH = os.path.join(BASE, "logs", "monitor_stats.json")
SUPPRESSION_LOG_PATH = os.path.join(BASE, "logs", "monitor_suppressions.jsonl")
SUPPRESSION_THRESHOLD = 0.5  # suppress if ineffective_rate > 50%

WINDOW_50 = 50
WINDOW_200 = 200
ACTION_EXPIRY = 50  # samples before auto-expire

# ── thresholds ──
RISK_SPIKE_THRESHOLD = 2
FALSE_ACCEPT_THRESHOLD = 3
DK_HI_RATE_THRESHOLD = 0.5
CONFIDENCE_GAP_THRESHOLD = 0.15
TIGHTENED_TAU_H = 0.80  # elevated accept threshold for domain_knowledge


# ═══════════════════════════════════════════════════════════════
# MONITOR STATE — alert-triggered routing actions
# ═══════════════════════════════════════════════════════════════

class MonitorState:
    """Holds active routing interventions triggered by alerts.

    All actions are:
      - temporary (auto-expire after ACTION_EXPIRY samples)
      - scoped to domain_knowledge failure mode only
      - routing-layer only (no model/scoring/gating changes)

    v2.1.3: tracks per-action effectiveness (pre/post metrics + expiry evaluation).
    v2.1.4: persistent cross-run memory (monitor_stats.json) + suppression.

    Usage in batch runner:
      monitor = MonitorState()
      # before each eval:
      action = monitor.get_action(failure_mode)
      # after each eval (feed sample signals):
      monitor.record_sample(is_dk, factuality_risk)
      monitor.tick()
      # after each batch:
      monitor.update(alerts, pre_false_accept, pre_risk_spike)
    """

    def __init__(self):
        self._active = {}       # action_name -> remaining_ticks
        self._tracking = {}     # action_name -> {false_accept: int, risk_spike: int, samples: int}
        self._consecutive_fail = {"forced_review": 0, "tightened_threshold": 0}
        # v2.1.4: persistent cross-run stats
        self._stats = self._load_stats()
        self._suppressed_events = []  # log of suppressions this run

    @property
    def active_actions(self):
        return dict(self._active)

    @property
    def consecutive_failures(self):
        return dict(self._consecutive_fail)

    @property
    def persistent_stats(self):
        """Return the current persistent stats dict (for inspection/logging)."""
        return dict(self._stats)

    @property
    def suppressed_events(self):
        """Return list of suppression events this run."""
        return list(self._suppressed_events)

    # ── v2.1.4: persistence layer ──

    def _load_stats(self) -> dict:
        """Load cumulative effectiveness stats from monitor_stats.json.

        Returns:
            {"forced_review": {"effective": N, "ineffective": M}, ...}
        """
        default = {
            "forced_review": {"effective": 0, "ineffective": 0},
            "tightened_threshold": {"effective": 0, "ineffective": 0},
        }
        if not os.path.exists(MONITOR_STATS_PATH):
            return default
        try:
            with open(MONITOR_STATS_PATH) as f:
                data = json.load(f)
            # Merge with defaults (handles missing keys)
            for action_name in default:
                if action_name not in data:
                    data[action_name] = {"effective": 0, "ineffective": 0}
                for key in ("effective", "ineffective"):
                    if key not in data[action_name]:
                        data[action_name][key] = 0
            return data
        except (json.JSONDecodeError, OSError):
            return default

    def _save_stats(self):
        """Write current cumulative stats to monitor_stats.json."""
        try:
            os.makedirs(os.path.dirname(MONITOR_STATS_PATH), exist_ok=True)
            with open(MONITOR_STATS_PATH, "w") as f:
                json.dump(self._stats, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _is_suppressed(self, action_name: str) -> bool:
        """Check if action should be suppressed due to historical ineffectiveness.

        Suppress if ineffective_rate > SUPPRESSION_THRESHOLD (50%)
        AND at least 2 total evaluations (avoid suppressing on first result).
        """
        s = self._stats.get(action_name, {})
        effective = s.get("effective", 0)
        ineffective = s.get("ineffective", 0)
        total = effective + ineffective
        if total < 2:
            return False  # not enough data to suppress
        if total == 0:
            return False
        return (ineffective / total) > SUPPRESSION_THRESHOLD

    def _log_suppression(self, action_name: str, reason: str):
        """Log a suppression event to monitor_suppressions.jsonl."""
        entry = {
            "action": action_name,
            "reason": reason,
            "stats_snapshot": dict(self._stats.get(action_name, {})),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._suppressed_events.append(entry)
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        s = self._stats.get(action_name, {})
        eff = s.get("effective", 0)
        ineff = s.get("ineffective", 0)
        total = eff + ineff
        rate = ineff / total if total > 0 else 0
        print(f"  [{ts}] [SUPPRESSED] {action_name}: {reason} "
              f"(ineffective_rate={rate:.0%}, {eff}e/{ineff}i)")
        try:
            os.makedirs(os.path.dirname(SUPPRESSION_LOG_PATH), exist_ok=True)
            with open(SUPPRESSION_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def get_action(self, failure_mode: str) -> str:
        """Return the routing action for this failure mode.

        Returns:
          "forced_review"       — RISK_SPIKE active, dk -> override to shadow review
          "tightened_threshold" — FALSE_ACCEPT active, dk -> require S >= 0.80
          "suppressed"          — action was suppressed due to historical ineffectiveness
          "none"                — no active intervention
        """
        if failure_mode != "domain_knowledge":
            return "none"
        if self._active.get("forced_review", 0) > 0:
            return "forced_review"
        if self._active.get("tightened_threshold", 0) > 0:
            return "tightened_threshold"
        return "none"

    def record_sample(self, is_dk: bool, factuality_risk: bool):
        """Feed per-sample signals for active action tracking.

        Call BEFORE tick() so the sample counts against the current window.

        Args:
            is_dk: True if this sample was classified as domain_knowledge
            factuality_risk: True if factuality_risk_flag was set
        """
        for action_name in self._active:
            if action_name not in self._tracking:
                self._tracking[action_name] = {"false_accept": 0, "risk_spike": 0, "samples": 0}
            t = self._tracking[action_name]
            t["samples"] += 1
            if factuality_risk:
                t["false_accept"] += 1
                if is_dk:
                    t["risk_spike"] += 1

    def tick(self):
        """Decrement all counters by 1. Evaluate + log expired actions. Call after record_sample."""
        expired = [k for k, v in self._active.items() if v <= 1]
        for k in expired:
            self._evaluate_action(k)
            del self._active[k]
        for k in self._active:
            self._active[k] -= 1

    def update(self, alerts: list[str], pre_false_accept: int = 0, pre_risk_spike: int = 0):
        """Activate new interventions from alert strings.

        When both RISK_SPIKE and FALSE_ACCEPT fire simultaneously,
        tightened_threshold is staggered to activate after forced_review expires.

        v2.1.4: checks historical suppression before activating.
        If action is suppressed, logs event and does NOT activate.

        Args:
            alerts: list of alert strings from scan()
            pre_false_accept: current false_accept count in window_200 (snapshot before action)
            pre_risk_spike: current risk_spike count in window_50 (snapshot before action)
        """
        has_spike = any("RISK_SPIKE" in a for a in alerts)
        has_fa = any("FALSE_ACCEPT" in a for a in alerts)

        if has_spike and "forced_review" not in self._active:
            # v2.1.4: check suppression before activating
            if self._is_suppressed("forced_review"):
                self._log_suppression("forced_review",
                    "historical ineffective_rate > 50%")
            else:
                self._active["forced_review"] = ACTION_EXPIRY
                self._tracking["forced_review"] = {
                    "pre_false_accept": pre_false_accept,
                    "pre_risk_spike": pre_risk_spike,
                    "false_accept": 0,
                    "risk_spike": 0,
                    "samples": 0,
                }

        if has_fa and "tightened_threshold" not in self._active:
            # v2.1.4: check suppression before activating
            if self._is_suppressed("tightened_threshold"):
                self._log_suppression("tightened_threshold",
                    "historical ineffective_rate > 50%")
            else:
                if "forced_review" in self._active:
                    self._active["tightened_threshold"] = self._active["forced_review"] + ACTION_EXPIRY
                else:
                    self._active["tightened_threshold"] = ACTION_EXPIRY
                self._tracking["tightened_threshold"] = {
                    "pre_false_accept": pre_false_accept,
                    "pre_risk_spike": pre_risk_spike,
                    "false_accept": 0,
                    "risk_spike": 0,
                    "samples": 0,
                }

    def _evaluate_action(self, action_name: str):
        """Evaluate action effectiveness on expiry. Log + update persistent stats."""
        t = self._tracking.pop(action_name, {})
        if not t or "pre_false_accept" not in t:
            return

        post_fa = t.get("false_accept", 0)
        post_spike = t.get("risk_spike", 0)
        pre_fa = t.get("pre_false_accept", 0)
        pre_spike = t.get("pre_risk_spike", 0)
        samples = t.get("samples", 0)

        # Effectiveness: did the action reduce its target metric?
        if action_name == "forced_review":
            delta = post_spike - pre_spike
            result = "effective" if delta < 0 else "neutral" if delta == 0 else "ineffective"
        else:  # tightened_threshold
            delta = post_fa - pre_fa
            result = "effective" if delta < 0 else "neutral" if delta == 0 else "ineffective"

        # Track consecutive failures
        if result == "ineffective":
            self._consecutive_fail[action_name] = self._consecutive_fail.get(action_name, 0) + 1
        else:
            self._consecutive_fail[action_name] = 0

        redesign_flag = self._consecutive_fail[action_name] >= 2

        entry = {
            "action": action_name,
            "result": result,
            "delta": delta,
            "samples": samples,
            "pre_false_accept": pre_fa,
            "post_false_accept": post_fa,
            "pre_risk_spike": pre_spike,
            "post_risk_spike": post_spike,
            "consecutive_failures": self._consecutive_fail[action_name],
            "needs_redesign": redesign_flag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            os.makedirs(os.path.dirname(EFFECTIVENESS_PATH), exist_ok=True)
            with open(EFFECTIVENESS_PATH, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

        # v2.1.4: update persistent stats
        if action_name not in self._stats:
            self._stats[action_name] = {"effective": 0, "ineffective": 0}
        if result == "effective":
            self._stats[action_name]["effective"] += 1
        elif result == "ineffective":
            self._stats[action_name]["ineffective"] += 1
        # neutral does not increment either counter
        self._save_stats()

        # Print outcome
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        redesign_tag = " NEEDS_REDESIGN" if redesign_flag else ""
        s = self._stats[action_name]
        total = s["effective"] + s["ineffective"]
        rate = s["ineffective"] / total if total > 0 else 0
        print(f"  [{ts}] [EFFECTIVENESS] {action_name}: {result} (delta={delta:+d}, samples={samples})"
              f" [cumulative: {s['effective']}e/{s['ineffective']}i, rate={rate:.0%}]{redesign_tag}")

        return result


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def scan(cases):
    """Scan disagreement cases over rolling windows.

    Returns:
        alerts: list of alert strings
        pre_metrics: dict with current false_accept_count and risk_spike_count
            (snapshot before any action — fed to monitor.update())
    """
    alerts = []
    pre_metrics = {"false_accept_count": 0, "risk_spike_count": 0}

    if not cases:
        return alerts, pre_metrics

    w200 = cases[-WINDOW_200:]
    w50 = cases[-WINDOW_50:]

    # ── window_200 metrics ──
    dk_cases = [c for c in w200 if c.get("failure_mode") == "domain_knowledge"]
    dk_hi = [c for c in dk_cases if c.get("is_high_impact")]
    dk_hi_rate = len(dk_hi) / len(dk_cases) if dk_cases else 0.0

    false_accepts = [c for c in w200 if c.get("factuality_risk_flag")]

    gaps = [c.get("confidence_gap", c.get("S_delta", 0)) for c in w200]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0

    # ── window_50 metrics ──
    risk_spike_count = sum(1 for c in w50 if c.get("factuality_risk_flag"))

    # Snapshot pre-action metrics
    pre_metrics["false_accept_count"] = len(false_accepts)
    pre_metrics["risk_spike_count"] = risk_spike_count

    # ── alert checks ──
    if risk_spike_count >= RISK_SPIKE_THRESHOLD:
        alerts.append(f"[ALERT] type=RISK_SPIKE count={risk_spike_count} window=50")

    if len(false_accepts) >= FALSE_ACCEPT_THRESHOLD:
        alerts.append(f"[ALERT] type=FALSE_ACCEPT count={len(false_accepts)} window=200")

    if dk_hi_rate > DK_HI_RATE_THRESHOLD:
        alerts.append(f"[ALERT] type=DK_DRIFT rate={dk_hi_rate:.2f} window=200")

    if gap_mean > CONFIDENCE_GAP_THRESHOLD:
        alerts.append(f"[ALERT] type=GAP_DRIFT gap={gap_mean:.4f} window=200")

    return alerts, pre_metrics


def log_alerts(alerts, path=ALERT_LOG_PATH):
    """Append alerts to JSONL log."""
    if not alerts:
        return
    ts = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for a in alerts:
                entry = {"timestamp": ts, "alert": a}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def print_status(cases):
    """Print one-line status (always, even with no alerts)."""
    w200 = cases[-WINDOW_200:]
    w50 = cases[-WINDOW_50:]

    dk = [c for c in w200 if c.get("failure_mode") == "domain_knowledge"]
    dk_hi = [c for c in dk if c.get("is_high_impact")]
    dk_rate = len(dk_hi) / len(dk) if dk else 0.0

    fa = sum(1 for c in w200 if c.get("factuality_risk_flag"))
    spike = sum(1 for c in w50 if c.get("factuality_risk_flag"))
    gaps = [c.get("confidence_gap", c.get("S_delta", 0)) for c in w200]
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] monitor  n={len(cases):>4}  "
          f"dk_HI={dk_rate:.0%}  fa={fa}  spike={spike}  gap={gap_mean:+.4f}")


def main():
    watch = "--watch" in sys.argv

    cases = load_jsonl(DISAGREEMENT_PATH)
    print(f"Loaded {len(cases)} disagreement cases")

    if not watch:
        # One-shot scan
        alerts, pre_metrics = scan(cases)
        if alerts:
            for a in alerts:
                print(a)
            log_alerts(alerts)
        else:
            print_status(cases)
            print("OK — no alerts")
        return

    # Watch mode: poll for new entries
    print("Watching for new entries (Ctrl+C to stop)...")
    print_status(cases)

    last_n = len(cases)
    try:
        while True:
            time.sleep(5)
            cases = load_jsonl(DISAGREEMENT_PATH)
            if len(cases) > last_n:
                new_count = len(cases) - last_n
                last_n = len(cases)
                alerts, pre_metrics = scan(cases)
                print_status(cases)
                for a in alerts:
                    print(a)
                log_alerts(alerts)
                if not alerts:
                    print(f"  +{new_count} new — OK")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

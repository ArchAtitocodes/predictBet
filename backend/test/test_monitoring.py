"""
Tests for PredictBet monitoring module.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest


class TestMetricSnapshot:
    def test_no_severity_when_below_threshold(self):
        from monitoring import MetricSnapshot
        snap = MetricSnapshot(name="latency", value=100.0, unit="ms", timestamp=time.time(),
                              threshold_warning=500.0, threshold_critical=2000.0)
        assert snap.severity() is None

    def test_warning_severity(self):
        from monitoring import MetricSnapshot
        snap = MetricSnapshot(name="latency", value=750.0, unit="ms", timestamp=time.time(),
                              threshold_warning=500.0, threshold_critical=2000.0)
        assert snap.severity() == "WARNING"

    def test_critical_severity(self):
        from monitoring import MetricSnapshot
        snap = MetricSnapshot(name="latency", value=3000.0, unit="ms", timestamp=time.time(),
                              threshold_warning=500.0, threshold_critical=2000.0)
        assert snap.severity() == "CRITICAL"


class TestSlidingWindowMetric:
    def test_record_and_snapshot(self):
        from monitoring import SlidingWindowMetric
        metric = SlidingWindowMetric(name="test", window_seconds=60, threshold_warning=10.0, threshold_critical=20.0)
        metric.record(5.0)
        snap = metric.snapshot()
        assert snap.value == 5.0
        assert snap.severity() is None

    def test_eviction(self):
        from monitoring import SlidingWindowMetric
        metric = SlidingWindowMetric(name="test", window_seconds=1)
        metric.record(10.0, timestamp=time.time() - 2)
        metric.record(5.0, timestamp=time.time())
        snap = metric.snapshot()
        assert snap.value == 5.0


class TestSystemMonitor:
    def test_register_and_record(self, tmp_db_path):
        from monitoring import SystemMonitor
        monitor = SystemMonitor(alert_log_path=str(tmp_db_path))
        monitor.register("custom_metric", window_seconds=60, threshold_warning=10.0, threshold_critical=20.0)
        monitor.record("custom_metric", 15.0)
        report = monitor.report()
        assert "custom_metric" in report

    def test_default_registration(self):
        from monitoring import SystemMonitor
        monitor = SystemMonitor()
        assert "prediction_latency_ms" in monitor.metrics
        assert "cache_hit_rate" in monitor.metrics

    def test_evaluate_no_alerts(self):
        from monitoring import SystemMonitor
        monitor = SystemMonitor()
        monitor.record("prediction_latency_ms", 100.0)
        alerts = monitor.evaluate()
        assert isinstance(alerts, list)

    def test_alert_to_dict(self):
        from monitoring import Alert
        alert = Alert(
            severity="WARNING", metric="latency", message="High latency",
            value=750.0, threshold=500.0, timestamp=time.time(),
        )
        d = alert.to_dict()
        assert d["severity"] == "WARNING"
        assert d["metric"] == "latency"
        assert "timestamp" in d


class TestAlertLog:
    def test_alert_log_writes_and_reads(self, tmp_db_path):
        from monitoring import AlertLog
        log = AlertLog(db_path=str(tmp_db_path))
        from monitoring import Alert
        alert = Alert("CRITICAL", "cpu", "CPU high", 99.0, 90.0, time.time())
        log.append(alert)
        rows = log.recent(limit=10)
        assert len(rows) == 1
        assert rows[0]["metric"] == "cpu"

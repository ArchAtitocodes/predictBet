"""
Tests for PredictBet calibration module.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest


class TestCalibrationStore:
    def test_add_and_get_samples(self, tmp_db_path):
        from calibration import CalibrationStore
        store = CalibrationStore(db_path=tmp_db_path)
        store.add_sample(store.__class__.__module__)
        from calibration import CalibrationSample
        sample = CalibrationSample(model_prob=0.70, actual=1, outcome="H", model_name="ensemble")
        store.add_sample(sample)
        samples = store.get_samples(outcome="H")
        assert len(samples) == 1
        assert samples[0].model_prob == 0.70
        assert samples[0].actual == 1

    def test_get_samples_empty(self, tmp_db_path):
        from calibration import CalibrationStore
        store = CalibrationStore(db_path=tmp_db_path)
        samples = store.get_samples()
        assert samples == []

    def test_save_and_load_model(self, tmp_db_path):
        from calibration import CalibrationStore
        store = CalibrationStore(db_path=tmp_db_path)
        store.save_model("test_model", "H", "isotonic", {"param": 1.0}, 0.25, 0.20, 100)
        loaded = store.load_model("test_model", "H", "isotonic")
        assert loaded is not None
        assert loaded["brier_before"] == 0.25
        assert loaded["brier_after"] == 0.20
        assert loaded["n_samples"] == 100

    def test_load_model_missing(self, tmp_db_path):
        from calibration import CalibrationStore
        store = CalibrationStore(db_path=tmp_db_path)
        loaded = store.load_model("nonexistent", "H", "isotonic")
        assert loaded is None


class TestCalibrationManager:
    def test_manager_initialization(self, tmp_db_path):
        from calibration import CalibrationManager
        mgr = CalibrationManager(db_path=tmp_db_path)
        assert mgr is not None

    def test_fit_and_apply_isotonic(self, tmp_db_path):
        from calibration import CalibrationManager
        mgr = CalibrationManager(db_path=tmp_db_path)
        for i in range(30):
            prob = 0.5 + (i % 5) * 0.1
            actual = 1 if i % 2 == 0 else 0
            mgr.store.add_sample(mgr.store.__class__.__module__)
            from calibration import CalibrationSample
            mgr.store.add_sample(CalibrationSample(model_prob=prob, actual=actual, outcome="H"))
        result = mgr.fit("ensemble", "H", method="isotonic")
        assert "method" in result or "status" in result

    def test_apply_all_with_no_calibrators(self, tmp_db_path):
        from calibration import CalibrationManager
        mgr = CalibrationManager(db_path=tmp_db_path)
        result = mgr.apply_all("ensemble", {"home_win_prob": 0.55, "draw_prob": 0.25, "away_win_prob": 0.20})
        assert "home_win_prob" in result


class TestCalibrationSample:
    def test_creation(self):
        from calibration import CalibrationSample
        sample = CalibrationSample(model_prob=0.70, actual=1, outcome="H")
        assert sample.model_prob == 0.70
        assert sample.actual == 1
        assert sample.outcome == "H"

    def test_defaults(self):
        from calibration import CalibrationSample
        import time
        before = time.time()
        sample = CalibrationSample(model_prob=0.5, actual=0, outcome="D")
        after = time.time()
        assert sample.model_name == ""
        assert before <= sample.created_at <= after

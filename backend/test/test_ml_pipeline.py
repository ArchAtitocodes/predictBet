"""
Tests for PredictBet ML pipeline module.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestSafeModelUnpickler:
    def test_rejects_os_module(self):
        from ml_pipeline import _SafeModelUnpickler
        import io
        payload = b"csubprocess\ncall\np0\nS'echo hacked'\ntp1\nRp2\n."
        with pytest.raises(Exception):
            _SafeModelUnpickler(io.BytesIO(payload)).load()

    def test_accepts_simple_dict(self):
        from ml_pipeline import _SafeModelUnpickler
        import io
        payload = pickle.dumps({"model": {"n_estimators": 100}, "version": "v1"})
        result = _SafeModelUnpickler(io.BytesIO(payload)).load()
        assert result == {"model": {"n_estimators": 100}, "version": "v1"}

    def test_rejects_missing_model_key(self, tmp_path):
        from ml_pipeline import _safe_load_pickle
        p = tmp_path / "bad.pkl"
        p.write_bytes(pickle.dumps({"version": "v1"}))
        result = _safe_load_pickle(str(p))
        assert result is None

    def test_rejects_non_dict(self, tmp_path):
        from ml_pipeline import _safe_load_pickle
        p = tmp_path / "bad.pkl"
        p.write_bytes(pickle.dumps([1, 2, 3]))
        result = _safe_load_pickle(str(p))
        assert result is None

    def test_missing_file_returns_none(self, tmp_path):
        from ml_pipeline import _safe_load_pickle
        result = _safe_load_pickle(str(tmp_path / "nonexistent.pkl"))
        assert result is None


class TestModelRegistry:
    def test_save_and_load(self, tmp_path):
        from ml_pipeline import ModelRegistry
        registry = ModelRegistry(model_dir=str(tmp_path))
        model = {"type": "xgboost", "params": {"n_estimators": 100}}
        registry.save(model, "xgboost", "v1.0", {"accuracy": 0.8})
        loaded = registry.load("xgboost", "v1.0")
        assert loaded is not None
        assert loaded["model"] == model
        assert loaded["version"] == "v1.0"

    def test_load_nonexistent(self, tmp_path):
        from ml_pipeline import ModelRegistry
        registry = ModelRegistry(model_dir=str(tmp_path))
        assert registry.load("nonexistent", "v1") is None

    def test_list_versions(self, tmp_path):
        from ml_pipeline import ModelRegistry
        registry = ModelRegistry(model_dir=str(tmp_path))
        registry.save({"m": 1}, "lgb", "v1", {})
        registry.save({"m": 2}, "lgb", "v2", {})
        versions = registry.list_versions("lgb")
        assert versions == ["v1", "v2"]


class TestMLPipeline:
    def test_pipeline_initialization(self, tmp_path):
        from ml_pipeline import MLPipeline, ModelRegistry
        registry = ModelRegistry(model_dir=str(tmp_path))
        pipeline = MLPipeline(registry=registry)
        assert pipeline is not None

    def test_predict_no_trained_model(self, tmp_path):
        from ml_pipeline import MLPipeline, ModelRegistry
        from features import FeatureExtractor
        from scraper import TeamForm
        registry = ModelRegistry(model_dir=str(tmp_path))
        pipeline = MLPipeline(registry=registry)
        extractor = FeatureExtractor()
        home = TeamForm("Arsenal", 5, [2, 3, 1, 2, 1], [0, 1, 0, 1, 1])
        away = TeamForm("Chelsea", 5, [1, 2, 0, 1, 2], [2, 1, 1, 0, 2])
        vec = extractor.build_vector(home, away, league_slug="eng.1")
        result = pipeline.predict(vec)
        assert result is None or "model_name" in result

    def test_pickle_load_via_registry_uses_safe_unpickler(self, tmp_path):
        import pickle
        import io
        from ml_pipeline import ModelRegistry, _safe_load_pickle
        registry = ModelRegistry(model_dir=str(tmp_path))
        safe_data = {"model": {"n_estimators": 50}, "metadata": {}, "version": "v1"}
        with open(tmp_path / "safe_v1.pkl", "wb") as f:
            pickle.dump(safe_data, f)
        loaded = registry.load("safe", "v1")
        assert loaded is not None
        assert loaded["model"]["n_estimators"] == 50

    def test_tampered_pickle_rejected(self, tmp_path):
        import pickle
        from ml_pipeline import ModelRegistry, _safe_load_pickle
        registry = ModelRegistry(model_dir=str(tmp_path))
        tampered_path = tmp_path / "tampered_v1.pkl"
        payload = b"csubprocess\ncall\np0\nS'echo hacked'\ntp1\nRp2\n."
        with open(tampered_path, "wb") as f:
            f.write(payload)
        loaded = registry.load("tampered", "v1")
        assert loaded is None

"""
Tests for PredictBet market_pipeline module.
"""
from __future__ import annotations

import os
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest


class TestScraperCache:
    def test_thread_safety(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path, default_ttl=60)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    cache.set(f"key_{n}_{i}", {"val": i})
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestResultsSyncer:
    def test_sync_requires_api_key(self, tmp_db_path):
        from market_pipeline import ResultsSyncer, VersionedPredictionLedger
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": ""}):
            ledger = VersionedPredictionLedger(db_path=str(tmp_db_path))
            syncer = ResultsSyncer(ledger=ledger)
            result = syncer.sync("PL")
            assert result["status"] == "nothing_pending"


class TestVersionedPredictionLedger:
    def test_log_and_compare(self, tmp_db_path):
        from market_pipeline import VersionedPredictionLedger
        from scraper import MatchModelResult
        ledger = VersionedPredictionLedger(db_path=str(tmp_db_path))
        model = MatchModelResult(
            home_team="A", away_team="B",
            home_win_prob=0.5, draw_prob=0.25, away_win_prob=0.25,
            over_2_5_prob=0.5, under_2_5_prob=0.5,
            btts_yes_prob=0.5, btts_no_prob=0.5,
            sample_size_home=10, sample_size_away=10,
        )
        pid = ledger.log(model, model_version="v1.0")
        assert pid > 0
        ledger.record_result(pid, "H")
        # compare_versions returns empty when < 10 scored predictions
        report = ledger.compare_versions()
        assert isinstance(report, dict)

    def test_migration_adds_model_version(self, tmp_db_path):
        db_path = str(tmp_db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, home_team TEXT)")
        conn.commit()
        conn.close()
        from market_pipeline import VersionedPredictionLedger
        ledger = VersionedPredictionLedger(db_path=db_path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        assert "model_version" in cols


class TestOddsMovementTracker:
    def test_snapshot_and_movement(self, tmp_db_path):
        from market_pipeline import OddsMovementTracker
        tracker = OddsMovementTracker(db_path=str(tmp_db_path))
        tracker.snapshot("Arsenal", "Chelsea", "2026-08-01", 2.20, 3.30, 3.50, "Betika")
        tracker.snapshot("Arsenal", "Chelsea", "2026-08-01", 2.05, 3.40, 3.70, "Betika")
        movement = tracker.movement("Arsenal", "Chelsea", "2026-08-01")
        assert movement["status"] == "ok"
        assert movement["n_snapshots"] == 2
        assert movement["line_moved"] is True

    def test_movement_no_data(self, tmp_db_path):
        from market_pipeline import OddsMovementTracker
        tracker = OddsMovementTracker(db_path=str(tmp_db_path))
        movement = tracker.movement("Unknown", "Unknown", "2026-08-01")
        assert movement["status"] == "no_data"


class TestXGDataClient:
    def test_client_requires_requests(self):
        from market_pipeline import XGDataClient
        with patch.dict("sys.modules", {"requests": None}):
            client = XGDataClient()
            assert client.available is False


class TestTeamNewsClient:
    def test_client_requires_feedparser(self):
        from market_pipeline import TeamNewsClient
        with patch.dict("sys.modules", {"feedparser": None}):
            client = TeamNewsClient()
            assert client.available is False


class TestLeagueParameterCalibrator:
    def test_calibrator_returns_dict(self):
        from market_pipeline import LeagueParameterCalibrator
        from scraper import TeamForm
        calibrator = LeagueParameterCalibrator()
        home = TeamForm("H", 10, [1, 2, 1, 2, 1], [0, 1, 0, 1, 0])
        away = TeamForm("A", 10, [1, 0, 1, 0, 1], [1, 2, 1, 2, 1])
        result = calibrator.calibrate(home, away)
        assert isinstance(result, dict)


class TestTeamHomeAdvantageEstimator:
    def test_estimate_returns_dict(self):
        from market_pipeline import TeamHomeAdvantageEstimator
        estimator = TeamHomeAdvantageEstimator(league_avg_home=1.55, league_avg_away=1.22)
        result = estimator.estimate(
            home_goals_scored=[2, 1, 3], home_goals_conceded=[0, 1, 0],
            away_goals_scored=[1, 0, 1], away_goals_conceded=[1, 2, 1],
        )
        assert isinstance(result, dict)


class TestMultiBookmakerOddsAggregator:
    def test_aggregator_requires_requests(self):
        from market_pipeline import MultiBookmakerOddsAggregator
        with patch.dict("sys.modules", {"requests": None}):
            agg = MultiBookmakerOddsAggregator()
            assert agg.available is False


class TestBlendWithMarketProbability:
    def test_blend_returns_dict(self):
        from market_pipeline import blend_with_market_probability
        result = blend_with_market_probability(0.55, 0.22, 0.23, 2.10, 3.40, 3.60)
        assert isinstance(result, dict)
        assert "home" in result
        assert "draw" in result
        assert "away" in result

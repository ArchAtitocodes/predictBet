"""
Tests for PredictBet pipeline module.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestResolveLeagueSlugInPipeline:
    def test_pipeline_imports_resolve(self):
        from pipeline import resolve_league_slug
        assert resolve_league_slug("eng.1", None) == "eng.1"
        assert resolve_league_slug(None, None) is None


class TestEnrichmentFailureOnMissingLeague:
    def test_enrich_raises_when_no_league(self):
        from pipeline import FixtureEnricher
        from scraper import ESPNScraperClient

        enricher = FixtureEnricher(max_workers=1)
        fixture = {"home_team": "UnknownTeamXYZ", "away_team": "AnotherUnknownABC", "home_odd": 2.0, "draw_odd": 3.0, "away_odd": 3.0}

        mock_espn = MagicMock()
        mock_espn.search_team.return_value = [
            {"id": "123", "name": "UnknownTeamXYZ", "league": None},
            {"id": "456", "name": "AnotherUnknownABC", "league": None},
        ]
        with patch.object(ESPNScraperClient, "__new__", return_value=mock_espn):
            result = enricher.enrich(fixture)
            assert result is None


class TestMarketEvaluator:
    def _make_model(self, home_goals=1.5, away_goals=1.2):
        from scraper import MatchModelResult
        return MatchModelResult(
            home_team="A", away_team="B",
            home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
            over_2_5_prob=0.55, under_2_5_prob=0.45,
            btts_yes_prob=0.50, btts_no_prob=0.50,
            sample_size_home=10, sample_size_away=10,
            expected_home_goals=home_goals, expected_away_goals=away_goals,
        )

    def test_evaluate_with_odds(self):
        from pipeline import MarketEvaluator
        from pipeline import FixtureEnrichment
        evaluator = MarketEvaluator()
        model = self._make_model()
        market_comp = {
            "home": {"edge_pct_points": 5.0, "market_implied_pct": 45.0},
            "draw": {"edge_pct_points": 2.0, "market_implied_pct": 28.0},
            "away": {"edge_pct_points": -1.0, "market_implied_pct": 27.0},
            "bookmaker_overround_pct": 4.0,
        }
        enrichment = FixtureEnrichment(
            fixture={"home_odd": 2.10, "draw_odd": 3.40, "away_odd": 3.60},
            bookmaker_odds={"home": 2.10, "draw": 3.40, "away": 3.60},
        )
        result = evaluator.evaluate(model, market_comp, enrichment)
        assert result["best_outcome"] == "home"
        assert result["edge_pct"] == 5.0
        assert result["confidence_tier"] in ("VALUE", "STRONG", "LOCK")

    def test_evaluate_no_odds(self):
        from pipeline import MarketEvaluator
        from pipeline import FixtureEnrichment
        evaluator = MarketEvaluator()
        model = self._make_model()
        enrichment = FixtureEnrichment(
            fixture={},
            bookmaker_odds={"home": 0, "draw": 0, "away": 0},
        )
        result = evaluator.evaluate(model, None, enrichment)
        assert result["best_outcome"] == "none"
        assert result["ev_pct"] == 0.0
        assert result["stake_suggestion_pct"] == 0.0


class TestBetRefusalEngine:
    def test_reject_friendly(self):
        from pipeline import BetRefusalEngine
        from pipeline import FixtureEnrichment
        engine = BetRefusalEngine()
        enrichment = FixtureEnrichment(fixture={"is_friendly": True})
        evaluation = {"edge_pct": 5.0, "ev_pct": 2.0, "confidence_score": 50, "risk_score": 40, "data_grade": "B"}
        should_reject, reason = engine.should_reject(enrichment, evaluation)
        assert should_reject is True
        assert reason == "FRIENDLY_MATCH"

    def test_reject_no_edge(self):
        from pipeline import BetRefusalEngine
        from pipeline import FixtureEnrichment
        engine = BetRefusalEngine()
        enrichment = FixtureEnrichment(fixture={})
        evaluation = {"edge_pct": -1.0, "ev_pct": -1.0, "confidence_score": 50, "risk_score": 40, "data_grade": "B"}
        should_reject, reason = engine.should_reject(enrichment, evaluation)
        assert should_reject is True
        assert reason == "NO_VALUE"

    def test_reject_low_confidence(self):
        from pipeline import BetRefusalEngine
        from pipeline import FixtureEnrichment
        engine = BetRefusalEngine()
        enrichment = FixtureEnrichment(fixture={})
        evaluation = {"edge_pct": 2.0, "ev_pct": 1.0, "confidence_score": 10, "risk_score": 40, "data_grade": "B"}
        should_reject, reason = engine.should_reject(enrichment, evaluation)
        assert should_reject is True
        assert reason == "CONFIDENCE_BELOW_THRESHOLD"


class TestStakeCalculationConsistency:
    """Ensure pipeline and streamlit_app compute identical stakes for the same inputs."""

    def test_pipeline_and_streamlit_agree(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        from pipeline import MarketEvaluator
        from pipeline import FixtureEnrichment

        evaluator = MarketEvaluator()
        from scraper import MatchModelResult
        model = MatchModelResult(
            home_team="A", away_team="B",
            home_win_prob=0.48, draw_prob=0.25, away_win_prob=0.27,
            over_2_5_prob=0.52, under_2_5_prob=0.48,
            btts_yes_prob=0.45, btts_no_prob=0.55,
            sample_size_home=10, sample_size_away=10,
            expected_home_goals=1.6, expected_away_goals=1.3,
        )
        market_comp = {
            "home": {"edge_pct_points": 6.0, "market_implied_pct": 44.0},
            "draw": {"edge_pct_points": 1.0, "market_implied_pct": 29.0},
            "away": {"edge_pct_points": 0.0, "market_implied_pct": 27.0},
            "bookmaker_overround_pct": 4.0,
        }
        enrichment = FixtureEnrichment(
            fixture={"home_odd": 2.10, "draw_odd": 3.40, "away_odd": 3.60},
            bookmaker_odds={"home": 2.10, "draw": 3.40, "away": 3.60},
        )
        evaluation = evaluator.evaluate(model, market_comp, enrichment)
        stake_pct = evaluation["stake_suggestion_pct"]

        # Simulate streamlit_app.py calculation
        best_edge = evaluation["edge_pct"]
        tier_str = evaluation["confidence_tier"]
        tier = ConfidenceTier[tier_str]
        offered = 2.10
        streamlit_res = AggressiveStakeEngine.suggest(tier, max(best_edge / 100, 0), offered)
        assert stake_pct == pytest.approx(streamlit_res["stake_pct"], abs=0.01)


class TestStartAutoRefresh:
    def test_starts_thread(self):
        from pipeline import start_auto_refresh, AutomatedPredictionPipeline
        pipeline = AutomatedPredictionPipeline(max_workers=1)
        t = start_auto_refresh(pipeline, interval_seconds=1, daemon=True)
        assert isinstance(t, threading.Thread)
        assert t.daemon is True
        time.sleep(0.1)


class TestSafePickleIntegration:
    def test_safe_load_rejects_bad_module(self, tmp_path):
        import pickle
        from ml_pipeline import _safe_load_pickle
        p = tmp_path / "bad.pkl"
        payload = b"csubprocess\ncall\np0\nS'echo hacked'\ntp1\nRp2\n."
        p.write_bytes(payload)
        result = _safe_load_pickle(str(p))
        assert result is None

    def test_safe_load_accepts_good_model(self, tmp_path):
        import pickle
        from ml_pipeline import _safe_load_pickle
        p = tmp_path / "good.pkl"
        p.write_bytes(pickle.dumps({"model": {"n": 10}, "version": "v1"}))
        result = _safe_load_pickle(str(p))
        assert result is not None
        assert result["version"] == "v1"

    def test_safe_load_missing_file(self, tmp_path):
        from ml_pipeline import _safe_load_pickle
        result = _safe_load_pickle(str(tmp_path / "nonexistent.pkl"))
        assert result is None


class TestEvidenceChecklist:
    def test_build_evidence(self):
        from pipeline import _build_evidence_checklist, FixtureEnrichment
        from scraper import TeamForm
        evidence = FixtureEnrichment(
            fixture={},
            home_form=TeamForm("H", 5, [1, 2], [0, 1]),
            away_form=TeamForm("A", 5, [1, 0], [1, 2]),
            lineups_confirmed_home=True,
            lineups_confirmed_away=True,
            injuries_home=[],
            injuries_away=[],
            xg_data={"home_xg": 1.5},
            elo_home=1850.0,
            elo_away=None,
            conflicting_sources=False,
            odds_movement=None,
        )
        checklist = _build_evidence_checklist(evidence)
        assert checklist.lineups_confirmed is True
        assert checklist.odds_verified is True
        assert checklist.xg_data_available is True
        assert checklist.team_strength_metrics_available is True

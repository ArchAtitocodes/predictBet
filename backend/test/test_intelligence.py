"""
Tests for PredictBet intelligence module.
"""
from __future__ import annotations

import math
import time

import pytest


class TestConfidenceTier:
    def test_enum_values(self):
        from intelligence import ConfidenceTier
        assert ConfidenceTier.LOCK.value == "LOCK"
        assert ConfidenceTier.STRONG.value == "STRONG"
        assert ConfidenceTier.VALUE.value == "VALUE"
        assert ConfidenceTier.LEAN.value == "LEAN"
        assert ConfidenceTier.NO_BET.value == "NO_BET"


class TestAggressiveStakeEngine:
    def test_no_bet_returns_zero(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.NO_BET, 0.05, 2.0)
        assert result["stake_pct"] == 0.0

    def test_odds_le_one_returns_zero(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LOCK, 0.05, 1.0)
        assert result["stake_pct"] == 0.0

    def test_negative_edge_returns_zero(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LOCK, -0.01, 2.0)
        assert result["stake_pct"] == 0.0

    def test_lock_cap(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LOCK, 0.50, 2.0)
        assert result["stake_pct"] <= 10.0

    def test_strong_cap(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.STRONG, 0.50, 2.0)
        assert result["stake_pct"] <= 5.0

    def test_value_cap(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.VALUE, 0.50, 2.0)
        assert result["stake_pct"] <= 3.0

    def test_lean_cap(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LEAN, 0.50, 2.0)
        assert result["stake_pct"] <= 1.0

    def test_kelly_formula_positive_edge(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        edge = 0.05
        odds = 2.0
        net = odds - 1.0
        expected_kelly = edge / net
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LOCK, edge, odds)
        assert result["stake_pct"] == pytest.approx(min(expected_kelly, 0.10) * 100, abs=0.01)

    def test_returns_note(self):
        from intelligence import AggressiveStakeEngine, ConfidenceTier
        result = AggressiveStakeEngine.suggest(ConfidenceTier.LOCK, 0.05, 2.0)
        assert "note" in result
        assert isinstance(result["note"], str)


class TestEstimateOutputAndEnsemble:
    def test_glm_estimate_returns_none_when_small_sample(self, sample_team_form_home):
        from intelligence import _fit_glm_estimate
        small_home = type("TF", (), {"team_name": "H", "matches_played": 2, "goals_scored": [1, 2], "goals_conceded": [0, 1]})
        small_away = type("TF", (), {"team_name": "A", "matches_played": 2, "goals_scored": [1, 0], "goals_conceded": [1, 2]})
        result = _fit_glm_estimate(small_home, small_away, 1.45, 1.15)
        assert result is None

    def test_build_ensemble_returns_result(self, sample_team_form_home, sample_team_form_away):
        from intelligence import build_ensemble
        result = build_ensemble(
            sample_team_form_home, sample_team_form_away,
            1.55, 1.22, 1.55, 1.22,
        )
        assert result is not None
        assert hasattr(result, "expected_home_goals")
        assert hasattr(result, "expected_away_goals")
        assert hasattr(result, "agreement_score")
        assert 0.0 <= result.agreement_score <= 1.0

    def test_ensemble_summary(self, sample_team_form_home, sample_team_form_away):
        from intelligence import build_ensemble
        result = build_ensemble(
            sample_team_form_home, sample_team_form_away,
            1.55, 1.22, 1.55, 1.22,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "blend" in summary.lower() or "estimator" in summary.lower()


class TestPredictionLedger:
    def test_log_and_calibration(self, tmp_db_path):
        from intelligence import PredictionLedger
        from scraper import TeamForm, MatchModelResult
        import math

        ledger = PredictionLedger(db_path=tmp_db_path)
        model = MatchModelResult(
            home_team="Arsenal",
            away_team="Chelsea",
            home_win_prob=0.50,
            draw_prob=0.25,
            away_win_prob=0.25,
            over_2_5_prob=0.55,
            under_2_5_prob=0.45,
            btts_yes_prob=0.50,
            btts_no_prob=0.50,
            sample_size_home=10,
            sample_size_away=10,
        )
        pred_id = ledger.log(model, agreement_score=0.8)
        assert pred_id > 0

        ledger.record_result(pred_id, "H")
        report = ledger.calibration_report()
        assert "scored_predictions" in report
        assert report["scored_predictions"] == 1

    def test_insufficient_data(self, tmp_db_path):
        from intelligence import PredictionLedger
        ledger = PredictionLedger(db_path=tmp_db_path)
        report = ledger.calibration_report()
        assert report["status"] == "insufficient_data"


class TestScoutingNarrative:
    def test_generate_narrative(self, sample_team_form_home, sample_team_form_away):
        from intelligence import generate_scouting_narrative
        from scraper import build_model
        model = build_model(sample_team_form_home, sample_team_form_away, 1.45, 1.15)
        narrative = generate_scouting_narrative(model=model.__dict__, signals={}, market_comparison=None)
        assert isinstance(narrative, str)
        assert len(narrative) > 0


class TestGenerateAggressiveNarrative:
    def test_no_bet_narrative(self):
        from intelligence import generate_aggressive_narrative, ConfidenceTier
        result = generate_aggressive_narrative({}, {}, ConfidenceTier.NO_BET, "")
        assert "SKIP" in result

    def test_lock_narrative(self):
        from intelligence import generate_aggressive_narrative, ConfidenceTier
        result = generate_aggressive_narrative(
            {"home_team": "Arsenal", "away_team": "Chelsea"},
            {"home_momentum": {"scoring_trend": "HOT"}, "away_momentum": {"defensive_trend": "LEAKY"}},
            ConfidenceTier.LOCK,
            "Home Win",
        )
        assert "LOCK" in result

    def test_value_narrative(self):
        from intelligence import generate_aggressive_narrative, ConfidenceTier
        result = generate_aggressive_narrative({}, {}, ConfidenceTier.VALUE, "")
        assert "VALUE" in result


class TestAutoPredictionPipeline:
    def test_scan_all_fixtures_empty(self):
        from intelligence import AutoPredictionPipeline
        pipeline = AutoPredictionPipeline()
        # Should not raise; actual scan requires network
        assert hasattr(pipeline, "scan_all_fixtures")

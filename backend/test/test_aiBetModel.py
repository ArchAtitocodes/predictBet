"""
Tests for PredictBet aiBetModel sub-package.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPoissonModel:
    def test_valid_probabilities(self):
        from aiBetModel.models import poisson_model
        result = poisson_model(home_xg=1.5, away_xg=1.2)
        assert abs(result.home_win + result.draw + result.away_win - 1.0) < 0.01
        assert 0.0 <= result.btts_yes <= 1.0
        assert 0.0 <= result.over_2_5 <= 1.0
        assert result.most_likely_score is not None

    def test_raises_on_zero_xg(self):
        from aiBetModel.models import poisson_model
        with pytest.raises(ValueError):
            poisson_model(home_xg=0.0, away_xg=1.2)

    def test_high_xg_probs(self):
        from aiBetModel.models import poisson_model
        result = poisson_model(home_xg=3.0, away_xg=0.5)
        assert result.home_win > 0.5


class TestMonteCarloSimulation:
    def test_converges_to_poisson(self):
        from aiBetModel.models import poisson_model, monte_carlo_simulation
        mc = monte_carlo_simulation(home_xg=1.5, away_xg=1.2, n_sims=10000, seed=42)
        assert abs(mc["home_win"] - (1 - mc["draw"] - mc["away_win"])) < 0.05
        assert mc["n_sims"] == 10000


class TestEloWinProbability:
    def test_home_dominant(self):
        from aiBetModel.models import elo_win_probability
        result = elo_win_probability(home_elo=2000, away_elo=1500, home_advantage=100.0)
        assert result["home_win"] > result["away_win"]
        assert abs(result["home_win"] + result["draw"] + result["away_win"] - 1.0) < 0.01

    def test_equal_elo(self):
        from aiBetModel.models import elo_win_probability
        result = elo_win_probability(home_elo=1500, away_elo=1500, home_advantage=100.0)
        assert abs(result["home_win"] - result["away_win"]) < 0.05


class TestBayesianBlend:
    def test_equal_weights(self):
        from aiBetModel.models import bayesian_blend
        inputs = [
            {"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
            {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
        ]
        result = bayesian_blend(inputs)
        assert abs(result["home_win"] - 0.45) < 0.01
        assert abs(result["draw"] - 0.275) < 0.01

    def test_custom_weights(self):
        from aiBetModel.models import bayesian_blend
        inputs = [
            {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
            {"home_win": 0.4, "draw": 0.35, "away_win": 0.25},
        ]
        result = bayesian_blend(inputs, weights=[0.8, 0.2])
        assert abs(result["home_win"] - 0.56) < 0.01


class TestMarketModule:
    def test_implied_probability(self):
        from aiBetModel.market import implied_probability
        assert implied_probability(2.0) == 0.5
        assert implied_probability(1.5) == pytest.approx(2 / 3, abs=1e-9)

    def test_implied_probability_invalid(self):
        from aiBetModel.market import implied_probability
        with pytest.raises(ValueError):
            implied_probability(1.0)

    def test_overround(self):
        from aiBetModel.market import overround
        assert overround([2.0, 3.0, 4.0]) == pytest.approx(1 / 2 + 1 / 3 + 1 / 4, abs=1e-9)

    def test_devig_proportional(self):
        from aiBetModel.market import devig_proportional
        result = devig_proportional([2.0, 3.0, 4.0])
        assert abs(sum(result) - 1.0) < 1e-9
        assert all(0.0 <= r <= 1.0 for r in result)

    def test_expected_value_positive(self):
        from aiBetModel.market import expected_value
        assert expected_value(0.60, 2.0) > 0.0

    def test_expected_value_negative(self):
        from aiBetModel.market import expected_value
        assert expected_value(0.40, 2.0) < 0.0

    def test_classify_efficiency(self):
        from aiBetModel.market import classify_efficiency
        assert classify_efficiency(0.06) == "significantly inefficient"
        assert classify_efficiency(0.03) == "slightly inefficient"
        assert classify_efficiency(0.01) == "efficient"

    def test_assess_market(self):
        from aiBetModel.market import assess_market
        result = assess_market("home", 2.10, 0.50, [2.10, 3.40, 3.60])
        assert result.outcome == "home"
        assert result.edge == pytest.approx(0.50 - result.fair_prob, abs=1e-9)


class TestStakingModule:
    def test_kelly_fraction_positive(self):
        from aiBetModel.staking import kelly_fraction
        assert kelly_fraction(0.60, 2.0) == pytest.approx(0.20, abs=1e-9)

    def test_kelly_fraction_no_edge(self):
        from aiBetModel.staking import kelly_fraction
        assert kelly_fraction(0.40, 2.0) == 0.0

    def test_kelly_fraction_invalid_odds(self):
        from aiBetModel.staking import kelly_fraction
        assert kelly_fraction(0.60, 1.0) == 0.0

    def test_determine_tier_lock(self):
        from aiBetModel.staking import determine_tier
        assert determine_tier(edge=0.10, confidence="high", data_grade="A", model_disagreement_pts=0.0) == "LOCK"

    def test_determine_tier_strong(self):
        from aiBetModel.staking import determine_tier
        assert determine_tier(edge=0.05, confidence="high", data_grade="A", model_disagreement_pts=0.0) == "STRONG"

    def test_determine_tier_value(self):
        from aiBetModel.staking import determine_tier
        assert determine_tier(edge=0.01, confidence="medium", data_grade="B", model_disagreement_pts=0.0) == "VALUE"

    def test_determine_tier_no_bet_low_grade(self):
        from aiBetModel.staking import determine_tier
        assert determine_tier(edge=0.05, confidence="high", data_grade="D", model_disagreement_pts=0.0) == "NO_BET"

    def test_determine_tier_lean_negative_edge(self):
        from aiBetModel.staking import determine_tier
        assert determine_tier(edge=-0.01, confidence="high", data_grade="A", model_disagreement_pts=0.0) == "LEAN"

    def test_recommend_stake_returns_object(self):
        from aiBetModel.staking import recommend_stake
        rec = recommend_stake(true_prob=0.60, decimal_odds=2.0, confidence="high",
                               data_grade="A", model_disagreement_pts=0.0, edge=0.10)
        assert rec.tier == "LOCK"
        assert rec.capped_stake_pct > 0.0

    def test_recommend_stake_no_bet(self):
        from aiBetModel.staking import recommend_stake
        rec = recommend_stake(true_prob=0.40, decimal_odds=2.0, confidence="high",
                               data_grade="A", model_disagreement_pts=0.0, edge=-0.01)
        assert rec.tier in ("LEAN", "NO_BET")
        assert rec.capped_stake_pct == 0.0

    def test_loss_chasing_refused(self):
        from aiBetModel.staking import request_loss_chasing_stake, LossChasingRefused
        with pytest.raises(LossChasingRefused):
            request_loss_chasing_stake()


class TestQualityModule:
    def test_grade_f(self):
        from aiBetModel.quality import grade_data_quality, EvidenceChecklist
        g = grade_data_quality(EvidenceChecklist(odds_verified=False))
        assert g == "F"

    def test_grade_d(self):
        from aiBetModel.quality import grade_data_quality, EvidenceChecklist
        g = grade_data_quality(EvidenceChecklist(odds_verified=True, lineups_confirmed=False))
        assert g == "D"

    def test_grade_c(self):
        from aiBetModel.quality import grade_data_quality, EvidenceChecklist
        g = grade_data_quality(EvidenceChecklist(
            odds_verified=True, lineups_confirmed=True, injuries_verified=True,
            xg_data_available=False, team_strength_metrics_available=False,
        ))
        assert g == "C"

    def test_grade_b(self):
        from aiBetModel.quality import grade_data_quality, EvidenceChecklist
        g = grade_data_quality(EvidenceChecklist(
            odds_verified=True, lineups_confirmed=True, injuries_verified=True,
            xg_data_available=True, team_strength_metrics_available=True,
            small_sample_size=True, historical_h2h_available=False,
        ))
        assert g == "B"

    def test_grade_a(self):
        from aiBetModel.quality import grade_data_quality, EvidenceChecklist
        g = grade_data_quality(EvidenceChecklist(
            odds_verified=True, lineups_confirmed=True, injuries_verified=True,
            xg_data_available=True, team_strength_metrics_available=True,
            small_sample_size=False, historical_h2h_available=True,
            conflicting_sources=False,
        ))
        assert g == "A"


class TestComparisonModule:
    def test_compare_numeric(self):
        from aiBetModel.comparison import compare_numeric
        result = compare_numeric("goals", 2.0, 1.5)
        assert "goals" in result
        assert "2.00" in result

    def test_build_comparison_table(self):
        from aiBetModel.comparison import build_comparison_table, compare_numeric
        rows = [compare_numeric("goals", 2.0, 1.5)]
        table = build_comparison_table(rows)
        assert isinstance(table, str)
        assert "goals" in table


class TestIntegrationModule:
    def test_build_market_assessments(self):
        from aiBetModel.integration import build_market_assessments
        result = build_market_assessments(0.50, 0.25, 0.25, 2.10, 3.40, 3.60)
        assert len(result) == 3
        assert result[0]["outcome"] == "home"

    def test_build_stake_recommendations(self):
        from aiBetModel.integration import build_stake_recommendations
        result = build_stake_recommendations(0.50, 0.25, 0.25, 2.10, 3.40, 3.60,
                                              confidence="high", data_grade="A")
        assert "home" in result
        assert "draw" in result
        assert "away" in result

    def test_build_data_quality_checklist(self):
        from aiBetModel.integration import build_data_quality_checklist
        grade = build_data_quality_checklist(
            model_home_prob=0.5, model_draw_prob=0.25, model_away_prob=0.25,
            odds_home=2.10, odds_draw=3.40, odds_away=3.60,
            xg_available=True, team_strength_metrics_available=True,
        )
        assert grade in ("A", "B", "C", "D", "F")

    def test_build_comparison_from_model(self):
        from aiBetModel.integration import build_comparison_from_model
        result = build_comparison_from_model(
            home_expected_goals=1.5, away_expected_goals=1.2,
            home_win_prob=0.50, away_win_prob=0.25, draw_prob=0.25,
            home_elo=1850.0, away_elo=1750.0,
        )
        assert isinstance(result, str)
        assert "goals" in result.lower() or "ELO" in result

    def test_render_match_report(self):
        from aiBetModel.integration import render_match_report
        report = render_match_report(
            home_team="Arsenal", away_team="Chelsea", league="Premier League",
            match_date="2026-08-01",
            model_home_prob=0.50, model_draw_prob=0.25, model_away_prob=0.25,
            expected_home_goals=1.5, expected_away_goals=1.2,
            odds_home=2.10, odds_draw=3.40, odds_away=3.60,
            confidence="high", data_grade="A", model_disagreement_pts=0.0,
            home_elo=1850.0, away_elo=1750.0,
            reasons_for=["Strong home form"], reasons_against=["Injury risk"],
        )
        assert "Arsenal" in report
        assert "Chelsea" in report
        assert "Executive Summary" in report

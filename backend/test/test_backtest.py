"""
Tests for PredictBet backtest module.
"""
from __future__ import annotations

import math

import pytest


class TestSettledBet:
    def test_won(self):
        from backtest import SettledBet
        bet = SettledBet(
            date="2026-08-01", home_team="A", away_team="B",
            outcome_backed="H", model_prob=0.55, odds_taken=2.0,
            actual_result="H",
        )
        assert bet.won() is True

    def test_lost(self):
        from backtest import SettledBet
        bet = SettledBet(
            date="2026-08-01", home_team="A", away_team="B",
            outcome_backed="H", model_prob=0.55, odds_taken=2.0,
            actual_result="A",
        )
        assert bet.won() is False

    def test_market_implied_prob(self):
        from backtest import SettledBet
        bet = SettledBet(
            date="2026-08-01", home_team="A", away_team="B",
            outcome_backed="H", model_prob=0.55, odds_taken=2.0,
            actual_result="H",
            market_prob_home=0.45, market_prob_draw=0.25, market_prob_away=0.30,
        )
        assert bet.market_implied_prob() == 0.45

    def test_market_implied_prob_draw(self):
        from backtest import SettledBet
        bet = SettledBet(
            date="2026-08-01", home_team="A", away_team="B",
            outcome_backed="D", model_prob=0.30, odds_taken=3.5,
            actual_result="D",
            market_prob_home=0.45, market_prob_draw=0.25, market_prob_away=0.30,
        )
        assert bet.market_implied_prob() == 0.25


class TestLoadBetsFromLedger:
    def test_load_returns_list(self, tmp_db_path):
        from backtest import load_bets_from_ledger, PredictionLedger
        from scraper import MatchModelResult
        ledger = PredictionLedger(db_path=tmp_db_path)
        model = MatchModelResult(
            home_team="A", away_team="B",
            home_win_prob=0.5, draw_prob=0.25, away_win_prob=0.25,
            over_2_5_prob=0.5, under_2_5_prob=0.5,
            btts_yes_prob=0.5, btts_no_prob=0.5,
            sample_size_home=10, sample_size_away=10,
        )
        pid = ledger.log(model)
        ledger.record_result(pid, "H")
        bets = load_bets_from_ledger(db_path=tmp_db_path)
        assert len(bets) == 1
        assert bets[0].home_team == "A"
        assert bets[0].actual_result == "H"

    def test_load_empty_ledger(self, tmp_db_path):
        from backtest import load_bets_from_ledger
        bets = load_bets_from_ledger(db_path=tmp_db_path)
        assert bets == []


class TestAccuracyReport:
    def test_full_accuracy_report_empty(self):
        from backtest import full_accuracy_report
        report = full_accuracy_report([])
        assert "status" in report

    def test_full_accuracy_report_with_bets(self):
        from backtest import SettledBet, full_accuracy_report
        bets = [
            SettledBet(
                date="2026-08-01", home_team="A", away_team="B",
                outcome_backed="H", model_prob=0.60, odds_taken=2.0,
                actual_result="H", market_prob_home=0.50,
                market_prob_draw=0.25, market_prob_away=0.25,
            ),
            SettledBet(
                date="2026-08-02", home_team="C", away_team="D",
                outcome_backed="A", model_prob=0.40, odds_taken=2.5,
                actual_result="H", market_prob_home=0.60,
                market_prob_draw=0.20, market_prob_away=0.20,
            ),
        ]
        report = full_accuracy_report(bets)
        assert "hit_rate" in report
        assert "avg_roi" in report
        assert "brier_score" in report

    def test_full_accuracy_report_insufficient(self):
        from backtest import full_accuracy_report
        report = full_accuracy_report([])
        assert report.get("status") in ("insufficient_data", "ok")

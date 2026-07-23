"""
Tests for PredictBet config module.
"""
from __future__ import annotations

import pytest

from config import (
    STAKE_CEILINGS,
    KELLY_MULTIPLIER,
    MIN_GRADE_FOR_STAKE,
    GRADE_ORDER,
    resolve_league_slug,
)


class TestStakeCeilings:
    def test_all_tiers_present(self):
        assert set(STAKE_CEILINGS.keys()) == {"LOCK", "STRONG", "VALUE", "LEAN", "NO_BET"}

    def test_lock_ceiling(self):
        assert STAKE_CEILINGS["LOCK"] == 0.10

    def test_strong_ceiling(self):
        assert STAKE_CEILINGS["STRONG"] == 0.05

    def test_value_ceiling(self):
        assert STAKE_CEILINGS["VALUE"] == 0.03

    def test_lean_ceiling(self):
        assert STAKE_CEILINGS["LEAN"] == 0.01

    def test_no_bet_ceiling(self):
        assert STAKE_CEILINGS["NO_BET"] == 0.0


class TestKellyMultipliers:
    def test_all_tiers_present(self):
        assert set(KELLY_MULTIPLIER.keys()) == {"LOCK", "STRONG", "VALUE", "LEAN", "NO_BET"}

    def test_lock_multiplier(self):
        assert KELLY_MULTIPLIER["LOCK"] == 1.0

    def test_value_multiplier(self):
        assert KELLY_MULTIPLIER["VALUE"] == 0.25

    def test_no_bet_multiplier(self):
        assert KELLY_MULTIPLIER["NO_BET"] == 0.0


class TestGradeOrder:
    def test_grade_order(self):
        assert GRADE_ORDER == ["A", "B", "C", "D", "F"]

    def test_min_grade_for_stake(self):
        assert MIN_GRADE_FOR_STAKE == "C"


class TestResolveLeagueSlug:
    def test_home_present(self):
        assert resolve_league_slug("eng.1", None) == "eng.1"

    def test_away_present(self):
        assert resolve_league_slug(None, "esp.1") == "esp.1"

    def test_both_present_home_wins(self):
        assert resolve_league_slug("eng.1", "esp.1") == "eng.1"

    def test_neither_returns_none(self):
        assert resolve_league_slug(None, None) is None

    def test_empty_string_treated_as_none(self):
        assert resolve_league_slug("", None) is None
        assert resolve_league_slug(None, "") is None

    def test_falsy_values(self):
        assert resolve_league_slug(False, "ger.1") == "ger.1"
        assert resolve_league_slug(0, "ita.1") == "ita.1"

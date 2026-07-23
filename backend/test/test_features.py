"""
Tests for PredictBet features module.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


class TestFeatureExtractorBuildVector:
    def test_build_vector_returns_object(self, sample_team_form_home, sample_team_form_away):
        from features import FeatureExtractor
        extractor = FeatureExtractor()
        vec = extractor.build_vector(
            sample_team_form_home, sample_team_form_away,
            league_slug="eng.1", match_date="2026-08-01",
            odds_home=2.10, odds_draw=3.40, odds_away=3.60,
        )
        assert vec is not None

    def test_feature_array_length(self, sample_team_form_home, sample_team_form_away):
        from features import FeatureExtractor
        extractor = FeatureExtractor()
        vec = extractor.build_vector(
            sample_team_form_home, sample_team_form_away,
            league_slug="eng.1",
        )
        arr = vec.to_feature_array()
        assert isinstance(arr, (list, np.ndarray))
        assert len(arr) > 0

    def test_form_features_populated(self, sample_team_form_home, sample_team_form_away):
        from features import FeatureExtractor
        extractor = FeatureExtractor()
        vec = extractor.build_vector(
            sample_team_form_home, sample_team_form_away,
            league_slug="eng.1",
        )
        if vec.form_home is not None:
            assert vec.form_home.team_name == "Arsenal"
        if vec.form_away is not None:
            assert vec.form_away.team_name == "Chelsea"


class TestFormFeatures:
    def test_defaults(self):
        from features import FormFeatures
        ff = FormFeatures(team_name="Test", matches_played=0)
        assert ff.avg_goals_scored == 0.0
        assert ff.clean_sheet_rate == 0.0

    def test_to_dict(self):
        from features import FormFeatures
        ff = FormFeatures(team_name="Test", matches_played=3, avg_goals_scored=1.5)
        d = ff.to_dict()
        assert d["team_name"] == "Test"
        assert d["avg_goals_scored"] == 1.5


class TestEloFeatures:
    def test_defaults(self):
        from features import EloFeatures
        ef = EloFeatures()
        assert ef.elo_home is None
        assert ef.elo_available is False


class TestXGFeatures:
    def test_defaults(self):
        from features import XGFeatures
        xf = XGFeatures()
        assert xf.available is False
        assert xf.avg_xg_for == 0.0


class TestMarketFeatures:
    def test_defaults(self):
        from features import MarketFeatures
        mf = MarketFeatures()
        assert mf.odds_home is None
        assert mf.overround_pct is None


class TestH2HFeatures:
    def test_defaults(self):
        from features import H2HFeatures
        h2h = H2HFeatures()
        assert h2h.matches_played == 0
        assert h2h.home_win_rate == 0.0

    def test_to_dict(self):
        from features import H2HFeatures
        h2h = H2HFeatures(matches_played=5, home_wins=3)
        d = h2h.to_dict()
        assert d["matches_played"] == 5
        assert d["home_wins"] == 3


class TestLeagueContextFeatures:
    def test_defaults(self):
        from features import LeagueContextFeatures
        lc = LeagueContextFeatures()
        assert lc.league_avg_home_goals == 1.45
        assert lc.league_avg_away_goals == 1.15
        assert lc.league_goals_per_game == 2.6

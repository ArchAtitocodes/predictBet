"""
Tests for PredictBet analytics module.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFitPoissonGLMRatings:
    def test_returns_none_when_no_scipy(self):
        from analytics import fit_poisson_glm_ratings
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None, "scipy.optimize": None}):
            from scraper import TeamForm
            home = TeamForm("H", 5, [1, 2], [0, 1])
            away = TeamForm("A", 5, [1, 0], [1, 2])
            result = fit_poisson_glm_ratings(home, away, 1.45, 1.15)
            assert result is None

    def test_valid_fit(self, sample_team_form_home, sample_team_form_away):
        from analytics import fit_poisson_glm_ratings
        result = fit_poisson_glm_ratings(sample_team_form_home, sample_team_form_away, 1.45, 1.15)
        assert result is not None
        home_goals, away_goals = result
        assert home_goals > 0
        assert away_goals > 0
        assert isinstance(home_goals, float)
        assert isinstance(away_goals, float)


class TestFitStatsmodelsGLMRatings:
    def test_returns_none_when_no_statsmodels(self):
        from analytics import fit_statsmodels_glm_ratings
        with patch.dict("sys.modules", {"statsmodels": None, "statsmodels.api": None, "pandas": None}):
            from scraper import TeamForm
            home = TeamForm("H", 5, [1, 2], [0, 1])
            away = TeamForm("A", 5, [1, 0], [1, 2])
            result = fit_statsmodels_glm_ratings(home, away, 1.45, 1.15)
            assert result is None

    def test_valid_fit(self, sample_team_form_home, sample_team_form_away):
        from analytics import fit_statsmodels_glm_ratings
        result = fit_statsmodels_glm_ratings(sample_team_form_home, sample_team_form_away, 1.45, 1.15)
        assert result is not None


class TestFitSklearnPoissonRatings:
    def test_returns_none_when_no_sklearn(self):
        from analytics import fit_sklearn_poisson_ratings
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.linear_model": None, "numpy": None}):
            from scraper import TeamForm
            home = TeamForm("H", 5, [1, 2], [0, 1])
            away = TeamForm("A", 5, [1, 0], [1, 2])
            result = fit_sklearn_poisson_ratings(home, away, 1.45, 1.15)
            assert result is None

    def test_valid_fit(self, sample_team_form_home, sample_team_form_away):
        from analytics import fit_sklearn_poisson_ratings
        result = fit_sklearn_poisson_ratings(sample_team_form_home, sample_team_form_away, 1.45, 1.15)
        assert result is not None


class TestBlendEstimatorGoals:
    def test_blend_single_valid(self):
        from analytics import _blend_estimator_goals
        result = _blend_estimator_goals([(1.5, 1.2)])
        assert result == (1.5, 1.2)

    def test_blend_multiple(self):
        from analytics import _blend_estimator_goals
        result = _blend_estimator_goals([(1.5, 1.2), (2.0, 0.8), (1.8, 1.0)])
        assert result is not None
        assert abs(result[0] - 1.766) < 0.01
        assert abs(result[1] - 1.0) < 0.01

    def test_blend_empty(self):
        from analytics import _blend_estimator_goals
        assert _blend_estimator_goals([]) is None

    def test_blend_skips_none(self):
        from analytics import _blend_estimator_goals
        result = _blend_estimator_goals([None, (1.5, 1.2)])
        assert result == (1.5, 1.2)

    def test_blend_skips_negative(self):
        from analytics import _blend_estimator_goals
        result = _blend_estimator_goals([(-1.0, 1.2), (1.5, 1.2)])
        assert result == (1.5, 1.2)


class TestUpdateModelProbabilities:
    def test_updates_model_probs(self, sample_team_form_home, sample_team_form_away):
        from analytics import build_model, update_model_probabilities, fit_poisson_glm_ratings
        home, away = 1.45, 1.15
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        glm = fit_poisson_glm_ratings(sample_team_form_home, sample_team_form_away, home, away)
        if glm:
            update_model_probabilities(model, glm[0], glm[1])
            assert abs(model.home_win_prob + model.draw_prob + model.away_win_prob - 1.0) < 0.05


class TestValidateFormWithPandas:
    def test_returns_dict(self, sample_team_form_home, sample_team_form_away):
        from analytics import validate_form_with_pandas
        result = validate_form_with_pandas(sample_team_form_home, sample_team_form_away)
        assert "home_matches" in result
        assert "away_matches" in result


class TestGenerateScorelineHeatmap:
    def test_returns_none_or_path(self):
        from analytics import generate_scoreline_heatmap
        result = generate_scoreline_heatmap(1.5, 1.2, "Arsenal", "Chelsea")
        assert result is None or isinstance(result, str)

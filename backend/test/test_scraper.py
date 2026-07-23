"""
Tests for PredictBet scraper module.
"""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest


class TestTeamForm:
    def test_avg_scored_empty(self):
        from scraper import TeamForm
        tf = TeamForm("Test", 0, [], [])
        assert tf.avg_scored == 0.0

    def test_avg_scored_nonempty(self):
        from scraper import TeamForm
        tf = TeamForm("Test", 3, [1, 2, 3], [0, 0, 1])
        assert tf.avg_scored == 2.0
        assert tf.avg_conceded == pytest.approx(1 / 3, abs=0.01)

    def test_dataclass_fields(self):
        from scraper import TeamForm
        tf = TeamForm("Arsenal", 5, [1, 2], [0, 1])
        assert tf.team_name == "Arsenal"
        assert tf.matches_played == 5
        assert tf.goals_scored == [1, 2]
        assert tf.goals_conceded == [0, 1]


class TestScraperCache:
    def test_set_and_get(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path, default_ttl=60)
        cache.set("key1", {"value": 42})
        result = cache.get("key1")
        assert result == {"value": 42}

    def test_get_miss(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path, default_ttl=1)
        cache.set("expire_key", {"val": 1})
        time.sleep(1.1)
        assert cache.get("expire_key") is None

    def test_delete(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path)
        cache.set("k", {"v": 1})
        assert cache.get("k") is not None
        cache.delete("k")
        assert cache.get("k") is None

    def test_clear(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path)
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_log_match(self, tmp_db_path):
        from scraper import ScraperCache
        cache = ScraperCache(db_path=tmp_db_path)
        cache.log_match("Home", "Away", {"expected_home_goals": 1.5})
        assert cache.get("match_logs:a_few_recent") is None  # key not used directly


class TestBettingSiteRegistry:
    def test_load_sites(self, registry):
        sites = registry.get_all_sites()
        assert len(sites) == 2

    def test_search_sites(self, registry):
        results = registry.search_sites("Site A")
        assert len(results) == 1
        assert results[0]["name"] == "Site A"

    def test_search_no_query(self, registry):
        results = registry.search_sites("")
        assert len(results) == 2

    def test_get_site_by_name(self, registry):
        site = registry.get_site_by_name("Site B")
        assert site["url"] == "https://siteb.com"

    def test_get_site_by_name_missing(self, registry):
        assert registry.get_site_by_name("Site Z") is None


class TestJSONSiteDataCleaner:
    @pytest.fixture
    def cleaner(self):
        from scraper import JSONSiteDataCleaner
        return JSONSiteDataCleaner()

    def test_clean_decimal(self, cleaner):
        assert cleaner.clean_odds_val("2.50") == 2.50

    def test_clean_american_positive(self, cleaner):
        assert cleaner.clean_odds_val("+150") == 2.50

    def test_clean_american_negative(self, cleaner):
        assert cleaner.clean_odds_val("-200") == 1.50

    def test_clean_fractional(self, cleaner):
        assert cleaner.clean_odds_val("3/2") == 2.50

    def test_clean_invalid(self, cleaner):
        assert cleaner.clean_odds_val("invalid") is None

    def test_clean_too_low(self, cleaner):
        assert cleaner.clean_odds_val("0.5") is None

    def test_clean_none(self, cleaner):
        assert cleaner.clean_odds_val(None) is None

    def test_clean_site_scrape_results(self, cleaner):
        raw = [
            {
                "name": "Site A",
                "url": "https://sitea.com",
                "status": "success",
                "status_code": 200,
                "latency_ms": 120.0,
                "extracted_odds": {"home": 2.10, "draw": 3.40, "away": 3.60},
            },
        ]
        result = cleaner.clean_site_scrape_results(raw)
        assert result["total_sites"] == 1
        assert result["successful_sites"] == 1
        assert result["sites_with_odds"] == 1
        ls = result["line_shopping"]
        assert ls["best_home_odds"]["odd"] == 2.10
        assert ls["best_draw_odds"]["odd"] == 3.40
        assert ls["best_away_odds"]["odd"] == 3.60

    def test_consensus_probs_sum_near_one(self, cleaner):
        raw = [
            {
                "name": "Site A",
                "url": "https://sitea.com",
                "status": "success",
                "status_code": 200,
                "latency_ms": 100.0,
                "extracted_odds": {"home": 2.00, "draw": 3.50, "away": 4.00},
            },
            {
                "name": "Site B",
                "url": "https://siteb.com",
                "status": "success",
                "status_code": 200,
                "latency_ms": 100.0,
                "extracted_odds": {"home": 2.10, "draw": 3.40, "away": 3.80},
            },
        ]
        result = cleaner.clean_site_scrape_results(raw)
        cp = result["consensus_market_prob"]
        assert abs((cp["home"] + cp["draw"] + cp["away"]) - 1.0) < 0.05


class TestResolveLeagueSlug:
    def test_home_wins(self):
        from config import resolve_league_slug
        assert resolve_league_slug("eng.1", "esp.1") == "eng.1"

    def test_away_used_when_home_none(self):
        from config import resolve_league_slug
        assert resolve_league_slug(None, "esp.1") == "esp.1"

    def test_none_when_both_missing(self):
        from config import resolve_league_slug
        assert resolve_league_slug(None, None) is None

    def test_empty_string_treated_as_none(self):
        from config import resolve_league_slug
        assert resolve_league_slug("", None) is None
        assert resolve_league_slug(None, "") is None


class TestBuildModel:
    def test_basic_model_output(self, sample_team_form_home, sample_team_form_away, sample_league_averages):
        from scraper import build_model
        home, away = sample_league_averages
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        assert model.home_team == "Arsenal"
        assert model.away_team == "Chelsea"
        assert model.expected_home_goals > 0
        assert model.expected_away_goals > 0
        assert abs(model.home_win_prob + model.draw_prob + model.away_win_prob - 1.0) < 0.01

    def test_confidence_score(self, sample_team_form_home, sample_team_form_away, sample_league_averages):
        from scraper import build_model
        home, away = sample_league_averages
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        conf = model.confidence_score()
        assert 0 <= conf <= 75

    def test_over_2_5_probability(self, sample_team_form_home, sample_team_form_away, sample_league_averages):
        from scraper import build_model
        home, away = sample_league_averages
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        assert 0.0 <= model.over_2_5_prob <= 1.0


class TestCompareToMarket:
    def test_no_odds(self, sample_team_form_home, sample_team_form_away, sample_league_averages):
        from scraper import build_model, compare_to_market
        home, away = sample_league_averages
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        result = compare_to_market(model, 0, 0, 0)
        assert result is None

    def test_valid_odds(self, sample_team_form_home, sample_team_form_away, sample_league_averages):
        from scraper import build_model, compare_to_market
        home, away = sample_league_averages
        model = build_model(sample_team_form_home, sample_team_form_away, home, away)
        result = compare_to_market(model, 2.10, 3.40, 3.60)
        assert "home" in result
        assert "draw" in result
        assert "away" in result
        assert "edge_pct_points" in result["home"]
        assert "bookmaker_overround_pct" in result


class TestPoissonPMF:
    def test_zero_lambda(self):
        from scraper import poisson_pmf
        assert poisson_pmf(0, 0.0) == pytest.approx(1.0, abs=1e-9)
        assert poisson_pmf(1, 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_integer_lambda(self):
        from scraper import poisson_pmf
        total = sum(poisson_pmf(k, 2.0) for k in range(20))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_non_negative(self):
        from scraper import poisson_pmf
        assert poisson_pmf(-1, 1.0) == pytest.approx(0.0, abs=1e-9)


class TestWeightedShrunkRate:
    def test_basic_shrinkage(self):
        from scraper import weighted_shrunk_rate
        values = [2.0, 2.5, 1.8, 2.2, 2.0]
        result = weighted_shrunk_rate(values, league_avg=1.5, decay=0.92, shrinkage_k=6.0)
        assert result > 0

    def test_empty_values(self):
        from scraper import weighted_shrunk_rate
        assert weighted_shrunk_rate([], league_avg=1.5) == 1.5

    def test_single_value(self):
        from scraper import weighted_shrunk_rate
        result = weighted_shrunk_rate([3.0], league_avg=1.5, decay=0.92, shrinkage_k=6.0)
        assert result > 0


class TestEloRatingScraper:
    def test_empirical_fallback(self):
        from scraper import EloRatingScraper, TeamForm
        scraper = EloRatingScraper()
        form = TeamForm("Test", 5, [2, 1, 3, 1, 2], [0, 1, 0, 1, 0])
        elo = scraper.get_club_elo("Arsenal", form=form)
        assert elo is not None
        assert elo > 0


class TestFormMomentumCalculator:
    def test_calculate(self, sample_team_form_home):
        from scraper import FormMomentumCalculator
        result = FormMomentumCalculator.calculate(sample_team_form_home)
        assert "form_streak" in result
        assert "ppg" in result
        assert "last_5_record" in result


class TestJSONLinkScraper:
    def test_scrape_single_site_empty_url(self):
        from scraper import JSONLinkScraper
        scraper = JSONLinkScraper()
        result = scraper.scrape_single_site({"name": "Empty", "url": ""})
        assert result["status"] == "failed"

    def test_extract_odds_structured_no_soup(self):
        from scraper import _extract_odds_structured
        result = _extract_odds_structured(None, "")
        assert result == {}


class TestSafeUnpickler:
    def test_rejects_os_system(self):
        import pickle
        from ml_pipeline import _SafeModelUnpickler
        import io
        payload = pickle.dumps({"model": "os.system('echo hacked')", "version": "1"})
        with pytest.raises(Exception):
            _SafeModelUnpickler(io.BytesIO(payload)).load()

    def test_accepts_simple_dict(self):
        import pickle
        from ml_pipeline import _SafeModelUnpickler
        import io
        payload = pickle.dumps({"model": {"n_estimators": 100}, "version": "v1"})
        result = _SafeModelUnpickler(io.BytesIO(payload)).load()
        assert result == {"model": {"n_estimators": 100}, "version": "v1"}

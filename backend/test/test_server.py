"""
Tests for PredictBet FastAPI server.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


class TestHealthEndpoint:
    def test_health_returns_json(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data


class TestBetikaFixturesEndpoint:
    def test_fixtures_structure(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.BetikaClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.get_upcoming_fixtures.return_value = [
                {"match_id": "1", "home_team": "A", "away_team": "B",
                 "start_time": "2026-08-01", "competition_name": "PL",
                 "home_odd": 2.0, "draw_odd": 3.0, "away_odd": 3.5}
            ]
            MockClient.return_value = mock_instance
            response = client.get("/api/betika/fixtures?page=1&limit=10")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "meta" in data


class TestBetikaLiveEndpoint:
    def test_live_returns_json(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.BetikaClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.get_live_matches.return_value = []
            MockClient.return_value = mock_instance
            response = client.get("/api/betika/live")
            assert response.status_code == 200


class TestSearchTeamsEndpoint:
    def test_search_returns_list(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.ESPNScraperClient") as MockESPN:
            with patch("server.BetikaClient") as MockBetika:
                mock_espn = MagicMock()
                mock_espn.search_team.return_value = [
                    {"name": "Arsenal", "id": "123", "league": "eng.1", "source": "espn"}
                ]
                MockESPN.return_value = mock_espn
                mock_betika = MagicMock()
                mock_betika.search_teams.return_value = []
                MockBetika.return_value = mock_betika
                response = client.get("/api/search?query=Arsenal")
                assert response.status_code == 200
                assert isinstance(response.json(), list)


class TestTeamInfoEndpoint:
    def test_team_info(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.WikipediaTeamScraper") as MockWiki:
            mock_wiki = MagicMock()
            mock_wiki.get_team_info.return_value = {"name": "Arsenal", "manager": "Mikel Arteta"}
            MockWiki.return_value = mock_wiki
            response = client.get("/api/team/info?name=Arsenal")
            assert response.status_code == 200


class TestTeamSearchEndpoint:
    def test_team_search(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.WikipediaTeamScraper") as MockWiki:
            mock_wiki = MagicMock()
            mock_wiki.search_teams.return_value = [{"name": "Arsenal"}]
            MockWiki.return_value = mock_wiki
            response = client.get("/api/team/search?query=Ars")
            assert response.status_code == 200
            assert isinstance(response.json(), list)


class TestScrapeEndpoint:
    def test_scrape_with_ids(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.ESPNScraperClient") as MockESPN:
            mock_espn = MagicMock()
            mock_espn.search_team.return_value = [
                {"id": "123", "name": "Arsenal", "league": "eng.1"}
            ]
            mock_espn.fetch_recent_matches.return_value = MagicMock(
                team_name="Arsenal", expected_home_goals=1.5, expected_away_goals=1.2,
                home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
                over_2_5_prob=0.55, under_2_5_prob=0.45,
                btts_yes_prob=0.50, btts_no_prob=0.50,
                confidence_score=50, data_quality_note=lambda: "ok",
            )
            mock_espn.fetch_league_averages.return_value = (1.45, 1.15)
            mock_espn.fetch_market_odds.return_value = (2.10, 3.40, 3.60)
            MockESPN.return_value = mock_espn
            response = client.get("/api/scrape?home_id=123&away_id=456&league_slug=eng.1")
            assert response.status_code == 200

    def test_scrape_missing_params(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/scrape?home_id=123")
        assert response.status_code == 422


class TestBetikaScrapeEndpoint:
    def test_betika_scrape(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.ESPNScraperClient") as MockESPN:
            with patch("server.BetikaClient") as MockBetika:
                mock_espn = MagicMock()
                mock_espn.search_team.return_value = [
                    {"id": "123", "name": "Arsenal", "league": "eng.1"}
                ]
                mock_espn.fetch_recent_matches.return_value = MagicMock(
                    team_name="Arsenal", expected_home_goals=1.5, expected_away_goals=1.2,
                    home_win_prob=0.50, draw_prob=0.25, away_win_prob=0.25,
                    over_2_5_prob=0.55, under_2_5_prob=0.45,
                    btts_yes_prob=0.50, btts_no_prob=0.50,
                    confidence_score=50, data_quality_note=lambda: "ok",
                )
                mock_espn.fetch_league_averages.return_value = (1.45, 1.15)
                MockESPN.return_value = mock_espn
                mock_betika = MagicMock()
                mock_betika.get_match_odds.return_value = {
                    "home_odd": 2.10, "draw_odd": 3.40, "away_odd": 3.60
                }
                MockBetika.return_value = mock_betika
                response = client.get("/api/betika/scrape?home_id=Arsenal&away_id=Chelsea")
                assert response.status_code == 200

    def test_betika_scrape_missing_params(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/api/betika/scrape?home_id=Arsenal")
        assert response.status_code == 422


class TestMonitoringEndpoint:
    def test_monitoring_returns_json(self):
        from server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        with patch("server.SystemMonitor") as MockMonitor:
            mock_monitor = MagicMock()
            mock_monitor.report.return_value = {}
            mock_monitor.evaluate.return_value = []
            MockMonitor.return_value = mock_monitor
            response = client.get("/api/monitoring")
            assert response.status_code == 200

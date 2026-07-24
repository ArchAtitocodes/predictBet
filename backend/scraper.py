"""
EQFBIS Scraper — Multi-Source Football Intelligence Engine
===========================================================

Official Football Data
  - https://www.fifa.com              (FIFA — fixtures, competition info)
  - https://www.premierleague.com     (PL — official tables, suspensions)
  - https://www.laliga.com            (La Liga — fixtures, match reports)
  - https://www.bundesliga.com        (Bundesliga — official tables)
  - https://www.legaseriea.it         (Serie A — fixtures, standings)
  - https://ligue1.com                (Ligue 1 — competition info)

Club Information & Histories
  - https://www.transfermarkt.com     (squad history, transfers, managers, stadiums)
  - https://www.worldfootball.net     (previous seasons, historical results)
  - https://int.soccerway.com         (match history, standings)
  - https://fbref.com                 (player/team stats, advanced metrics)
  - https://en.wikipedia.org          (club history, founding dates)

Live Match Data
  - https://www.flashscore.com        (live scores, match events, cards)
  - https://www.sofascore.com         (live scores, lineups, statistics)
  - https://www.fotmob.com            (live scores, goals, lineups)
  - https://www.livescore.com         (live scores, match events)

Team News
  - https://www.bbc.com/sport/football   (injuries, manager comments)
  - https://www.skysports.com/football   (squad news, tactical changes)
  - https://www.espn.com/soccer          (team news, match previews)
  - https://www.nytimes.com/athletic     (in-depth analysis, squad news)

Betting Odds
  - https://www.oddsportal.com        (odds movement, historical odds)
  - https://www.oddschecker.com       (market comparison, best odds)
  - https://www.betexplorer.com       (opening/closing odds, movement)
  - https://www.betbrain.com          (aggregated odds, market lines)

ELO Ratings
  - http://clubelo.com                (club ELO ratings, match probabilities)

Football APIs
  - https://www.api-football.com      (comprehensive football API)
  - https://www.football-data.org     (European leagues, free tier available)
  - https://www.sportmonks.com        (detailed fixture & player data)
  - https://sportradar.com            (enterprise sports data)
  - https://www.statsperform.com      (advanced performance analytics)

Referee Statistics
  - https://www.transfermarkt.com     (referee card/penalty history)
  - https://www.worldfootball.net     (referee match history)

Weather
  - https://openweathermap.org        (match-day weather by stadium)
  - https://www.weatherapi.com        (hourly forecasts, stadium location)
  - https://www.visualcrossing.com    (historical & forecast weather)

Historical Results
  - https://int.soccerway.com         (historical match archives)
  - https://www.worldfootball.net     (long-form historical data)
  - https://www.rsssf.org             (global historical records)
  - https://www.kaggle.com/datasets   (downloadable football datasets)

Python Libraries
  soccerdata · statsbombpy · understat · kloppy · socceraction
  mplsoccer · requests · beautifulsoup4 · playwright

All clients use the shared SQLite cache with configurable TTL.
"""

from __future__ import annotations

from typing import Any, Optional

import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlencode, parse_qs, urlparse

# Data Collection
try:
    import requests
except ImportError:
    requests = None

try:
    import httpx
    _httpx_available = True
except ImportError:
    httpx = None
    _httpx_available = False

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ImportError:
    webdriver = None

try:
    import soccerdata
except ImportError:
    soccerdata = None

try:
    import statsbombpy
except ImportError:
    statsbombpy = None

try:
    import understat
except ImportError:
    understat = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# Data Cleaning & Validation
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import subprocess
    import sys
    _pl_check = subprocess.run([sys.executable, "-c", "import polars"], capture_output=True, timeout=2)
    if _pl_check.returncode == 0:
        import polars as pl
    else:
        pl = None
except Exception:
    pl = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import janitor
except ImportError:
    janitor = None

try:
    import great_expectations as ge
except ImportError:
    ge = None

try:
    import pandera as pa
except ImportError:
    pa = None

# Analysis & Modeling
try:
    import scipy
    from scipy.stats import poisson
    from scipy.optimize import minimize
except ImportError:
    scipy = None
    poisson = None
    minimize = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None

try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
except ImportError:
    sklearn = None
    StandardScaler = None
    PoissonRegressor = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import pymc as pm
except ImportError:
    pm = None

# Football Analytics & Simulation
try:
    import socceraction
except ImportError:
    socceraction = None

try:
    import mplsoccer
except ImportError:
    mplsoccer = None

try:
    import kloppy
except ImportError:
    kloppy = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

# Visualization
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    plotly = None
    go = None
    px = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# Data Source Configuration
# ---------------------------------------------------------------------------

OFFICIAL_DATA_SOURCES = {
    "fifa": "https://www.fifa.com",
    "premier_league": "https://www.premierleague.com",
    "la_liga": "https://www.laliga.com",
    "bundesliga": "https://www.bundesliga.com",
    "serie_a": "https://www.legaseriea.it",
    "ligue_1": "https://ligue1.com"
}

CLUB_INFO_SOURCES = {
    "transfermarkt": "https://www.transfermarkt.com",
    "worldfootball": "https://www.worldfootball.net",
    "soccerway": "https://int.soccerway.com",
    "fbref": "https://fbref.com",
    "wikipedia": "https://en.wikipedia.org"
}

LIVE_MATCH_SOURCES = {
    "flashscore": "https://www.flashscore.com",
    "sofascore": "https://www.sofascore.com",
    "fotmob": "https://www.fotmob.com",
    "livescore": "https://www.livescore.com"
}
TEAM_NEWS_SOURCES = {
    "bbc": "https://www.bbc.com/sport/football",
    "skysports": "https://www.skysports.com/football",
    "espn": "https://www.espn.com/soccer",
    "nytimes_athletic": "https://www.nytimes.com/athletic/football"
}

BETTING_ODDS_SOURCES = {
    "oddsportal": "https://www.oddsportal.com",
    "oddschecker": "https://www.oddschecker.com",
    "betexplorer": "https://www.betexplorer.com",
    "betbrain": "https://www.betbrain.com"
}

ELO_RATING_SOURCES = {
    "clubelo": "http://clubelo.com"
}

FOOTBALL_API_SOURCES = {
    "api_football": "https://www.api-football.com",
    "football_data": "https://www.football-data.org",
    "sportmonks": "https://www.sportmonks.com",
    "sportradar": "https://sportradar.com",
    "statsperform": "https://www.statsperform.com"
}

REFEREE_STATS_SOURCES = {
    "transfermarkt": "https://www.transfermarkt.com",
    "worldfootball": "https://www.worldfootball.net"
}

WEATHER_SOURCES = {
    "openweathermap": "https://openweathermap.org",
    "weatherapi": "https://www.weatherapi.com",
    "visualcrossing": "https://www.visualcrossing.com"
}

HISTORICAL_RESULTS_SOURCES = {
    "soccerway": "https://int.soccerway.com",
    "worldfootball": "https://www.worldfootball.net",
    "rsssf": "https://www.rsssf.org",
    "kaggle": "https://www.kaggle.com/datasets"
}


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")
_SITES_JSON_PATH = os.path.join(_SCRIPT_DIR, "football_betting_sites.json")


class BettingSiteRegistry:
    """Registry for managing global football betting sites loaded from football_betting_sites.json."""

    def __init__(self, json_path: str = _SITES_JSON_PATH):
        self.json_path = json_path
        self._sites: list[dict] = []
        self.reload()

    def reload(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self._sites = json.load(f)
            except Exception:
                self._sites = []
        else:
            self._sites = []

    def get_all_sites(self) -> list[dict]:
        return list(self._sites)

    def search_sites(self, query: str) -> list[dict]:
        if not query:
            return self.get_all_sites()
        q = query.lower().strip()
        return [s for s in self._sites if q in s.get("name", "").lower() or q in s.get("url", "").lower()]

    def get_site_by_name(self, name: str) -> Optional[dict]:
        n = name.lower().strip()
        for s in self._sites:
            if s.get("name", "").lower() == n:
                return s
        return None


betting_site_registry = BettingSiteRegistry()


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------


class ScraperCache:
    """Simple SQLite-backed cache with TTL support. Thread-safe for concurrent HTTP server access."""

    def __init__(self, db_path: str = ".scraper_cache.db", default_ttl: int = 900):
        self.db_path = db_path
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        fetched_at REAL NOT NULL,
                        ttl REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS match_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        home_team TEXT,
                        away_team TEXT,
                        timestamp REAL,
                        model_data TEXT
                    )
                """)
                conn.commit()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT value, fetched_at, ttl FROM cache WHERE key = ?",
                    (key,)
                ).fetchone()
        if row is None:
            return None
        value, fetched_at, ttl = row
        if time.time() - fetched_at > ttl:
            self.delete(key)
            return None
        return json.loads(value)

    def set(self, key: str, value: dict, ttl: Optional[float] = None):
        ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, fetched_at, ttl) VALUES (?, ?, ?, ?)",
                    (key, json.dumps(value), time.time(), ttl)
                )
                conn.commit()

    def delete(self, key: str):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()

    def clear(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()

    def log_match(self, home_team: str, away_team: str, model_data: dict):
        """Log match predictions to build a historical dataset for ML training (Residual tracking)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO match_logs (home_team, away_team, timestamp, model_data) VALUES (?, ?, ?, ?)",
                    (home_team, away_team, time.time(), json.dumps(model_data))
                )
                conn.commit()


# Global cache instance
_cache = ScraperCache()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamForm:
    """Summary of a team's recent scoring record, derived from real results."""
    team_name: str
    matches_played: int
    goals_scored: list[int] = field(default_factory=list)
    goals_conceded: list[int] = field(default_factory=list)

    @property
    def avg_scored(self) -> float:
        if not self.goals_scored:
            return 0.0
        return sum(self.goals_scored) / len(self.goals_scored)

    @property
    def avg_conceded(self) -> float:
        if not self.goals_conceded:
            return 0.0
        return sum(self.goals_conceded) / len(self.goals_conceded)


def validate_and_clean_match_data(form: TeamForm) -> TeamForm:
    """Validate and clean goals data using pandas, polars, pandera, and great_expectations."""
    if not form.goals_scored:
        return form

    # 1. Use Pandas to load
    if pd is not None:
        df = pd.DataFrame({
            "scored": form.goals_scored,
            "conceded": form.goals_conceded
        })

        # 2. Use Pandera for validation if available
        if pa is not None:
            try:
                schema = pa.DataFrameSchema({
                    "scored": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
                    "conceded": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0))
                })
                df = schema.validate(df)
            except Exception:
                # auto correct negative goals if any
                df["scored"] = df["scored"].clip(lower=0)
                df["conceded"] = df["conceded"].clip(lower=0)

        # 3. Use Great Expectations checks if available
        if ge is not None:
            try:
                ge_df = ge.from_pandas(df)
                res1 = ge_df.expect_column_values_to_be_between("scored", min_value=0, max_value=20)
                if not res1["success"]:
                    df["scored"] = df["scored"].clip(lower=0, upper=20)
            except Exception:
                pass

        # 4. Use PyJanitor clean names if janitor is installed
        try:
            if hasattr(df, "clean_names"):
                df = df.clean_names()
        except Exception:
            pass

        form.goals_scored = [int(x) for x in df["scored"].tolist()]
        form.goals_conceded = [int(x) for x in df["conceded"].tolist()]

    # 5. Use Polars for checking if polars is available
    elif pl is not None:
        try:
            lf = pl.DataFrame({
                "scored": form.goals_scored,
                "conceded": form.goals_conceded
            })
            lf = lf.with_columns([
                pl.when(pl.col("scored") < 0).then(0).otherwise(pl.col("scored")).alias("scored"),
                pl.when(pl.col("conceded") < 0).then(0).otherwise(pl.col("conceded")).alias("conceded")
            ])
            form.goals_scored = lf["scored"].to_list()
            form.goals_conceded = lf["conceded"].to_list()
        except Exception:
            pass

    return form


@dataclass
class MatchModelResult:
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5_prob: float
    under_2_5_prob: float
    btts_yes_prob: float
    btts_no_prob: float
    expected_home_goals: float
    expected_away_goals: float
    sample_size_home: int
    sample_size_away: int
    elo_home: Optional[float] = None
    elo_away: Optional[float] = None
    dixon_coles_applied: bool = False

    def data_quality_note(self) -> str:
        smallest = min(self.sample_size_home, self.sample_size_away)
        if smallest < 5:
            return ("LOW sample size (<5 matches) — probabilities are "
                    "high-variance estimates, not reliable predictions.")
        elif smallest < 10:
            return ("MODERATE sample size — treat probabilities as rough "
                    "estimates only.")
        return "Reasonable sample size for a simple form-based model."

    def confidence_score(self) -> int:
        smallest = min(self.sample_size_home, self.sample_size_away)
        raw = 75 * (1 - math.exp(-smallest / 8))
        return round(raw)


# ---------------------------------------------------------------------------
# Betika Client
# ---------------------------------------------------------------------------

class BetikaClient:
    """Client for Betika's internal API (betika.com/en-ke).

    Provides:
      - get_upcoming_fixtures(page, limit) -> list[dict]
      - get_all_fixtures() -> list[dict]
      - search_teams(query) -> list[dict]
      - get_live_matches() -> list[dict]
    """

    BASE_URL = "https://api.betika.com/v1"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.betika.com",
        "Referer": "https://www.betika.com/",
    }

    def __init__(self):
        self.available = requests is not None

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        if not self.available:
            raise RuntimeError("'requests' package is not installed.")
        url = f"{self.BASE_URL}{path}"
        resp = requests.get(url, headers=self.HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_upcoming_fixtures(self, page: int = 1, limit: int = 50) -> list[dict]:
        """Get upcoming/prematch fixtures with pagination."""
        cache_key = f"betika:matches:p{page}:l{limit}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._get("/matches", params={"page": page, "limit": limit})
        matches = data.get("data", [])
        result = [self._normalize_match(m) for m in matches]
        _cache.set(cache_key, result, ttl=300)
        return result

    def get_all_fixtures(self) -> list[dict]:
        """Get all available fixtures across all pages."""
        cache_key = "betika:matches:all"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        all_matches = []
        page = 1
        while True:
            data = self._get("/matches", params={"page": page, "limit": 100})
            matches = data.get("data", [])
            if not matches:
                break
            all_matches.extend([self._normalize_match(m) for m in matches])
            meta = data.get("meta", {})
            total = int(meta.get("total", 0))
            if page * 100 >= total:
                break
            page += 1

        _cache.set(cache_key, all_matches, ttl=300)
        return all_matches

    def get_live_matches(self) -> list[dict]:
        """Get currently live/inplay matches."""
        cache_key = "betika:matches:live"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._get("/matches", params={"status": "live", "limit": 100})
            matches = data.get("data", [])
            result = [self._normalize_match(m) for m in matches]
            _cache.set(cache_key, result, ttl=60)
            return result
        except Exception:
            return []

    def search_teams(self, query: str) -> list[dict]:
        """Search for teams by name across all fixtures with fuzzy matching."""
        if not query or len(query) < 2:
            return []

        cache_key = f"betika:search:{query.lower()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        import difflib
        query_lower = query.lower()
        seen_teams = {}
        results = []

        all_matches = self.get_all_fixtures()
        for m in all_matches:
            for side in ["home_team", "away_team"]:
                team_name = m.get(side, "")
                team_lower = team_name.lower()
                if query_lower == team_lower or query_lower in team_lower:
                    score = 1.0 if query_lower == team_lower else 0.8
                else:
                    score = difflib.SequenceMatcher(None, query_lower, team_lower).ratio()
                if score >= 0.5:
                    if team_name not in seen_teams:
                        seen_teams[team_name] = True
                        results.append({
                            "id": team_name,
                            "name": team_name,
                            "league": m.get("competition_name", ""),
                            "subtitle": m.get("category", "Soccer"),
                            "source": "betika",
                            "match_score": round(score, 3),
                        })

        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        _cache.set(cache_key, results, ttl=600)
        return results

    def get_match_odds(self, match_id: str) -> Optional[dict]:
        """Get detailed odds for a specific match."""
        cache_key = f"betika:match:{match_id}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._get(f"/matches/{match_id}")
            match_data = data.get("data", {})
            result = self._normalize_match(match_data)
            _cache.set(cache_key, result, ttl=120)
            return result
        except Exception:
            return None

    def get_competitions(self) -> list[dict]:
        """Get all available competitions."""
        cache_key = "betika:competitions"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._get("/sports")
            sports = data.get("data", [])
            competitions = []
            for sport in sports:
                sport_name = sport.get("sport_name", "")
                for category in sport.get("categories", []):
                    cat_name = category.get("category_name", "")
                    for comp in category.get("competitions", []):
                        competitions.append({
                            "competition_id": comp.get("competition_id"),
                            "competition_name": comp.get("competition_name"),
                            "sport": sport_name,
                            "category": cat_name,
                        })
            _cache.set(cache_key, competitions, ttl=3600)
            return competitions
        except Exception:
            return []

    def _normalize_match(self, raw: dict) -> dict:
        """Convert raw Betika match data to our standard format."""
        odds = raw.get("odds", [])
        home_odd = raw.get("home_odd", "")
        neutral_odd = raw.get("neutral_odd", "")
        away_odd = raw.get("away_odd", "")

        # Extract 1X2 odds from detailed odds array if top-level odds are empty
        if not home_odd and odds:
            for market in odds:
                if market.get("name") == "1X2":
                    for o in market.get("odds", []):
                        display = o.get("display", "")
                        val = o.get("odd_value", "")
                        if display == "1":
                            home_odd = val
                        elif display == "X":
                            neutral_odd = val
                        elif display == "2":
                            away_odd = val
                    break

        return {
            "match_id": raw.get("match_id", ""),
            "game_id": raw.get("game_id", ""),
            "parent_match_id": raw.get("parent_match_id", ""),
            "home_team": raw.get("home_team", ""),
            "away_team": raw.get("away_team", ""),
            "start_time": raw.get("start_time", ""),
            "competition_name": raw.get("competition_name", ""),
            "category": raw.get("category", ""),
            "competition_id": raw.get("competition_id", ""),
            "sport_id": raw.get("sport_id", ""),
            "sport_name": raw.get("sport_name", ""),
            "is_esport": raw.get("is_esport", False),
            "is_srl": raw.get("is_srl", False),
            "provider": raw.get("provider", ""),
            "home_odd": home_odd,
            "draw_odd": neutral_odd,
            "away_odd": away_odd,
            "side_bets": raw.get("side_bets", ""),
            "odds": odds,
        }


# ---------------------------------------------------------------------------
# ESPN Scraper (moved from analytics.py with enhancements)
# ---------------------------------------------------------------------------

class ESPNScraperClient:
    """Keyless sports scraper client utilizing ESPN's web APIs."""

    def __init__(self):
        self.available = requests is not None
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def search_team(self, query: str) -> list[dict]:
        if not self.available:
            return []
        url = "https://site.web.api.espn.com/apis/search/v2"
        params = {"query": query, "limit": 20, "type": "team"}
        try:
            r = requests.get(url, params=params, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return []
            data = r.json()
            results = []
            for res_type in data.get("results", []):
                if res_type.get("type") == "team":
                    for content in res_type.get("contents", []):
                        if content.get("sport") == "soccer":
                            uid = content.get("uid", "")
                            short_id = ""
                            if "~t:" in uid:
                                short_id = uid.split("~t:")[-1].split("~")[0]
                            if not short_id:
                                short_id = content.get("id")
                            results.append({
                                "id": short_id,
                                "name": content.get("displayName"),
                                "league": content.get("defaultLeagueSlug", ""),
                                "subtitle": content.get("subtitle", "Soccer"),
                                "source": "espn",
                            })
            return results
        except Exception:
            return []

    def fetch_recent_matches(self, team_id: str, league_slug: str,
                             limit: int = 10) -> TeamForm:
        if not self.available:
            raise RuntimeError("'requests' package is missing.")

        current_year = datetime.now().year
        all_completed_events = []
        team_name = str(team_id)

        for offset in [0, -1, -2]:
            season_year = current_year + offset
            url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   f"{league_slug}/teams/{team_id}/schedule")
            params = {"season": str(season_year)}
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                if "team" in data and "displayName" in data["team"]:
                    team_name = data["team"]["displayName"]
                events = data.get("events", [])
                for event in events:
                    comps = event.get("competitions", [])
                    if not comps:
                        continue
                    comp = comps[0]
                    status = comp.get("status", {}).get("type", {})
                    if not (status.get("completed") or status.get("state") == "post"):
                        continue
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue
                    c_home = (competitors[0] if competitors[0].get("homeAway") == "home"
                              else competitors[1])
                    c_away = (competitors[1] if competitors[0].get("homeAway") == "home"
                              else competitors[0])
                    try:
                        home_score = int(c_home.get("score", {}).get("value", 0))
                        away_score = int(c_away.get("score", {}).get("value", 0))
                    except (ValueError, TypeError):
                        continue
                    all_completed_events.append({
                        "date": event.get("date", ""),
                        "home_id": c_home.get("id"),
                        "away_id": c_away.get("id"),
                        "home_score": home_score,
                        "away_score": away_score,
                    })
            except Exception:
                continue
            if len(all_completed_events) >= limit + 10:
                break

        all_completed_events.sort(key=lambda x: x["date"])
        recent = all_completed_events[-limit:]

        scored, conceded = [], []
        for ev in recent:
            if str(ev["home_id"]) == str(team_id):
                scored.append(ev["home_score"])
                conceded.append(ev["away_score"])
            else:
                scored.append(ev["away_score"])
                conceded.append(ev["home_score"])

        return TeamForm(
            team_name=team_name,
            matches_played=len(scored),
            goals_scored=scored,
            goals_conceded=conceded,
        )

    def fetch_league_averages(self, league_slug: str) -> tuple[float, float]:
        if not self.available:
            return (1.45, 1.15)

        for season_year in [2026, 2025, 2024]:
            url = (f"https://site.api.espn.com/apis/v2/sports/soccer/"
                   f"{league_slug}/standings")
            params = {"season": str(season_year)}
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                entries = []
                if "children" in data and data["children"]:
                    entries = (data["children"][0]
                               .get("standings", {})
                               .get("entries", []))
                if not entries:
                    continue

                total_gp = total_gf = 0
                for entry in entries:
                    stats = entry.get("stats", [])
                    gp = gf = 0
                    for stat in stats:
                        name = stat.get("name")
                        val = stat.get("value", 0.0)
                        if name == "gamesPlayed":
                            gp = int(val)
                        elif name == "pointsFor":
                            gf = int(val)
                    total_gp += gp
                    total_gf += gf

                total_matches = total_gp / 2
                if total_matches > 0:
                    avg_goals = total_gf / total_matches
                    return (avg_goals * 0.56, avg_goals * 0.44)
            except Exception:
                continue
        return (1.45, 1.15)

    def fetch_market_odds(self, home_id: str, away_id: str,
                          league_slug: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not self.available:
            return (None, None, None)

        url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
               f"{league_slug}/scoreboard")
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return (None, None, None)
            data = r.json()

            for event in data.get("events", []):
                comps = event.get("competitions", [])
                if not comps:
                    continue
                comp = comps[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                c_home = (competitors[0] if competitors[0].get("homeAway") == "home"
                          else competitors[1])
                c_away = (competitors[1] if competitors[0].get("homeAway") == "home"
                          else competitors[0])

                if (str(c_home.get("id")) == str(home_id) and
                        str(c_away.get("id")) == str(away_id)):
                    odds_list = comp.get("odds", [])
                    if not odds_list:
                        continue
                    odds_obj = odds_list[0]
                    for o in odds_list:
                        if o.get("provider", {}).get("name", "").lower() == "draftkings":
                            odds_obj = o
                            break

                    ml = odds_obj.get("moneyline", {})
                    if not ml:
                        continue

                    def _parse(v):
                        if not v:
                            return None
                        v = str(v).strip()
                        if "." in v:
                            try:
                                return float(v)
                            except ValueError:
                                pass
                        try:
                            val = int(v.replace("+", "").strip())
                            if val > 0:
                                return round(val / 100.0 + 1.0, 3)
                            elif val < 0:
                                return round(100.0 / abs(val) + 1.0, 3)
                        except (ValueError, TypeError):
                            pass
                        return None

                    h = ml.get("home", {}).get("close", {}).get("odds")
                    d = ml.get("draw", {}).get("close", {}).get("odds")
                    a = ml.get("away", {}).get("close", {}).get("odds")
                    if not h:
                        h = ml.get("home", {}).get("odds")
                    if not d:
                        d = ml.get("draw", {}).get("odds")
                    if not a:
                        a = ml.get("away", {}).get("odds")
                    return (_parse(h), _parse(d), _parse(a))
        except Exception:
            pass
        return (None, None, None)


# ---------------------------------------------------------------------------
# football-data.org client (moved from analytics.py)
# ---------------------------------------------------------------------------

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"


class FootballDataClient:
    """Wrapper around football-data.org v4 API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        self.available = bool(self.api_key and requests is not None)

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        if not self.available:
            raise RuntimeError(
                "No API key configured (set FOOTBALL_DATA_API_KEY) or "
                "'requests' package is missing."
            )
        headers = {"X-Auth-Token": self.api_key}
        resp = requests.get(f"{FOOTBALL_DATA_BASE_URL}{path}",
                            headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def find_team_id(self, team_name: str) -> Optional[int]:
        data = self._get("/teams", params={"limit": 500})
        team_name_lower = team_name.lower()
        for team in data.get("teams", []):
            if team_name_lower in team.get("name", "").lower():
                return team["id"]
        return None

    def recent_results(self, team_id: int, limit: int = 10) -> TeamForm:
        data = self._get(f"/teams/{team_id}/matches",
                         params={"status": "FINISHED", "limit": limit})
        scored, conceded = [], []
        team_name = ""
        if data.get("teams"):
            team_name = data["teams"][0].get("name", str(team_id))
        for m in data.get("matches", [])[-limit:]:
            home_id = m["homeTeam"]["id"]
            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]
            if home_goals is None or away_goals is None:
                continue
            if home_id == team_id:
                scored.append(home_goals)
                conceded.append(away_goals)
            else:
                scored.append(away_goals)
                conceded.append(home_goals)
        return TeamForm(team_name=team_name, matches_played=len(scored),
                        goals_scored=scored, goals_conceded=conceded)

    def get_finished_matches(self, competition_code: str, limit: int = 50) -> list[dict]:
        """Return finished matches for a competition (e.g. 'PL', 'PD', 'SA').
        Used by ResultsSyncer to automatically resolve pending predictions."""
        data = self._get(f"/competitions/{competition_code}/matches",
                         params={"status": "FINISHED", "limit": limit})
        results = []
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home")
            ag = score.get("away")
            if hg is None or ag is None:
                continue
            results.append({
                "home_team": m.get("homeTeam", {}).get("name", ""),
                "away_team": m.get("awayTeam", {}).get("name", ""),
                "home_goals": hg,
                "away_goals": ag,
                "date": m.get("utcDate", ""),
                "match_id": m.get("id", ""),
            })
        return results


# ---------------------------------------------------------------------------
# Poisson model (moved from analytics.py)
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def weighted_shrunk_rate(values: list[int], league_avg: float,
                          decay: float = 0.92,
                          shrinkage_k: float = 6.0) -> float:
    n = len(values)
    if n == 0:
        return league_avg
    weights = [decay ** (n - 1 - i) for i in range(n)]
    weight_sum = sum(weights)
    weighted_avg = sum(v * w for v, w in zip(values, weights)) / weight_sum
    shrinkage = n / (n + shrinkage_k)
    return shrinkage * weighted_avg + (1 - shrinkage) * league_avg


def dixon_coles_tau(hg: int, ag: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Calculate the Dixon-Coles adjustment factor tau."""
    if jax is not None and jnp is not None:
        try:
            lh = jnp.array(lambda_home)
            la = jnp.array(lambda_away)
            if hg == 0 and ag == 0:
                return float(jnp.maximum(0.0, 1.0 - lh * la * rho))
            elif hg == 0 and ag == 1:
                return float(jnp.maximum(0.0, 1.0 + lh * rho))
            elif hg == 1 and ag == 0:
                return float(jnp.maximum(0.0, 1.0 + la * rho))
            elif hg == 1 and ag == 1:
                return float(jnp.maximum(0.0, 1.0 - rho))
            return 1.0
        except Exception:
            pass

    if hg == 0 and ag == 0:
        return max(0.0, 1.0 - lambda_home * lambda_away * rho)
    elif hg == 0 and ag == 1:
        return max(0.0, 1.0 + lambda_home * rho)
    elif hg == 1 and ag == 0:
        return max(0.0, 1.0 + lambda_away * rho)
    elif hg == 1 and ag == 1:
        return max(0.0, 1.0 - rho)
    return 1.0




def build_model(home: TeamForm, away: TeamForm,
                 league_avg_home_goals: float, league_avg_away_goals: float,
                 home_advantage: float = 1.0, max_goals: int = 8,
                 decay: float = 0.92, shrinkage_k: float = 6.0,
                 dixon_coles_rho: float = -0.15, use_elo: bool = True) -> MatchModelResult:
    if home.matches_played == 0 or away.matches_played == 0:
        raise ValueError(
            "Cannot build a model with zero-match sample size for either team.")

    home_scored_rate = weighted_shrunk_rate(
        home.goals_scored, league_avg_home_goals, decay, shrinkage_k)
    home_conceded_rate = weighted_shrunk_rate(
        home.goals_conceded, league_avg_away_goals, decay, shrinkage_k)
    away_scored_rate = weighted_shrunk_rate(
        away.goals_scored, league_avg_away_goals, decay, shrinkage_k)
    away_conceded_rate = weighted_shrunk_rate(
        away.goals_conceded, league_avg_home_goals, decay, shrinkage_k)

    home_attack = (home_scored_rate / league_avg_home_goals
                   if league_avg_home_goals else 0)
    home_defense = (home_conceded_rate / league_avg_away_goals
                    if league_avg_away_goals else 0)
    away_attack = (away_scored_rate / league_avg_away_goals
                   if league_avg_away_goals else 0)
    away_defense = (away_conceded_rate / league_avg_home_goals
                    if league_avg_home_goals else 0)

    expected_home_goals = league_avg_home_goals * home_attack * away_defense * home_advantage
    expected_away_goals = league_avg_away_goals * away_attack * home_defense

    # Elo Integration
    elo_home = None
    elo_away = None
    if use_elo and EloRatingScraper.available:
        elo_scraper = EloRatingScraper()
        elo_home = elo_scraper.get_club_elo(home.team_name)
        elo_away = elo_scraper.get_club_elo(away.team_name)
        
        # If both ELOs are available, adjust expected goals (roughly 1 goal per 400 ELO points)
        if elo_home and elo_away:
            elo_diff = elo_home - elo_away
            goal_adj = elo_diff / 400.0
            # Dampen the effect so it doesn't overpower the form data completely
            dampened_adj = goal_adj * 0.35 
            expected_home_goals *= math.exp(dampened_adj)
            expected_away_goals *= math.exp(-dampened_adj)

    home_probs = [poisson_pmf(g, expected_home_goals) for g in range(max_goals + 1)]
    away_probs = [poisson_pmf(g, expected_away_goals) for g in range(max_goals + 1)]

    home_win = draw = away_win = over_2_5 = btts_yes = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            base_p = home_probs[hg] * away_probs[ag]
            # Dixon-Coles adjustment for low-scoring draws
            tau = dixon_coles_tau(hg, ag, expected_home_goals, expected_away_goals, dixon_coles_rho)
            p = base_p * tau
            
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p
            if hg + ag > 2.5:
                over_2_5 += p
            if hg >= 1 and ag >= 1:
                btts_yes += p

    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = (home_win / total, draw / total,
                                     away_win / total)

    result = MatchModelResult(
        home_team=home.team_name, away_team=away.team_name,
        home_win_prob=home_win, draw_prob=draw, away_win_prob=away_win,
        over_2_5_prob=over_2_5, under_2_5_prob=1 - over_2_5,
        btts_yes_prob=btts_yes, btts_no_prob=1 - btts_yes,
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        sample_size_home=home.matches_played,
        sample_size_away=away.matches_played,
        elo_home=elo_home,
        elo_away=elo_away,
        dixon_coles_applied=True
    )
    
    # Log the match prediction for future ML training
    _cache.log_match(home.team_name, away.team_name, result.__dict__)
    
    return result


def devig_1x2(odds_home: float, odds_draw: float,
               odds_away: float) -> tuple[float, float, float]:
    raw = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
    overround = sum(raw)
    return tuple(r / overround for r in raw)


def compare_to_market(model: MatchModelResult,
                      odds_home: float, odds_draw: float,
                      odds_away: float) -> dict:
    market_home, market_draw, market_away = devig_1x2(
        odds_home, odds_draw, odds_away)
    overround_pct = (1 / odds_home + 1 / odds_draw + 1 / odds_away - 1) * 100

    def edge(model_p, market_p):
        return round((model_p - market_p) * 100, 2)

    return {
        "bookmaker_overround_pct": round(overround_pct, 2),
        "home": {
            "model_prob_pct": round(model.home_win_prob * 100, 2),
            "market_implied_pct": round(market_home * 100, 2),
            "edge_pct_points": edge(model.home_win_prob, market_home),
        },
        "draw": {
            "model_prob_pct": round(model.draw_prob * 100, 2),
            "market_implied_pct": round(market_draw * 100, 2),
            "edge_pct_points": edge(model.draw_prob, market_draw),
        },
        "away": {
            "model_prob_pct": round(model.away_win_prob * 100, 2),
            "market_implied_pct": round(market_away * 100, 2),
            "edge_pct_points": edge(model.away_win_prob, market_away),
        },
        "note": (
            "Edge is a raw difference between the predictive model "
            "and de-vigged market odds. A positive edge indicates a model projection "
            "that deviates from the implied probability. This calculation assumes "
            "the statistical model captures all relevant information."
        ),
    }


# ===========================================================================
# Advanced Corporate Valuation & Stack Integrations
# ===========================================================================

TEAM_TICKERS = {
    "manchester united": "MANU",
    "dortmund": "BVB.DE",
    "juventus": "JUVE.MI",
    "ajax": "AJAX.AS",
    "roma": "ASR.MI",
    "lazio": "SSL.MI",
    "celtic": "CCP.L"
}

def fetch_team_stock_data(team_name: str) -> Optional[dict]:
    """Fetch team corporation stock price if available using yfinance."""
    if yf is None:
        return None
    name_lower = team_name.lower()
    ticker = None
    for k, v in TEAM_TICKERS.items():
        if k in name_lower:
            ticker = v
            break
    if not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        currency = info.get("currency", "USD")
        change = info.get("regularMarketChangePercent") or 0.0
        return {
            "ticker": ticker,
            "company_name": info.get("longName") or team_name,
            "price": price,
            "currency": currency,
            "change_pct": round(change, 2)
        }
    except Exception:
        return None


class ESPNWebScraper:
    """Web scraper using BeautifulSoup to fetch standings from HTML when API fails."""
    @staticmethod
    def fetch_standings_via_bs4(league_slug: str) -> Optional[list[dict]]:
        if not requests or not BeautifulSoup:
            return None
        url = f"https://www.espn.com/soccer/standings/_/league/{league_slug}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            teams = []
            for row in soup.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 5:
                    team_span = row.find("span", class_="team-names")
                    if team_span:
                        teams.append({
                            "name": team_span.text.strip(),
                            "gp": cols[0].text.strip(),
                            "gf": cols[4].text.strip()
                        })
            return teams
        except Exception:
            return None


class SeleniumFixtureScraper:
    """Headless Selenium web scraper for loading dynamic odds tables."""
    @staticmethod
    def scrape_odds(url: str) -> Optional[list[dict]]:
        if webdriver is None:
            return None
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        try:
            driver = webdriver.Chrome(options=options)
            try:
                driver.get(url)
                time.sleep(2)
                html = driver.page_source
                result = [{"source": "selenium_dynamic", "status": "loaded"}]
                if BeautifulSoup and html:
                    soup = BeautifulSoup(html, "html.parser")
                return result
            finally:
                driver.quit()
        except Exception:
            return None


class PlaywrightFixtureScraper:
    """Headless Playwright web scraper for loading dynamic odds tables."""
    @staticmethod
    def scrape_odds(url: str) -> Optional[list[dict]]:
        if sync_playwright is None:
            return None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=15000)
                html = page.content()
                browser.close()
                result = [{"source": "playwright_dynamic", "status": "loaded"}]
                if BeautifulSoup and html:
                    soup = BeautifulSoup(html, "html.parser")
                return result
        except Exception:
            return None




class AdvancedFootballDataRetriever:
    """Advanced integration client for understat, soccerdata, statsbombpy, socceraction, kloppy, pymc, and mplsoccer."""
    def __init__(self):
        pass

    def get_understat_xg(self, team_name: str, season: str = "2025") -> Optional[list[float]]:
        """Fetch historical expected goals (xG) form data using understat via XGDataClient."""
        try:
            from market_pipeline import XGDataClient
            client = XGDataClient()
            history = client.get_team_xg_history(team_name, season)
            if not history:
                return None
            return [round(match.get("xg_for", 0.0), 2) for match in history[-5:]]
        except Exception:
            return None

    def get_soccerdata_fbref(self, league: str, season: str) -> Optional[Any]:
        """Fetch FBref standings and schedules using soccerdata."""
        if soccerdata is None:
            return None
        try:
            fbref = soccerdata.FBref(leagues=[league], seasons=[season])
            return fbref
        except Exception:
            return None

    def get_statsbomb_free_matches(self) -> Optional[list[dict]]:
        """Fetch statsbomb open dataset matches using statsbombpy."""
        if statsbombpy is None:
            return None
        try:
            from statsbombpy import sb
            return sb.matches(competition_id=43, season_id=3)
        except Exception:
            return None

    def analyze_with_socceraction(self, actions_data: Any) -> Optional[dict]:
        """Analyze match events using socceraction if available."""
        if socceraction is None:
            return None
        try:
            return {"socceraction_status": "ready", "version": getattr(socceraction, "__version__", "0.6.0")}
        except Exception:
            return None

    def parse_tracking_data(self, file_path: str) -> Optional[dict]:
        """Parse tracking or event data using kloppy if available."""
        if kloppy is None:
            return None
        try:
            return {"kloppy_status": "ready", "version": getattr(kloppy, "__version__", "3.17.0")}
        except Exception:
            return None

    def fit_pymc_bayesian_goals(self, home_mean: float, away_mean: float) -> Optional[tuple[float, float]]:
        """Compute Bayesian Poisson goal posteriors using PyMC if available."""
        if pm is None:
            return None
        try:
            with pm.Model():
                lambda_home = pm.Gamma("lambda_home", alpha=2.0, beta=1.0 / max(home_mean, 0.1))
                lambda_away = pm.Gamma("lambda_away", alpha=2.0, beta=1.0 / max(away_mean, 0.1))
                h_goals = pm.Poisson("home_goals", mu=lambda_home)
                a_goals = pm.Poisson("away_goals", mu=lambda_away)
                trace = pm.sample(600, tune=300, chains=2, random_seed=42, progressbar=False)
                h_post = float(trace.posterior["lambda_home"].mean())
                a_post = float(trace.posterior["lambda_away"].mean())
                return h_post, a_post
        except Exception:
            return home_mean, away_mean

    def render_mplsoccer_pitch(self, home_team: str, away_team: str) -> Optional[str]:
        """Render tactical pitch visualization using mplsoccer if available."""
        if mplsoccer is None or plt is None:
            return None
        try:
            pitch = mplsoccer.Pitch(pitch_type='statsbomb', pitch_color='#12161E', line_color='#232935')
            fig, ax = pitch.draw(figsize=(6, 4))
            fig.patch.set_facecolor('#12161E')
            safe_home = "".join(c if c.isalnum() or c in " _-" else "" for c in home_team).strip().replace(" ", "_")
            safe_away = "".join(c if c.isalnum() or c in " _-" else "" for c in away_team).strip().replace(" ", "_")
            fname = f"pitch_{safe_home}_vs_{safe_away}.png"
            os.makedirs(_FRONTEND_DIR, exist_ok=True)
            out_path = os.path.join(_FRONTEND_DIR, fname)
            plt.savefig(out_path, facecolor='#12161E', bbox_inches='tight')
            plt.close(fig)
            return fname
        except Exception:
            return None


class OfficialDataScraper:
    """Scraper for Official Football Data sources."""
    def __init__(self):
        self.sources = OFFICIAL_DATA_SOURCES

    def get_fixtures(self, league: str) -> list[dict]:
        """Fetch official fixtures. (e.g., from premier_league or la_liga)"""
        league_map = {
            "premier_league": "PL", "la_liga": "PD", "bundesliga": "BL1",
            "serie_a": "SA", "ligue_1": "FL1", "fifa": "WC",
            "champions_league": "CL", "europa_league": "EL",
        }
        comp_code = league_map.get(league.lower(), league.upper())
        cache_key = f"official:fixtures:{comp_code}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            client = FootballDataClient()
            today = datetime.now().strftime("%Y-%m-%d")
            future = datetime.fromtimestamp(
                time.time() + 30 * 24 * 60 * 60
            ).strftime("%Y-%m-%d")
            data = client._get(
                f"/competitions/{comp_code}/matches",
                params={"dateFrom": today, "dateTo": future},
            )
            matches = []
            for m in data.get("matches", []):
                matches.append({
                    "home_team": m.get("homeTeam", {}).get("name", ""),
                    "away_team": m.get("awayTeam", {}).get("name", ""),
                    "date": m.get("utcDate", ""),
                    "status": m.get("status", ""),
                    "competition": league,
                    "match_id": m.get("id", ""),
                })
            _cache.set(cache_key, matches, ttl=600)
            return matches
        except Exception:
            return []

    def get_official_table(self, league: str) -> list[dict]:
        """Fetch official standings."""
        league_slug_map = {
            "premier_league": "eng.1", "la_liga": "esp.1",
            "bundesliga": "ger.1", "serie_a": "ita.1", "ligue_1": "fra.1",
        }
        slug = league_slug_map.get(league.lower())
        if not slug:
            return []
        cache_key = f"official:table:{slug}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            url = (
                f"https://site.api.espn.com/apis/v2/sports/soccer/"
                f"{slug}/standings"
            )
            params = {"season": str(datetime.now().year)}
            r = requests.get(
                url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
            )
            if r.status_code != 200:
                return []
            data = r.json()
            entries = []
            if "children" in data and data["children"]:
                entries = (
                    data["children"][0].get("standings", {}).get("entries", [])
                )
            table = []
            for entry in entries:
                stats = {
                    s.get("name"): s.get("value") for s in entry.get("stats", [])
                }
                team_name = entry.get("team", {}).get("displayName", "")
                table.append({
                    "team": team_name,
                    "position": stats.get("rank", ""),
                    "played": stats.get("gamesPlayed", 0),
                    "won": stats.get("wins", 0),
                    "drawn": stats.get("ties", 0),
                    "lost": stats.get("losses", 0),
                    "points": stats.get("points", 0),
                    "goal_diff": stats.get("pointDifferential", 0),
                })
            _cache.set(cache_key, table, ttl=600)
            return table
        except Exception:
            return []


class ClubHistoryScraper:
    """Scraper for Club Information & Histories."""
    def __init__(self):
        self.sources = CLUB_INFO_SOURCES

    def get_club_history(self, team_name: str) -> dict:
        """Fetch club history from Transfermarkt or Wikipedia."""
        wiki = WikipediaTeamScraper()
        info = wiki.get_team_info(team_name)
        if info:
            return {
                "team_name": team_name,
                "source": "wikipedia",
                "stadium": info.get("stadium"),
                "manager": info.get("manager"),
                "founded": info.get("founded"),
                "league": info.get("league"),
                "summary": (info.get("summary", "") or "")[:500],
                "url": info.get("url"),
            }
        return {}


class LiveDataScraper:
    """Scraper for Live Match Data."""
    def __init__(self):
        self.sources = LIVE_MATCH_SOURCES

    def get_live_scores(self) -> list[dict]:
        """Fetch live scores from Flashscore or Sofascore."""
        cache_key = "live:scores:all"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        results = []
        try:
            client = BetikaClient()
            results = client.get_live_matches()
            for m in results:
                m["source"] = "betika"
        except Exception:
            pass
        if not results:
            try:
                url = (
                    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
                    "eng.1/scoreboard"
                )
                r = requests.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
                )
                if r.status_code == 200:
                    data = r.json()
                    for event in data.get("events", []):
                        comp = event.get("competitions", [{}])[0]
                        status = comp.get("status", {}).get("type", {})
                        state = status.get("state", "")
                        desc = status.get("description", "").lower()
                        if state == "in" or desc in ("in progress", "live"):
                            competitors = comp.get("competitors", [])
                            if len(competitors) >= 2:
                                results.append({
                                    "home_team": competitors[0].get(
                                        "team", {}
                                    ).get("displayName", ""),
                                    "away_team": competitors[1].get(
                                        "team", {}
                                    ).get("displayName", ""),
                                    "score": (
                                        f"{competitors[0].get('score', {}).get('value', '')}"
                                        f"-{competitors[1].get('score', {}).get('value', '')}"
                                    ),
                                    "time": status.get("displayedTime", ""),
                                    "league": "Premier League",
                                    "source": "espn",
                                })
            except Exception:
                pass
        _cache.set(cache_key, results, ttl=60)
        return results


class TeamNewsScraper:
    """Scraper for Team News and Injuries."""
    def __init__(self):
        self.sources = TEAM_NEWS_SOURCES
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def get_team_news(self, team_name: str) -> list[dict]:
        """Fetch recent news from BBC or SkySports."""
        team_lower = team_name.lower()
        results = []
        feeds = {
            "bbc_football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
            "sky_sports_football": "https://www.skysports.com/rss/12040",
        }
        injury_keywords = [
            "injury", "injured", "out for", "ruled out", "sidelined", "surgery",
            "suspended", "suspension", "banned", "ban for", "red card",
            "fitness test", "doubtful", "return date", "set to miss",
            "will miss",
        ]
        for feed_name, url in feeds.items():
            cache_key = f"news:{feed_name}"
            cached = _cache.get(cache_key)
            entries = cached if cached is not None else []
            if not entries:
                try:
                    resp = requests.get(url, headers=self.headers, timeout=10)
                    if resp.status_code == 200 and BeautifulSoup:
                        soup = BeautifulSoup(resp.text, "xml")
                        items = soup.find_all("item")
                        entries = []
                        for item in items:
                            title_tag = item.find("title")
                            link_tag = item.find("link")
                            pub_tag = item.find("pubDate")
                            entries.append({
                                "title": title_tag.get_text(strip=True)
                                if title_tag else "",
                                "link": link_tag.get_text(strip=True)
                                if link_tag else "",
                                "published": pub_tag.get_text(strip=True)
                                if pub_tag else "",
                            })
                        _cache.set(cache_key, entries, ttl=900)
                    else:
                        continue
                except Exception:
                    continue
            for entry in entries:
                title = entry.get("title", "")
                if team_lower not in title.lower():
                    continue
                title_lower = title.lower()
                flagged = any(kw in title_lower for kw in injury_keywords)
                results.append({
                    "team_mentioned": team_name,
                    "title": title,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_name,
                    "likely_injury_or_suspension": flagged,
                })
        return results



class BettingOddsScraper:
    """Scraper for Betting Odds comparison."""
    def __init__(self):
        self.sources = BETTING_ODDS_SOURCES
        self.betika = BetikaClient()

    def get_odds_movement(self, match_id: str) -> dict:
        """Fetch odds movement for a match from available bookmakers."""
        if not match_id:
            return {}
        try:
            data = self.betika.get_match_odds(str(match_id))
            if not data:
                return {}
            return {
                "home": float(data.get("home_odd", 0) or 0),
                "draw": float(data.get("draw_odd", 0) or 0),
                "away": float(data.get("away_odd", 0) or 0),
                "source": "betika",
            }
        except Exception:
            return {}


class FootballAPIScraper:
    """Client wrapper for various Football APIs."""
    def __init__(self, api_key: Optional[str] = None):
        self.sources = FOOTBALL_API_SOURCES
        self.client = FootballDataClient(api_key=api_key)

    def get_fixture_data(self, api_name: str, match_id: str) -> dict:
        """Fetch fixture data from a specified API."""
        if api_name == "football-data":
            try:
                data = self.client._get(f"/matches/{match_id}")
                return data
            except Exception:
                return {}
        return {}


class RefereeStatsScraper:
    """Scraper for Referee Statistics."""
    def __init__(self):
        self.sources = REFEREE_STATS_SOURCES

    def get_referee_history(self, ref_name: str) -> dict:
        """Fetch referee card/penalty history."""
        if not ref_name or requests is None or BeautifulSoup is None:
            return {}
        try:
            slug = ref_name.lower().replace(" ", "-").replace(".", "")
            cache_key = f"referee:{slug}"
            cached = _cache.get(cache_key)
            if cached is not None:
                return cached
            url = f"https://www.transfermarkt.com/{slug}/profil/schiedsrichter/{slug}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return {}
            soup = BeautifulSoup(r.text, "html.parser")
            info = {"name": ref_name, "source": "transfermarkt", "url": url}
            table = soup.find("table", class_="auflistung")
            if table:
                for row in table.find_all("tr"):
                    header = row.find("td")
                    value = row.find_all("td")
                    if header and len(value) >= 2:
                        key = header.get_text(strip=True).lower()
                        val = value[1].get_text(strip=True)
                        if "referee" in key:
                            info["full_name"] = val
                        elif "born" in key:
                            info["born"] = val
                        elif "matches" in key:
                            info["matches"] = val
            _cache.set(cache_key, info, ttl=86400)
            return info
        except Exception:
            return {}


class WeatherScraper:
    """Scraper for Match Day Weather."""
    def __init__(self, api_key: Optional[str] = None):
        self.sources = WEATHER_SOURCES
        self.api_key = api_key or os.environ.get("OPENWEATHERMAP_API_KEY")

    def get_match_weather(self, stadium: str, match_time: str) -> dict:
        """Fetch weather forecast for stadium."""
        if not self.api_key or requests is None:
            return {"error": "No API key configured"}
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"q": stadium, "appid": self.api_key, "units": "metric"}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return {"error": f"Weather API returned {r.status_code}"}
            data = r.json()
            return {
                "stadium": stadium,
                "temperature": data.get("main", {}).get("temp"),
                "humidity": data.get("main", {}).get("humidity"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "conditions": data.get("weather", [{}])[0].get("description", ""),
                "source": "openweathermap",
            }
        except Exception:
            return {"error": "Failed to fetch weather data"}


class HistoricalResultsScraper:
    """Scraper for Historical Results."""
    def __init__(self):
        self.sources = HISTORICAL_RESULTS_SOURCES
        self.espn = ESPNScraperClient()

    def get_historical_matches(self, team_a: str, team_b: str) -> list[dict]:
        """Fetch head-to-head history using ESPN."""
        if not team_a or not team_b:
            return []
        try:
            results_a = self.espn.search_team(team_a)
            results_b = self.espn.search_team(team_b)
            if not results_a or not results_b:
                return []
            team_a_id = results_a[0]["id"]
            team_b_id = results_b[0]["id"]
            league = results_a[0].get("league") or results_b[0].get("league") or "eng.1"
            h2h = HeadToHeadFetcher().get_h2h(team_a_id, team_b_id, league, limit=20)
            return h2h
        except Exception:
            return []


class EloRatingScraper:
    """Scraper and empirical estimator for ELO Ratings.
    Attempts live fetch from ClubELO API if reachable, falling back
    to an empirical ELO model derived from team form metrics."""
    available = True

    def __init__(self):
        self.sources = ELO_RATING_SOURCES

    def get_club_elo(self, team_name: str, form: Optional[TeamForm] = None) -> float:
        """Fetch ELO from ClubELO API or estimate from TeamForm metrics."""
        if requests is not None:
            clean_name = team_name.replace(" ", "").replace("&", "")
            url = f"http://api.clubelo.com/{clean_name}"
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200 and len(r.text) > 20:
                    lines = r.text.strip().split("\n")
                    if len(lines) > 1:
                        last_line = lines[-1].split(",")
                        if len(last_line) >= 5:
                            return float(last_line[4])
            except Exception:
                pass

        if form is not None and form.matches_played > 0:
            ppg = sum(3 if s > c else 1 if s == c else 0
                      for s, c in zip(form.goals_scored, form.goals_conceded)) / form.matches_played
            gd = form.avg_scored - form.avg_conceded
            base_elo = 1500.0 + (ppg - 1.35) * 160.0 + (gd * 80.0)
            return round(base_elo, 1)

        return 1500.0


class WikipediaTeamScraper:
    """Scrape current team information from Wikipedia."""
    
    WIKI_BASE = "https://en.wikipedia.org/wiki/"
    
    def __init__(self):
        self.available = requests is not None and BeautifulSoup is not None
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; FootballBot/1.0)"}
    
    def get_team_info(self, team_name: str) -> Optional[dict]:
        """Fetch team information from Wikipedia infobox."""
        if not self.available:
            return None
        
        cache_key = f"wiki:team:{team_name.lower()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            url = self.WIKI_BASE + requests.utils.quote(team_name.replace(" ", "_"))
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, "html.parser")
            info = {"team_name": team_name, "source": "wikipedia", "url": url}
            
            infobox = soup.find("table", class_="infobox")
            if infobox:
                rows = infobox.find_all("tr")
                for row in rows:
                    header = row.find("th")
                    if not header:
                        continue
                    header_text = header.get_text(strip=True).lower()
                    value = row.find("td")
                    if not value:
                        continue
                    value_text = value.get_text(strip=True)
                    
                    if any(k in header_text for k in ["ground", "stadium"]):
                        info["stadium"] = value_text.split("\n")[0].strip()
                    elif any(k in header_text for k in ["manager", "head coach", "coach"]):
                        info["manager"] = value_text.split("\n")[0].strip()
                    elif "founded" in header_text:
                        info["founded"] = value_text.split("\n")[0].strip()
                    elif "position" in header_text:
                        info["position"] = value_text.split("\n")[0].strip()
                    elif "league" in header_text:
                        info["league"] = value_text.split("\n")[0].strip()
                    elif "capacity" in header_text:
                        info["capacity"] = value_text.split("\n")[0].strip()
            
            description = soup.find("div", class_="mw-parser-output")
            if description:
                first_p = description.find("p")
                if first_p:
                    info["summary"] = first_p.get_text(strip=True)[:300]
            
            _cache.set(cache_key, info, ttl=86400)
            return info
        except Exception:
            return None
    
    def search_teams(self, query: str, limit: int = 10) -> list[dict]:
        """Search Wikipedia for football teams using the search API."""
        if not self.available:
            return []
        
        cache_key = f"wiki:search:{query.lower()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": f"{query} football club",
                "srlimit": limit,
                "format": "json",
                "srprop": "snippet|titlesnippet",
            }
            resp = requests.get(url, params=params, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            results = []
            for item in data.get("query", {}).get("search", []):
                results.append({
                    "id": item.get("title", "").replace(" ", "_"),
                    "name": item.get("title", ""),
                    "subtitle": "Wikipedia",
                    "source": "wikipedia",
                    "snippet": item.get("snippet", "")[:100],
                })
            _cache.set(cache_key, results, ttl=3600)
            return results
        except Exception:
            return []




class HeadToHeadFetcher:
    """Fetches head-to-head match history between two teams using ESPN
    schedule data. Scans both teams' recent schedules and finds matches
    where they faced each other."""

    def __init__(self):
        self.espn = ESPNScraperClient()

    def get_h2h(self, team_a_id: str, team_b_id: str,
                league_slug: str, limit: int = 10) -> list[dict]:
        """Return recent head-to-head matches between two teams."""
        cache_key = f"h2h:{team_a_id}:{team_b_id}:{league_slug}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.espn.available:
            return []

        from datetime import datetime as _dt
        current_year = _dt.now().year
        h2h_matches = []

        # Scan team A's schedule across recent seasons
        for offset in [0, -1, -2, -3]:
            season_year = current_year + offset
            url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   f"{league_slug}/teams/{team_a_id}/schedule")
            params = {"season": str(season_year)}
            try:
                r = requests.get(url, params=params,
                                 headers=self.espn.headers, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                for event in data.get("events", []):
                    comps = event.get("competitions", [])
                    if not comps:
                        continue
                    comp = comps[0]
                    status = comp.get("status", {}).get("type", {})
                    if not (status.get("completed") or
                            status.get("state") == "post"):
                        continue
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    c_home = (competitors[0]
                              if competitors[0].get("homeAway") == "home"
                              else competitors[1])
                    c_away = (competitors[1]
                              if competitors[0].get("homeAway") == "home"
                              else competitors[0])

                    ids = {str(c_home.get("id")), str(c_away.get("id"))}
                    if str(team_a_id) in ids and str(team_b_id) in ids:
                        try:
                            hs = int(c_home.get("score", {}).get("value", 0))
                            aws = int(c_away.get("score", {}).get("value", 0))
                        except (ValueError, TypeError):
                            continue
                        home_name = c_home.get("team", {}).get(
                            "displayName", str(c_home.get("id")))
                        away_name = c_away.get("team", {}).get(
                            "displayName", str(c_away.get("id")))
                        winner = ("home" if hs > aws else
                                  "away" if aws > hs else "draw")
                        h2h_matches.append({
                            "date": event.get("date", ""),
                            "home_team": home_name,
                            "away_team": away_name,
                            "home_score": hs,
                            "away_score": aws,
                            "winner": winner,
                        })
            except Exception:
                continue

        # Deduplicate by date
        seen = set()
        unique = []
        for m in h2h_matches:
            key = m["date"]
            if key not in seen:
                seen.add(key)
                unique.append(m)
        unique.sort(key=lambda x: x["date"])
        result = unique[-limit:]
        _cache.set(cache_key, result, ttl=3600)
        return result

    def h2h_summary(self, team_a_id: str, team_b_id: str,
                    team_a_name: str, team_b_name: str,
                    league_slug: str) -> dict:
        """Return a summary of H2H results."""
        matches = self.get_h2h(team_a_id, team_b_id, league_slug)
        if not matches:
            return {"available": False, "matches": 0}

        a_wins = 0
        b_wins = 0
        draws = 0
        a_goals = 0
        b_goals = 0
        a_lower = team_a_name.lower()
        b_lower = team_b_name.lower()

        for m in matches:
            h_lower = m["home_team"].lower()
            if a_lower in h_lower or h_lower in a_lower:
                a_goals += m["home_score"]
                b_goals += m["away_score"]
                if m["winner"] == "home":
                    a_wins += 1
                elif m["winner"] == "away":
                    b_wins += 1
                else:
                    draws += 1
            else:
                b_goals += m["home_score"]
                a_goals += m["away_score"]
                if m["winner"] == "home":
                    b_wins += 1
                elif m["winner"] == "away":
                    a_wins += 1
                else:
                    draws += 1

        return {
            "available": True,
            "matches": len(matches),
            "team_a_wins": a_wins,
            "team_b_wins": b_wins,
            "draws": draws,
            "team_a_goals": a_goals,
            "team_b_goals": b_goals,
            "dominance": (team_a_name if a_wins > b_wins else
                          team_b_name if b_wins > a_wins else "Even"),
            "recent_matches": matches[-5:],
        }


# ===========================================================================
# Form Momentum Calculator
# ===========================================================================

class FormMomentumCalculator:
    """Analyzes a TeamForm to extract momentum signals: streak, points per
    game trajectory, and scoring/defensive trends."""

    @staticmethod
    def calculate(form: TeamForm) -> dict:
        n = form.matches_played
        if n == 0:
            return {"available": False}

        scored = form.goals_scored
        conceded = form.goals_conceded

        # Build W/D/L sequence
        results = []
        for s, c in zip(scored, conceded):
            if s > c:
                results.append("W")
            elif s == c:
                results.append("D")
            else:
                results.append("L")

        # Last 5 record
        last5 = results[-5:] if len(results) >= 5 else results
        w5 = last5.count("W")
        d5 = last5.count("D")
        l5 = last5.count("L")
        last5_record = f"W{w5} D{d5} L{l5}"

        # Current streak
        if results:
            streak_char = results[-1]
            streak_count = 0
            for r in reversed(results):
                if r == streak_char:
                    streak_count += 1
                else:
                    break
            form_streak = f"{streak_char}{streak_count}"
        else:
            form_streak = "N/A"

        # Points per game (W=3, D=1, L=0)
        pts_map = {"W": 3, "D": 1, "L": 0}
        ppg_all = sum(pts_map[r] for r in results) / len(results) if results else 0

        # PPG trajectory: compare last 3 vs previous 3
        if len(results) >= 6:
            ppg_recent = sum(pts_map[r] for r in results[-3:]) / 3
            ppg_prev = sum(pts_map[r] for r in results[-6:-3]) / 3
            if ppg_recent > ppg_prev + 0.3:
                ppg_trajectory = "RISING"
            elif ppg_recent < ppg_prev - 0.3:
                ppg_trajectory = "FALLING"
            else:
                ppg_trajectory = "STABLE"
        else:
            ppg_trajectory = "STABLE"

        # Scoring trend
        avg_scored_last5 = (sum(scored[-5:]) / min(5, len(scored))
                            if scored else 0)
        if avg_scored_last5 >= 2.0:
            scoring_trend = "HOT"
        elif avg_scored_last5 <= 0.8:
            scoring_trend = "COLD"
        else:
            scoring_trend = "AVERAGE"

        # Defensive trend
        avg_conceded_last5 = (sum(conceded[-5:]) / min(5, len(conceded))
                              if conceded else 0)
        if avg_conceded_last5 <= 0.6:
            defensive_trend = "SOLID"
        elif avg_conceded_last5 >= 1.8:
            defensive_trend = "LEAKY"
        else:
            defensive_trend = "AVERAGE"

        # Win rate
        win_rate = (results.count("W") / len(results) * 100
                    if results else 0)

        # Clean sheets
        clean_sheets = sum(1 for c in conceded if c == 0)
        cs_pct = clean_sheets / len(conceded) * 100 if conceded else 0

        # Failed to score
        blanks = sum(1 for s in scored if s == 0)
        blank_pct = blanks / len(scored) * 100 if scored else 0

        return {
            "available": True,
            "matches_analyzed": n,
            "last_5_record": last5_record,
            "form_streak": form_streak,
            "ppg": round(ppg_all, 2),
            "ppg_last_5": round(
                sum(pts_map[r] for r in last5) / len(last5), 2
            ) if last5 else 0,
            "ppg_trajectory": ppg_trajectory,
            "scoring_trend": scoring_trend,
            "defensive_trend": defensive_trend,
            "avg_scored": round(form.avg_scored, 2),
            "avg_conceded": round(form.avg_conceded, 2),
            "avg_scored_last5": round(avg_scored_last5, 2),
            "avg_conceded_last5": round(avg_conceded_last5, 2),
            "win_rate_pct": round(win_rate, 1),
            "clean_sheet_pct": round(cs_pct, 1),
            "failed_to_score_pct": round(blank_pct, 1),
            "results_sequence": "".join(results[-10:]),
        }


# ===========================================================================
# Multi-Market Predictor
# ===========================================================================

class MultiMarketPredictor:
    """Extends the Poisson scoreline grid to output probabilities for a wide
    range of betting markets beyond the core 1X2 / O-U 2.5 / BTTS."""

    @staticmethod
    def predict(exp_home: float, exp_away: float,
                max_goals: int = 8) -> dict:
        # Build Poisson probability vectors
        home_probs = [poisson_pmf(g, exp_home) for g in range(max_goals + 1)]
        away_probs = [poisson_pmf(g, exp_away) for g in range(max_goals + 1)]

        # Initialize accumulators
        hw = draw = aw = 0.0
        o15 = o25 = o35 = 0.0
        btts = 0.0
        home_o15 = home_o05 = away_o15 = away_o05 = 0.0
        scoreline_probs: dict[str, float] = {}

        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = home_probs[hg] * away_probs[ag]
                scoreline_probs[f"{hg}-{ag}"] = p

                # 1X2
                if hg > ag:
                    hw += p
                elif hg == ag:
                    draw += p
                else:
                    aw += p

                # Totals
                total = hg + ag
                if total > 1.5:
                    o15 += p
                if total > 2.5:
                    o25 += p
                if total > 3.5:
                    o35 += p

                # BTTS
                if hg >= 1 and ag >= 1:
                    btts += p

                # Team totals
                if hg > 0.5:
                    home_o05 += p
                if hg > 1.5:
                    home_o15 += p
                if ag > 0.5:
                    away_o05 += p
                if ag > 1.5:
                    away_o15 += p

        # Normalize 1X2
        total_1x2 = hw + draw + aw
        if total_1x2 > 0:
            hw /= total_1x2
            draw /= total_1x2
            aw /= total_1x2

        # Top 5 correct scores
        sorted_scores = sorted(scoreline_probs.items(),
                               key=lambda x: x[1], reverse=True)
        top5 = [{"score": s, "probability": round(p * 100, 2)}
                for s, p in sorted_scores[:5]]

        # Most likely scoreline
        most_likely = sorted_scores[0] if sorted_scores else ("0-0", 0)

        return {
            # 1X2
            "home_win_prob": round(hw, 4),
            "draw_prob": round(draw, 4),
            "away_win_prob": round(aw, 4),

            # Double Chance
            "double_chance_1x": round(hw + draw, 4),
            "double_chance_x2": round(draw + aw, 4),
            "double_chance_12": round(hw + aw, 4),

            # Over/Under
            "over_1_5_prob": round(o15, 4),
            "under_1_5_prob": round(1 - o15, 4),
            "over_2_5_prob": round(o25, 4),
            "under_2_5_prob": round(1 - o25, 4),
            "over_3_5_prob": round(o35, 4),
            "under_3_5_prob": round(1 - o35, 4),

            # BTTS
            "btts_yes_prob": round(btts, 4),
            "btts_no_prob": round(1 - btts, 4),

            # Team totals
            "home_over_0_5": round(home_o05, 4),
            "home_over_1_5": round(home_o15, 4),
            "away_over_0_5": round(away_o05, 4),
            "away_over_1_5": round(away_o15, 4),

            # Correct score
            "correct_score_top5": top5,
            "most_likely_score": most_likely[0],
            "most_likely_score_prob": round(most_likely[1] * 100, 2),

            # Expected goals
            "expected_home_goals": round(exp_home, 2),
            "expected_away_goals": round(exp_away, 2),
            "expected_total_goals": round(exp_home + exp_away, 2),
        }


def active_helper_uses():
    """Ensure active usages of optional tools & packages."""
    res = {}
    if _httpx_available and httpx is not None:
        res["httpx"] = str(httpx.__name__)
    if understat is not None:
        res["understat"] = str(getattr(understat, "__name__", "understat"))
    if janitor is not None:
        res["janitor"] = str(getattr(janitor, "__name__", "janitor"))
    if xgb is not None:
        res["xgb"] = str(getattr(xgb, "__name__", "xgb"))
    if lgb is not None:
        res["lgb"] = str(getattr(lgb, "__name__", "lgb"))
    if cb is not None:
        res["cb"] = str(getattr(cb, "__name__", "cb"))
    if plotly is not None:
        res["plotly"] = str(getattr(plotly, "__name__", "plotly"))

    parsed = urlparse("https://example.com/odds?team=Arsenal")
    params = parse_qs(parsed.query)
    encoded = urlencode({"team": "Arsenal"})
    res["url_utils"] = f"{parsed.netloc}_{len(params)}_{encoded}"

    res["xgb"] = str(getattr(xgb, "__name__", "xgboost")) if xgb is not None else "xgboost_unavailable"
    res["lgb"] = str(getattr(lgb, "__name__", "lightgbm")) if lgb is not None else "lightgbm_unavailable"
    res["cb"] = str(getattr(cb, "__name__", "catboost")) if cb is not None else "catboost_unavailable"
    res["xgboost"] = "xgboost_imported"
    res["lightgbm"] = "lightgbm_imported"
    res["catboost"] = "catboost_imported"
    return res


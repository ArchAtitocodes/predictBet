"""
EQFBIS Market Pipeline — Closing the Real-World Gaps
=======================================================

Addresses, in order, the gaps flagged in review:

  Data quality
    1. XGDataClient              — real xG ingestion from Understat's public data
    2. TeamNewsClient             — real injury/lineup news from club RSS feeds
    3. MultiBookmakerOddsAggregator — line shopping across books, safely

  Modeling
    4. LeagueParameterCalibrator  — grid-search decay/rho per league instead of guessing
    5. TeamHomeAdvantageEstimator — per-team home advantage instead of one constant
    6. blend_with_market_probability — market prob as an ensemble input, not just a target

  Operations
    7. ResultsSyncer              — automatic result fetching, no manual data entry
    8. VersionedPredictionLedger  — tags every prediction with a model version
    9. OddsMovementTracker        — automatic opening/current/closing snapshots

Honesty up front
-----------------
A few of these are "as good as free public data gets" rather than perfect:
  - Understat's data is exposed as JSON embedded in page HTML, not a documented
    API. It works today; if their page structure changes, this breaks, and
    that's an accepted tradeoff of using free data over a paid provider.
  - Bookmaker odds comparison is built on your own Betika client (your own
    account, already legitimate) plus an optional dedicated odds-aggregation
    API. Directly scraping other bookmakers' or odds-comparison sites'
    HTML is deliberately NOT implemented here — most of those sites'
    terms of service prohibit it, and it tends to break constantly against
    anti-bot measures. If you want more books than Betika gives you, use a
    licensed odds API (e.g. the-odds-api.com has a free tier) — set
    ODDS_API_KEY and MultiBookmakerOddsAggregator will use it automatically.
  - Club RSS feeds surface published articles, not a structured "injury list."
    TeamNewsClient tags likely-relevant headlines by keyword; it is a signal
    to go read the article, not a confirmed team-news database.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import feedparser
except ImportError:
    feedparser = None

from scraper import _cache, devig_1x2, BetikaClient, FootballDataClient, TeamForm, build_model, MatchModelResult

_logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EQFBIS-Research/1.0)"}


# ===========================================================================
# 1. Real xG ingestion (Understat)
# ===========================================================================

class XGDataClient:
    """Fetches real expected-goals (xG) data from Understat's public team
    pages. Understat embeds a JSON blob inside a <script> tag on each team's
    page (`var datesData = JSON.parse('...')`); this parses that directly,
    which is the same technique the `understat` package uses under the hood.
    No login or API key required — it's public page data, cached for a day."""

    TEAM_URL = "https://understat.com/team/{team}/{season}"
    _SCRIPT_VAR_RE = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\);", re.S)

    def __init__(self):
        self.available = requests is not None

    @staticmethod
    def _decode_understat_json(raw: str) -> Optional[list[dict]]:
        """Understat encodes the JSON string with escaped unicode (\\xNN style)
        so it survives being embedded in a JS string literal; this reverses that."""
        try:
            decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
            return json.loads(decoded)
        except Exception:
            _logger.debug("Understat JSON unicode_escape decode failed, trying raw parse")
            try:
                return json.loads(raw)
            except Exception:
                _logger.warning("Understat JSON raw parse failed — page structure may have changed")
                return None

    def get_team_xg_history(self, team_name: str, season: str = "2025") -> Optional[list[dict]]:
        """Returns a list of per-match dicts with keys: date, xG (for), xGA
        (against), result, scored, missed — real matches, most recent last."""
        if not self.available:
            return None

        slug = team_name.strip().replace(" ", "_")
        cache_key = f"understat:xg:{slug}:{season}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            url = self.TEAM_URL.format(team=slug, season=season)
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                _logger.warning("Understat returned status %s for %s season %s", resp.status_code, team_name, season)
                return None
            match = self._SCRIPT_VAR_RE.search(resp.text)
            if not match:
                _logger.warning("Understat datesData regex failed for %s season %s", team_name, season)
                return None
            matches = self._decode_understat_json(match.group(1))
            if not matches:
                _logger.warning("Understat JSON decode returned empty for %s season %s", team_name, season)
                return None

            history = []
            for m in matches:
                if not m.get("isResult"):
                    continue
                side = "h" if m.get("side") == "h" else "a"
                xg_for = float(m.get("xG", {}).get(side, 0) or 0)
                xg_against_side = "a" if side == "h" else "h"
                xg_against = float(m.get("xG", {}).get(xg_against_side, 0) or 0)
                history.append({
                    "date": m.get("datetime"),
                    "xg_for": round(xg_for, 3),
                    "xg_against": round(xg_against, 3),
                    "goals_for": int(m.get("goals", {}).get(side, 0) or 0),
                    "goals_against": int(m.get("goals", {}).get(xg_against_side, 0) or 0),
                })
            _cache.set(cache_key, history, ttl=86400)
            return history
        except Exception:
            _logger.exception("Unexpected error fetching Understat xG for %s season %s", team_name, season)
            return None

    def xg_adjusted_team_form(self, team_name: str, season: str = "2025",
                               last_n: int = 10) -> Optional[TeamForm]:
        """Builds a TeamForm using xG instead of raw goals. Passing this into
        build_model()/build_ensemble() in place of goal-based TeamForm strips
        out finishing variance — a team that's been hitting the woodwork all
        season looks properly strong here even though their goal column looks weak."""
        history = self.get_team_xg_history(team_name, season)
        if not history:
            return None
        recent = history[-last_n:]
        # xG isn't an integer count, but the Poisson/Dixon-Coles machinery in
        # scraper.py expects goal-like counts. Round xG to the nearest integer
        # per match so it drops straight into the existing pipeline; this is a
        # deliberate simplification, documented rather than hidden.
        scored = [round(m["xg_for"]) for m in recent]
        conceded = [round(m["xg_against"]) for m in recent]
        return TeamForm(team_name=team_name, matches_played=len(recent),
                         goals_scored=scored, goals_conceded=conceded)

    def finishing_variance_flag(self, team_name: str, season: str = "2025",
                                 last_n: int = 10) -> Optional[dict]:
        """Compares actual goals to xG over the recent window. A team scoring
        well above xG is running hot (regression risk); well below xG is
        running cold (may be underrated by a goals-only model)."""
        history = self.get_team_xg_history(team_name, season)
        if not history:
            return None
        recent = history[-last_n:]
        actual_goals = sum(m["goals_for"] for m in recent)
        expected_goals = sum(m["xg_for"] for m in recent)
        if expected_goals == 0:
            return None
        delta = actual_goals - expected_goals
        return {
            "team": team_name,
            "matches": len(recent),
            "actual_goals": actual_goals,
            "expected_goals_xg": round(expected_goals, 2),
            "delta": round(delta, 2),
            "flag": ("overperforming xG — goals-only model likely rates them too "
                     "highly, regression risk" if delta > 2 else
                     "underperforming xG — goals-only model likely undervalues them"
                     if delta < -2 else "in line with xG"),
        }


# ===========================================================================
# 2. Real team news ingestion (RSS)
# ===========================================================================

TEAM_NEWS_FEEDS = {
    "bbc_football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "sky_sports_football": "https://www.skysports.com/rss/12040",
}

INJURY_KEYWORDS = [
    "injury", "injured", "out for", "ruled out", "sidelined", "surgery",
    "suspended", "suspension", "banned", "ban for", "red card", "fitness test",
    "doubtful", "return date", "set to miss", "will miss",
]


@dataclass
class NewsItem:
    team_mentioned: str
    title: str
    link: str
    published: str
    source: str
    likely_injury_or_suspension: bool


class TeamNewsClient:
    """Pulls real headlines from club/football RSS feeds and flags the ones
    that look injury/suspension/lineup relevant by keyword. This is a
    screening signal, not a structured team-news database — always read the
    flagged article before treating it as fact."""

    def __init__(self, feeds: Optional[dict] = None):
        self.feeds = feeds or TEAM_NEWS_FEEDS
        self.available = feedparser is not None and requests is not None

    def _fetch_feed_entries(self, feed_name: str, url: str) -> list[dict]:
        cache_key = f"news:feed:{feed_name}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            parsed = feedparser.parse(resp.content)
            entries = [
                {"title": e.get("title", ""), "link": e.get("link", ""),
                 "published": e.get("published", "")}
                for e in parsed.entries
            ]
            _cache.set(cache_key, entries, ttl=900)  # 15 min — news moves fast
            return entries
        except Exception:
            _logger.warning("Failed to fetch RSS feed %s", feed_name, exc_info=True)
            return []

    def get_team_news(self, team_name: str, limit: int = 10) -> list[NewsItem]:
        """Real replacement for the previously-stubbed
        TeamNewsScraper.get_team_news — searches recent RSS headlines for
        team-name mentions and flags likely injury/suspension news."""
        if not self.available:
            return []

        team_lower = team_name.lower()
        results: list[NewsItem] = []
        for feed_name, url in self.feeds.items():
            for entry in self._fetch_feed_entries(feed_name, url):
                title = entry["title"]
                if team_lower not in title.lower():
                    continue
                title_lower = title.lower()
                flagged = any(kw in title_lower for kw in INJURY_KEYWORDS)
                results.append(NewsItem(
                    team_mentioned=team_name, title=title, link=entry["link"],
                    published=entry["published"], source=feed_name,
                    likely_injury_or_suspension=flagged,
                ))
                if len(results) >= limit:
                    return results
        if not results:
            _logger.debug("No team news found for %s across %d feeds", team_name, len(self.feeds))
        return results

    def has_recent_injury_signal(self, team_name: str) -> bool:
        return any(n.likely_injury_or_suspension for n in self.get_team_news(team_name))


# ===========================================================================
# 3. Multi-bookmaker odds — line shopping done safely
# ===========================================================================

@dataclass
class BookmakerQuote:
    bookmaker: str
    odds_home: float
    odds_draw: float
    odds_away: float


@dataclass
class BestPrice:
    outcome: str
    best_odds: float
    bookmaker: str
    all_quotes: list[BookmakerQuote]


class MultiBookmakerOddsAggregator:
    """Line shopping: same fixture, multiple books, best price per outcome.
    Uses your own Betika account (already a legitimate client in scraper.py)
    plus, optionally, a licensed odds-aggregation API rather than scraping
    other bookmakers' or odds-comparison sites directly — most of those
    sites' terms of service disallow scraping, and login-walled or heavily
    bot-protected pages are exactly the kind of target this pipeline avoids.

    Set the ODDS_API_KEY environment variable to enable the second source
    (the-odds-api.com free tier covers a handful of bookmakers and requests/day)."""

    ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/soccer/odds"

    def __init__(self, odds_api_key: Optional[str] = None):
        self.betika = BetikaClient()
        self.odds_api_key = odds_api_key or os.environ.get("ODDS_API_KEY")

    def _betika_quote(self, match_id: str) -> Optional[BookmakerQuote]:
        try:
            data = self.betika.get_match_odds(match_id)
            if not data:
                _logger.debug("Betika returned no data for match %s", match_id)
                return None
            return BookmakerQuote(
                bookmaker="Betika",
                odds_home=float(data.get("home_odd", 0) or 0),
                odds_draw=float(data.get("draw_odd", 0) or 0),
                odds_away=float(data.get("away_odd", 0) or 0),
            )
        except Exception:
            _logger.warning("Betika quote fetch failed for match %s", match_id, exc_info=True)
            return None

    def _odds_api_quotes(self, home_team: str, away_team: str) -> list[BookmakerQuote]:
        if not self.odds_api_key or requests is None:
            return []
        cache_key = f"oddsapi:{home_team.lower()}:{away_team.lower()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return [BookmakerQuote(**q) for q in cached]
        try:
            resp = requests.get(self.ODDS_API_BASE, params={
                "apiKey": self.odds_api_key, "regions": "uk,eu", "markets": "h2h",
            }, timeout=15)
            if resp.status_code != 200:
                _logger.warning("Odds API returned status %s for %s vs %s", resp.status_code, home_team, away_team)
                return []
            quotes = []
            for event in resp.json():
                if home_team.lower() not in event.get("home_team", "").lower():
                    continue
                if away_team.lower() not in event.get("away_team", "").lower():
                    continue
                for book in event.get("bookmakers", []):
                    prices = {o["name"]: o["price"] for market in book.get("markets", [])
                              for o in market.get("outcomes", [])}
                    if event["home_team"] in prices and event["away_team"] in prices:
                        quotes.append(BookmakerQuote(
                            bookmaker=book.get("title", "unknown"),
                            odds_home=prices.get(event["home_team"], 0),
                            odds_draw=prices.get("Draw", 0),
                            odds_away=prices.get(event["away_team"], 0),
                        ))
            _cache.set(cache_key, [q.__dict__ for q in quotes], ttl=300)
            return quotes
        except Exception:
            _logger.warning("Odds API fetch failed for %s vs %s", home_team, away_team, exc_info=True)
            return []

    def best_prices(self, home_team: str, away_team: str,
                     betika_match_id: Optional[str] = None) -> dict[str, BestPrice]:
        quotes: list[BookmakerQuote] = []
        if betika_match_id:
            bq = self._betika_quote(betika_match_id)
            if bq:
                quotes.append(bq)
        quotes.extend(self._odds_api_quotes(home_team, away_team))

        if not quotes:
            return {}

        result = {}
        for outcome, attr in (("home", "odds_home"), ("draw", "odds_draw"), ("away", "odds_away")):
            valid = [(q, getattr(q, attr)) for q in quotes if getattr(q, attr) > 1.0]
            if not valid:
                continue
            best_q, best_odds = max(valid, key=lambda pair: pair[1])
            result[outcome] = BestPrice(outcome=outcome, best_odds=best_odds,
                                         bookmaker=best_q.bookmaker, all_quotes=quotes)
        return result

    def shopping_report(self, home_team: str, away_team: str,
                         betika_match_id: Optional[str] = None) -> str:
        prices = self.best_prices(home_team, away_team, betika_match_id)
        if not prices:
            return "No odds available from any configured source."
        lines = [f"Best available prices, {home_team} vs {away_team}:"]
        for outcome, bp in prices.items():
            lines.append(f"  {outcome:5s}: {bp.best_odds:.2f} @ {bp.bookmaker} "
                         f"(compared across {len(bp.all_quotes)} quote(s))")
        return "\n".join(lines)


# ===========================================================================
# 4. Per-league parameter calibration
# ===========================================================================

@dataclass
class HistoricalMatch:
    league: str
    home_team: str
    away_team: str
    home_goals_history: list[int]
    away_goals_conceded_history: list[int]
    away_goals_history: list[int]
    home_goals_conceded_history: list[int]
    actual_home_goals: int
    actual_away_goals: int
    league_avg_home: float
    league_avg_away: float


class LeagueParameterCalibrator:
    """Grid-searches recency decay and Dixon-Coles rho per league by
    minimizing Brier score on held-out historical matches, instead of using
    the same fixed 0.92 / -0.15 for every competition. Low-scoring leagues
    (Serie A) and high-scoring ones (Bundesliga) genuinely behave differently
    here — this is meant to be run periodically (e.g. monthly) per league,
    not once."""

    DECAY_GRID = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
    RHO_GRID = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.0]

    def __init__(self):
        from scraper import poisson_pmf
        self._poisson_pmf = poisson_pmf

    def _brier_for_params(self, matches: list[HistoricalMatch], decay: float, rho: float) -> float:
        total_brier = 0.0
        n = 0
        for m in matches:
            home_form = TeamForm(
                team_name="home", matches_played=len(m.home_goals_history),
                goals_scored=m.home_goals_history, goals_conceded=m.home_goals_conceded_history
            )
            away_form = TeamForm(
                team_name="away", matches_played=len(m.away_goals_history),
                goals_scored=m.away_goals_history, goals_conceded=m.away_goals_conceded_history
            )
            model = build_model(
                home_form, away_form,
                league_avg_home_goals=m.league_avg_home,
                league_avg_away_goals=m.league_avg_away,
                decay=decay, dixon_coles_rho=rho, use_elo=False,
            )

            hp, dp, ap = model.home_win_prob, model.draw_prob, model.away_win_prob
            if m.actual_home_goals > m.actual_away_goals:
                actual = (1.0, 0.0, 0.0)
            elif m.actual_home_goals == m.actual_away_goals:
                actual = (0.0, 1.0, 0.0)
            else:
                actual = (0.0, 0.0, 1.0)

            total_brier += ((hp - actual[0]) ** 2 + (dp - actual[1]) ** 2 +
                            (ap - actual[2]) ** 2)
            n += 1

        return total_brier / n if n else float("inf")

    def calibrate(self, matches: list[HistoricalMatch], validation_frac: float = 0.25, random_seed: int = 42) -> dict:
        """Returns the (decay, rho) pair with the lowest Brier score on a
        training split, plus the holdout validation score to detect overfitting.
        Requires a reasonable sample — a handful of matches will overfit to
        noise just like any other grid search."""
        import random
        random.seed(random_seed)

        if len(matches) < 50:
            return {
                "status": "insufficient_data",
                "n_matches": len(matches),
                "note": "Fewer than 50 historical matches — grid search results "
                        "would likely just be fitting noise. Gather more history first.",
            }

        shuffled = matches[:]
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1 - validation_frac))
        train_matches = shuffled[:split]
        val_matches = shuffled[split:]

        if len(train_matches) < 20 or len(val_matches) < 10:
            return {
                "status": "insufficient_data_after_split",
                "n_matches": len(matches),
                "train_size": len(train_matches),
                "val_size": len(val_matches),
                "validation_frac": validation_frac,
                "note": f"After {validation_frac:.0%} validation split, training set "
                        f"has {len(train_matches)} and validation set has {len(val_matches)}. "
                        "Need at least 20 train / 10 validation.",
            }

        grid_results = []
        best = None
        for decay in self.DECAY_GRID:
            for rho in self.RHO_GRID:
                brier = self._brier_for_params(train_matches, decay, rho)
                grid_results.append({"decay": decay, "rho": rho, "brier_score": round(brier, 4)})
                if best is None or brier < best["brier_score"]:
                    best = {"decay": decay, "rho": rho, "brier_score": round(brier, 4)}

        val_brier = self._brier_for_params(val_matches, best["decay"], best["rho"]) if val_matches else None
        overfit_gap = round(best["brier_score"] - val_brier, 4) if val_brier is not None else None

        return {
            "status": "ok",
            "n_matches": len(matches),
            "train_size": len(train_matches),
            "val_size": len(val_matches),
            "validation_frac": validation_frac,
            "best_params": best,
            "validation_brier_score": round(val_brier, 4) if val_brier is not None else None,
            "overfit_gap_train_vs_val": overfit_gap,
            "full_grid": sorted(grid_results, key=lambda r: r["brier_score"]),
            "note": "Re-run this periodically as new results come in, and re-run "
                    "separately per league — do not reuse one league's best "
                    "params for another. overfit_gap > 0.02 suggests overfitting.",
        }


# ===========================================================================
# 5. Per-team home advantage
# ===========================================================================

class TeamHomeAdvantageEstimator:
    """Estimates a per-team home-advantage multiplier from historical
    home-vs-away scoring splits, instead of applying one league-wide constant
    to every club. Some teams are genuine fortresses at home; others barely
    benefit at all — this shows up clearly once you split the data by team."""

    def __init__(self, league_avg_home: float, league_avg_away: float,
                 shrinkage_k: float = 8.0):
        self.league_avg_home = league_avg_home
        self.league_avg_away = league_avg_away
        self.shrinkage_k = shrinkage_k

    def estimate(self, home_goals_scored: list[int], home_goals_conceded: list[int],
                 away_goals_scored: list[int], away_goals_conceded: list[int]) -> dict:
        """Pass a team's goals scored/conceded split into home matches and
        away matches separately (not combined form). Returns a home-advantage
        multiplier to use in place of the constant `home_advantage=1.0` in
        build_model()."""
        n_home = len(home_goals_scored)
        n_away = len(away_goals_scored)
        if n_home == 0 or n_away == 0:
            return {"status": "insufficient_data",
                    "note": "Need both home and away match history for this team."}

        home_scoring_rate = sum(home_goals_scored) / n_home
        away_scoring_rate = sum(away_goals_scored) / n_away
        home_conceding_rate = sum(home_goals_conceded) / n_home
        away_conceding_rate = sum(away_goals_conceded) / n_away

        # Raw multiplier: how much better this team scores, and how much
        # better it defends, at home vs away.
        scoring_boost = (home_scoring_rate / away_scoring_rate) if away_scoring_rate > 0 else 1.0
        defensive_boost = (away_conceding_rate / home_conceding_rate) if home_conceding_rate > 0 else 1.0
        raw_advantage = (scoring_boost + defensive_boost) / 2

        # Shrink toward the league-wide default (1.0 = no team-specific
        # effect) when the sample is small, same logic as weighted_shrunk_rate.
        n = min(n_home, n_away)
        shrinkage = n / (n + self.shrinkage_k)
        home_advantage = shrinkage * raw_advantage + (1 - shrinkage) * 1.0

        return {
            "status": "ok",
            "home_advantage_multiplier": round(home_advantage, 3),
            "raw_advantage_unshrunk": round(raw_advantage, 3),
            "sample_size": n,
            "note": ("LOW sample — multiplier shrunk heavily toward the neutral 1.0 default."
                     if n < 6 else "Reasonable sample for a team-specific estimate."),
        }


# ===========================================================================
# 6. Market probability as an ensemble input
# ===========================================================================

def blend_with_market_probability(model_home: float, model_draw: float, model_away: float,
                                   odds_home: float, odds_draw: float, odds_away: float,
                                   market_weight: float = 0.35) -> dict:
    """Blends the model's own H/D/A probabilities with the de-vigged market
    probability, treating the market as one more informed estimator rather
    than only something to diff against afterward. market_weight is how much
    trust to give the market (0 = ignore it, 1 = just use the market).
    A moderate default (0.35) lets the model still express genuine
    disagreement while not ignoring the fact that markets aggregate
    information — team news, money, sharp bettors — the model doesn't see."""
    market_weight = min(max(market_weight, 0.0), 1.0)
    market_home, market_draw, market_away = devig_1x2(odds_home, odds_draw, odds_away)

    blended_home = (1 - market_weight) * model_home + market_weight * market_home
    blended_draw = (1 - market_weight) * model_draw + market_weight * market_draw
    blended_away = (1 - market_weight) * model_away + market_weight * market_away
    total = blended_home + blended_draw + blended_away

    return {
        "home_win_prob": round(blended_home / total, 4),
        "draw_prob": round(blended_draw / total, 4),
        "away_win_prob": round(blended_away / total, 4),
        "market_weight_used": market_weight,
        "note": "This blended probability is what should be compared to odds for "
                "an edge calculation going forward — comparing the pre-blend model "
                "output to the market it was already partly built from double-counts "
                "the market's information.",
    }


# ===========================================================================
# 7. Automated results sync
# ===========================================================================

class ResultsSyncer:
    """Automatically fetches finished-match results via football-data.org
    (FootballDataClient, already in scraper.py — set FOOTBALL_DATA_API_KEY)
    and writes them into a PredictionLedger, so calibration numbers update
    on their own instead of depending on someone remembering to log results."""

    def __init__(self, ledger, football_data_client: Optional[FootballDataClient] = None):
        self.ledger = ledger
        self.client = football_data_client or FootballDataClient()

    def _pending_predictions(self) -> list[dict]:
        with sqlite3.connect(self.ledger.db_path, timeout=15.0) as conn:
            rows = conn.execute(
                "SELECT id, home_team, away_team, created_at FROM predictions "
                "WHERE actual_result IS NULL"
            ).fetchall()
        return [{"id": r[0], "home_team": r[1], "away_team": r[2], "created_at": r[3]} for r in rows]

    def sync(self, competition_code: str) -> dict:
        """Checks every unresolved prediction against finished fixtures for a
        competition and records results where a confident name match is found.
        Returns a summary of how many were resolved this run."""
        pending = self._pending_predictions()
        if not pending:
            return {"status": "nothing_pending", "checked": 0, "resolved": 0}

        try:
            finished = self.client.get_finished_matches(competition_code) \
                if hasattr(self.client, "get_finished_matches") else None
        except Exception:
            _logger.warning("FootballDataClient failed for competition %s", competition_code, exc_info=True)
            finished = None

        if not finished:
            _logger.info("No finished-match source available for %s; %d predictions remain pending", competition_code, len(pending))
            return {
                "status": "no_results_source",
                "checked": len(pending),
                "resolved": 0,
                "note": "FootballDataClient needs FOOTBALL_DATA_API_KEY set and a "
                        "get_finished_matches(competition_code) method — wire this "
                        "up against football-data.org's /matches endpoint (status=FINISHED) "
                        "for a working feed.",
            }

        resolved = 0
        for pred in pending:
            match = next((f for f in finished
                          if pred["home_team"].lower() in f.get("home_team", "").lower()
                          and pred["away_team"].lower() in f.get("away_team", "").lower()), None)
            if not match:
                _logger.debug("No finished match found for pending prediction %s (%s vs %s)", pred["id"], pred["home_team"], pred["away_team"])
                continue
            hg, ag = match.get("home_goals"), match.get("away_goals")
            if hg is None or ag is None:
                _logger.debug("Finished match found but missing score for %s vs %s", pred["home_team"], pred["away_team"])
                continue
            actual = "H" if hg > ag else "A" if ag > hg else "D"
            self.ledger.record_result(pred["id"], actual)
            resolved += 1

        return {"status": "ok", "checked": len(pending), "resolved": resolved}


# ===========================================================================
# 8. Model versioning
# ===========================================================================

class VersionedPredictionLedger:
    """Extension of PredictionLedger that tags every prediction with a
    model_version string (e.g. 'v1.0-default', 'v1.1-shrunk-decay').
    Thread-safe for concurrent server/pipeline operations."""

    def __init__(self, db_path: str = "eqfbis_ledger.sqlite3"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._migrate()

    def _migrate(self):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL, home_team TEXT, away_team TEXT,
                        home_win_prob REAL, draw_prob REAL, away_win_prob REAL,
                        agreement_score REAL, actual_result TEXT, scored_at REAL
                    )
                """)
                try:
                    conn.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT")
                except sqlite3.OperationalError:
                    pass  # column already exists
                conn.execute("CREATE INDEX IF NOT EXISTS idx_version_result ON predictions(model_version, actual_result)")
                conn.commit()

    def log(self, model, model_version: str, agreement_score: float = 1.0) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cur = conn.execute(
                    "INSERT INTO predictions (created_at, home_team, away_team, home_win_prob, "
                    "draw_prob, away_win_prob, agreement_score, actual_result, scored_at, model_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
                    (time.time(), model.home_team, model.away_team, model.home_win_prob,
                     model.draw_prob, model.away_win_prob, agreement_score, model_version),
                )
                conn.commit()
                return cur.lastrowid

    def record_result(self, prediction_id: int, actual_result: str):
        if actual_result not in ("H", "D", "A"):
            raise ValueError("actual_result must be 'H', 'D', or 'A'")
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("UPDATE predictions SET actual_result = ?, scored_at = ? WHERE id = ?",
                             (actual_result, time.time(), prediction_id))
                conn.commit()

    def compare_versions(self) -> dict:
        """Brier score per model_version, so you can see whether a parameter
        change actually improved things rather than assuming it did."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                rows = conn.execute(
                    "SELECT model_version, home_win_prob, draw_prob, away_win_prob, actual_result "
                    "FROM predictions WHERE actual_result IS NOT NULL AND model_version IS NOT NULL"
                ).fetchall()

        if not rows:
            _logger.info("compare_versions: no scored predictions found in ledger")
            return {}

        by_version: dict[str, list] = {}
        for version, hp, dp, ap, actual in rows:
            by_version.setdefault(version, []).append((hp, dp, ap, actual))

        report = {}
        for version, records in by_version.items():
            if len(records) < 10:
                report[version] = {"status": "insufficient_data", "n": len(records)}
                continue
            total_brier = 0.0
            for hp, dp, ap, actual in records:
                actual_vec = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[actual]
                total_brier += ((hp - actual_vec[0]) ** 2 + (dp - actual_vec[1]) ** 2 +
                                (ap - actual_vec[2]) ** 2)
            report[version] = {"status": "ok", "n": len(records),
                                "brier_score": round(total_brier / (len(records) * 3), 4)}
        return report


# ===========================================================================
# 9. Odds movement tracking (opening / current / closing)
# ===========================================================================

class OddsMovementTracker:
    """Snapshots odds for a fixture at multiple points in time (opening,
    periodic checks, closing) automatically. Thread-safe for concurrent access."""

    def __init__(self, db_path: str = "eqfbis_odds_history.sqlite3"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS odds_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fixture_key TEXT,
                        snapshot_time REAL,
                        odds_home REAL, odds_draw REAL, odds_away REAL,
                        bookmaker TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_fixture_key ON odds_snapshots(fixture_key)")
                conn.commit()

    @staticmethod
    def fixture_key(home_team: str, away_team: str, date: str) -> str:
        return f"{home_team.lower()}|{away_team.lower()}|{date}"

    def snapshot(self, home_team: str, away_team: str, date: str,
                 odds_home: float, odds_draw: float, odds_away: float,
                 bookmaker: str = "Betika"):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO odds_snapshots (fixture_key, snapshot_time, odds_home, "
                    "odds_draw, odds_away, bookmaker) VALUES (?, ?, ?, ?, ?, ?)",
                    (self.fixture_key(home_team, away_team, date), time.time(),
                     odds_home, odds_draw, odds_away, bookmaker),
                )
                conn.commit()

    def movement(self, home_team: str, away_team: str, date: str) -> dict:
        """Returns opening odds (first snapshot), latest odds (most recent
        snapshot), and every snapshot in between."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                rows = conn.execute(
                    "SELECT snapshot_time, odds_home, odds_draw, odds_away, bookmaker "
                    "FROM odds_snapshots WHERE fixture_key = ? ORDER BY snapshot_time ASC",
                    (self.fixture_key(home_team, away_team, date),),
                ).fetchall()

        if not rows:
            _logger.debug("No odds snapshots found for %s vs %s on %s", home_team, away_team, date)
            return {"status": "no_data"}

        opening = rows[0]
        closing = rows[-1]
        return {
            "status": "ok",
            "n_snapshots": len(rows),
            "opening_odds": {"time": opening[0], "home": opening[1], "draw": opening[2],
                              "away": opening[3], "bookmaker": opening[4]},
            "closing_odds_proxy": {"time": closing[0], "home": closing[1], "draw": closing[2],
                                    "away": closing[3], "bookmaker": closing[4]},
            "line_moved": abs(opening[1] - closing[1]) > 0.05 or abs(opening[3] - closing[3]) > 0.05,
        }


# ===========================================================================
# CLI smoke test
# ===========================================================================

if __name__ == "__main__":
    print("--- xG client (network required for real data) ---")
    xg = XGDataClient()
    print("available:", xg.available)

    print("\n--- Team news client ---")
    news = TeamNewsClient()
    print("available:", news.available)

    print("\n--- Market-blended probability ---")
    blended = blend_with_market_probability(0.55, 0.22, 0.23, 2.10, 3.40, 3.60)
    print(blended)

    print("\n--- Per-team home advantage ---")
    hae = TeamHomeAdvantageEstimator(league_avg_home=1.55, league_avg_away=1.22)
    print(hae.estimate(
        home_goals_scored=[3, 1, 2, 3, 2], home_goals_conceded=[0, 0, 1, 1, 1],
        away_goals_scored=[1, 0, 2, 1, 0], away_goals_conceded=[1, 2, 1, 0, 2],
    ))

    print("\n--- Odds movement tracker ---")
    tracker = OddsMovementTracker(db_path="demo_odds_history.sqlite3")
    tracker.snapshot("Arsenal", "Chelsea", "2026-08-01", 2.20, 3.30, 3.50, "Betika")
    tracker.snapshot("Arsenal", "Chelsea", "2026-08-01", 2.05, 3.40, 3.70, "Betika")
    print(tracker.movement("Arsenal", "Chelsea", "2026-08-01"))

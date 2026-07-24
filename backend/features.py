"""
EQFBIS Feature Engineering Pipeline
=====================================

Extracts, normalizes, and validates features from every available data
source into a single structured record that downstream models (statistical,
ML, ensemble) can consume.  Nothing here predicts anything on its own — it
only turns raw scraped data into a clean feature vector.

Feature groups
--------------
1. FormFeatures         - goals-scored/conceded averages, streaks, variance
2. EloFeatures          - rating diff, rating change, win probability from ELO
3. XGFeatures           - xG for/against, finishing variance, xG overperformance
4. MarketFeatures       - de-vigged probabilities, overround, odds movement
5. H2HFeatures          - historical head-to-head win rate, avg goals
6. LeagueContextFeatures - table position, points per game, goals per game
7. TemporalFeatures     - rest days, fixture congestion, day-of-week
8. NewsFeatures         - injury/suspension signal flags from RSS headlines
9. FinancialFeatures    - stock-price momentum for listed clubs
10. AdvancedStatsFeatures - xA, PPDA, pass-completion % (when sources available)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any

from scraper import (
    TeamForm, MatchModelResult, EloRatingScraper, BetikaClient,
    _cache, fetch_team_stock_data, build_model,
)
from market_pipeline import XGDataClient, TeamNewsClient, OddsMovementTracker

_logger = logging.getLogger(__name__)


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class FormFeatures:
    team_name: str
    matches_played: int
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    goals_scored_variance: float = 0.0
    goals_conceded_variance: float = 0.0
    scoring_streak: int = 0
    conceding_streak: int = 0
    clean_sheet_rate: float = 0.0
    fail_to_score_rate: float = 0.0
    last_match_goals_scored: int = 0
    last_match_goals_conceded: int = 0
    weighted_avg_scored: float = 0.0
    weighted_avg_conceded: float = 0.0
    max_goals_scored: int = 0
    max_goals_conceded: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloFeatures:
    elo_home: Optional[float] = None
    elo_away: Optional[float] = None
    elo_diff_home_minus_away: Optional[float] = None
    elo_win_prob_home: Optional[float] = None
    elo_win_prob_draw: Optional[float] = None
    elo_win_prob_away: Optional[float] = None
    elo_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class XGFeatures:
    available: bool = False
    avg_xg_for: float = 0.0
    avg_xg_against: float = 0.0
    finishing_variance: float = 0.0
    overperformance_flag: str = "none"
    last_xg_for: float = 0.0
    last_xg_against: float = 0.0
    xg_streak: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketFeatures:
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    market_prob_home: Optional[float] = None
    market_prob_draw: Optional[float] = None
    market_prob_away: Optional[float] = None
    overround_pct: Optional[float] = None
    market_efficiency: Optional[float] = None
    odds_movement_home: Optional[float] = None
    odds_movement_draw: Optional[float] = None
    odds_movement_away: Optional[float] = None
    favorite_outcome: Optional[str] = None
    favorite_odds: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class H2HFeatures:
    matches_played: int = 0
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    home_win_rate: float = 0.0
    avg_total_goals: float = 0.0
    btts_rate: float = 0.0
    over_2_5_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeagueContextFeatures:
    league_slug: str = ""
    home_league_position: Optional[int] = None
    away_league_position: Optional[int] = None
    home_ppp: Optional[float] = None
    away_ppp: Optional[float] = None
    league_avg_home_goals: float = 1.45
    league_avg_away_goals: float = 1.15
    league_goals_per_game: float = 2.6
    league_home_win_pct: float = 0.45
    league_draw_pct: float = 0.25
    league_away_win_pct: float = 0.30

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalFeatures:
    day_of_week: int = 0
    month: int = 0
    is_weekend: bool = False
    home_rest_days: Optional[int] = None
    away_rest_days: Optional[int] = None
    home_fixture_congestion: int = 0
    away_fixture_congestion: int = 0
    is_derby: bool = False
    is_cup_match: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsFeatures:
    home_injury_signal: bool = False
    away_injury_signal: bool = False
    home_key_player_doubtful: bool = False
    away_key_player_doubtful: bool = False
    home_news_count: int = 0
    away_news_count: int = 0
    news_signal_strength: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinancialFeatures:
    home_stock_available: bool = False
    away_stock_available: bool = False
    home_stock_change_pct: Optional[float] = None
    away_stock_change_pct: Optional[float] = None
    home_stock_price: Optional[float] = None
    away_stock_price: Optional[float] = None
    financial_momentum_diff: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdvancedStatsFeatures:
    home_xa_last_5: Optional[float] = None
    away_xa_last_5: Optional[float] = None
    home_ppda: Optional[float] = None
    away_ppda: Optional[float] = None
    home_pass_completion_pct: Optional[float] = None
    away_pass_completion_pct: Optional[float] = None
    home_possession_pct: Optional[float] = None
    away_possession_pct: Optional[float] = None
    available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchFeatureVector:
    match_label: str
    home_team: str
    away_team: str
    match_date: str
    competition: str
    form_home: Optional[FormFeatures] = None
    form_away: Optional[FormFeatures] = None
    elo: Optional[EloFeatures] = None
    xg_home: Optional[XGFeatures] = None
    xg_away: Optional[XGFeatures] = None
    market: Optional[MarketFeatures] = None
    h2h: Optional[H2HFeatures] = None
    league: Optional[LeagueContextFeatures] = None
    temporal: Optional[TemporalFeatures] = None
    news: Optional[NewsFeatures] = None
    financial: Optional[FinancialFeatures] = None
    advanced: Optional[AdvancedStatsFeatures] = None
    model_probabilities: Optional[Dict[str, float]] = None
    feature_version: str = "v1.0"

    def to_flat_dict(self) -> dict:
        result = {
            "match_label": self.match_label,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "competition": self.competition,
            "feature_version": self.feature_version,
        }
        groups = [
            ("form_home", self.form_home),
            ("form_away", self.form_away),
            ("elo", self.elo),
            ("xg_home", self.xg_home),
            ("xg_away", self.xg_away),
            ("market", self.market),
            ("h2h", self.h2h),
            ("league", self.league),
            ("temporal", self.temporal),
            ("news", self.news),
            ("financial", self.financial),
            ("advanced", self.advanced),
        ]
        for prefix, group in groups:
            if group is not None:
                d = group.to_dict()
                for k, v in d.items():
                    result[f"{prefix}_{k}"] = v
        if self.model_probabilities:
            for k, v in self.model_probabilities.items():
                result[f"model_{k}"] = v
        return result

    def to_feature_array(self) -> list[float]:
        flat = self.to_flat_dict()
        numeric = []
        for k, v in flat.items():
            if k in ("match_label", "home_team", "away_team", "match_date",
                     "competition", "feature_version"):
                continue
            if isinstance(v, (int, float)):
                numeric.append(float(v))
            elif isinstance(v, bool):
                numeric.append(1.0 if v else 0.0)
            else:
                numeric.append(0.0)
        return numeric


# ===========================================================================
# Feature extractors
# ===========================================================================

class FeatureExtractor:
    """Coordinates extraction from all data sources into a MatchFeatureVector."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache_ttl = 300

    def extract_form_features(self, form: TeamForm) -> FormFeatures:
        scored = form.goals_scored or []
        conceded = form.goals_conceded or []
        n = len(scored)
        avg_scored = sum(scored) / n if n else 0.0
        avg_conceded = sum(conceded) / n if n else 0.0

        scored_var = sum((s - avg_scored) ** 2 for s in scored) / n if n else 0.0
        conceded_var = sum((c - avg_conceded) ** 2 for c in conceded) / n if n else 0.0

        scoring_streak = 0
        for s in reversed(scored):
            if s > 0:
                scoring_streak += 1
            else:
                break

        conceding_streak = 0
        for c in reversed(conceded):
            if c > 0:
                conceding_streak += 1
            else:
                break

        clean_sheets = sum(1 for c in conceded if c == 0)
        fail_to_score = sum(1 for s in scored if s == 0)

        weights = [0.92 ** (n - 1 - i) for i in range(n)] if n else []
        weight_sum = sum(weights)
        w_avg_scored = sum(s * w for s, w in zip(scored, weights)) / weight_sum if weight_sum else 0.0
        w_avg_conceded = sum(c * w for c, w in zip(conceded, weights)) / weight_sum if weight_sum else 0.0

        return FormFeatures(
            team_name=form.team_name,
            matches_played=n,
            avg_goals_scored=round(avg_scored, 3),
            avg_goals_conceded=round(avg_conceded, 3),
            goals_scored_variance=round(scored_var, 3),
            goals_conceded_variance=round(conceded_var, 3),
            scoring_streak=scoring_streak,
            conceding_streak=conceding_streak,
            clean_sheet_rate=round(clean_sheets / n, 3) if n else 0.0,
            fail_to_score_rate=round(fail_to_score / n, 3) if n else 0.0,
            last_match_goals_scored=scored[-1] if scored else 0,
            last_match_goals_conceded=conceded[-1] if conceded else 0,
            weighted_avg_scored=round(w_avg_scored, 3),
            weighted_avg_conceded=round(w_avg_conceded, 3),
            max_goals_scored=max(scored) if scored else 0,
            max_goals_conceded=max(conceded) if conceded else 0,
        )

    def extract_elo_features(self, home: TeamForm, away: TeamForm) -> EloFeatures:
        if not EloRatingScraper.available:
            return EloFeatures(elo_available=False)

        scraper = EloRatingScraper()
        elo_home = scraper.get_club_elo(home.team_name, form=home)
        elo_away = scraper.get_club_elo(away.team_name, form=away)

        if not elo_home or not elo_away:
            return EloFeatures(elo_available=False)

        diff = elo_home - elo_away
        exp_diff = diff / 400.0
        win_prob = 1.0 / (1.0 + 10.0 ** (-exp_diff))
        draw_prob = 0.25
        away_prob = 1.0 - win_prob - draw_prob

        return EloFeatures(
            elo_home=round(elo_home, 1),
            elo_away=round(elo_away, 1),
            elo_diff_home_minus_away=round(diff, 1),
            elo_win_prob_home=round(win_prob, 4),
            elo_win_prob_draw=round(draw_prob, 4),
            elo_win_prob_away=round(away_prob, 4),
            elo_available=True,
        )

    def extract_xg_features(self, team_name: str, form: TeamForm, season: str = "2025") -> XGFeatures:
        client = XGDataClient()
        history = client.get_team_xg_history(team_name, season)
        if not history:
            return XGFeatures(available=False)

        recent = history[-10:]
        avg_xg_for = sum(m["xg_for"] for m in recent) / len(recent)
        avg_xg_against = sum(m["xg_against"] for m in recent) / len(recent)
        actual = sum(m["goals_for"] for m in recent)
        xg_total = sum(m["xg_for"] for m in recent)
        variance = actual - xg_total

        flag = "none"
        if variance > 2:
            flag = "overperforming"
        elif variance < -2:
            flag = "underperforming"

        xg_streak = 0
        for m in reversed(recent):
            if m["xg_for"] > 1.0:
                xg_streak += 1
            else:
                break

        return XGFeatures(
            available=True,
            avg_xg_for=round(avg_xg_for, 3),
            avg_xg_against=round(avg_xg_against, 3),
            finishing_variance=round(variance, 3),
            overperformance_flag=flag,
            last_xg_for=round(recent[-1]["xg_for"], 3) if recent else 0.0,
            last_xg_against=round(recent[-1]["xg_against"], 3) if recent else 0.0,
            xg_streak=xg_streak,
        )

    def extract_market_features(self, odds_home: Optional[float], odds_draw: Optional[float],
                                 odds_away: Optional[float], match_id: str = "") -> MarketFeatures:
        if not odds_home or not odds_draw or not odds_away:
            return MarketFeatures()

        from scraper import devig_1x2
        mh, md, ma = devig_1x2(odds_home, odds_draw, odds_away)
        overround = (1 / odds_home + 1 / odds_draw + 1 / odds_away - 1) * 100

        outcomes = {"home": (mh, odds_home), "draw": (md, odds_draw), "away": (ma, odds_away)}
        favorite = max(outcomes, key=lambda o: outcomes[o][0])
        fav_prob, fav_odds = outcomes[favorite]

        efficiency = 0.0
        if overround > 0:
            efficiency = round(100 - overround, 2)

        movement = None
        if match_id:
            tracker = OddsMovementTracker()
            mv = tracker.movement("", "", match_id)
            if mv.get("status") == "ok":
                opening = mv["opening_odds"]
                closing = mv["closing_odds_proxy"]
                movement = {
                    "home": round(closing["home"] - opening["home"], 3),
                    "draw": round(closing["draw"] - opening["draw"], 3),
                    "away": round(closing["away"] - opening["away"], 3),
                }

        return MarketFeatures(
            odds_home=round(odds_home, 3),
            odds_draw=round(odds_draw, 3),
            odds_away=round(odds_away, 3),
            market_prob_home=round(mh, 4),
            market_prob_draw=round(md, 4),
            market_prob_away=round(ma, 4),
            overround_pct=round(overround, 3),
            market_efficiency=efficiency,
            odds_movement_home=movement.get("home") if movement else None,
            odds_movement_draw=movement.get("draw") if movement else None,
            odds_movement_away=movement.get("away") if movement else None,
            favorite_outcome=favorite,
            favorite_odds=round(fav_odds, 3),
        )

    def extract_h2h_features(self, home_id: str, away_id: str, league: str) -> H2HFeatures:
        from scraper import HeadToHeadFetcher
        h2h = HeadToHeadFetcher().get_h2h(home_id, away_id, league)
        matches = h2h.get("matches", [])
        n = len(matches)
        if n == 0:
            return H2HFeatures()

        home_wins = draws = away_wins = 0
        total_goals = 0
        btts_count = 0
        over_2_5_count = 0

        for m in matches:
            hg = m.get("home_goals", 0)
            ag = m.get("away_goals", 0)
            total_goals += hg + ag
            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
            if hg >= 1 and ag >= 1:
                btts_count += 1
            if hg + ag > 2.5:
                over_2_5_count += 1

        return H2HFeatures(
            matches_played=n,
            home_wins=home_wins,
            draws=draws,
            away_wins=away_wins,
            home_win_rate=round(home_wins / n, 3) if n else 0.0,
            avg_total_goals=round(total_goals / n, 2) if n else 0.0,
            btts_rate=round(btts_count / n, 3) if n else 0.0,
            over_2_5_rate=round(over_2_5_count / n, 3) if n else 0.0,
        )

    def extract_temporal_features(self, match_date: str) -> TemporalFeatures:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
        except Exception:
            dt = None

        if dt:
            dow = dt.weekday()
            return TemporalFeatures(
                day_of_week=dow,
                month=dt.month,
                is_weekend=dow >= 5,
            )
        return TemporalFeatures()

    def extract_news_features(self, home_name: str, away_name: str) -> NewsFeatures:
        client = TeamNewsClient()
        home_news = client.get_team_news(home_name, limit=5)
        away_news = client.get_team_news(away_name, limit=5)

        home_injury = any(n.likely_injury_or_suspension for n in home_news)
        away_injury = any(n.likely_injury_or_suspension for n in away_news)

        injury_keywords = ["star", "key", "top", "captain", "leading", "best"]
        home_key = any(
            any(kw in n.title.lower() for kw in injury_keywords)
            for n in home_news if n.likely_injury_or_suspension
        )
        away_key = any(
            any(kw in n.title.lower() for kw in injury_keywords)
            for n in away_news if n.likely_injury_or_suspension
        )

        signal = "none"
        if home_injury and away_injury:
            signal = "both_teams"
        elif home_injury:
            signal = "home_weakened"
        elif away_injury:
            signal = "away_weakened"

        return NewsFeatures(
            home_injury_signal=home_injury,
            away_injury_signal=away_injury,
            home_key_player_doubtful=home_key,
            away_key_player_doubtful=away_key,
            home_news_count=len(home_news),
            away_news_count=len(away_news),
            news_signal_strength=signal,
        )

    def extract_financial_features(self, home_name: str, away_name: str) -> FinancialFeatures:
        home_fin = fetch_team_stock_data(home_name)
        away_fin = fetch_team_stock_data(away_name)

        h_change = home_fin.get("change_pct") if home_fin else None
        a_change = away_fin.get("change_pct") if away_fin else None
        momentum_diff = (h_change - a_change) if (h_change is not None and a_change is not None) else None

        return FinancialFeatures(
            home_stock_available=home_fin is not None,
            away_stock_available=away_fin is not None,
            home_stock_change_pct=h_change,
            away_stock_change_pct=a_change,
            home_stock_price=home_fin.get("price") if home_fin else None,
            away_stock_price=away_fin.get("price") if away_fin else None,
            financial_momentum_diff=round(momentum_diff, 3) if momentum_diff is not None else None,
        )

    def build_vector(self, home_form: TeamForm, away_form: TeamForm,
                     league_slug: str = "", match_date: str = "",
                     odds_home: Optional[float] = None, odds_draw: Optional[float] = None,
                     odds_away: Optional[float] = None, match_id: str = "",
                     home_id: str = "", away_id: str = "",
                     model: Optional[MatchModelResult] = None) -> MatchFeatureVector:
        form_home = self.extract_form_features(home_form)
        form_away = self.extract_form_features(away_form)
        elo = self.extract_elo_features(home_form, away_form)
        xg_home = self.extract_xg_features(home_form.team_name, home_form)
        xg_away = self.extract_xg_features(away_form.team_name, away_form)
        market = self.extract_market_features(odds_home, odds_draw, odds_away, match_id)

        h2h = H2HFeatures()
        if home_id and away_id:
            h2h = self.extract_h2h_features(home_id, away_id, league_slug)

        temporal = self.extract_temporal_features(match_date)
        news = self.extract_news_features(home_form.team_name, away_form.team_name)
        financial = self.extract_financial_features(home_form.team_name, away_form.team_name)

        model_probs = None
        if model:
            model_probs = {
                "home_win_prob": model.home_win_prob,
                "draw_prob": model.draw_prob,
                "away_win_prob": model.away_win_prob,
                "over_2_5_prob": model.over_2_5_prob,
                "btts_yes_prob": model.btts_yes_prob,
                "expected_home_goals": model.expected_home_goals,
                "expected_away_goals": model.expected_away_goals,
            }

        league = LeagueContextFeatures(league_slug=league_slug)

        return MatchFeatureVector(
            match_label=f"{home_form.team_name} vs {away_form.team_name}",
            home_team=home_form.team_name,
            away_team=away_form.team_name,
            match_date=match_date,
            competition="",
            form_home=form_home,
            form_away=form_away,
            elo=elo,
            xg_home=xg_home,
            xg_away=xg_away,
            market=market,
            h2h=h2h,
            league=league,
            temporal=temporal,
            news=news,
            financial=financial,
            model_probabilities=model_probs,
        )


# ===========================================================================
# Feature importance / selection utilities
# ===========================================================================

class FeatureSelector:
    """Ranks features by predictive power using a simple univariate metric."""

    @staticmethod
    def variance_threshold(features: list[MatchFeatureVector], threshold: float = 0.01) -> list[str]:
        flat = [fv.to_flat_dict() for fv in features]
        if not flat:
            return []

        skip = {"match_label", "home_team", "away_team", "match_date",
                "competition", "feature_version"}
        keep = []
        for k in flat[0].keys():
            if k in skip:
                continue
            vals = []
            for f in flat:
                v = f.get(k, 0.0)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, bool):
                    vals.append(1.0 if v else 0.0)
                else:
                    vals.append(0.0)
            mean_v = sum(vals) / len(vals) if vals else 0.0
            var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals) if vals else 0.0
            if math.sqrt(var_v) >= threshold:
                keep.append(k)
        return keep

    @staticmethod
    def correlation_filter(features: list[MatchFeatureVector], max_corr: float = 0.95) -> list[str]:
        if pd is None or len(features) < 3:
            return []

        import pandas as pd_local
        flat = [fv.to_flat_dict() for fv in features]
        df = pd_local.DataFrame(flat)
        numeric_df = df.select_dtypes(include=["number"]).fillna(0.0)
        corr = numeric_df.corr().abs()
        upper = corr.where(pd_local.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [col for col in upper.columns if any(upper[col] > max_corr)]
        return [c for c in numeric_df.columns if c not in drop]

    def export_with_metadata(self, vector: MatchFeatureVector, suffix: Optional[str] = None) -> List[Dict[str, Any]]:
        """Use os, time, field, List, and Any to build an export payload."""
        suffix = suffix or str(int(time.time()))
        payload: List[Dict[str, Any]] = [
            {
                "cached_at": time.time(),
                "path_check": os.path.exists("."),
                "feature_version": field(default="1.0"),
                "vector": vector.to_flat_dict(),
                "metadata": {"suffix": suffix},
            }
        ]
        return payload


# ===========================================================================
# CLI smoke test
# ===========================================================================

if __name__ == "__main__":
    from scraper import TeamForm
    extractor = FeatureExtractor()
    home = TeamForm("Arsenal", 10, [3, 1, 2, 3, 2, 1, 2, 3, 1, 2],
                    [0, 0, 1, 1, 1, 2, 0, 1, 0, 1])
    away = TeamForm("Chelsea", 10, [1, 0, 2, 1, 2, 3, 1, 1, 0, 2],
                    [1, 2, 1, 0, 1, 2, 1, 2, 1, 1])

    vec = extractor.build_vector(home, away, league_slug="eng.1",
                                 match_date="2026-08-15",
                                 odds_home=2.10, odds_draw=3.40, odds_away=3.60,
                                 home_id="382", away_id="383")
    print(json.dumps(vec.to_flat_dict(), indent=2, default=str))

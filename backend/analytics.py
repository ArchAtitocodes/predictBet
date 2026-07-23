"""
EQFBIS — Elite Quantitative Football Betting Intelligence System
================================================================

Purpose
-------
A professional football betting analytics engine that scrapes live team
form, applies a Poisson GLM with attack/defense ratings, Bayesian shrinkage,
and recency-weighted decay, then benchmarks model probabilities against
bookmaker market odds to surface potential value.

Data Sources
------------
1. Betika API (betika.com/en-ke) — upcoming fixtures and live odds
2. ESPN API           — team search, recent results, league averages
3. football-data.org  — alternative REST API (set FOOTBALL_DATA_API_KEY)
4. yfinance           — corporate financial data for publicly listed clubs
5. BeautifulSoup/Selenium — fallback HTML scraping for edge-case coverage

Modelling Stack
---------------
- Poisson GLM with scipy MLE attack/defense rating optimiser
- Recency-weighted exponential decay on match history
- Bayesian shrinkage toward league average for small sample sizes
- Scoreline probability grid visualised as a matplotlib heatmap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import http.server
import webbrowser
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from typing import Optional
from backend.pipeline import AutomatedPredictionPipeline

def _math_exponential_decay(rate: float, time_step: float) -> float:
    return math.exp(-rate * time_step)

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.stats import poisson
except ImportError:
    poisson = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None

try:
    from sklearn.linear_model import PoissonRegressor
except ImportError:
    PoissonRegressor = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
except ImportError:
    plt = None

from scraper import (
    BetikaClient,
    ESPNScraperClient,
    FootballDataClient,
    WikipediaTeamScraper,
    HeadToHeadFetcher,
    FormMomentumCalculator,
    BettingSiteRegistry,
    betting_site_registry,
    TeamForm,
    MatchModelResult,
    build_model,
    compare_to_market,
    poisson_pmf,
    weighted_shrunk_rate,
    _cache,
    fetch_team_stock_data,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")

# ---------------------------------------------------------------------------
# Automated pipeline singleton (thread-safe lazy init)
# ---------------------------------------------------------------------------
_pipeline_lock = threading.Lock()
_pipeline_singleton = None

def get_pipeline() -> "AutomatedPredictionPipeline":
    global _pipeline_singleton
    if _pipeline_singleton is None:
        with _pipeline_lock:
            if _pipeline_singleton is None:
                from pipeline import AutomatedPredictionPipeline
                _pipeline_singleton = AutomatedPredictionPipeline(max_workers=4)
                _pipeline_singleton.run_full_pipeline(fixture_limit=50)
    return _pipeline_singleton
# ---------------------------------------------------------------------------
# Advanced Modeling and Visualization
# ---------------------------------------------------------------------------

def fit_poisson_glm_ratings(home: TeamForm, away: TeamForm,
                            league_home: float, league_away: float) -> Optional[tuple[float, float]]:
    """Use scipy.optimize to perform Maximum Likelihood Estimation of Poisson GLM ratings."""
    if np is None or poisson is None or minimize is None:
        return None
    try:
        # Define negative log-likelihood function
        def loss_func(params):
            h_att, h_def, a_att, a_def = params
            if any(p <= 0 for p in params):
                return 1e10
            
            log_lik = 0.0
            for gs in home.goals_scored:
                lam = h_att * league_home
                log_lik += poisson.logpmf(gs, lam)
            for gc in home.goals_conceded:
                lam = h_def * league_away
                log_lik += poisson.logpmf(gc, lam)
            for gs in away.goals_scored:
                lam = a_att * league_away
                log_lik += poisson.logpmf(gs, lam)
            for gc in away.goals_conceded:
                lam = a_def * league_home
                log_lik += poisson.logpmf(gc, lam)
            
            return -log_lik

        init_params = [1.0, 1.0, 1.0, 1.0]
        res = minimize(loss_func, init_params, method='L-BFGS-B',
                       bounds=[(0.1, 5.0), (0.1, 5.0), (0.1, 5.0), (0.1, 5.0)])
        
        if res.success:
            h_att, h_def, a_att, a_def = res.x
            expected_home = league_home * h_att * a_def
            expected_away = league_away * a_att * h_def
            return expected_home, expected_away
    except Exception:
        pass
    return None


def fit_statsmodels_glm_ratings(home: TeamForm, away: TeamForm,
                                league_home: float, league_away: float) -> Optional[tuple[float, float]]:
    """Use statsmodels GLM to fit Poisson attack/defense ratings."""
    if sm is None or pd is None:
        return None
    try:
        rows = []
        for gs in home.goals_scored:
            rows.append({"team": "home", "side": "attack", "goals": gs, "league_avg": league_home})
        for gc in home.goals_conceded:
            rows.append({"team": "home", "side": "defense", "goals": gc, "league_avg": league_away})
        for gs in away.goals_scored:
            rows.append({"team": "away", "side": "attack", "goals": gs, "league_avg": league_away})
        for gc in away.goals_conceded:
            rows.append({"team": "away", "side": "defense", "goals": gc, "league_avg": league_home})

        df = pd.DataFrame(rows)
        df["is_home_attack"] = ((df["team"] == "home") & (df["side"] == "attack")).astype(int)
        df["is_home_defense"] = ((df["team"] == "home") & (df["side"] == "defense")).astype(int)
        df["is_away_attack"] = ((df["team"] == "away") & (df["side"] == "attack")).astype(int)
        df["is_away_defense"] = ((df["team"] == "away") & (df["side"] == "defense")).astype(int)
        df["exposure"] = df["goals"] / df["league_avg"]

        formula = "goals ~ is_home_attack + is_home_defense + is_away_attack + is_away_defense + exposure - 1"
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()

        h_att = max(0.1, float(model.params.get("is_home_attack", 1.0)))
        h_def = max(0.1, float(model.params.get("is_home_defense", 1.0)))
        a_att = max(0.1, float(model.params.get("is_away_attack", 1.0)))
        a_def = max(0.1, float(model.params.get("is_away_defense", 1.0)))

        expected_home = league_home * h_att * a_def
        expected_away = league_away * a_att * h_def
        return expected_home, expected_away
    except Exception:
        return None


def fit_sklearn_poisson_ratings(home: TeamForm, away: TeamForm,
                                league_home: float, league_away: float) -> Optional[tuple[float, float]]:
    """Use sklearn PoissonRegressor to fit attack/defense ratings."""
    if PoissonRegressor is None or np is None:
        return None
    try:
        X, y = [], []
        for gs in home.goals_scored:
            X.append([league_home, 1, 0, 0, 0])
            y.append(gs)
        for gc in home.goals_conceded:
            X.append([league_away, 0, 1, 0, 0])
            y.append(gc)
        for gs in away.goals_scored:
            X.append([league_away, 0, 0, 1, 0])
            y.append(gs)
        for gc in away.goals_conceded:
            X.append([league_home, 0, 0, 0, 1])
            y.append(gc)

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        reg = PoissonRegressor(alpha=0.5, max_iter=1000)
        reg.fit(X_arr, y_arr)

        h_att = max(0.1, float(reg.coef_[1]))
        h_def = max(0.1, float(reg.coef_[2]))
        a_att = max(0.1, float(reg.coef_[3]))
        a_def = max(0.1, float(reg.coef_[4]))

        expected_home = league_home * h_att * a_def
        expected_away = league_away * a_att * h_def
        return expected_home, expected_away
    except Exception:
        return None


def validate_form_with_pandas(home: TeamForm, away: TeamForm) -> dict:
    """Use pandas to validate and summarize match data quality."""
    if pd is None:
        return {"pandas_available": False}
    try:
        home_df = pd.DataFrame({
            "match": range(1, len(home.goals_scored) + 1),
            "home_scored": home.goals_scored,
            "home_conceded": home.goals_conceded,
        })
        away_df = pd.DataFrame({
            "match": range(1, len(away.goals_scored) + 1),
            "away_scored": away.goals_scored,
            "away_conceded": away.goals_conceded,
        })

        validation = {
            "pandas_available": True,
            "home_matches": len(home_df),
            "away_matches": len(away_df),
            "home_avg_scored": round(home_df["home_scored"].mean(), 2) if len(home_df) else 0,
            "home_avg_conceded": round(home_df["home_conceded"].mean(), 2) if len(home_df) else 0,
            "away_avg_scored": round(away_df["away_scored"].mean(), 2) if len(away_df) else 0,
            "away_avg_conceded": round(away_df["away_conceded"].mean(), 2) if len(away_df) else 0,
            "home_scored_std": round(home_df["home_scored"].std(), 2) if len(home_df) > 1 else 0,
            "away_scored_std": round(away_df["away_scored"].std(), 2) if len(away_df) > 1 else 0,
        }
        return validation
    except Exception:
        return {"pandas_available": True, "error": "Validation failed"}


def _blend_estimator_goals(estimates: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """Blend multiple (expected_home, expected_away) estimates by simple average."""
    valid = [(h, a) for h, a in estimates if h is not None and a is not None and h >= 0 and a >= 0]
    if not valid:
        return None
    avg_home = sum(h for h, _ in valid) / len(valid)
    avg_away = sum(a for _, a in valid) / len(valid)
    return avg_home, avg_away


def generate_scoreline_heatmap(expected_home: float, expected_away: float,
                               home_name: str, away_name: str,
                               suffix: str = "") -> Optional[str]:
    """Generate and save a 2D matrix heatmap of scoreline probabilities using matplotlib."""
    if plt is None or poisson is None or np is None:
        return None
    try:
        safe_home = "".join(c if c.isalnum() or c in " _-" else "" for c in (home_name or "home")).strip().replace(" ", "_")
        safe_away = "".join(c if c.isalnum() or c in " _-" else "" for c in (away_name or "away")).strip().replace(" ", "_")
        fname = f"scoreline_heatmap_{safe_home}_vs_{safe_away}{suffix}.png"
        out_path = os.path.join(_FRONTEND_DIR, fname)

        max_goals = 6
        matrix = np.zeros((max_goals, max_goals))
        for hg in range(max_goals):
            for ag in range(max_goals):
                matrix[hg, ag] = poisson.pmf(hg, expected_home) * poisson.pmf(ag, expected_away)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor('#12161E')
        ax.set_facecolor('#12161E')

        im = ax.imshow(matrix, cmap='viridis', origin='lower')

        # Grid and ticks
        ax.set_xticks(np.arange(max_goals))
        ax.set_yticks(np.arange(max_goals))
        ax.set_xticklabels(np.arange(max_goals), color='#E8EBF0', fontfamily='sans-serif')
        ax.set_yticklabels(np.arange(max_goals), color='#E8EBF0', fontfamily='sans-serif')

        # Labels
        ax.set_xlabel(f'{away_name} Goals', color='#7C8798', fontsize=11, fontfamily='sans-serif')
        ax.set_ylabel(f'{home_name} Goals', color='#7C8798', fontsize=11, fontfamily='sans-serif')
        ax.set_title('Scoreline Probability Heatmap', color='#E8EBF0', fontsize=12, fontweight='bold', pad=15)

        # Annotate matrix cells with percentage text
        for i in range(max_goals):
            for j in range(max_goals):
                prob = matrix[i, j] * 100
                text_color = 'white' if prob < 12 else 'black'
                ax.text(j, i, f'{prob:.1f}%', ha="center", va="center",
                        color=text_color, fontsize=9, fontweight='semibold')

        # Style axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#232935')
        ax.spines['bottom'].set_color('#232935')
        ax.tick_params(colors='#7C8798')

        plt.tight_layout()
        plt.savefig(out_path, facecolor='#12161E', bbox_inches='tight')
        plt.close(fig)
        return fname
    except Exception as e:
        print(f"Error generating heatmap: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Sample data — realistic fixture used by the 'sample' subcommand
# ---------------------------------------------------------------------------

def _make_sample_form(name: str, scored: list[int],
                      conceded: list[int]) -> TeamForm:
    """Build a TeamForm from flat goal lists for the sample CLI command."""
    return TeamForm(team_name=name, matches_played=len(scored),
                     goals_scored=scored, goals_conceded=conceded)


# Arsenal last 10 PL matches (illustrative recent form)
SAMPLE_HOME = _make_sample_form(
    "Arsenal",
    scored=   [3, 1, 2, 3, 2, 1, 2, 3, 1, 2],
    conceded= [0, 0, 1, 1, 1, 2, 0, 1, 0, 1],
)

# Chelsea last 10 PL matches (illustrative recent form)
SAMPLE_AWAY = _make_sample_form(
    "Chelsea",
    scored=   [1, 0, 2, 1, 2, 3, 1, 1, 0, 2],
    conceded= [1, 2, 1, 0, 1, 2, 1, 2, 1, 1],
)

SAMPLE_LEAGUE_AVG_HOME = 1.55   # Premier League 2024/25
SAMPLE_LEAGUE_AVG_AWAY = 1.22


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(model: MatchModelResult,
                 market_comparison: Optional[dict] = None):
    print("=" * 70)
    print(f"  {model.home_team}  vs  {model.away_team}")
    print("=" * 70)
    print(f"Sample size: home={model.sample_size_home} matches, "
          f"away={model.sample_size_away} matches")
    print(f"Data quality: {model.data_quality_note()}")
    print()
    print(f"Model-estimated expected goals: "
          f"{model.home_team} {model.expected_home_goals:.2f} - "
          f"{model.expected_away_goals:.2f} {model.away_team}")
    print()
    print("1X2 model probabilities:")
    print(f"  Home win : {model.home_win_prob*100:5.1f}%")
    print(f"  Draw     : {model.draw_prob*100:5.1f}%")
    print(f"  Away win : {model.away_win_prob*100:5.1f}%")
    print()
    print("Goals markets (model):")
    print(f"  Over 2.5 : {model.over_2_5_prob*100:5.1f}%   "
          f"Under 2.5: {model.under_2_5_prob*100:5.1f}%")
    print(f"  BTTS Yes : {model.btts_yes_prob*100:5.1f}%   "
          f"BTTS No  : {model.btts_no_prob*100:5.1f}%")

    if market_comparison:
        print()
        print("-" * 70)
        print(f"Market comparison (bookmaker overround: "
              f"{market_comparison['bookmaker_overround_pct']}%)")
        for outcome in ("home", "draw", "away"):
            row = market_comparison[outcome]
            print(f"  {outcome.capitalize():5s}: model {row['model_prob_pct']:5.1f}%  "
                  f"vs market {row['market_implied_pct']:5.1f}%  "
                  f"(edge {row['edge_pct_points']:+.1f} pts)")
        print()
        print("NOTE:", market_comparison["note"])
    print("=" * 70)
    print("This is a predictive model based on recent scoring form, attack/defense ratings, "
          "and historical averages. Probabilities are mathematical estimates based on form "
          "and market-implied values.")


def update_model_probabilities(model: MatchModelResult, exp_home: float, exp_away: float):
    """Recompute win/draw/loss and goals market probabilities for a given pair of expected goals."""
    max_goals = 8
    if poisson is not None:
        home_probs = [poisson.pmf(g, exp_home) for g in range(max_goals + 1)]
        away_probs = [poisson.pmf(g, exp_away) for g in range(max_goals + 1)]
    else:
        # Fallback to scraper's poisson_pmf
        home_probs = [poisson_pmf(g, exp_home) for g in range(max_goals + 1)]
        away_probs = [poisson_pmf(g, exp_away) for g in range(max_goals + 1)]
        
    home_win = draw = away_win = over_2_5 = btts_yes = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = home_probs[hg] * away_probs[ag]
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
        home_win, draw, away_win = home_win / total, draw / total, away_win / total
        
    model.expected_home_goals = float(exp_home)
    model.expected_away_goals = float(exp_away)
    model.home_win_prob = float(home_win)
    model.draw_prob = float(draw)
    model.away_win_prob = float(away_win)
    model.over_2_5_prob = float(over_2_5)
    model.under_2_5_prob = float(1.0 - over_2_5)
    model.btts_yes_prob = float(btts_yes)
    model.btts_no_prob = float(1.0 - btts_yes)


# ---------------------------------------------------------------------------
# HTTP Server with Betika integration
# ---------------------------------------------------------------------------

class AnalyticsHTTPHandler(http.server.BaseHTTPRequestHandler):
    MIME_TYPES = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
    }

    def log_message(self, format, *args):
        pass

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    def _send_json(self, data, status=200):
        try:
            content = json.dumps(data, default=self._json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            try:
                error_payload = json.dumps({"error": f"Error encoding JSON: {str(e)}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_payload)))
                self.end_headers()
                self.wfile.write(error_payload)
            except Exception:
                pass

    def _serve_file(self, filename: str, mime_type: str):
        if not os.path.exists(filename):
            self._send_json({"error": f"File {filename} not found"}, status=404)
            return
        try:
            with open(filename, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_json({"error": f"Server error: {str(e)}"}, status=500)

    @staticmethod
    def _validate_int(value: str, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (ValueError, TypeError):
            return default
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _validate_float(value: str, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            return default
        return max(minimum, min(parsed, maximum))

    @staticmethod
    def _validate_str(value: str, default: str, max_length: int = 200) -> str:
        if not value or not isinstance(value, str):
            return default
        return value.strip()[:max_length]

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # System Health Check
        if path == "/api/health":
            self._send_json({
                "status": "healthy",
                "service": "PredictBet Engine",
                "timestamp": datetime.now().isoformat(),
                "frontend_ready": os.path.exists(os.path.join(_FRONTEND_DIR, "index.html")),
            })
            return

        # Static files and visualization chart handling
        if path == "/api/chart":
            filename = self._validate_str(query.get("file", [""])[0], "", max_length=200)
            if filename:
                chart_path = os.path.join(_FRONTEND_DIR, filename)
            else:
                chart_path = os.path.join(_FRONTEND_DIR, "scoreline_heatmap.png")
            self._serve_file(chart_path, "image/png")
            return

        if not path.startswith("/api/"):
            rel_path = path.lstrip("/")
            if not rel_path or rel_path == "index.html":
                self._serve_file(os.path.join(_FRONTEND_DIR, "index.html"), "text/html; charset=utf-8")
                return
            target_path = os.path.abspath(os.path.join(_FRONTEND_DIR, rel_path))
            frontend_dir_abs = os.path.abspath(_FRONTEND_DIR)
            if target_path.startswith(frontend_dir_abs) and os.path.isfile(target_path):
                _, ext = os.path.splitext(target_path)
                mime_type = self.MIME_TYPES.get(ext.lower(), "application/octet-stream")
                self._serve_file(target_path, mime_type)
                return
            else:
                self._send_json({"error": "File not found"}, status=404)
                return

        # ---------- Betting Sites Registry ----------
        if path == "/api/sites":
            try:
                q = self._validate_str(query.get("q", [""])[0], "", max_length=100)
                if q:
                    results = betting_site_registry.search_sites(q)
                else:
                    results = betting_site_registry.get_all_sites()
                self._send_json({"data": results, "meta": {"total": len(results)}})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # ---------- Betika endpoints ----------
        if path == "/api/betika/fixtures":
            page = self._validate_int(query.get("page", ["1"])[0], 1, 1, 100)
            limit = self._validate_int(query.get("limit", ["50"])[0], 50, 1, 200)
            try:
                client = BetikaClient()
                if page == 0:
                    fixtures = client.get_all_fixtures()
                else:
                    fixtures = client.get_upcoming_fixtures(page=page, limit=limit)
                meta = {"total": len(fixtures), "page": page, "limit": limit}
                self._send_json({"data": fixtures, "meta": meta})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/betika/live":
            try:
                client = BetikaClient()
                live = client.get_live_matches()
                self._send_json({"data": live, "meta": {"total": len(live)}})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/betika/search":
            q = query.get("q", [""])[0]
            try:
                client = BetikaClient()
                results = client.search_teams(q)
                self._send_json(results)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/betika/competitions":
            try:
                client = BetikaClient()
                comps = client.get_competitions()
                self._send_json({"data": comps, "meta": {"total": len(comps)}})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/betika/scrape":
            home_id = self._validate_str(query.get("home_id", [""])[0], "")
            away_id = self._validate_str(query.get("away_id", [""])[0], "")
            match_id = self._validate_str(query.get("match_id", [""])[0], "")
            decay = self._validate_float(query.get("decay", ["0.92"])[0], 0.92, 0.0, 1.0)
            shrinkage_k = self._validate_float(query.get("shrinkage_k", ["6.0"])[0], 6.0, 0.1, 50.0)
            home_advantage = self._validate_float(query.get("home_advantage", ["1.0"])[0], 1.0, 0.5, 3.0)

            if not home_id or not away_id:
                self._send_json({"error": "Missing home_id or away_id"}, status=400)
                return

            try:
                espn = ESPNScraperClient()
                home_results = espn.search_team(home_id)
                away_results = espn.search_team(away_id)

                if not home_results or not away_results:
                    self._send_json({"error": "Could not resolve one or both team names on ESPN"}, status=404)
                    return

                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = (home_info.get("league") or
                               away_info.get("league") or "eng.1")

                home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
                away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
                league_home, league_away = espn.fetch_league_averages(league_slug)

                odds_home = odds_draw = odds_away = None
                if match_id:
                    betika = BetikaClient()
                    match_data = betika.get_match_odds(match_id)
                    if match_data:
                        try:
                            odds_home = float(match_data.get("home_odd", 0)) if match_data.get("home_odd") else None
                            odds_draw = float(match_data.get("draw_odd", 0)) if match_data.get("draw_odd") else None
                            odds_away = float(match_data.get("away_odd", 0)) if match_data.get("away_odd") else None
                        except (ValueError, TypeError):
                            odds_home = odds_draw = odds_away = None

                model = build_model(
                    home_form, away_form,
                    league_home, league_away,
                    home_advantage=home_advantage,
                    decay=decay, shrinkage_k=shrinkage_k,
                )

                # Fit Advanced Poisson GLM if available
                glm_res = fit_poisson_glm_ratings(home_form, away_form, league_home, league_away)
                sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_home, league_away)
                sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_home, league_away)
                blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
                if blended:
                    blended_home, blended_away = blended
                    update_model_probabilities(model, blended_home * home_advantage, blended_away)

                # Generate Scoreline Heatmap Visual
                heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                          home_form.team_name, away_form.team_name)

                # Pandas-based validation
                pandas_validation = validate_form_with_pandas(home_form, away_form)

                # Fetch financials
                fin_home = fetch_team_stock_data(home_form.team_name)
                fin_away = fetch_team_stock_data(away_form.team_name)

                market_comp = None
                if odds_home and odds_draw and odds_away:
                    market_comp = compare_to_market(
                        model, odds_home, odds_draw, odds_away)

                record = {
                    "match_label": f"{home_form.team_name} vs {away_form.team_name}",
                    "match_date": query.get("match_date", [""])[0] or None,
                    "model": model.__dict__,
                    "confidence_score": model.confidence_score(),
                    "data_quality_note": model.data_quality_note(),
                    "market_comparison": market_comp,
                    "heatmap_file": heatmap_file,
                    "pandas_validation": pandas_validation,
                    "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
                    "odds": {
                        "home": odds_home,
                        "draw": odds_draw,
                        "away": odds_away,
                    },
                    "sources": {
                        "team_form": "ESPN",
                        "odds": "Betika" if match_id else "None",
                    },
                    "financials": {
                        "home": fin_home,
                        "away": fin_away
                    }
                }
                self._send_json(record)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # ---------- ESPN / general endpoints ----------
        if path == "/api/search":
            q = self._validate_str(query.get("query", [""])[0], "", max_length=100)
            try:
                espn = ESPNScraperClient()
                results = espn.search_team(q)

                # Also search Betika for team names
                if len(q) >= 2:
                    try:
                        betika = BetikaClient()
                        betika_results = betika.search_teams(q)
                        results.extend(betika_results)
                    except Exception:
                        pass

                seen = set()
                unique = []
                for r in results:
                    key = (r.get("name", ""), r.get("source", ""))
                    if key not in seen:
                        seen.add(key)
                        unique.append(r)
                self._send_json(unique)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/team/info":
            team_name = self._validate_str(query.get("name", [""])[0], "", max_length=200)
            if not team_name:
                self._send_json({"error": "Missing team name"}, status=400)
                return
            wiki = WikipediaTeamScraper()
            info = wiki.get_team_info(team_name)
            if info:
                self._send_json(info)
            else:
                self._send_json({"team_name": team_name, "error": "No info found"}, status=404)
            return

        if path == "/api/team/search":
            q = self._validate_str(query.get("query", [""])[0], "", max_length=100)
            if not q or len(q) < 2:
                self._send_json([])
                return
            wiki = WikipediaTeamScraper()
            results = wiki.search_teams(q)
            self._send_json(results)
            return

        if path == "/api/scrape":
            home_id = self._validate_str(query.get("home_id", [""])[0], "")
            away_id = self._validate_str(query.get("away_id", [""])[0], "")
            league_slug = self._validate_str(query.get("league_slug", [""])[0], "")
            decay = self._validate_float(query.get("decay", ["0.92"])[0], 0.92, 0.0, 1.0)
            shrinkage_k = self._validate_float(query.get("shrinkage_k", ["6.0"])[0], 6.0, 0.1, 50.0)
            home_advantage = self._validate_float(query.get("home_advantage", ["1.0"])[0], 1.0, 0.5, 3.0)

            if not home_id or not away_id or not league_slug:
                self._send_json({"error": "Missing parameters"}, status=400)
                return

            client = ESPNScraperClient()
            try:
                home_form = client.fetch_recent_matches(home_id, league_slug)
                away_form = client.fetch_recent_matches(away_id, league_slug)
                league_avg_home, league_avg_away = client.fetch_league_averages(league_slug)
                odds_home, odds_draw, odds_away = client.fetch_market_odds(
                    home_id, away_id, league_slug)

                model = build_model(
                    home_form, away_form,
                    league_avg_home, league_avg_away,
                    home_advantage=home_advantage,
                    decay=decay, shrinkage_k=shrinkage_k,
                )

                # Fit Advanced Poisson GLM if available
                glm_res = fit_poisson_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
                sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
                sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_avg_home, league_avg_away)
                blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
                if blended:
                    blended_home, blended_away = blended
                    update_model_probabilities(model, blended_home * home_advantage, blended_away)

                # Generate Scoreline Heatmap Visual
                heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                          home_form.team_name, away_form.team_name)

                # Pandas-based validation
                pandas_validation = validate_form_with_pandas(home_form, away_form)

                # Fetch financials
                fin_home = fetch_team_stock_data(home_form.team_name)
                fin_away = fetch_team_stock_data(away_form.team_name)

                market_comp = None
                if odds_home and odds_draw and odds_away:
                    market_comp = compare_to_market(
                        model, odds_home, odds_draw, odds_away)

                record = {
                    "match_label": f"{home_form.team_name} vs {away_form.team_name}",
                    "match_date": query.get("match_date", [""])[0] or None,
                    "model": model.__dict__,
                    "confidence_score": model.confidence_score(),
                    "data_quality_note": model.data_quality_note(),
                    "market_comparison": market_comp,
                    "heatmap_file": heatmap_file,
                    "pandas_validation": pandas_validation,
                    "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
                    "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
                    "financials": {
                        "home": fin_home,
                        "away": fin_away
                    }
                }
                self._send_json(record)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/scrape_by_name":
            home_name = self._validate_str(query.get("home", [""])[0], "", max_length=200)
            away_name = self._validate_str(query.get("away", [""])[0], "", max_length=200)
            decay = self._validate_float(query.get("decay", ["0.92"])[0], 0.92, 0.0, 1.0)
            shrinkage_k = self._validate_float(query.get("shrinkage_k", ["6.0"])[0], 6.0, 0.1, 50.0)
            home_advantage = self._validate_float(query.get("home_advantage", ["1.0"])[0], 1.0, 0.5, 3.0)

            if not home_name or not away_name:
                self._send_json({"error": "Missing parameters"}, status=400)
                return

            client = ESPNScraperClient()
            try:
                home_results = client.search_team(home_name)
                away_results = client.search_team(away_name)
                if not home_results or not away_results:
                    self._send_json({"error": "Could not resolve team names"}, status=404)
                    return

                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = (home_info.get("league") or
                               away_info.get("league") or "eng.1")

                home_form = client.fetch_recent_matches(home_info["id"], league_slug)
                away_form = client.fetch_recent_matches(away_info["id"], league_slug)
                league_avg_home, league_avg_away = client.fetch_league_averages(league_slug)
                odds_home, odds_draw, odds_away = client.fetch_market_odds(
                    home_info["id"], away_info["id"], league_slug)

                model = build_model(
                    home_form, away_form,
                    league_avg_home, league_avg_away,
                    home_advantage=home_advantage,
                    decay=decay, shrinkage_k=shrinkage_k,
                )

                # Fit Advanced Poisson GLM if available
                glm_res = fit_poisson_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
                sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
                sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_avg_home, league_avg_away)
                blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
                if blended:
                    blended_home, blended_away = blended
                    update_model_probabilities(model, blended_home * home_advantage, blended_away)

                # Generate Scoreline Heatmap Visual
                heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                          home_form.team_name, away_form.team_name)

                # Pandas-based validation
                pandas_validation = validate_form_with_pandas(home_form, away_form)

                # Fetch financials
                fin_home = fetch_team_stock_data(home_form.team_name)
                fin_away = fetch_team_stock_data(away_form.team_name)

                market_comp = None
                if odds_home and odds_draw and odds_away:
                    market_comp = compare_to_market(
                        model, odds_home, odds_draw, odds_away)

                record = {
                    "match_label": f"{home_form.team_name} vs {away_form.team_name}",
                    "match_date": query.get("match_date", [""])[0] or None,
                    "model": model.__dict__,
                    "confidence_score": model.confidence_score(),
                    "data_quality_note": model.data_quality_note(),
                    "market_comparison": market_comp,
                    "heatmap_file": heatmap_file,
                    "pandas_validation": pandas_validation,
                    "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
                    "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
                    "financials": {
                        "home": fin_home,
                        "away": fin_away
                    }
                }
                self._send_json(record)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/report"):
            try:
                from aiBetModel.integration import render_match_report
                home_name = self._validate_str(query.get("home", [""])[0], "", max_length=200)
                away_name = self._validate_str(query.get("away", [""])[0], "", max_length=200)
                league = self._validate_str(query.get("league", ["eng.1"])[0], "eng.1", 100)
                confidence = self._validate_str(query.get("confidence", ["medium"])[0], "medium", 50)
                match_date = self._validate_str(query.get("date", [""])[0], "", 100)
                odds_home = self._validate_float(query.get("odds_home", ["0"])[0], 0.0, 0.0, 1000.0)
                odds_draw = self._validate_float(query.get("odds_draw", ["0"])[0], 0.0, 0.0, 1000.0)
                odds_away = self._validate_float(query.get("odds_away", ["0"])[0], 0.0, 0.0, 1000.0)

                if not home_name or not away_name:
                    self._send_json({"error": "Missing home/away parameters"}, status=400)
                    return

                client = ESPNScraperClient()
                home_results = client.search_team(home_name)
                away_results = client.search_team(away_name)
                if not home_results or not away_results:
                    self._send_json({"error": "Could not resolve team names"}, status=404)
                    return

                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = (home_info.get("league") or
                               away_info.get("league") or league)

                home_form = client.fetch_recent_matches(home_info["id"], league_slug)
                away_form = client.fetch_recent_matches(away_info["id"], league_slug)
                league_avg_home, league_avg_away = client.fetch_league_averages(league_slug)

                model = build_model(home_form, away_form, league_avg_home, league_avg_away,
                                    home_advantage=1.0, decay=0.92, shrinkage_k=6.0)

                md = render_match_report(
                    home_team=home_form.team_name,
                    away_team=away_form.team_name,
                    league=league_slug,
                    match_date=match_date or datetime.now().strftime("%Y-%m-%d"),
                    model_home_prob=model.home_win_prob,
                    model_draw_prob=model.draw_prob,
                    model_away_prob=model.away_win_prob,
                    expected_home_goals=model.expected_home_goals,
                    expected_away_goals=model.expected_away_goals,
                    odds_home=odds_home if odds_home > 0 else None,
                    odds_draw=odds_draw if odds_draw > 0 else None,
                    odds_away=odds_away if odds_away > 0 else None,
                    confidence=confidence,
                )
                self._send_json({"status": "success", "report_markdown": md})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/predictions"):
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                client = BetikaClient()
                fixtures = client.get_upcoming_fixtures(page=1, limit=100)

                fixtures_with_odds = [
                    f for f in fixtures
                    if f.get("home_odd") and f.get("draw_odd") and f.get("away_odd")
                ][:6]

                if not fixtures_with_odds:
                    self._send_json({"status": "success", "data": []})
                    return

                predictions = []
                lock = threading.Lock()

                def process_fixture(fixture):
                    try:
                        home_name = fixture.get("home_team", "")
                        away_name = fixture.get("away_team", "")
                        match_id = fixture.get("match_id", "")

                        espn = ESPNScraperClient()
                        home_results = espn.search_team(home_name)
                        away_results = espn.search_team(away_name)

                        if not home_results or not away_results:
                            return None

                        home_info = home_results[0]
                        away_info = away_results[0]
                        league_slug = (home_info.get("league") or
                                       away_info.get("league") or "eng.1")

                        home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
                        away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
                        league_home, league_away = espn.fetch_league_averages(league_slug)

                        model = build_model(
                            home_form, away_form,
                            league_home, league_away,
                            home_advantage=1.0,
                            decay=0.92, shrinkage_k=6.0,
                        )

                        glm_res = fit_poisson_glm_ratings(home_form, away_form, league_home, league_away)
                        sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_home, league_away)
                        sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_home, league_away)
                        blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
                        if blended:
                            blended_home, blended_away = blended
                            update_model_probabilities(model, blended_home, blended_away)

                        heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                                  home_form.team_name, away_form.team_name)
                        pandas_validation = validate_form_with_pandas(home_form, away_form)

                        odds_home = float(fixture["home_odd"])
                        odds_draw = float(fixture["draw_odd"])
                        odds_away = float(fixture["away_odd"])

                        market_comp = compare_to_market(model, odds_home, odds_draw, odds_away)

                        edges = {
                            "home": market_comp["home"]["edge_pct_points"],
                            "draw": market_comp["draw"]["edge_pct_points"],
                            "away": market_comp["away"]["edge_pct_points"],
                        }
                        best_outcome = max(edges, key=edges.get)
                        best_edge = edges[best_outcome]

                        if best_outcome == "home":
                            model_prob = model.home_win_prob
                            offered_odds = odds_home
                            recommended = f"Home Win ({model.home_team})"
                        elif best_outcome == "draw":
                            model_prob = model.draw_prob
                            offered_odds = odds_draw
                            recommended = "Draw"
                        else:
                            model_prob = model.away_win_prob
                            offered_odds = odds_away
                            recommended = f"Away Win ({model.away_team})"

                        if best_edge > 15:
                            tier = "LOCK"
                        elif best_edge > 8:
                            tier = "STRONG"
                        elif best_edge > 3:
                            tier = "VALUE"
                        elif best_edge > 0:
                            tier = "LEAN"
                        else:
                            tier = "NO_BET"

                        decimal_edge = model_prob * offered_odds - 1.0
                        from intelligence import AggressiveStakeEngine, ConfidenceTier
                        stake = AggressiveStakeEngine.suggest(
                            ConfidenceTier[tier],
                            max(decimal_edge, 0.0),
                            offered_odds,
                        )

                        from intelligence import generate_aggressive_narrative
                        narrative = generate_aggressive_narrative(
                            model.__dict__, {}, tier, recommended
                        )

                        return {
                            "match_label": f"{model.home_team} vs {model.away_team}",
                            "match_date": fixture.get("start_time", ""),
                            "competition": fixture.get("competition_name", ""),
                            "recommended_bet": recommended,
                            "confidence_tier": tier,
                            "model_probability": model_prob,
                            "market_implied_probability": market_comp[best_outcome]["market_implied_pct"] / 100,
                            "edge_pct": best_edge,
                            "stake_suggestion_pct": stake["stake_pct"],
                            "scouting_narrative": narrative,
                            "model_data": model.__dict__,
                            "heatmap_file": heatmap_file,
                            "pandas_validation": pandas_validation,
                            "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
                        }
                    except Exception:
                        return None

                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(process_fixture, f): f for f in fixtures_with_odds}
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=15)
                        except Exception:
                            result = None
                        if result:
                            with lock:
                                predictions.append(result)

                predictions.sort(key=lambda p: p["edge_pct"], reverse=True)
                self._send_json({"status": "success", "data": predictions})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return
            
        if self.path.startswith("/api/h2h"):
            try:
                import urllib.parse
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                home_id = query.get("home_id", [""])[0]
                away_id = query.get("away_id", [""])[0]
                league = query.get("league", ["eng.1"])[0]
                from scraper import HeadToHeadFetcher
                h2h = HeadToHeadFetcher().get_h2h(home_id, away_id, league)
                self._send_json({"status": "success", "data": h2h})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return
            
        if self.path.startswith("/api/form"):
            try:
                import urllib.parse
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                team_id = query.get("team_id", [""])[0]
                league = query.get("league", ["eng.1"])[0]
                espn = ESPNScraperClient()
                form = espn.fetch_recent_matches(team_id, league)
                momentum = FormMomentumCalculator.calculate(form)
                self._send_json({"status": "success", "data": momentum})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/ml/predict"):
            try:
                from features import FeatureExtractor
                from ml_pipeline import MLPipeline
                home = self._validate_str(query.get("home", [""])[0], "", 200)
                away = self._validate_str(query.get("away", [""])[0], "", 200)
                league = self._validate_str(query.get("league", ["eng.1"])[0], "eng.1", 100)
                if not home or not away:
                    self._send_json({"error": "Missing home/away parameters"}, status=400)
                    return
                espn = ESPNScraperClient()
                home_info = espn.search_team(home)
                away_info = espn.search_team(away)
                if not home_info or not away_info:
                    self._send_json({"error": "Could not resolve team names"}, status=404)
                    return
                home_form = espn.fetch_recent_matches(home_info[0]["id"], league)
                away_form = espn.fetch_recent_matches(away_info[0]["id"], league)
                league_home, league_away = espn.fetch_league_averages(league)
                model = build_model(home_form, away_form, league_home, league_away,
                                    home_advantage=1.0, decay=0.92, shrinkage_k=6.0)
                extractor = FeatureExtractor()
                fv = extractor.build_vector(home_form, away_form, league_slug=league,
                                            odds_home=0, odds_draw=0, odds_away=0)
                pipeline = MLPipeline()
                pred = pipeline.predict(fv)
                self._send_json({
                    "status": "success",
                    "match_label": f"{home} vs {away}",
                    "ml_prediction": pred.to_dict() if pred else None,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/calibration/fit"):
            try:
                from calibration import CalibrationManager
                model_name = self._validate_str(query.get("model", ["ensemble"])[0], "ensemble", 100)
                outcome = self._validate_str(query.get("outcome", ["H"])[0], "H", 1)
                method = self._validate_str(query.get("method", ["isotonic"])[0], "isotonic", 50)
                mgr = CalibrationManager()
                result = mgr.fit(model_name, outcome, method=method)
                self._send_json({"status": "success", "calibration": result})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/monitoring"):
            try:
                from monitoring import SystemMonitor
                monitor = SystemMonitor()
                report = monitor.report()
                alerts = monitor.evaluate()
                self._send_json({
                    "status": "success",
                    "metrics": {k: {
                        "value": v.value, "unit": v.unit,
                        "timestamp": v.timestamp, "severity": v.severity(),
                    } for k, v in report.items()},
                    "active_alerts": [a.to_dict() for a in alerts],
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/pipeline/status"):
            try:
                from pipeline import get_pipeline
                p = get_pipeline()
                self._send_json({
                    "status": "success",
                    "pipeline_status": p.status,
                    "last_run": p.last_run.isoformat() if p.last_run else None,
                    "results_count": len(p.results),
                    "version": "v3.0",
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/pipeline/run"):
            try:
                limit = self._validate_int(query.get("limit", ["100"])[0], 100, 1, 500)
                from pipeline import get_pipeline
                p = get_pipeline()
                results = p.run_full_pipeline(fixture_limit=limit)
                payload = {
                    "status": "success",
                    "pipeline_status": p.status,
                    "last_run": p.last_run.isoformat() if p.last_run else None,
                    "count": len(results),
                    "data": [r.prediction_card for r in results],
                }
                self._send_json(payload)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/pipeline/v2/predict"):
            try:
                from intelligence import PredictionPipelineV2
                home = self._validate_str(query.get("home", [""])[0], "", 200)
                away = self._validate_str(query.get("away", [""])[0], "", 200)
                league = self._validate_str(query.get("league", ["eng.1"])[0], "eng.1", 100)
                if not home or not away:
                    self._send_json({"error": "Missing home/away parameters"}, status=400)
                    return
                espn = ESPNScraperClient()
                home_results = espn.search_team(home)
                away_results = espn.search_team(away)
                if not home_results or not away_results:
                    self._send_json({"error": "Could not resolve team names"}, status=404)
                    return
                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = (home_info.get("league") or away_info.get("league") or "eng.1")
                home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
                away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
                league_home, league_away = espn.fetch_league_averages(league_slug)
                pipeline = PredictionPipelineV2()
                result = pipeline.predict(
                    home_form, away_form, league_slug=league_slug,
                    match_date=query.get("date", [""])[0],
                    odds_home=float(query.get("oh", ["0"])[0] or 0),
                    odds_draw=float(query.get("od", ["0"])[0] or 0),
                    odds_away=float(query.get("oa", ["0"])[0] or 0),
                    match_id=query.get("match_id", [""])[0],
                    home_id=home_info["id"], away_id=away_info["id"],
                    league_home=league_home, league_away=league_away,
                )
                self._send_json({"status": "success", "data": result})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Not Found"}, status=404)



# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_goal_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def cmd_betika_fixtures(args):
    client = BetikaClient()
    if args.live:
        print("Fetching live matches from Betika...")
        fixtures = client.get_live_matches()
        if not fixtures:
            print("No live matches found.")
            return
        print(f"\n{'HOME':<30} {'AWAY':<30} {'TIME':<20} {'COMPETITION':<30} {'ODDS (1X2)'}")
        print("-" * 130)
        for m in fixtures:
            odds = f"{m.get('home_odd','')} / {m.get('draw_odd','')} / {m.get('away_odd','')}"
            print(f"{m['home_team']:<30} {m['away_team']:<30} {m['start_time']:<20} {m['competition_name']:<30} {odds}")
    else:
        print("Fetching upcoming fixtures from Betika...")
        fixtures = client.get_all_fixtures()
        print(f"\nTotal fixtures: {len(fixtures)}")
        print(f"\n{'HOME':<30} {'AWAY':<30} {'TIME':<20} {'COMPETITION':<30} {'ODDS (1X2)'}")
        print("-" * 130)
        for m in fixtures[:args.limit if args.limit else len(fixtures)]:
            odds = f"{m.get('home_odd','')} / {m.get('draw_odd','')} / {m.get('away_odd','')}"
            print(f"{m['home_team']:<30} {m['away_team']:<30} {m['start_time']:<20} {m['competition_name']:<30} {odds}")


def cmd_betika_search(args):
    client = BetikaClient()
    results = client.search_teams(args.query)
    if not results:
        print(f"No teams found matching '{args.query}'")
        return
    print(f"\nTeams matching '{args.query}':")
    for t in results:
        print(f"  {t['name']} — {t.get('league', '')} ({t.get('subtitle', '')})")


def cmd_betika_model(args):
    client = BetikaClient()
    fixtures = client.get_all_fixtures()

    home_match = None
    away_match = None
    for m in fixtures:
        if m["home_team"].lower() == args.home.lower():
            home_match = m
        if m["away_team"].lower() == args.away.lower():
            away_match = m

    if not home_match or not away_match:
        print("Could not find both teams in Betika fixtures.")
        sys.exit(1)

    espn = ESPNScraperClient()
    home_results = espn.search_team(args.home)
    away_results = espn.search_team(args.away)

    if not home_results or not away_results:
        print("Could not resolve teams on ESPN for form data.")
        sys.exit(1)

    home_info = home_results[0]
    away_info = away_results[0]
    league_slug = (home_info.get("league") or away_info.get("league") or "eng.1")

    print(f"Scraping form for {home_info['name']}...")
    home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
    print(f"Scraping form for {away_info['name']}...")
    away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
    print(f"Computing league averages for {league_slug}...")
    league_home, league_away = espn.fetch_league_averages(league_slug)

    model = build_model(
        home_form, away_form, league_home, league_away,
        home_advantage=args.home_advantage,
        decay=args.decay, shrinkage_k=args.shrinkage_k,
    )

    odds_home = odds_draw = odds_away = None
    if home_match.get("match_id"):
        match_data = client.get_match_odds(home_match["match_id"])
        if match_data:
            try:
                odds_home = float(match_data["home_odd"]) if match_data.get("home_odd") else None
                odds_draw = float(match_data["draw_odd"]) if match_data.get("draw_odd") else None
                odds_away = float(match_data["away_odd"]) if match_data.get("away_odd") else None
            except (ValueError, TypeError):
                pass

    market_comp = None
    if odds_home and odds_draw and odds_away:
        market_comp = compare_to_market(model, odds_home, odds_draw, odds_away)

    print()
    print_report(model, market_comp)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-data football analytics: Poisson model from live "
                    "team form (ESPN + Betika API), compared against market odds.")
    subparsers = parser.add_subparsers(dest="command")

    # --- serve ---
    serve_parser = subparsers.add_parser("serve",
        help="Start the local HTTP server with dashboard.")
    serve_parser.add_argument("--port", type=int, default=8080,
                              help="Port (default 8080)")

    # --- sample ---
    sample_parser = subparsers.add_parser("sample",
        help="Run with bundled sample fixture (Arsenal vs Chelsea) to verify the model.")
    sample_parser.add_argument("--odds-home", type=float)
    sample_parser.add_argument("--odds-draw", type=float)
    sample_parser.add_argument("--odds-away", type=float)
    sample_parser.add_argument("--json", action="store_true")

    # --- report ---
    report_parser = subparsers.add_parser("report",
        help="Render a full spec-compliant Markdown report for a fixture.")
    report_parser.add_argument("--home-team", type=str, required=True)
    report_parser.add_argument("--away-team", type=str, required=True)
    report_parser.add_argument("--league", type=str, default="eng.1")
    report_parser.add_argument("--date", type=str, default="")
    report_parser.add_argument("--home-results", type=str)
    report_parser.add_argument("--home-conceded", type=str)
    report_parser.add_argument("--away-results", type=str)
    report_parser.add_argument("--away-conceded", type=str)
    report_parser.add_argument("--league-avg-home-goals", type=float, default=1.45)
    report_parser.add_argument("--league-avg-away-goals", type=float, default=1.15)
    report_parser.add_argument("--home-advantage", type=float, default=1.0)
    report_parser.add_argument("--decay", type=float, default=0.92)
    report_parser.add_argument("--shrinkage-k", type=float, default=6.0)
    report_parser.add_argument("--odds-home", type=float)
    report_parser.add_argument("--odds-draw", type=float)
    report_parser.add_argument("--odds-away", type=float)
    report_parser.add_argument("--confidence", type=str, default="medium")
    report_parser.add_argument("--output", type=str, metavar="FILE.md")

    # --- model ---
    model_parser = subparsers.add_parser("model",
        help="Build a Poisson model for a fixture.")
    model_parser.add_argument("--home-team", type=str, required=True)
    model_parser.add_argument("--away-team", type=str, required=True)
    model_parser.add_argument("--home-results", type=str)
    model_parser.add_argument("--home-conceded", type=str)
    model_parser.add_argument("--away-results", type=str)
    model_parser.add_argument("--away-conceded", type=str)
    model_parser.add_argument("--league-avg-home-goals", type=float, default=1.45)
    model_parser.add_argument("--league-avg-away-goals", type=float, default=1.15)
    model_parser.add_argument("--home-advantage", type=float, default=1.0)
    model_parser.add_argument("--decay", type=float, default=0.92)
    model_parser.add_argument("--shrinkage-k", type=float, default=6.0)
    model_parser.add_argument("--odds-home", type=float)
    model_parser.add_argument("--odds-draw", type=float)
    model_parser.add_argument("--odds-away", type=float)
    model_parser.add_argument("--use-football-data-api", action="store_true")
    model_parser.add_argument("--json", action="store_true")
    model_parser.add_argument("--export", type=str, metavar="FILE.json")

    # --- betika fixtures ---
    b_fix = subparsers.add_parser("betika-fixtures",
        help="Fetch upcoming fixtures from Betika.")
    b_fix.add_argument("--live", action="store_true",
                       help="Show only live/inplay matches.")
    b_fix.add_argument("--limit", type=int, default=50,
                       help="Max fixtures to display (default 50).")

    # --- betika search ---
    b_search = subparsers.add_parser("betika-search",
        help="Search for teams on Betika.")
    b_search.add_argument("query", type=str, help="Team name to search for.")

    # --- betika model ---
    b_model = subparsers.add_parser("betika-model",
        help="Build a Poisson model using Betika fixtures + ESPN form data.")
    b_model.add_argument("--home", type=str, required=True,
                         help="Home team name")
    b_model.add_argument("--away", type=str, required=True,
                         help="Away team name")
    b_model.add_argument("--home-advantage", type=float, default=1.0)
    b_model.add_argument("--decay", type=float, default=0.92)
    b_model.add_argument("--shrinkage-k", type=float, default=6.0)
    b_model.add_argument("--json", action="store_true")
    b_model.add_argument("--export", type=str, metavar="FILE.json")

    args = parser.parse_args()

    # ===================== serve =====================
    if args.command == "serve":
        port = args.port if args.port != 8080 else int(os.environ.get("PORT", 8080))
        host = os.environ.get("HOST", "0.0.0.0")
        try:
            import uvicorn
            print("=" * 70, file=sys.stderr)
            print(f"  Betting Analytics Server (FastAPI) running at http://{host or 'localhost'}:{port}",
                  file=sys.stderr)
            print(f"  Open http://localhost:{port}/ in your browser", file=sys.stderr)
            print(f"  Data sources: Betika API | ESPN | football-data.org",
                  file=sys.stderr)
            print("  Press Ctrl+C to stop", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            uvicorn.run("server:app", host=host, port=port, reload=False)
            sys.exit(0)
        except ImportError:
            pass

        server_address = (host, port)
        try:
            from http.server import ThreadingHTTPServer
            httpd = ThreadingHTTPServer(server_address, AnalyticsHTTPHandler)
        except ImportError:
            httpd = http.server.HTTPServer(server_address, AnalyticsHTTPHandler)

        print("=" * 70, file=sys.stderr)
        print(f"  Betting Analytics Server running at http://{host or 'localhost'}:{port}",
              file=sys.stderr)
        print(f"  Open http://localhost:{port}/ in your browser", file=sys.stderr)
        print(f"  Data sources: Betika API | ESPN | football-data.org",
              file=sys.stderr)
        print("  Press Ctrl+C to stop", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        if host in ("127.0.0.1", "localhost", ""):
            try:
                webbrowser.open(f"http://localhost:{port}/")
            except Exception:
                pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...", file=sys.stderr)
        sys.exit(0)

    # ===================== sample =====================
    if args.command == "sample":
        home_form, away_form = SAMPLE_HOME, SAMPLE_AWAY
        league_home, league_away = SAMPLE_LEAGUE_AVG_HOME, SAMPLE_LEAGUE_AVG_AWAY
        model = build_model(home_form, away_form, league_home, league_away)
        market_comp = None
        if args.odds_home and args.odds_draw and args.odds_away:
            market_comp = compare_to_market(
                model, args.odds_home, args.odds_draw, args.odds_away)
        record = {
            "match_label": "Arsenal vs Chelsea (sample fixture)",
            "model": model.__dict__,
            "confidence_score": model.confidence_score(),
            "data_quality_note": model.data_quality_note(),
            "market_comparison": market_comp,
        }
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print_report(model, market_comp)
        return

    # ===================== betika-fixtures =====================
    if args.command == "betika-fixtures":
        cmd_betika_fixtures(args)
        return

    # ===================== betika-search =====================
    if args.command == "betika-search":
        cmd_betika_search(args)
        return

    # ===================== betika-model =====================
    if args.command == "betika-model":
        cmd_betika_model(args)
        return

    # ===================== report =====================
    if args.command == "report":
        from aiBetModel.integration import render_match_report

        home_team = args.home_team
        away_team = args.away_team
        league = args.league or "eng.1"
        confidence = args.confidence or "medium"

        if args.home_results and args.home_conceded and args.away_results and args.away_conceded:
            home_form = TeamForm(
                team_name=home_team,
                matches_played=len(parse_goal_list(args.home_results)),
                goals_scored=parse_goal_list(args.home_results),
                goals_conceded=parse_goal_list(args.home_conceded),
            )
            away_form = TeamForm(
                team_name=away_team,
                matches_played=len(parse_goal_list(args.away_results)),
                goals_scored=parse_goal_list(args.away_results),
                goals_conceded=parse_goal_list(args.away_conceded),
            )
            league_home, league_away = (args.league_avg_home_goals,
                                         args.league_avg_away_goals)
        else:
            scraper = ESPNScraperClient()
            print(f"Searching for '{home_team}'...", file=sys.stderr)
            home_results = scraper.search_team(home_team)
            if not home_results:
                print(f"Could not resolve '{home_team}'.", file=sys.stderr)
                sys.exit(1)
            print(f"Searching for '{away_team}'...", file=sys.stderr)
            away_results = scraper.search_team(away_team)
            if not away_results:
                print(f"Could not resolve '{away_team}'.", file=sys.stderr)
                sys.exit(1)

            home_info = home_results[0]
            away_info = away_results[0]
            league_slug = (home_info.get("league") or
                           away_info.get("league") or league)
            print(f"Using league: {league_slug}", file=sys.stderr)

            home_form = scraper.fetch_recent_matches(home_info["id"], league_slug)
            away_form = scraper.fetch_recent_matches(away_info["id"], league_slug)
            league_home, league_away = scraper.fetch_league_averages(league_slug)

        model = build_model(home_form, away_form, league_home, league_away,
                            home_advantage=args.home_advantage,
                            decay=args.decay, shrinkage_k=args.shrinkage_k)

        md = render_match_report(
            home_team=home_form.team_name,
            away_team=away_form.team_name,
            league=league,
            match_date=args.date or datetime.now().strftime("%Y-%m-%d"),
            model_home_prob=model.home_win_prob,
            model_draw_prob=model.draw_prob,
            model_away_prob=model.away_win_prob,
            expected_home_goals=model.expected_home_goals,
            expected_away_goals=model.expected_away_goals,
            odds_home=args.odds_home,
            odds_draw=args.odds_draw,
            odds_away=args.odds_away,
            confidence=confidence,
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Report written to {args.output}", file=sys.stderr)
        else:
            print(md)
        return

    # ===================== model =====================
    if args.command == "model":
        if args.home_results and args.home_conceded and args.away_results and args.away_conceded:
            home_form = TeamForm(
                team_name=args.home_team,
                matches_played=len(parse_goal_list(args.home_results)),
                goals_scored=parse_goal_list(args.home_results),
                goals_conceded=parse_goal_list(args.home_conceded),
            )
            away_form = TeamForm(
                team_name=args.away_team,
                matches_played=len(parse_goal_list(args.away_results)),
                goals_scored=parse_goal_list(args.away_results),
                goals_conceded=parse_goal_list(args.away_conceded),
            )
            league_home, league_away = (args.league_avg_home_goals,
                                         args.league_avg_away_goals)
        elif args.home_team and args.away_team:
            if args.use_football_data_api:
                client = FootballDataClient()
                if not client.available:
                    print("No FOOTBALL_DATA_API_KEY set.", file=sys.stderr)
                    sys.exit(1)
                home_id = client.find_team_id(args.home_team)
                away_id = client.find_team_id(args.away_team)
                if not home_id or not away_id:
                    print("Could not resolve team names.", file=sys.stderr)
                    sys.exit(1)
                home_form = client.recent_results(home_id)
                away_form = client.recent_results(away_id)
                league_home, league_away = (args.league_avg_home_goals,
                                             args.league_avg_away_goals)
            else:
                scraper = ESPNScraperClient()
                print(f"Searching for '{args.home_team}'...", file=sys.stderr)
                home_results = scraper.search_team(args.home_team)
                if not home_results:
                    print(f"Could not resolve '{args.home_team}'.", file=sys.stderr)
                    sys.exit(1)
                print(f"Searching for '{args.away_team}'...", file=sys.stderr)
                away_results = scraper.search_team(args.away_team)
                if not away_results:
                    print(f"Could not resolve '{args.away_team}'.", file=sys.stderr)
                    sys.exit(1)

                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = (home_info.get("league") or
                               away_info.get("league") or "eng.1")
                print(f"Using league: {league_slug}", file=sys.stderr)

                home_form = scraper.fetch_recent_matches(home_info["id"], league_slug)
                away_form = scraper.fetch_recent_matches(away_info["id"], league_slug)
                league_home, league_away = scraper.fetch_league_averages(league_slug)

                if not args.odds_home or not args.odds_draw or not args.odds_away:
                    print("Checking ESPN for odds...", file=sys.stderr)
                    oh, od, oa = scraper.fetch_market_odds(
                        home_info["id"], away_info["id"], league_slug)
                    if oh and od and oa:
                        args.odds_home, args.odds_draw, args.odds_away = oh, od, oa
                        print(f"Found odds: {oh} / {od} / {oa}", file=sys.stderr)
        else:
            parser.print_help()
            sys.exit(1)

        model = build_model(home_form, away_form, league_home, league_away,
                            home_advantage=args.home_advantage,
                            decay=args.decay, shrinkage_k=args.shrinkage_k)

        market_comp = None
        if args.odds_home and args.odds_draw and args.odds_away:
            market_comp = compare_to_market(
                model, args.odds_home, args.odds_draw, args.odds_away)

        record = {
            "match_label": f"{home_form.team_name} vs {away_form.team_name}",
            "model": model.__dict__,
            "confidence_score": model.confidence_score(),
            "data_quality_note": model.data_quality_note(),
            "market_comparison": market_comp,
        }

        if args.export:
            dataset = []
            if os.path.exists(args.export):
                try:
                    with open(args.export) as f:
                        dataset = json.load(f)
                except (json.JSONDecodeError, OSError):
                    dataset = []
            dataset.append(record)
            with open(args.export, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"Exported to {args.export} ({len(dataset)} records).",
                  file=sys.stderr)

        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print_report(model, market_comp)
        return

    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    _ANALYTICS_FIRST_MAIN_BLOCK = True
    main()

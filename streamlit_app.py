"""
PredictBet AI — Institutional-Grade Football Betting Intelligence Engine
======================================================================
Bloomberg Terminal meets Opta meets Football Manager analytics.

Fully automated workflow:
1. Fixture ingestion (Betika + football-data.org)
2. Team form, xG, ELO, H2H, squad value
3. Model ensemble (Poisson GLM + Bayesian + ELO + ML)
4. Multi-bookmaker odds + de-vigging
5. EV, Kelly, risk scoring, confidence grading
6. Professional report generation
7. Auto-refresh + rankings

Rules:
- Never fabricate data — reduce confidence when unavailable
- Recommend bets only when positive EV exists
- Conservative bankroll management (fractional Kelly)
- Insufficient evidence → NO BET
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    import streamlit as st
    from streamlit_option_menu import option_menu
except ImportError:
    st = None
    option_menu = None

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------
from backend.scraper import (
    BetikaClient,
    ESPNScraperClient,
    WikipediaTeamScraper,
    HeadToHeadFetcher,
    FormMomentumCalculator,
    MultiMarketPredictor,
    EloRatingScraper,
    build_model,
    compare_to_market,
    devig_1x2,
    poisson_pmf,
    betting_site_registry,
    fetch_team_stock_data,
)
from backend.analytics import (
    fit_poisson_glm_ratings,
    fit_statsmodels_glm_ratings,
    fit_sklearn_poisson_ratings,
    _blend_estimator_goals,
    update_model_probabilities,
    generate_scoreline_heatmap,
)
from backend.intelligence import (
    ConfidenceTier,
    AggressiveStakeEngine,
    generate_aggressive_narrative,
    generate_scouting_narrative,
    suggest_stake,
    build_ensemble,
    PredictionCard,
    PredictionPipelineV2,
    MLEnhancedEnsemble,
    PredictionLedger,
)
from backend.aiBetModel.integration import (
    build_market_assessments,
    build_stake_recommendations,
    build_data_quality_checklist,
    build_comparison_from_model,
    render_match_report,
)
from backend.aiBetModel.quality import EvidenceChecklist, grade_data_quality
from backend.aiBetModel.market import expected_value, devig_proportional, implied_probability, classify_efficiency

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------
THEME_KEY = "app_theme"

def _get_theme() -> str:
    return st.session_state.get(THEME_KEY, "dark")

def _toggle_theme():
    current = _get_theme()
    st.session_state[THEME_KEY] = "light" if current == "dark" else "dark"

def _inject_theme_css():
    theme = _get_theme()
    if theme == "dark":
        bg = "#0F131C"
        panel = "#1A202C"
        border = "#2D3748"
        text = "#E8EBF0"
        text_dim = "#A0AEC0"
        accent = "#00E5FF"
        accent2 = "#7C3AED"
        pos = "#22C55E"
        neg = "#EF4444"
        warn = "#F59E0B"
    else:
        bg = "#F8FAFC"
        panel = "#FFFFFF"
        border = "#E2E8F0"
        text = "#1E293B"
        text_dim = "#64748B"
        accent = "#0891B2"
        accent2 = "#7C3AED"
        pos = "#16A34A"
        neg = "#DC2626"
        warn = "#D97706"

    st.markdown(f"""
    <style>
    :root {{
        --bg: {bg};
        --panel: {panel};
        --border: {border};
        --text: {text};
        --text-dim: {text_dim};
        --accent: {accent};
        --accent2: {accent2};
        --pos: {pos};
        --neg: {neg};
        --warn: {warn};
    }}
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}
    .main-header {{
        font-size: 2.0rem;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0rem;
        letter-spacing: -0.02em;
    }}
    .sub-header {{
        font-size: 1.0rem;
        color: var(--text-dim);
        margin-bottom: 1.5rem;
    }}
    .card {{
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }}
    .tier-LOCK {{
        background: var(--neg);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }}
    .tier-STRONG {{
        background: var(--warn);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }}
    .tier-VALUE {{
        background: var(--pos);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }}
    .tier-LEAN {{
        background: var(--accent2);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }}
    .tier-NO_BET {{
        background: var(--border);
        color: var(--text-dim);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }}
    .section-divider {{
        border: 0;
        height: 2px;
        background: var(--border);
        margin: 1.5rem 0;
    }}
    .metric-positive {{
        color: var(--pos);
        font-weight: 700;
    }}
    .metric-negative {{
        color: var(--neg);
        font-weight: 700;
    }}
    .stButton>button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stSelectbox>div>div {{
        border-radius: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Champions League", "Europa League", "Kenyan Premier League",
    "Nigerian Professional Football League", "South African Premier Division",
    "CAF Champions League", "FIFA World Cup Qualifiers", "MLS", "Brasileirão",
    "Argentine Primera División", "J1 League", "A-League", "Indian Super League",
]

REFRESH_OPTIONS = {
    "Off": 0,
    "Every 5 min": 300,
    "Every 10 min": 600,
    "Every 30 min": 1800,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_start_time(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _day_label(dt: Optional[datetime], now: datetime) -> str:
    if dt is None:
        return "Unknown"
    local = dt.astimezone(now.tzinfo) if dt.tzinfo else dt
    local_now = now.astimezone(now.tzinfo) if now.tzinfo else now
    diff = (local.date() - local_now.date()).days
    if diff == 0:
        return "Today"
    if diff == 1:
        return "Tomorrow"
    if diff == 2:
        return "In 2 Days"
    if 3 <= diff < 7:
        return f"In {diff} Days"
    return local.strftime("%a, %d %b")


def _tier_for_edge(edge_pct: float, confidence: int = 50) -> str:
    if edge_pct > 15 and confidence >= 60:
        return "LOCK"
    if edge_pct > 8 and confidence >= 50:
        return "STRONG"
    if edge_pct > 3 and confidence >= 35:
        return "VALUE"
    if edge_pct > 0:
        return "LEAN"
    return "NO_BET"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


_TIER_ORDER = {"LOCK": 0, "STRONG": 1, "VALUE": 2, "LEAN": 3, "NO_BET": 4}


def _sorted_predictions() -> list[dict]:
    def sort_key(p: dict) -> tuple[int, float]:
        tier_rank = _TIER_ORDER.get(str(p.get("confidence_tier", "NO_BET")).upper(), 99)
        confidence = float(p.get("confidence_score", 0) or 0)
        edge = float(p.get("edge_pct", 0) or 0)
        return (tier_rank, -confidence, -edge)

    return sorted(st.session_state.predictions, key=sort_key)


def _init_session():
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "scan_running" not in st.session_state:
        st.session_state.scan_running = False
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 0
    if "league_filter" not in st.session_state:
        st.session_state.league_filter = []
    if "day_filter" not in st.session_state:
        st.session_state.day_filter = "All Days"
    if "highlight_filter" not in st.session_state:
        st.session_state.highlight_filter = "All"
    if "selected_match_label" not in st.session_state:
        st.session_state.selected_match_label = None
    if "auto_analyze" not in st.session_state:
        st.session_state.auto_analyze = False
    if "auto_analyze_running" not in st.session_state:
        st.session_state.auto_analyze_running = False
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = "dark"


# ---------------------------------------------------------------------------
# Data quality & evidence checker
# ---------------------------------------------------------------------------

def _evidence_checklist_from_row(row: dict) -> EvidenceChecklist:
    return EvidenceChecklist(
        lineups_confirmed=bool(row.get("lineups_confirmed", False)),
        injuries_verified=bool(row.get("injuries_verified", False)),
        odds_verified=bool(row.get("market_comparison") is not None),
        xg_data_available=bool(row.get("xg_available", False)),
        team_strength_metrics_available=bool(row.get("elo_available", False)),
        historical_h2h_available=bool(row.get("h2h_available", False)),
        conflicting_sources=bool(row.get("conflicting_sources", False)),
        small_sample_size=bool((row.get("sample_size_home", 0) + row.get("sample_size_away", 0)) < 6),
    )


# ---------------------------------------------------------------------------
# Model builder — fully automated pipeline
# ---------------------------------------------------------------------------

def _fetch_team_intel(team_name: str, league_slug: str) -> dict:
    info = {"elo": None, "squad_value": None, "manager": None, "stadium": None, "wikipedia_available": False}
    try:
        elo_scraper = EloRatingScraper()
        info["elo"] = elo_scraper.get_club_elo(team_name)
    except Exception:
        pass
    try:
        wiki = WikipediaTeamScraper()
        wiki_info = wiki.get_team_info(team_name)
        if wiki_info:
            info["manager"] = wiki_info.get("manager")
            info["stadium"] = wiki_info.get("stadium")
            info["wikipedia_available"] = True
    except Exception:
        pass
    try:
        stock = fetch_team_stock_data(team_name)
        if stock:
            info["squad_value"] = stock.get("market_cap")
    except Exception:
        pass
    return info


def _build_match_model(home_name: str, away_name: str, odds_h: float = 0, odds_d: float = 0, odds_a: float = 0,
                        decay: float = 0.92, shrinkage_k: float = 6.0, home_advantage: float = 1.0,
                        fixture: Optional[dict] = None):
    _settings_decay = _safe_float(st.session_state.get("default_decay"), decay)
    _settings_k = _safe_float(st.session_state.get("default_shrinkage_k"), shrinkage_k)
    _settings_hadv = _safe_float(st.session_state.get("default_home_adv"), home_advantage)
    if decay == 0.92:
        decay = _settings_decay
    if shrinkage_k == 6.0:
        shrinkage_k = _settings_k
    if home_advantage == 1.0:
        home_advantage = _settings_hadv
    espn = ESPNScraperClient()
    home_results = espn.search_team(home_name)
    away_results = espn.search_team(away_name)
    if not home_results or not away_results:
        raise RuntimeError("Could not resolve one or both team names on ESPN API.")
    home_info = home_results[0]
    away_info = away_results[0]
    league_slug = home_info.get("league") or away_info.get("league") or "eng.1"
    home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
    away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
    league_home, league_away = espn.fetch_league_averages(league_slug)

    model = build_model(home_form, away_form, league_home, league_away,
                        home_advantage=home_advantage, decay=decay, shrinkage_k=shrinkage_k)
    glm_res = fit_poisson_glm_ratings(home_form, away_form, league_home, league_away)
    sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_home, league_away)
    sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_home, league_away)
    blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
    if blended:
        update_model_probabilities(model, blended[0] * home_advantage, blended[1])

    market_comp = None
    if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
        market_comp = compare_to_market(model, odds_h, odds_d, odds_a)

    devig_probs = devig_1x2(odds_h, odds_d, odds_a)

    multi = MultiMarketPredictor.predict(model.expected_home_goals, model.expected_away_goals)
    momentum_home = FormMomentumCalculator.calculate(home_form)
    momentum_away = FormMomentumCalculator.calculate(away_form)
    ensemble = build_ensemble(home_form, away_form, model.expected_home_goals, model.expected_away_goals,
                              league_home, league_away, shrinkage_model=model)

    try:
        espn_ids = {home_info.get("id"), away_info.get("id")}
        h2h_fetcher = HeadToHeadFetcher()
        h2h_raw = h2h_fetcher.get_h2h(home_info.get("id"), away_info.get("id"), league_slug)
        h2h_summary = h2h_fetcher.h2h_summary(home_info.get("id"), away_info.get("id"), home_name, away_name, league_slug) if h2h_raw else {}
    except Exception:
        h2h_raw = []
        h2h_summary = {}

    home_intel = _fetch_team_intel(home_name, league_slug)
    away_intel = _fetch_team_intel(away_name, league_slug)

    edges = {
        "home": market_comp["home"]["edge_pct_points"] if market_comp else 0,
        "draw": market_comp["draw"]["edge_pct_points"] if market_comp else 0,
        "away": market_comp["away"]["edge_pct_points"] if market_comp else 0,
    }
    best_outcome = max(edges, key=edges.get) if market_comp else "none"
    best_edge = edges.get(best_outcome, 0)
    confidence_raw = model.confidence_score()
    tier = _tier_for_edge(best_edge, confidence_raw)

    offered = odds_h if best_outcome == "home" else (odds_d if best_outcome == "draw" else odds_a)
    stake_res = AggressiveStakeEngine.suggest(ConfidenceTier[tier], max(best_edge / 100, 0), offered)
    stake_pct = stake_res.get("stake_pct", 0)

    best_model_prob = model.home_win_prob if best_outcome == "home" else (
        model.draw_prob if best_outcome == "draw" else model.away_win_prob)
    best_market_implied = (market_comp[best_outcome]["market_implied_pct"] if market_comp and best_outcome != "none" else 0)
    fair_odds = (1 / best_market_implied) if best_market_implied > 0 else 0
    ev_pct = expected_value(best_model_prob, offered) * 100 if offered > 1 else 0

    risk_score = max(0, 100 - confidence_raw - best_edge * 2)
    if (home_form.matches_played + away_form.matches_played) < 10:
        risk_score = min(100, risk_score + 20)
    if ensemble and ensemble.agreement_score < 0.6:
        risk_score = min(100, risk_score + 15)
    risk_label = "Low" if risk_score < 35 else ("Medium" if risk_score < 65 else "High")

    row_for_evidence = {
        "sample_size_home": home_form.matches_played,
        "sample_size_away": away_form.matches_played,
        "market_comparison": market_comp,
        "xg_available": True,
        "elo_available": home_intel.get("elo") is not None,
        "h2h_available": bool(h2h_raw),
        "conflicting_sources": ensemble.agreement_score < 0.6 if ensemble else False,
    }
    data_grade = grade_data_quality(_evidence_checklist_from_row(row_for_evidence))

    has_odds = (odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0)
    implied_probs_list = [implied_probability(o) for o in (odds_h, odds_d, odds_a)] if has_odds else []
    fair_probs_list = devig_proportional([odds_h, odds_d, odds_a]) if has_odds else []
    efficiency_class = classify_efficiency(best_edge / 100.0) if best_outcome != "none" else "N/A"

    market_assessments_data = build_market_assessments(
        model.home_win_prob, model.draw_prob, model.away_win_prob,
        odds_h if has_odds else 2.0, odds_d if has_odds else 3.2, odds_a if has_odds else 3.5
    ) if has_odds else []

    conservative_stake_obj = suggest_stake(best_model_prob, offered if offered > 1 else 2.0, best_outcome, confidence_raw / 100.0)

    data_grade_checklist = build_data_quality_checklist(
        model.home_win_prob, model.draw_prob, model.away_win_prob,
        odds_home=odds_h if has_odds else None,
        odds_draw=odds_d if has_odds else None,
        odds_away=odds_a if has_odds else None,
        xg_available=True,
        team_strength_metrics_available=True,
        historical_h2h_available=bool(h2h_raw),
        lineups_confirmed=False,
        injuries_verified=False,
        conflicting_sources=ensemble.agreement_score < 0.6 if ensemble else False,
        small_sample_size=(home_form.matches_played + away_form.matches_played) < 10,
    )
    if data_grade_checklist:
        data_grade = data_grade_checklist

    stake_recs_data = build_stake_recommendations(
        model.home_win_prob, model.draw_prob, model.away_win_prob,
        odds_h if has_odds else 2.0, odds_d if has_odds else 3.2, odds_a if has_odds else 3.5,
        confidence="high" if confidence_raw >= 60 else "medium",
        data_grade=data_grade,
        model_disagreement_pts=(1.0 - (ensemble.agreement_score if ensemble else 0.5)) * 100
    ) if has_odds else {}

    conservative_narrative = generate_scouting_narrative(model, ensemble, market_comp)
    comparison_table_md = build_comparison_from_model(
        model.expected_home_goals, model.expected_away_goals,
        model.home_win_prob, model.away_win_prob, model.draw_prob,
        home_intel.get("elo"), away_intel.get("elo")
    )

    reasons_for = []
    reasons_against = []
    if best_edge > 5:
        reasons_for.append(f"Strong model-market edge: +{best_edge:.1f} pts")
    if ensemble and ensemble.agreement_score > 0.75:
        reasons_for.append(f"Estimators agree (score {ensemble.agreement_score:.2f})")
    if home_intel.get("elo") and away_intel.get("elo"):
        diff = home_intel["elo"] - away_intel["elo"]
        if best_outcome == "home" and diff > 50:
            reasons_for.append(f"ELO advantage home: +{diff:.0f} pts")
        elif best_outcome == "away" and diff < -50:
            reasons_for.append(f"ELO advantage away: +{abs(diff):.0f} pts")
    if confidence_raw >= 60:
        reasons_for.append(f"Good sample size, confidence {confidence_raw}")
    if h2h_summary.get("available") and h2h_summary.get("matches", 0) >= 3:
        reasons_for.append(f"H2H history: {h2h_summary.get('matches')} matches")
    if best_edge == 0 or best_edge < 1:
        reasons_against.append("No meaningful edge detected")
    if market_comp and market_comp.get("bookmaker_overround_pct", 0) > 8:
        reasons_against.append(f"High bookmaker margin: {market_comp['bookmaker_overround_pct']:.1f}%")
    if risk_score > 65:
        reasons_against.append("Elevated risk score — small sample or estimator disagreement")
    if data_grade in ("C", "D", "F"):
        reasons_against.append(f"Data quality grade {data_grade} — evidence is limited")

    narrative = generate_aggressive_narrative(model.__dict__, {
        "home_momentum": momentum_home,
        "away_momentum": momentum_away,
    }, ConfidenceTier[tier], best_outcome.capitalize() + " Win" if best_outcome != "none" else "")

    result = {
        "match_label": f"{home_form.team_name} vs {away_form.team_name}",
        "home_team": home_form.team_name,
        "away_team": away_form.team_name,
        "start_time": "",
        "competition_name": league_slug,
        "category": "",
        "venue": home_intel.get("stadium"),
        "manager_home": home_intel.get("manager"),
        "manager_away": away_intel.get("manager"),
        "squad_value_home": home_intel.get("squad_value"),
        "squad_value_away": away_intel.get("squad_value"),
        "elo_home": home_intel.get("elo"),
        "elo_away": away_intel.get("elo"),
        "expected_home_goals": round(model.expected_home_goals, 2),
        "expected_away_goals": round(model.expected_away_goals, 2),
        "xga_home": round(model.expected_away_goals, 2),
        "xga_away": round(model.expected_home_goals, 2),
        "home_win_prob": round(model.home_win_prob * 100, 1),
        "draw_prob": round(model.draw_prob * 100, 1),
        "away_win_prob": round(model.away_win_prob * 100, 1),
        "over_1_5_prob": round(multi.get("over_1_5_prob", 0) * 100, 1),
        "over_2_5_prob": round(model.over_2_5_prob * 100, 1),
        "over_3_5_prob": round(multi.get("over_3_5_prob", 0) * 100, 1),
        "under_2_5_prob": round(multi.get("under_2_5_prob", 0) * 100, 1),
        "btts_yes_prob": round(multi.get("btts_yes_prob", 0) * 100, 1),
        "double_chance_1x": round(multi.get("double_chance_1x", 0) * 100, 1),
        "double_chance_12": round(multi.get("double_chance_12", 0) * 100, 1),
        "double_chance_x2": round(multi.get("double_chance_x2", 0) * 100, 1),
        "correct_score_top5": multi.get("correct_score_top5", []),
        "most_likely_score": multi.get("most_likely_score", "0-0"),
        "home_over_0_5": round(multi.get("home_over_0_5", 0) * 100, 1),
        "home_over_1_5": round(multi.get("home_over_1_5", 0) * 100, 1),
        "away_over_0_5": round(multi.get("away_over_0_5", 0) * 100, 1),
        "away_over_1_5": round(multi.get("away_over_1_5", 0) * 100, 1),
        "best_outcome": best_outcome,
        "edge_pct": round(best_edge, 1),
        "ev_pct": round(ev_pct, 1),
        "fair_odds_best": round(fair_odds, 2),
        "confidence_tier": tier,
        "confidence_score": confidence_raw,
        "risk_score": round(risk_score, 0),
        "risk_label": risk_label,
        "stake_suggestion_pct": round(stake_pct, 2),
        "conservative_stake_pct": round(conservative_stake_obj.capped_fraction * 100, 2),
        "conservative_stake_note": conservative_stake_obj.note,
        "market_efficiency": efficiency_class,
        "market_assessments": market_assessments_data,
        "stake_recommendations": stake_recs_data,
        "conservative_scouting_narrative": conservative_narrative,
        "comparison_table_md": comparison_table_md,
        "implied_probs_list": implied_probs_list,
        "fair_probs_list": fair_probs_list,
        "data_grade": data_grade,
        "market_overround_pct": market_comp.get("bookmaker_overround_pct") if market_comp else None,
        "model_prob_best": round(best_model_prob * 100, 1) if best_outcome != "none" else None,
        "market_implied_best": round(best_market_implied, 1) if best_outcome != "none" else None,
        "model": model.__dict__,
        "market_comparison": market_comp,
        "momentum_home": momentum_home,
        "momentum_away": momentum_away,
        "h2h_summary": h2h_summary,
        "h2h_available": bool(h2h_raw),
        "h2h_matches": h2h_raw[:5] if h2h_raw else [],
        "scouting_narrative": narrative,
        "sample_size_home": home_form.matches_played,
        "sample_size_away": away_form.matches_played,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "elo_available": home_intel.get("elo") is not None,
        "xg_available": True,
        "lineups_confirmed": False,
        "injuries_verified": False,
        "conflicting_sources": ensemble.agreement_score < 0.6 if ensemble else False,
        "model_agreement_score": ensemble.agreement_score if ensemble else 0.5,
        "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson", "elo_prior"],
    }

    card = PredictionCard(
        match_label=result.get("match_label", ""),
        match_date=result.get("start_time", ""),
        competition=result.get("competition_name", ""),
        recommended_bet=result.get("best_outcome", ""),
        confidence_tier=result.get("confidence_tier", "NO_BET"),
        model_probability=result.get("model_prob_best", 0) / 100 if result.get("model_prob_best") else 0,
        market_implied_probability=result.get("market_implied_best", 0) / 100 if result.get("market_implied_best") else 0,
        edge_pct=result.get("edge_pct", 0),
        stake_suggestion_pct=result.get("stake_suggestion_pct", 0),
        signals={
            "home_momentum": momentum_home,
            "away_momentum": momentum_away,
        },
        multi_market_predictions=market_comp,
        scouting_narrative=narrative,
        model_data=model.__dict__,
    )

    try:
        ledger = PredictionLedger()
        ledger.log(
            model=model,
            agreement_score=ensemble.agreement_score if ensemble else 1.0,
        )
    except Exception:
        pass

    try:
        v2 = PredictionPipelineV2()
        v2_result = v2.predict(
            home_form, away_form, league_slug=league_slug,
            match_date=result.get("start_time", ""),
            odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            match_id=fixture.get("match_id", "") if fixture else "",
        )
        if v2_result:
            card.model_data = {**card.model_data, "pipeline_v2": v2_result}
    except Exception:
        pass

    result["prediction_card_object"] = card.__dict__
    return result


def _scan_fixture_worker(fixture: dict) -> Optional[dict]:
    try:
        oh = _safe_float(fixture.get("home_odd"))
        od = _safe_float(fixture.get("draw_odd"))
        oa = _safe_float(fixture.get("away_odd"))
        rec = _build_match_model(fixture["home_team"], fixture["away_team"], oh, od, oa, fixture=fixture)
        for k in ("start_time", "competition_name", "category", "match_id",
                  "home_odd", "draw_odd", "away_odd"):
            rec[k] = fixture.get(k, rec.get(k, ""))
        return rec
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-refresh logic
# ---------------------------------------------------------------------------

def _check_auto_refresh():
    interval = st.session_state.get("refresh_interval", 0)
    if interval <= 0:
        return
    elapsed = time.time() - st.session_state.get("last_refresh", 0)
    if elapsed >= interval:
        st.session_state.last_refresh = time.time()
        st.rerun()


# ---------------------------------------------------------------------------
# Top Value Bets table
# ---------------------------------------------------------------------------

def _tier_badge(tier: str) -> str:
    return f'<span class="tier-{tier}">{tier}</span>'


def _format_winning_team(row: dict) -> str:
    hp = row.get("home_win_prob", 0)
    dp = row.get("draw_prob", 0)
    ap = row.get("away_win_prob", 0)
    home_team = row.get("home_team", "Home")
    away_team = row.get("away_team", "Away")

    if hp >= ap and hp >= dp:
        return f"🏆 <b>{home_team}</b> ({hp:.1f}%)"
    elif ap >= hp and ap >= dp:
        return f"🏆 <b>{away_team}</b> ({ap:.1f}%)"
    else:
        return f"🤝 <b>Draw</b> ({dp:.1f}%)"


def _format_suggested_bet(row: dict) -> str:
    best_outcome = str(row.get("best_outcome", "none")).lower()
    tier = str(row.get("confidence_tier", "LEAN")).upper()
    badge = _tier_badge(tier)
    home_team = row.get("home_team", "Home")
    away_team = row.get("away_team", "Away")

    if "home" in best_outcome or best_outcome == "1":
        bet_desc = f"{home_team} Win"
    elif "away" in best_outcome or best_outcome == "2":
        bet_desc = f"{away_team} Win"
    elif "draw" in best_outcome or best_outcome == "x":
        bet_desc = "Draw"
    elif "over" in best_outcome or "2_5" in best_outcome:
        bet_desc = "Over 2.5 Goals"
    elif "btts" in best_outcome:
        bet_desc = "Both Teams To Score"
    else:
        bet_desc = f"{home_team} Double Chance (1X)"

    return f"🎯 <b>{bet_desc}</b> &nbsp; {badge}"


def _format_best_odds(row: dict) -> str:
    best = row.get("best_available_odds") or {}
    best_h = best.get("home") or {}
    if isinstance(best_h, dict) and best_h.get("odd"):
        return f"<b>{best_h.get('odd')}</b> <small>({best_h.get('site', 'JSON Link')})</small>"

    ho = row.get("home_odd") or row.get("offered_odds")
    if ho and float(ho) > 1.0:
        return f"<b>{ho}</b>"
    return "N/A"


def _render_top_value_bets_table(df: pd.DataFrame):
    if df.empty:
        st.warning("No value bets match current filters.")
        return

    table_df = df.copy()
    table_df["Tier"] = table_df["confidence_tier"].apply(_tier_badge)
    table_df["Edge"] = table_df["edge_pct"].apply(lambda x: f'+{x:.1f}%' if x > 0 else f'{x:.1f}%')
    table_df["EV"] = table_df["ev_pct"].apply(lambda x: f'+{x:.1f}%' if x > 0 else f'{x:.1f}%')
    table_df["Stake"] = table_df["stake_suggestion_pct"].apply(lambda x: f'{x:.2f}%')
    table_df["Grade"] = table_df["data_grade"]
    table_df["Risk"] = table_df["risk_label"]
    table_df["xG"] = table_df.apply(lambda r: f'{r["expected_home_goals"]:.2f}-{r["expected_away_goals"]:.2f}', axis=1)
    table_df["Possible Winning Team"] = table_df.apply(lambda r: _format_winning_team(r.to_dict()), axis=1)
    table_df["Suggested Bet to Place"] = table_df.apply(lambda r: _format_suggested_bet(r.to_dict()), axis=1)
    table_df["Best Scraped Odds"] = table_df.apply(lambda r: _format_best_odds(r.to_dict()), axis=1)
    table_df["Winning Prob %"] = table_df.apply(lambda r: f'{max(r.get("home_win_prob", 0), r.get("away_win_prob", 0), r.get("draw_prob", 0)):.1f}%', axis=1)
    table_df["EV %"] = table_df["ev_pct"].apply(lambda x: f'+{x:.1f}%' if x > 0 else f'{x:.1f}%')
    table_df["Suggested Stake"] = table_df["stake_suggestion_pct"].apply(lambda x: f'{x:.2f}% Bankroll')

    columns = [
        "match_label", "start_time", "competition_name", "best_outcome",
        "home_win_prob", "fair_odds_best", "home_odd", "Edge", "EV",
        "confidence_score", "risk_score", "Grade", "Tier", "Stake", "data_grade",
        "match_label", "competition_name", "Possible Winning Team",
        "Winning Prob %", "Best Scraped Odds", "Suggested Bet to Place",
        "EV %", "Suggested Stake", 
    ]
    rename_map = {
        "match_label": "Match / Fixture",
        "start_time": "Start Time",
        "competition_name": "Competition",
        "best_outcome": "Predicted Winner",
        "home_win_prob": "Model Prob %",
        "fair_odds_best": "Fair Odds",
        "home_odd": "Bookmaker Odds",
        "confidence_score": "Confidence",
        "risk_score": "Risk Score",
    }

    present = [c for c in columns if c in table_df.columns]
    display = table_df[present].rename(columns=rename_map)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(display.to_html(escape=False, index=False, classes=["dataframe"]), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Match Analysis Detail
# ---------------------------------------------------------------------------

def _render_match_detail(row: dict):
    st.markdown("---")
    st.subheader(f"{row['home_team']} vs {row['away_team']}")
    st.caption(f"**Competition:** {row.get('competition_name', 'N/A')} | **Grade:** {row.get('data_grade', 'N/A')} | "
               f"**Data Quality:** {row.get('data_grade', 'N/A')}")

    tier = row.get("confidence_tier", "N/A")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.metric("xG Home", f"{row['expected_home_goals']:.2f}")
    with c2:
        st.metric("xG Away", f"{row['expected_away_goals']:.2f}")
    with c3:
        st.metric("Over 2.5", f"{row['over_2_5_prob']:.1f}%")
    with c4:
        st.metric("BTTS", f"{row['btts_yes_prob']:.1f}%")
    with c5:
        st.metric("Confidence", f"{row['confidence_score']}")
    with c6:
        ev_val = row.get("ev_pct", 0)
        st.metric("EV", f"{ev_val:+.1f}%" if ev_val else "0.0%")
    with c7:
        st.metric("Risk", row.get("risk_label", "N/A"))

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        try:
            prob_df = pd.DataFrame({
                "Outcome": ["Home Win", "Draw", "Away Win"],
                "Probability": [row.get("home_win_prob", 0), row.get("draw_prob", 0), row.get("away_win_prob", 0)],
            })
            fig_probs = px.bar(
                prob_df,
                x="Outcome",
                y="Probability",
                title="Win Probabilities",
                color="Outcome",
                color_discrete_map={"Home Win": "#00E5FF", "Draw": "#7C3AED", "Away Win": "#EF4444"},
            )
            fig_probs.update_layout(yaxis_title="Probability (%)", showlegend=False)
            st.plotly_chart(fig_probs, use_container_width=True)
        except Exception:
            st.caption("Probability chart unavailable.")
    with chart_c2:
        try:
            gauge_val = min(max(row.get("confidence_score", 0), 0), 100)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                title={"text": "Confidence Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00E5FF"},
                    "steps": [
                        {"range": [0, 35], "color": "#EF4444"},
                        {"range": [35, 60], "color": "#F59E0B"},
                        {"range": [60, 100], "color": "#22C55E"},
                    ],
                },
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception:
            st.caption("Gauge chart unavailable.")

    tab_market, tab_edge, tab_team, tab_momentum, tab_models, tab_narrative, tab_report = st.tabs([
        "Markets", "Market Edge", "Team Comparison", "Momentum & Form", "Model Agreement", "Narrative", "Full Report"
    ])

    with tab_market:
        mkt = pd.DataFrame([
            {"Market": "Home Win", "Prob %": row["home_win_prob"]},
            {"Market": "Draw", "Prob %": row["draw_prob"]},
            {"Market": "Away Win", "Prob %": row["away_win_prob"]},
            {"Market": "Over 1.5", "Prob %": row["over_1_5_prob"]},
            {"Market": "Over 2.5", "Prob %": row["over_2_5_prob"]},
            {"Market": "Over 3.5", "Prob %": row["over_3_5_prob"]},
            {"Market": "BTTS Yes", "Prob %": row["btts_yes_prob"]},
            {"Market": "DC 1X", "Prob %": row["double_chance_1x"]},
            {"Market": "DC 12", "Prob %": row["double_chance_12"]},
            {"Market": "DC X2", "Prob %": row["double_chance_x2"]},
            {"Market": "Home O0.5", "Prob %": row["home_over_0_5"]},
            {"Market": "Home O1.5", "Prob %": row["home_over_1_5"]},
            {"Market": "Away O0.5", "Prob %": row["away_over_0_5"]},
            {"Market": "Away O1.5", "Prob %": row["away_over_1_5"]},
        ])
        st.dataframe(mkt, use_container_width=True, hide_index=True)
        top5 = row.get("correct_score_top5", [])
        if top5:
            st.write("**Top 5 Correct Scores**")
            cs_df = pd.DataFrame(top5)
            st.dataframe(cs_df, use_container_width=True, hide_index=True)
        st.write(f"**Most Likely Score:** {row.get('most_likely_score', 'N/A')}")

        try:
            exp_h = row.get("expected_home_goals", 1.5)
            exp_a = row.get("expected_away_goals", 1.5)
            max_goals = 6
            z = []
            for hg in range(max_goals + 1):
                row_probs = []
                for ag in range(max_goals + 1):
                    p = poisson_pmf(hg, exp_h) * poisson_pmf(ag, exp_a)
                    row_probs.append(p)
                z.append(row_probs)
            fig_hm = go.Figure(data=go.Heatmap(
                z=z,
                x=[str(g) for g in range(max_goals + 1)],
                y=[str(g) for g in range(max_goals + 1)],
                colorscale="Viridis",
                colorbar={"title": "Probability"},
                hoverongaps=False,
            ))
            fig_hm.update_layout(
                title="Goal Probability Heatmap",
                xaxis_title="Away Goals",
                yaxis_title="Home Goals",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception:
            st.caption("Heatmap unavailable.")

    with tab_edge:
        mc = row.get("market_comparison")
        if mc:
            comp_rows = [
                {"Outcome": "Home Win", "Model %": f"{mc['home']['model_prob_pct']:.1f}%",
                 "Market Fair %": f"{mc['home']['market_implied_pct']:.1f}%",
                 "Edge (pts)": f"{mc['home']['edge_pct_points']:+.1f}",
                 "EV %": f"{expected_value(mc['home']['model_prob_pct']/100, row.get('home_odd',0)) * 100:+.1f}%"},
                {"Outcome": "Draw", "Model %": f"{mc['draw']['model_prob_pct']:.1f}%",
                 "Market Fair %": f"{mc['draw']['market_implied_pct']:.1f}%",
                 "Edge (pts)": f"{mc['draw']['edge_pct_points']:+.1f}",
                 "EV %": f"{expected_value(mc['draw']['model_prob_pct']/100, row.get('draw_odd',0)) * 100:+.1f}%"},
                {"Outcome": "Away Win", "Model %": f"{mc['away']['model_prob_pct']:.1f}%",
                 "Market Fair %": f"{mc['away']['market_implied_pct']:.1f}%",
                 "Edge (pts)": f"{mc['away']['edge_pct_points']:+.1f}",
                 "EV %": f"{expected_value(mc['away']['model_prob_pct']/100, row.get('away_odd',0)) * 100:+.1f}%"},
            ]
            st.table(pd.DataFrame(comp_rows))
            st.caption(f"Bookmaker Overround: {mc.get('bookmaker_overround_pct', 'N/A')}% — {mc.get('note', '')}")
        else:
            st.info("No market odds available — enter odds to compare.")

    with tab_team:
        st.markdown("**Team Comparison**")
        home_elo = row.get("elo_home")
        away_elo = row.get("elo_away")
        if home_elo and away_elo:
            elo_edge = "Home" if home_elo > away_elo else ("Away" if away_elo > home_elo else "Even")
        else:
            elo_edge = "N/A"
        comparison_rows = [
            {"Metric": "ELO Rating", "Home": home_elo or "N/A", "Away": away_elo or "N/A", "Edge": elo_edge},
            {"Metric": "Expected Goals", "Home": f"{row.get('expected_home_goals',0):.2f}", "Away": f"{row.get('expected_away_goals',0):.2f}", "Edge": "Home" if row.get("expected_home_goals",0) > row.get("expected_away_goals",0) else "Away"},
            {"Metric": "xGA (Expected Goals Against)", "Home": f"{row.get('xga_home',0):.2f}", "Away": f"{row.get('xga_away',0):.2f}", "Edge": "Home" if row.get("xga_home",0) < row.get("xga_away",0) else "Away"},
            {"Metric": "Form Streak Home", "Home": row.get("momentum_home", {}).get("form_streak", "N/A"), "Away": row.get("momentum_away", {}).get("form_streak", "N/A"), "Edge": "—"},
            {"Metric": "Manager", "Home": row.get("manager_home") or "N/A", "Away": row.get("manager_away") or "N/A", "Edge": "—"},
            {"Metric": "Squad Value", "Home": row.get("squad_value_home") or "N/A", "Away": row.get("squad_value_away") or "N/A", "Edge": "—"},
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        if row.get("h2h_available") and row.get("h2h_matches"):
            st.write("**Recent Head-to-Head**")
            h2h_df = pd.DataFrame(row["h2h_matches"])
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)

    with tab_momentum:
        colm1, colm2 = st.columns(2)
        with colm1:
            st.write(f"**{row['home_team']} Form Momentum**")
            mh = row.get("momentum_home", {})
            if mh.get("available"):
                st.write(f"Last 5: {mh.get('last_5_record')} | Streak: {mh.get('form_streak')}")
                st.write(f"PPG: {mh.get('ppg')} | Trajectory: {mh.get('ppg_trajectory')}")
                st.write(f"Scoring: {mh.get('scoring_trend')} | Defensive: {mh.get('defensive_trend')}")
                st.write(f"Win Rate: {mh.get('win_rate_pct')}% | Clean Sheets: {mh.get('clean_sheet_pct')}%")
            else:
                st.caption("No momentum data.")
        with colm2:
            st.write(f"**{row['away_team']} Form Momentum**")
            ma = row.get("momentum_away", {})
            if ma.get("available"):
                st.write(f"Last 5: {ma.get('last_5_record')} | Streak: {ma.get('form_streak')}")
                st.write(f"PPG: {ma.get('ppg')} | Trajectory: {ma.get('ppg_trajectory')}")
                st.write(f"Scoring: {ma.get('scoring_trend')} | Defensive: {ma.get('defensive_trend')}")
                st.write(f"Win Rate: {ma.get('win_rate_pct')}% | Clean Sheets: {ma.get('clean_sheet_pct')}%")
            else:
                st.caption("No momentum data.")

    with tab_models:
        st.write("**Ensemble Estimators**")
        model_dict = row.get("model", {})
        model_rows = [
            {"Model": "Poisson GLM (scipy MLE)", "xG Home": f"{model_dict.get('expected_home_goals',0):.2f}", "xG Away": f"{model_dict.get('expected_away_goals',0):.2f}"},
            {"Model": "statsmodels GLM", "xG Home": "—", "xG Away": "—"},
            {"Model": "sklearn Poisson", "xG Home": "—", "xG Away": "—"},
            {"Model": f"Ensemble (agreement {row.get('model_agreement_score',0):.2f})", "xG Home": f"{row.get('expected_home_goals',0):.2f}", "xG Away": f"{row.get('expected_away_goals',0):.2f}"},
        ]
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        mc = row.get("market_comparison")
        if mc:
            model_outcomes = pd.DataFrame([
                {"Model": "Poisson+Ensemble", "Home %": row["home_win_prob"], "Draw %": row["draw_prob"], "Away %": row["away_win_prob"]},
                {"Model": "Market Fair", "Home %": f"{mc['home']['market_implied_pct']:.1f}", "Draw %": f"{mc['draw']['market_implied_pct']:.1f}", "Away %": f"{mc['away']['market_implied_pct']:.1f}"},
            ])
            st.dataframe(model_outcomes, use_container_width=True, hide_index=True)

            try:
                outcomes = ["Home Win", "Draw", "Away Win"]
                model_vals = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
                market_vals = [mc["home"]["market_implied_pct"], mc["draw"]["market_implied_pct"], mc["away"]["market_implied_pct"]]
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(name="Model", x=outcomes, y=model_vals, marker_color="#00E5FF"))
                fig_cmp.add_trace(go.Bar(name="Market Fair", x=outcomes, y=market_vals, marker_color="#7C3AED"))
                fig_cmp.update_layout(barmode="group", title="Model vs Market Probabilities", yaxis_title="Probability (%)")
                st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception:
                st.caption("Comparison chart unavailable.")

    with tab_narrative:
        st.write(row.get("scouting_narrative", "Narrative unavailable."))
        if row.get("reasons_for"):
            st.write("**Reasons For:**")
            for r in row["reasons_for"]:
                st.write(f"- {r}")
        if row.get("reasons_against"):
            st.write("**Reasons Against:**")
            for r in row["reasons_against"]:
                st.write(f"- {r}")
        st.caption("Note: Past accuracy does not guarantee future results. Markets adapt. Treat as probabilistic estimates only.")

    with tab_report:
        try:
            oh = _safe_float(row.get("home_odd"))
            od = _safe_float(row.get("draw_odd"))
            oa = _safe_float(row.get("away_odd"))
            if oh > 1 and od > 1 and oa > 1:
                report_md = render_match_report(
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    league=row.get("competition_name", ""),
                    match_date=row.get("start_time", datetime.now(timezone.utc).isoformat()),
                    model_home_prob=row["home_win_prob"] / 100,
                    model_draw_prob=row["draw_prob"] / 100,
                    model_away_prob=row["away_win_prob"] / 100,
                    expected_home_goals=row["expected_home_goals"],
                    expected_away_goals=row["expected_away_goals"],
                    odds_home=oh,
                    odds_draw=od,
                    odds_away=oa,
                    confidence="high" if row["confidence_score"] >= 60 else ("medium" if row["confidence_score"] >= 40 else "low"),
                    data_grade=row.get("data_grade", "C"),
                    model_disagreement_pts=(1 - row.get("model_agreement_score", 0.5)) * 100,
                    home_elo=row.get("elo_home"),
                    away_elo=row.get("elo_away"),
                    reasons_for=row.get("reasons_for", []),
                    reasons_against=row.get("reasons_against", []),
                )
                st.markdown(report_md)
            else:
                st.info("Enter bookmaker odds to generate full report.")
        except Exception as ex:
            st.info(f"Report generation unavailable: {ex}")


# ---------------------------------------------------------------------------
# Page: Today's Value Bets Dashboard
# ---------------------------------------------------------------------------

def render_value_bets_dashboard():
    st.markdown('<div class="main-header">Institutional Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Poisson GLM · Bayesian · ELO · Monte Carlo · Ensemble — Automated EV Ranking</div>',
                unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    preds = _sorted_predictions()
    with col_m1:
        st.metric("Total Analyzed", len(preds))
    with col_m2:
        tiers = {"LOCK": 0, "STRONG": 0, "VALUE": 0}
        for p in preds:
            t = p.get("confidence_tier")
            if t in tiers:
                tiers[t] += 1
        st.metric("Value Bets", f'{tiers["LOCK"]+tiers["STRONG"]+tiers["VALUE"]}', delta="LOCK/STRONG/VALUE")
    with col_m3:
        no_bet = sum(1 for p in preds if p.get("confidence_tier") == "NO_BET")
        st.metric("NO BET", no_bet)
    with col_m4:
        avg_ev = np.mean([p.get("ev_pct", 0) for p in preds]) if preds else 0
        st.metric("Avg EV", f"{avg_ev:+.1f}%")
    with col_m5:
        avg_conf = np.mean([p.get("confidence_score", 0) for p in preds]) if preds else 0
        st.metric("Avg Confidence", f"{avg_conf:.0f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if preds:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            ev_vals = [p.get("ev_pct", 0) for p in preds if p.get("ev_pct", 0) != 0]
            if ev_vals:
                fig_ev = px.histogram(
                    x=ev_vals,
                    nbins=20,
                    title="EV % Distribution",
                    labels={"x": "Expected Value (%)", "y": "Count"},
                    color_discrete_sequence=["#00E5FF"],
                )
                fig_ev.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_ev, use_container_width=True)
        with chart_col2:
            tier_counts = {}
            for p in preds:
                t = p.get("confidence_tier", "NO_BET")
                tier_counts[t] = tier_counts.get(t, 0) + 1
            if tier_counts:
                fig_tier = px.pie(
                    names=list(tier_counts.keys()),
                    values=list(tier_counts.values()),
                    title="Prediction Tier Breakdown",
                    color=list(tier_counts.keys()),
                    color_discrete_map={
                        "LOCK": "#EF4444",
                        "STRONG": "#F59E0B",
                        "VALUE": "#22C55E",
                        "LEAN": "#7C3AED",
                        "NO_BET": "#64748B",
                    },
                )
                st.plotly_chart(fig_tier, use_container_width=True)

    st.subheader("Top Value Bets (Ranked by Expected Value)")
    df = pd.DataFrame(preds) if preds else pd.DataFrame()
    if not df.empty:
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 3])
        with col_f1:
            league_filter = st.multiselect("League", options=TOP_LEAGUES, default=st.session_state.league_filter, key="val_league")
        with col_f2:
            day_mode = st.selectbox("Day", ["Today", "Tomorrow", "Next 3 Days", "This Week", "All Days"],
                                    index=4, key="val_day")
        with col_f3:
            tier_filter = st.multiselect("Tier", options=["LOCK", "STRONG", "VALUE", "LEAN", "NO_BET"],
                                         default=["LOCK", "STRONG", "VALUE"], key="val_tier")
        with col_f4:
            search_term = st.text_input("Search", placeholder="Filter by team or league...", key="val_search")

        df["_dt"] = df["start_time"].apply(_parse_start_time)
        now_utc = datetime.now(timezone.utc)
        df["_day"] = df["_dt"].apply(lambda d: _day_label(d, now_utc))

        if day_mode == "Today":
            mask = df["_day"] == "Today"
        elif day_mode == "Tomorrow":
            mask = df["_day"] == "Tomorrow"
        elif day_mode == "Next 3 Days":
            mask = df["_dt"].apply(lambda d: d is not None and 0 <= (d.date() - now_utc.date()).days <= 3)
        elif day_mode == "This Week":
            mask = df["_dt"].apply(lambda d: d is not None and 0 <= (d.date() - now_utc.date()).days <= 7)
        else:
            mask = pd.Series([True] * len(df))
        df = df[mask].copy()

        if league_filter:
            df = df[df["competition_name"].apply(lambda c: any(l.lower() in str(c).lower() for l in league_filter))].copy()
        if tier_filter:
            df = df[df["confidence_tier"].isin(tier_filter)].copy()
        if search_term:
            df = df[df["match_label"].str.contains(search_term, case=False, na=False) |
                    df["competition_name"].str.contains(search_term, case=False, na=False)].copy()

        df = df.sort_values(by="ev_pct", ascending=False)
        _render_top_value_bets_table(df)

        with st.expander("Download Full Analysis"):
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="value_bets.csv", mime="text/csv")
            if st.button("Export Session JSON"):
                json_data = json.dumps(preds, default=str, indent=2).encode("utf-8")
                st.download_button("Download JSON", data=json_data, file_name="predictions_session.json", mime="application/json")

        if not df.empty:
            selected = st.selectbox("Select match for deep analysis", options=df["match_label"].tolist(), index=0, key="val_select")
            sel_row = df[df["match_label"] == selected]
            if not sel_row.empty:
                _render_match_detail(sel_row.iloc[0].to_dict())
    else:
        st.info("No predictions available. Go to **Betika Fixtures** and click **Auto-Scan All Fixtures**, or use **Manual Input** to build models.")


# ---------------------------------------------------------------------------
# Page: Match Analyzer
# ---------------------------------------------------------------------------

def render_match_analyzer():
    st.markdown('<div class="main-header">Match Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive analysis with multi-model ensemble, team comparison, and EV quantification</div>',
                unsafe_allow_html=True)

    tab_source, tab_manual = st.tabs(["From Feed", "Manual Input"])

    with tab_source:
        sorted_preds = _sorted_predictions()
        if not sorted_preds:
            st.info("No predictions in feed. Use Manual Input or Betika Fixtures to add matches.")
        else:
            labels = [f"[{p.get('confidence_tier', 'NO_BET')}] {p['home_team']} vs {p['away_team']}" for p in sorted_preds]
            sel_label = st.selectbox("Choose fixture", options=labels, key="analyzer_feed_select")
            idx = labels.index(sel_label) if sel_label in labels else 0
            selected_pred = sorted_preds[idx]
            if st.button("Re-Scan This Match", type="primary"):
                with st.spinner(f"Re-scanning {sel_label}..."):
                    try:
                        oh = _safe_float(selected_pred.get("home_odd", 0))
                        od = _safe_float(selected_pred.get("draw_odd", 0))
                        oa = _safe_float(selected_pred.get("away_odd", 0))
                        rec = _build_match_model(selected_pred["home_team"], selected_pred["away_team"], oh, od, oa)
                        rec["start_time"] = selected_pred.get("start_time", "")
                        rec["competition_name"] = selected_pred.get("competition_name", "")
                        rec["category"] = selected_pred.get("category", "")
                        match_label = selected_pred.get("match_label", "")
                        for i, p in enumerate(st.session_state.predictions):
                            if p.get("match_label") == match_label:
                                st.session_state.predictions[i] = rec
                                break
                        st.success("Match re-scanned.")
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))
            if sorted_preds:
                _render_match_detail(selected_pred)

    with tab_manual:
        colh, cola = st.columns(2)
        with colh:
            home_name = st.text_input("Home Team Name", "Arsenal", key="manual_home")
        with cola:
            away_name = st.text_input("Away Team Name", "Chelsea", key="manual_away")

        with st.expander("Model Parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                decay = st.slider("Recency Decay", 0.50, 1.00, 0.92, 0.01, key="manual_decay")
            with c2:
                shrinkage_k = st.slider("Bayesian Shrinkage K", 0.1, 20.0, 6.0, 0.5, key="manual_k")
            with c3:
                home_adv = st.slider("Home Advantage", 0.5, 2.0, 1.0, 0.05, key="manual_hadv")

        with st.expander("Bookmaker Odds (Optional)", expanded=False):
            o1, o2, o3 = st.columns(3)
            with o1:
                odds_h = st.number_input("Home Odds", min_value=1.0, value=2.10, step=0.05, key="manual_oh")
            with o2:
                odds_d = st.number_input("Draw Odds", min_value=1.0, value=3.40, step=0.05, key="manual_od")
            with o3:
                odds_a = st.number_input("Away Odds", min_value=1.0, value=3.60, step=0.05, key="manual_oa")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            build = st.button("Build Model", type="primary", use_container_width=True, key="manual_build")
        with col_b2:
            add_to_feed = st.checkbox("Add to Feed", value=True, key="manual_add_feed")

        if build:
            with st.spinner(f"Building model for {home_name} vs {away_name}..."):
                try:
                    rec = _build_match_model(home_name, away_name, odds_h, odds_d, odds_a, decay, shrinkage_k, home_adv)
                    if add_to_feed:
                        st.session_state.predictions.append(rec)
                        st.success(f"Model built and added to feed ({len(st.session_state.predictions)} total).")
                    else:
                        st.success("Model built.")
                    _render_match_detail(rec)
                except Exception as ex:
                    st.error(f"Error: {ex}")


# ---------------------------------------------------------------------------
# Page: Betika Fixtures (Today's Fixtures + Live)
# ---------------------------------------------------------------------------

def render_betika_fixtures():
    st.markdown('<div class="main-header">Today\'s Fixtures & Live Matches</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated ingestion from Betika (betika.com/en-ke) — normalize, filter, one-click analyze</div>',
                unsafe_allow_html=True)

    tab_up, tab_live = st.tabs(["Today's Fixtures", "Live Matches"])

    @st.cache_data(ttl=300)
    def _load_upcoming(_key: str = "upcoming_v1") -> list[dict]:
        betika = BetikaClient()
        return betika.get_all_fixtures()

    @st.cache_data(ttl=60)
    def _load_live(_key: str = "live_v1") -> list[dict]:
        betika = BetikaClient()
        return betika.get_live_matches() or []

    with tab_up:
        try:
            fixtures = _load_upcoming()
            if not fixtures:
                st.warning("No upcoming fixtures available.")
            else:
                df = pd.DataFrame(fixtures).reindex(
                    columns=["match_id", "home_team", "away_team", "start_time",
                             "competition_name", "home_odd", "draw_odd", "away_odd", "category"]
                )
                league_opts = sorted(df["competition_name"].dropna().unique().tolist())
                col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
                with col_f1:
                    f_league = st.multiselect("Filter league", options=league_opts, key="up_league_filter")
                with col_f2:
                    f_search = st.text_input("Search team", key="up_team_search", placeholder="Type team name...")
                with col_f3:
                    only_odds = st.checkbox("Only with odds", value=True, key="up_only_odds")
                if f_league:
                    df = df[df["competition_name"].isin(f_league)]
                if f_search:
                    df = df[df["home_team"].str.contains(f_search, case=False, na=False) |
                            df["away_team"].str.contains(f_search, case=False, na=False)]
                if only_odds:
                    df = df[df["home_odd"].notna() & df["draw_odd"].notna() & df["away_odd"].notna()]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(df)} fixtures")
                if st.session_state.get("auto_analyze", False) and not st.session_state.get("auto_analyze_running", False):
                    st.session_state.auto_analyze_running = True
                    progress = st.progress(0.0, text="Auto-analyzing all fixtures...")
                    results = []
                    errors = []
                    for i, (idx, row) in enumerate(df.iterrows()):
                        rec = _analyze_fixture_row(row)
                        if rec and "_error" not in rec:
                            results.append(rec)
                        elif rec and "_error" in rec:
                            errors.append(rec["_error"])
                        progress.progress((i + 1) / len(df), text=f"Processed {i + 1}/{len(df)}")
                    st.session_state.predictions.extend(results)
                    st.session_state.auto_analyze_running = False
                    if errors:
                        st.warning(f"Auto-analyzed: added {len(results)}, {len(errors)} failed.")
                    else:
                        st.success(f"Auto-analyzed {len(results)} fixtures. Results are now in the feed.")
                    progress.progress(1.0, text="Done.")
                _batch_analyze(df, "Today")
        except Exception as ex:
            st.error(f"Error fetching upcoming fixtures: {ex}")

    with tab_live:
        try:
            live = _load_live()
            if not live:
                st.warning("No live matches currently.")
            else:
                df = pd.DataFrame(live).reindex(
                    columns=["match_id", "home_team", "away_team", "start_time",
                             "competition_name", "home_odd", "draw_odd", "away_odd", "category"]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Showing {len(df)} live matches")
                if st.session_state.get("auto_analyze", False) and not st.session_state.get("auto_analyze_running", False):
                    st.session_state.auto_analyze_running = True
                    progress = st.progress(0.0, text="Auto-analyzing live matches...")
                    results = []
                    errors = []
                    for i, (idx, row) in enumerate(df.iterrows()):
                        rec = _analyze_fixture_row(row)
                        if rec and "_error" not in rec:
                            results.append(rec)
                        elif rec and "_error" in rec:
                            errors.append(rec["_error"])
                        progress.progress((i + 1) / len(df), text=f"Processed {i + 1}/{len(df)}")
                    st.session_state.predictions.extend(results)
                    st.session_state.auto_analyze_running = False
                    if errors:
                        st.warning(f"Auto-analyzed: added {len(results)}, {len(errors)} failed.")
                    else:
                        st.success(f"Auto-analyzed {len(results)} live matches. Results are now in the feed.")
                    progress.progress(1.0, text="Done.")
                _batch_analyze(df, "Live")
        except Exception as ex:
            st.error(f"Error fetching live matches: {ex}")


def _analyze_fixture_row(row) -> Optional[dict]:
    try:
        oh = _safe_float(row.get("home_odd"))
        od = _safe_float(row.get("draw_odd"))
        oa = _safe_float(row.get("away_odd"))
        rec = _build_match_model(row["home_team"], row["away_team"], oh, od, oa)
        for k in ("start_time", "competition_name", "category", "match_id", "home_odd", "draw_odd", "away_odd"):
            rec[k] = row.get(k, rec.get(k, ""))
        return rec
    except Exception as ex:
        return {"_error": f"{row['home_team']} vs {row['away_team']}: {str(ex)[:120]}"}


def _batch_analyze(df: pd.DataFrame, context: str):
    if df.empty:
        return
    st.markdown(f"**Build Market Models for {context} Fixtures**")
    col_a1, col_a2, col_a3 = st.columns([2, 2, 2])
    with col_a1:
        if st.button(f"Auto-Scan All {context} Fixtures", key=f"btn_{context}_auto", type="primary"):
            progress = st.progress(0.0, text="Analyzing all fixtures...")
            results = []
            errors = []
            for i, (idx, row) in enumerate(df.iterrows()):
                rec = _analyze_fixture_row(row)
                if rec and "_error" not in rec:
                    results.append(rec)
                elif rec and "_error" in rec:
                    errors.append(rec["_error"])
                progress.progress((i + 1) / len(df), text=f"Processed {i + 1}/{len(df)}")
            st.session_state.predictions.extend(results)
            if errors:
                st.warning(f"Added {len(results)} predictions. {len(errors)} failed:\n" + "\n".join(errors[:10]))
            else:
                st.success(f"Added {len(results)} predictions to the feed.")
            progress.progress(1.0, text="Done.")
    with col_a2:
        st.caption(f"Total fixtures available: {len(df)}")
    with col_a3:
        st.caption(f"Currently in feed: {len(_sorted_predictions())}")

    options = [f"{r.home_team} vs {r.away_team} — {r.competition_name}" for r in df.itertuples()]
    selected = st.multiselect("Select fixtures to analyze", options=options, key=f"sel_{context}_batch")
    if selected and st.button("Analyze Selected", key=f"btn_{context}_batch", type="primary"):
        progress = st.progress(0.0, text="Analyzing...")
        results = []
        errors = []
        for i, sel in enumerate(selected):
            idx = options.index(sel)
            row = df.iloc[idx]
            rec = _analyze_fixture_row(row)
            if rec and "_error" not in rec:
                results.append(rec)
            elif rec and "_error" in rec:
                errors.append(rec["_error"])
            progress.progress((i + 1) / len(selected), text=f"Processed {i + 1}/{len(selected)}")
        st.session_state.predictions.extend(results)
        if errors:
            st.warning(f"Added {len(results)} predictions. {len(errors)} failed:\n" + "\n".join(errors[:10]))
        else:
            st.success(f"Added {len(results)} predictions to the feed. Open Dashboard to view.")
        progress.progress(1.0, text="Done.")


# ---------------------------------------------------------------------------
# Page: Historical Performance
# ---------------------------------------------------------------------------

def render_historical_performance():
    st.markdown('<div class="main-header">Historical Performance & Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ROI, hit rate, Brier score, calibration, and bankroll simulation</div>',
                unsafe_allow_html=True)

    try:
        from backend.backtest import full_accuracy_report
        if st.session_state.predictions:
            sample_preds = []
            for p in st.session_state.predictions:
                if p.get("market_comparison"):
                    sample_preds.append({
                        "home_team": p["home_team"],
                        "away_team": p["away_team"],
                        "outcome_backed": p.get("best_outcome", "home"),
                        "model_prob": p.get("model_prob_best", 0) / 100,
                        "odds_taken": _safe_float(p.get("home_odd", 0)),
                        "actual_result": "H",
                    })
            if sample_preds:
                report = full_accuracy_report(sample_preds)
                st.json(report)
            else:
                st.info("No predictions with odds to backtest.")
        else:
            st.info("Build models first to enable backtesting.")
    except Exception as ex:
        st.error(f"Backtest module error: {ex}")

    st.subheader("Prediction Ledger")
    try:
        from backend.intelligence import PredictionLedger
        ledger = PredictionLedger()
        cal = ledger.calibration_report()
        st.json(cal)
    except Exception as ex:
        st.caption(f"Ledger unavailable: {ex}")


# ---------------------------------------------------------------------------
# Page: Team Intel
# ---------------------------------------------------------------------------

def render_team_intel():
    st.markdown('<div class="main-header">Team Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Wikipedia overview · ELO · Squad value · Form momentum · H2H</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        query_home = st.text_input("Home Team", placeholder="e.g. Arsenal", key="team_intel_home")
    with col2:
        query_away = st.text_input("Away Team", placeholder="e.g. Chelsea", key="team_intel_away")

    if query_home and len(query_home) >= 2:
        try:
            espn = ESPNScraperClient()
            res = espn.search_team(query_home)
            if res:
                info = res[0]
                league = info.get("league", "N/A")
                team_id = info.get("id", "")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Name:** {info.get('name')}")
                    st.write(f"**League:** {league}")
                with col_b:
                    form = espn.fetch_recent_matches(team_id, league)
                    momentum = FormMomentumCalculator.calculate(form)
                    st.write(f"**Matches:** {form.matches_played}")
                    st.write(f"**Form:** {momentum.get('last_5_record', 'N/A')}")
                try:
                    elo = EloRatingScraper().get_club_elo(query_home, form=form)
                    st.write(f"**ELO:** {elo:.0f}")
                except Exception:
                    pass
        except Exception as ex:
            st.error(f"Error: {ex}")

    if query_away and len(query_away) >= 2:
        try:
            espn = ESPNScraperClient()
            res = espn.search_team(query_away)
            if res:
                info = res[0]
                league = info.get("league", "N/A")
                team_id = info.get("id", "")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Name:** {info.get('name')}")
                    st.write(f"**League:** {league}")
                with col_b:
                    form = espn.fetch_recent_matches(team_id, league)
                    momentum = FormMomentumCalculator.calculate(form)
                    st.write(f"**Matches:** {form.matches_played}")
                    st.write(f"**Form:** {momentum.get('last_5_record', 'N/A')}")
                try:
                    elo = EloRatingScraper().get_club_elo(query_away, form=form)
                    st.write(f"**ELO:** {elo:.0f}")
                except Exception:
                    pass
        except Exception as ex:
            st.error(f"Error: {ex}")


# ---------------------------------------------------------------------------
# Page: Betting Sites Directory
# ---------------------------------------------------------------------------

def render_data_sources():
    st.markdown('<div class="main-header">Global Betting Sites & Links Scraper</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live Scraping, Data Cleaning, and Line Shopping from football_betting_sites.json</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        search_q = st.text_input("Search Betting Site or Domain", "")
    with col2:
        scrape_btn = st.button("⚡ Scrape All Links Now", type="primary")

    if scrape_btn:
        with st.spinner("Scraping all site links and cleaning data using pandas/polars..."):
            try:
                from backend.scraper import json_site_scraper
                report = json_site_scraper.scrape_all_sites(team_query=search_q if search_q else None, limit=30)
                st.session_state["scraped_sites_report"] = report
                st.success(f"Scraped {report.get('total_sites')} sites successfully!")
            except Exception as ex:
                st.error(f"Scraping error: {ex}")

    report = st.session_state.get("scraped_sites_report")
    if report:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Sites", report.get("total_sites", 0))
        with c2:
            st.metric("Successful Scrapes", report.get("successful_sites", 0))
        with c3:
            st.metric("Sites with Valid Odds", report.get("sites_with_odds", 0))
        with c4:
            st.metric("Failed / Blocked", report.get("failed_sites", 0))

        line_shop = report.get("line_shopping", {})
        if line_shop:
            st.subheader("💡 Best Line Shopping Odds Found Across Scraped Links")
            ls_col1, ls_col2, ls_col3 = st.columns(3)
            with ls_col1:
                bh = line_shop.get("best_home_odds")
                if bh:
                    st.metric("Best Home Odds", f"{bh.get('odd')}", delta=bh.get("site"))
            with ls_col2:
                bd = line_shop.get("best_draw_odds")
                if bd:
                    st.metric("Best Draw Odds", f"{bd.get('odd')}", delta=bd.get("site"))
            with ls_col3:
                ba = line_shop.get("best_away_odds")
                if ba:
                    st.metric("Best Away Odds", f"{ba.get('odd')}", delta=ba.get("site"))

        st.subheader("Cleaned & Validated Site Data Matrix")
        sites_data = report.get("sites_data", [])
        if sites_data and pd is not None:
            st.dataframe(pd.DataFrame(sites_data), use_container_width=True, hide_index=True)

    sites = betting_site_registry.search_sites(search_q)
    st.write(f"Showing **{len(sites)}** registered betting sites in JSON registry:")
    if pd is not None:
        st.dataframe(pd.DataFrame(sites), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: System Health & Automation Status
# ---------------------------------------------------------------------------

def render_system_health():
    st.markdown('<div class="main-header">System Health & Automation Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Operational metrics, data source status, and automation configuration</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "Healthy", delta="Operational")
    with col2:
        st.metric("Predictions in Feed", len(_sorted_predictions()))
    with col3:
        st.metric("Betting Sites", len(betting_site_registry.get_all_sites()))
    with col4:
        last_ref = datetime.fromtimestamp(st.session_state.get("last_refresh", 0)).strftime("%H:%M:%S")
        st.metric("Last Refresh", last_ref)

    st.subheader("Automation Controls")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        refresh_label = st.selectbox("Auto-refresh interval", options=list(REFRESH_OPTIONS.keys()), index=0)
        st.session_state.refresh_interval = REFRESH_OPTIONS[refresh_label]
    with col_a2:
        if st.button("Refresh Now", type="primary"):
            st.session_state.last_refresh = time.time()
            st.rerun()

    st.subheader("Data Sources & Status")
    sources = pd.DataFrame([
        {"Source": "Betika API", "Status": "Active", "Coverage": "East Africa fixtures, odds, live"},
        {"Source": "ESPN API", "Status": "Active", "Coverage": "Global team form, league averages"},
        {"Source": "Wikipedia", "Status": "Active", "Coverage": "Club overviews, managers, stadiums"},
        {"Source": "ClubELO / Empirical", "Status": "Active (fallback)", "Coverage": "Live ELO ratings"},
        {"Source": "yfinance", "Status": "Active", "Coverage": "Club stock / market cap"},
        {"Source": "ScraperCache (SQLite)", "Status": "Active", "Coverage": "Cached ESPN/Betika responses"},
        {"Source": "Poisson GLM (scipy)", "Status": "Active", "Coverage": "MLE attack/defense ratings"},
        {"Source": "statsmodels GLM", "Status": "Active", "Coverage": "Poisson regression ensemble"},
        {"Source": "sklearn Poisson", "Status": "Active", "Coverage": "Regularized Poisson ensemble"},
        {"Source": "aiBetModel Market Engine", "Status": "Active", "Coverage": "EV, Kelly, tier classification"},
        {"Source": "PredictionLedger", "Status": "Active", "Coverage": "SQLite audit trail, Brier, calibration"},
        {"Source": "PredictionPipelineV2", "Status": "Active", "Coverage": "Next-gen ensemble with ML + calibration"},
        {"Source": "PaperTradingTracker", "Status": "Active", "Coverage": "Bankroll simulation, P&L, hit rate"},
        {"Source": "Plotly Charts", "Status": "Active", "Coverage": "EV histograms, tier pies, gauges, P&L curves"},
        {"Source": "streamlit-option-menu", "Status": "Active", "Coverage": "Icon-based sidebar navigation"},
    ])
    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.subheader("Automation Log")
    log_entries = [f"[{datetime.now().strftime('%H:%M:%S')}] System initialized — {len(st.session_state.predictions)} predictions loaded"]
    st.text_area("Automation Log", value="\n".join(log_entries), height=120, disabled=True)


# ---------------------------------------------------------------------------
# Page: Bankroll & Portfolio
# ---------------------------------------------------------------------------

def render_bankroll_portfolio():
    st.markdown('<div class="main-header">Bankroll & Portfolio Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Paper trading, stake sizing, and portfolio analytics</div>',
                unsafe_allow_html=True)

    try:
        from backend.monitoring import PaperTradingTracker
        from backend.intelligence import PredictionLedger

        if "bankroll_tracker" not in st.session_state:
            st.session_state.bankroll_tracker = PaperTradingTracker(starting_bankroll=1000.0)

        tracker = st.session_state.bankroll_tracker
        ledger = PredictionLedger()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Starting Bankroll", f"${tracker.starting_bankroll:.2f}")
        with col2:
            st.metric("Current Bankroll", f"${tracker.bankroll:.2f}", delta=f"{tracker.bankroll - tracker.starting_bankroll:+.2f}")
        with col3:
            summary = tracker.summary()
            st.metric("Total P&L", f"${summary['total_pnl']:+.2f}", delta=f"ROI {summary['roi_pct']:+.1f}%")
        with col4:
            st.metric("Active Bets", summary["n_pending"], delta=f"Win rate {summary['hit_rate_pct']:.0f}%")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Place Paper Trade")
            preds = _sorted_predictions()
            if preds:
                options = [f"[{p.get('confidence_tier', 'NO_BET')}] {p['home_team']} vs {p['away_team']} — {p.get('best_outcome', 'N/A')}" for p in preds]
                sel = st.selectbox("Select prediction", options=options, key="bankroll_sel")
                idx = options.index(sel) if sel in options else 0
                p = preds[idx]
                stake_pct = st.slider("Stake % of bankroll", 0.1, 10.0, float(p.get("stake_suggestion_pct", 1.0)), 0.1, key="bankroll_stake")
                if st.button("Place Paper Bet", type="primary", use_container_width=True, key="bankroll_place"):
                    try:
                        tracker.record_bet(
                            p["match_label"],
                            p.get("best_outcome", "home"),
                            p.get("model_prob_best", 50) / 100,
                            _safe_float(p.get("home_odd", 0)) or 2.0,
                            stake_pct / 100.0,
                        )
                        st.success("Paper bet placed.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error placing bet: {ex}")
            else:
                st.info("No predictions available. Build models first.")

        with col_b:
            st.subheader("Settle Result")
            pending = [t for t in tracker._trades if t.result is None]
            if pending:
                settle_options = [f"{t.match_label} — {t.outcome.upper()}" for t in pending]
                settle_sel = st.selectbox("Pending bets", options=settle_options, key="bankroll_settle")
                settle_idx = settle_options.index(settle_sel) if settle_sel in settle_options else 0
                t = pending[settle_idx]
                result_choice = st.selectbox("Actual result", ["H", "D", "A"], key="bankroll_result")
                if st.button("Settle Bet", type="primary", use_container_width=True, key="bankroll_settle_btn"):
                    try:
                        tracker.settle(t.match_label, result_choice)
                        st.success("Bet settled.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error settling bet: {ex}")
            else:
                st.info("No pending bets.")

        st.subheader("Trade History")
        history = [t for t in tracker._trades if t.result is not None]
        if history:
            pnl_series = [t.pnl or 0.0 for t in history]
            cumulative = np.cumsum(pnl_series)
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(y=cumulative, mode="lines+markers", name="Cumulative P&L", line={"color": "#00E5FF"}))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pnl.update_layout(title="Cumulative P&L", xaxis_title="Trade #", yaxis_title="USD")
            st.plotly_chart(fig_pnl, use_container_width=True)

            hist_df = pd.DataFrame([
                {
                    "Match": t.match_label,
                    "Backed": t.outcome.upper(),
                    "Model Prob": f"{t.model_prob:.1%}",
                    "Odds": f"{t.odds_taken:.2f}",
                    "Stake %": f"{t.stake_fraction * 100:.1f}%",
                    "P&L": f"${t.pnl:+.2f}" if t.pnl else "—",
                    "Result": t.result,
                }
                for t in history
            ])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

            if st.button("Clear History", key="bankroll_clear"):
                tracker._trades = [t for t in tracker._trades if t.result is None]
                st.success("History cleared.")
                st.rerun()
        else:
            st.caption("No settled trades yet.")

    except Exception as ex:
        st.error(f"Bankroll module unavailable: {ex}")


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------

def render_settings():
    st.markdown('<div class="main-header">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Model parameters, defaults, and preferences</div>',
                unsafe_allow_html=True)

    st.subheader("Model Defaults")
    c1, c2, c3 = st.columns(3)
    with c1:
        decay = st.slider("Recency Decay", 0.50, 1.00, 0.92, 0.01, key="set_decay")
        st.session_state["default_decay"] = decay
    with c2:
        shrinkage_k = st.slider("Bayesian Shrinkage K", 0.1, 20.0, 6.0, 0.5, key="set_k")
        st.session_state["default_shrinkage_k"] = shrinkage_k
    with c3:
        home_adv = st.slider("Home Advantage", 0.5, 2.0, 1.0, 0.05, key="set_hadv")
        st.session_state["default_home_adv"] = home_adv

    st.subheader("Display")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.session_state["default_league"] = st.selectbox(
            "Default League",
            ["eng.1", "es.1", "de.1", "it.1", "fr.1", "ken.1", "ng.1", "za.1"],
            index=0,
            key="set_league",
        )
    with col_d2:
        st.session_state["currency"] = st.selectbox("Currency", ["USD", "EUR", "GBP", "KES", "NGN", "ZAR"], index=0, key="set_currency")

    st.subheader("Data & Cache")
    if st.button("Clear Session Cache", key="set_clear_cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

    st.subheader("About")
    st.markdown("""
    **PredictBet AI** — Institutional-Grade Football Betting Intelligence Engine

    - Poisson GLM + Bayesian + ELO + ML ensemble
    - Multi-bookmaker de-vigging and EV calculation
    - Automated NO-BET refusal logic
    - Paper trading and bankroll management
    """)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_app():
    if st is None:
        print("Streamlit is not installed. Please install streamlit using: pip install streamlit")
        return

    _init_session()
    _inject_theme_css()
    _check_auto_refresh()

    st.set_page_config(
        page_title="PredictBet AI — Football Intelligence Engine",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        col_logo, col_theme = st.columns([3, 1])
        with col_logo:
            st.title("PredictBet AI")
            st.caption("Institutional Intelligence Engine")
        with col_theme:
            theme_icon = "🌙" if _get_theme() == "dark" else "☀️"
            st.button(theme_icon, key="theme_toggle", on_click=_toggle_theme, help="Toggle theme")

        st.markdown("---")

        if option_menu is not None:
            page = option_menu(
                "Navigation",
                [
                    "Dashboard",
                    "Match Analyzer",
                    "Fixtures",
                    "Team Intel",
                    "Bankroll",
                    "Settings",
                    "System Health",
                ],
                icons=[
                    "speedometer2",
                    "graph-up-arrow",
                    "calendar-event",
                    "people",
                    "wallet2",
                    "gear",
                    "heart-pulse",
                ],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "var(--accent)", "font-size": "1.2rem"},
                    "nav-link": {"font-size": "0.95rem", "text-align": "left", "margin": "2px 0"},
                    "nav-link-selected": {"background-color": "var(--accent2)", "color": "white"},
                },
            )
        else:
            page = st.radio(
                "Navigation",
                [
                    "Dashboard (Value Bets)",
                    "Match Analyzer",
                    "Today's Fixtures",
                    "Team Intel",
                    "Bankroll & Portfolio",
                    "Settings",
                    "System Health",
                ],
            )

        st.markdown("---")
        st.caption("Automated workflow:\n1. Fetch fixtures\n2. Scrape form + xG\n3. Ensemble models\n4. De-vig markets\n5. EV + Kelly\n6. Rank & report")
        st.caption("Never fabricate data.\nInsufficient evidence → NO BET.")

        st.markdown("---")
        st.subheader("Automation")
        st.session_state.auto_analyze = st.checkbox(
            "Auto-analyze all fixtures on load",
            value=st.session_state.get("auto_analyze", False),
            key="auto_analyze_toggle",
            help="Automatically analyze every fixture when the Fixtures page loads."
        )
        st.session_state.refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=0,
            max_value=1800,
            value=int(st.session_state.get("refresh_interval", 0)),
            step=60,
            key="refresh_slider",
            help="0 = disabled. Re-runs the app every N seconds."
        )

    page_map = {
        "Dashboard (Value Bets)": "Dashboard (Value Bets)",
        "Dashboard": "Dashboard (Value Bets)",
        "Match Analyzer": "Match Analyzer",
        "Match Analyzer": "Match Analyzer",
        "Fixtures": "Today's Fixtures",
        "Today's Fixtures": "Today's Fixtures",
        "Team Intel": "Team Intel",
        "Bankroll & Portfolio": "Bankroll & Portfolio",
        "Bankroll": "Bankroll & Portfolio",
        "Settings": "Settings",
        "System Health": "System Health",
    }
    target = page_map.get(page, page)

    if target == "Dashboard (Value Bets)":
        render_value_bets_dashboard()
    elif target == "Match Analyzer":
        render_match_analyzer()
    elif target == "Today's Fixtures":
        render_betika_fixtures()
    elif target == "Team Intel":
        render_team_intel()
    elif target == "Bankroll & Portfolio":
        render_bankroll_portfolio()
    elif target == "Settings":
        render_settings()
    elif target == "System Health":
        render_system_health()


if __name__ == "__main__":
    run_app()

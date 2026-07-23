"""
EQFBIS FastAPI Server
=====================

Proper FastAPI/Flask-style app exposing the existing analytics endpoints.
The original `analytics.py` HTTP handler logic is preserved; this module
adds a modern ASGI server with automatic docs and structured routing.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from analytics import (
    AnalyticsHTTPHandler,
    _FRONTEND_DIR,
    BetikaClient,
    ESPNScraperClient,
    FootballDataClient,
    WikipediaTeamScraper,
    HeadToHeadFetcher,
    FormMomentumCalculator,
    build_model,
    compare_to_market,
    poisson_pmf,
    weighted_shrunk_rate,
    fetch_team_stock_data,
    fit_poisson_glm_ratings,
    fit_statsmodels_glm_ratings,
    fit_sklearn_poisson_ratings,
    _blend_estimator_goals,
    update_model_probabilities,
    generate_scoreline_heatmap,
    validate_form_with_pandas,
)
from market_pipeline import ResultsSyncer, VersionedPredictionLedger
from migrations import MigrationRunner

app = FastAPI(title="PredictBet Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "PredictBet Engine",
        "frontend_ready": os.path.exists(os.path.join(_FRONTEND_DIR, "index.html")),
    }


@app.get("/api/betika/fixtures")
def betika_fixtures(page: int = 1, limit: int = 50):
    client = BetikaClient()
    try:
        if page == 0:
            fixtures = client.get_all_fixtures()
        else:
            fixtures = client.get_upcoming_fixtures(page=page, limit=limit)
        return {"data": fixtures, "meta": {"total": len(fixtures), "page": page, "limit": limit}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/betika/live")
def betika_live():
    client = BetikaClient()
    try:
        live = client.get_live_matches()
        return {"data": live, "meta": {"total": len(live)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/betika/search")
def betika_search(q: str = ""):
    if not q:
        return []
    client = BetikaClient()
    try:
        return client.search_teams(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/betika/competitions")
def betika_competitions():
    client = BetikaClient()
    try:
        comps = client.get_competitions()
        return {"data": comps, "meta": {"total": len(comps)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
def search_teams(query: str = ""):
    if not query or len(query) < 2:
        return []
    espn = ESPNScraperClient()
    try:
        results = espn.search_team(query)
        try:
            betika = BetikaClient()
            results.extend(betika.search_teams(query))
        except Exception:
            pass
        seen = set()
        unique = []
        for r in results:
            key = (r.get("name", ""), r.get("source", ""))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/team/info")
def team_info(name: str = ""):
    if not name:
        raise HTTPException(status_code=400, detail="Missing team name")
    wiki = WikipediaTeamScraper()
    info = wiki.get_team_info(name)
    if info:
        return info
    raise HTTPException(status_code=404, detail="No info found")


@app.get("/api/team/search")
def team_search(query: str = ""):
    if not query or len(query) < 2:
        return []
    wiki = WikipediaTeamScraper()
    return wiki.search_teams(query)


@app.get("/api/scrape")
def scrape(
    home_id: str = "",
    away_id: str = "",
    league_slug: str = "",
    decay: float = 0.92,
    shrinkage_k: float = 6.0,
    home_advantage: float = 1.0,
    match_date: Optional[str] = None,
):
    if not home_id or not away_id or not league_slug:
        raise HTTPException(status_code=400, detail="Missing parameters")

    espn = ESPNScraperClient()
    try:
        home_form = espn.fetch_recent_matches(home_id, league_slug)
        away_form = espn.fetch_recent_matches(away_id, league_slug)
        league_avg_home, league_avg_away = espn.fetch_league_averages(league_slug)
        odds_home, odds_draw, odds_away = espn.fetch_market_odds(home_id, away_id, league_slug)

        model = build_model(
            home_form, away_form,
            league_avg_home, league_avg_away,
            home_advantage=home_advantage,
            decay=decay, shrinkage_k=shrinkage_k,
        )

        glm_res = fit_poisson_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
        sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_avg_home, league_avg_away)
        sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_avg_home, league_avg_away)
        blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
        if blended:
            blended_home, blended_away = blended
            update_model_probabilities(model, blended_home * home_advantage, blended_away)

        heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                  home_form.team_name, away_form.team_name)
        pandas_validation = validate_form_with_pandas(home_form, away_form)
        fin_home = fetch_team_stock_data(home_form.team_name)
        fin_away = fetch_team_stock_data(away_form.team_name)

        market_comp = None
        if odds_home and odds_draw and odds_away:
            market_comp = compare_to_market(model, odds_home, odds_draw, odds_away)

        return {
            "match_label": f"{home_form.team_name} vs {away_form.team_name}",
            "match_date": match_date,
            "model": model.__dict__,
            "confidence_score": model.confidence_score(),
            "data_quality_note": model.data_quality_note(),
            "market_comparison": market_comp,
            "heatmap_file": heatmap_file,
            "pandas_validation": pandas_validation,
            "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
            "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
            "financials": {"home": fin_home, "away": fin_away},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/betika/scrape")
def betika_scrape(
    home_id: str = "",
    away_id: str = "",
    match_id: str = "",
    decay: float = 0.92,
    shrinkage_k: float = 6.0,
    home_advantage: float = 1.0,
    match_date: Optional[str] = None,
):
    if not home_id or not away_id:
        raise HTTPException(status_code=400, detail="Missing home_id or away_id")

    espn = ESPNScraperClient()
    try:
        home_results = espn.search_team(home_id)
        away_results = espn.search_team(away_id)
        if not home_results or not away_results:
            raise HTTPException(status_code=404, detail="Could not resolve one or both team names on ESPN")

        home_info = home_results[0]
        away_info = away_results[0]
        league_slug = home_info.get("league") or away_info.get("league") or "eng.1"

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
        glm_res = fit_poisson_glm_ratings(home_form, away_form, league_home, league_away)
        sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_home, league_away)
        sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_home, league_away)
        blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
        if blended:
            blended_home, blended_away = blended
            update_model_probabilities(model, blended_home * home_advantage, blended_away)

        heatmap_file = generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                  home_form.team_name, away_form.team_name)
        pandas_validation = validate_form_with_pandas(home_form, away_form)
        fin_home = fetch_team_stock_data(home_form.team_name)
        fin_away = fetch_team_stock_data(away_form.team_name)

        market_comp = None
        if odds_home and odds_draw and odds_away:
            market_comp = compare_to_market(model, odds_home, odds_draw, odds_away)

        return {
            "match_label": f"{home_form.team_name} vs {away_form.team_name}",
            "match_date": match_date,
            "model": model.__dict__,
            "confidence_score": model.confidence_score(),
            "data_quality_note": model.data_quality_note(),
            "market_comparison": market_comp,
            "heatmap_file": heatmap_file,
            "pandas_validation": pandas_validation,
            "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
            "odds": {"home": odds_home, "draw": odds_draw, "away": odds_away},
            "financials": {"home": fin_home, "away": fin_away},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def predictions():
    client = BetikaClient()
    try:
        fixtures = client.get_upcoming_fixtures(page=1, limit=100)
        fixtures_with_odds = [f for f in fixtures if f.get("home_odd") and f.get("draw_odd") and f.get("away_odd")][:6]
        if not fixtures_with_odds:
            return {"status": "success", "data": []}

        predictions = []
        for fixture in fixtures_with_odds:
            try:
                home_name = fixture.get("home_team", "")
                away_name = fixture.get("away_team", "")
                match_id = fixture.get("match_id", "")

                espn = ESPNScraperClient()
                home_results = espn.search_team(home_name)
                away_results = espn.search_team(away_name)
                if not home_results or not away_results:
                    continue

                home_info = home_results[0]
                away_info = away_results[0]
                league_slug = home_info.get("league") or away_info.get("league") or "eng.1"

                home_form = espn.fetch_recent_matches(home_info["id"], league_slug)
                away_form = espn.fetch_recent_matches(away_info["id"], league_slug)
                league_home, league_away = espn.fetch_league_averages(league_slug)

                model = build_model(home_form, away_form, league_home, league_away,
                                    home_advantage=1.0, decay=0.92, shrinkage_k=6.0)
                glm_res = fit_poisson_glm_ratings(home_form, away_form, league_home, league_away)
                sm_res = fit_statsmodels_glm_ratings(home_form, away_form, league_home, league_away)
                sk_res = fit_sklearn_poisson_ratings(home_form, away_form, league_home, league_away)
                blended = _blend_estimator_goals([glm_res, sm_res, sk_res])
                if blended:
                    blended_home, blended_away = blended
                    update_model_probabilities(model, blended_home, blended_away)

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
                narrative = generate_aggressive_narrative(model.__dict__, {}, tier, recommended)

                predictions.append({
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
                    "heatmap_file": generate_scoreline_heatmap(model.expected_home_goals, model.expected_away_goals,
                                                               home_form.team_name, away_form.team_name),
                    "pandas_validation": validate_form_with_pandas(home_form, away_form),
                    "estimators_used": ["scipy_mle", "statsmodels_glm", "sklearn_poisson"],
                })
            except Exception:
                continue

        predictions.sort(key=lambda p: p["edge_pct"], reverse=True)
        return {"status": "success", "data": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/sync")
def results_sync(competition_code: str = "PL"):
    ledger_db = os.path.join(os.path.dirname(__file__), "eqfbis_ledger.sqlite3")
    runner = MigrationRunner(ledger_db)
    runner.run()

    ledger = VersionedPredictionLedger(db_path=ledger_db)
    syncer = ResultsSyncer(ledger=ledger)
    result = syncer.sync(competition_code)
    return result


@app.get("/api/monitoring")
def monitoring():
    from monitoring import SystemMonitor
    monitor = SystemMonitor()
    report = monitor.report()
    alerts = monitor.evaluate()
    return {
        "status": "success",
        "metrics": {k: {"value": v.value, "unit": v.unit,
                         "timestamp": v.timestamp, "severity": v.severity()} for k, v in report.items()},
        "active_alerts": [a.to_dict() for a in alerts],
    }


@app.get("/api/calibration/fit")
def calibration_fit(model: str = "ensemble", outcome: str = "H", method: str = "isotonic"):
    from calibration import CalibrationManager
    mgr = CalibrationManager()
    result = mgr.fit(model, outcome, method=method)
    return {"status": "success", "calibration": result}


@app.get("/")
def serve_frontend():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>PredictBet Engine</h1><p>Frontend not found.</p>")


@app.get("/api/chart")
def chart(file: str = ""):
    import urllib.parse
    file_path = os.path.join(_FRONTEND_DIR, file) if file else os.path.join(_FRONTEND_DIR, "scoreline_heatmap.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chart not found")
    with open(file_path, "rb") as f:
        content = f.read()
    return Response(content=content, media_type="image/png")

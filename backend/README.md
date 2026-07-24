# PredictBet — Elite Quantitative Football Betting Intelligence System

A professional football betting analytics engine that scrapes real-time team data from multiple sources, builds Poisson GLM models with attack/defense ratings, evaluates multi-market odds, and provides aggressive staking recommendations via the Confidence Tier Engine.

---

## Features

- **Aggressive Prediction Engine** — Full-Kelly staking capped by confidence tier (`LOCK`, `STRONG`, `VALUE`, `LEAN`).
- **Multi-Market Analysis** — Over/Under 1.5/3.5, Double Chance, Team Totals, Correct Score matrices.
- **Live Data Scraping** — Betika API, ESPN API (team form, league averages, head-to-head), football-data.org
- **Poisson GLM Modeling** — scipy MLE attack/defense optimizer, recency-weighted decay, Bayesian shrinkage
- **Ensemble Diversity** — Blends Shrinkage + Dixon-Coles, MLE GLM, and dynamic ELO priors.
- **Scoreline Heatmap** — matplotlib probability grid (0–5 × 0–5 scoreline matrix)
- **Market Edge Analysis** — de-vigged 1X2 comparison, overround calculation, edge quantification
- **Interactive Dashboard** — Streamlit UI with Predictions Feed and Match Analyzer
- **CLI Tools** — headless model building, JSON export, fixture listing
- **Automated Pipeline** — 9-step orchestrator from fixture ingestion to prediction output
- **Confidence Sorting** — All predictions ranked LOCK → STRONG → VALUE → LEAN → NO_BET

---

## Recent Changes

### Automation & UI
- **Auto-analyze toggle** in sidebar — automatically analyzes every fixture when the Fixtures page loads
- **Auto-refresh interval** — configurable 0–1800s slider for continuous automated updates
- **Auto-Scan All Fixtures** button — one-click analysis of all visible fixtures
- **Confidence-sorted feed** — all predictions ranked by confidence tier with edge % tiebreaker
- **Match Analyzer integration** — analyzed fixtures auto-populate the Match Analyzer dropdown

### Bug Fixes
- Fixed `PredictionCard` as proper `@dataclass` — was previously an uninstantiable annotated class
- Fixed `_build_match_model` dead code — full pipeline now executes end-to-end
- Fixed undefined `fixture` variable — added safe `Optional[dict]` parameter
- Fixed `PredictionLedger` import in `streamlit_app.py`
- Fixed missing `backend/requirements.txt` for Docker builds
- Fixed `streamlit_option_menu` dependency
- Fixed `Tuple` import in `backend/analytics.py`
- Fixed `bucket_size` scope bug in `backend/intelligence.py`
- Fixed `fair_odds` undefined in `backend/pipeline.py`
- Fixed `h2h_available` unexpected keyword in `backend/pipeline.py`
- Fixed `home_info`/`away_info` undefined in `backend/pipeline.py`
- Fixed `EvidenceChecklist` import in `backend/pipeline.py`
- Fixed `Optional`/`Any` imports in `backend/scraper.py`
- Removed invalid `hashlib>=4.0.0` from requirements (blocked deployment)
- Removed duplicate requirements sections
- Fixed silent exception swallowing — errors now reported per fixture

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit dashboard
streamlit run streamlit_app.py

# Or start the FastAPI server
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8080
```

---

## Data Sources

| Source | Coverage | Auth |
|--------|----------|------|
| Betika API | East Africa fixtures, odds, live | None (public) |
| ESPN API | Global team search, schedules, results | None (public) |
| ClubELO | Live ELO ratings with empirical fallback | None (public) |
| football-data.org | European leagues, historical data | Free API key |
| yfinance | Public club stock prices | None |
| BeautifulSoup | HTML fallback scraping | None |

Set `FOOTBALL_DATA_API_KEY=your_key_here` to enable football-data.org.

---

## Modelling Stack

### Data Collection
`requests`, `httpx`, `BeautifulSoup`, `soccerdata`, `statsbombpy`, `understat`, `yfinance`

### Data Cleaning & Validation
`pandas`, `polars`, `numpy`, `pyjanitor`, `great_expectations`, `pandera`

### Statistical Modelling
`scipy` (Poisson GLM MLE), `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`

### Football Analytics
`mplsoccer`, `kloppy`, `socceraction`

### Visualization
`matplotlib` (scoreline heatmap), `plotly`

---

## Architecture

```text
streamlit_app.py          — Primary UI (Dashboard, Analyzer, Fixtures, Intel, History, Sites, Health)
backend/server.py         — FastAPI server (alternative API layer with CORS)
backend/analytics.py      — Core engine: GLM fits, heatmaps, validation
backend/scraper.py        — Data ingestion: Betika, ESPN, Wikipedia, Understat, etc.
backend/intelligence.py   — Ensemble engine, PredictionLedger, staking, narratives
backend/pipeline.py       — 9-step automated orchestrator (Fixture → Prediction)
backend/market_pipeline.py — xG ingestion, odds tracking, results syncing, model versioning
backend/features.py       — Feature engineering (10 feature groups)
backend/ml_pipeline.py    — XGBoost/LightGBM gradient-boosted models
backend/backtest.py       — Historical ROI, Brier score, CLV, significance tests
backend/calibration.py    — Probability calibration (Platt, Isotonic, Bayesian binning)
backend/monitoring.py     — System observability, sliding-window metrics, alerting
backend/config.py         — Staking ceilings, Kelly multipliers, environment config
backend/aiBetModel/       — Spec-driven sub-package (models, market, staking, quality, reports)
backend/engine/           — Re-export of aiBetModel.staking
backend/frontend/         — Vanilla HTML/CSS/JS dashboard
```

**API Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/predictions` | Aggressive auto-prediction feed |
| GET | `/api/h2h` | Head-to-head historical data |
| GET | `/api/form` | Form momentum and PPG trajectory |
| GET | `/api/scrape` | Build model by team IDs |
| GET | `/api/scrape_by_name` | Build model by team names |
| GET | `/api/betika/scrape` | Build model from Betika fixture |
| GET | `/api/betika/fixtures` | Upcoming Betika fixtures |
| GET | `/api/betika/live` | Live Betika matches |
| GET | `/api/search` | Search teams across ESPN + Betika |
| GET | `/api/chart` | Serve scoreline heatmap PNG |

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FOOTBALL_DATA_API_KEY` | football-data.org API key | None |
| `BETIKA_BASE_URL` | Override Betika endpoint | `https://api.betika.com` |

---

## Requirements

Python 3.9+ and the packages in `requirements.txt`. Key packages:

```text
requests httpx beautifulsoup4
pandas polars numpy scipy statsmodels scikit-learn
xgboost lightgbm mplsoccer kloppy yfinance matplotlib
pandera great_expectations pyjanitor soccerdata
streamlit streamlit-option-menu streamlit-aggrid
```

---

## Project Structure

```text
PredictBet/
├── streamlit_app.py      — Primary Streamlit UI
├── requirements.txt       — Python dependencies
├── Dockerfile             — Container build
├── docker-compose.yml     — Compose config
├── .env                   — Environment variables
├── backend/
│   ├── server.py          — FastAPI server
│   ├── analytics.py       — Core engine: GLM fits, heatmaps
│   ├── scraper.py         — Data collection, GLM modeling
│   ├── intelligence.py    — Staking logic, confidence grading, PredictionLedger
│   ├── pipeline.py        — 9-step automated orchestrator
│   ├── market_pipeline.py — Ledger tracking, odds movement
│   ├── backtest.py        — ROI evaluation
│   ├── calibration.py     — Probability calibration
│   ├── monitoring.py      — System observability
│   ├── config.py          — Staking ceilings, Kelly multipliers
│   ├── aiBetModel/        — Models, market, staking, quality, reports
│   ├── engine/            — Re-export of aiBetModel.staking
│   └── frontend/          — Vanilla HTML/CSS/JS dashboard
```

---

## CLI Reference

```bash
# Start server
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8080

# Start Streamlit dashboard
streamlit run streamlit_app.py

# Build match model (scrapes ESPN automatically)
python -m backend.analytics model --home-team TEAM --away-team TEAM [OPTIONS]

# Options:
#   --league SLUG           League slug (required)
#   --odds-home FLOAT       Home win decimal odds
#   --odds-draw FLOAT       Draw decimal odds  
#   --odds-away FLOAT       Away win decimal odds
#   --home-advantage FLOAT  xG multiplier for home (default 1.0)
#   --decay FLOAT           Recency decay (default 0.92)
#   --shrinkage-k FLOAT     Bayesian shrinkage K (default 6.0)
#   --json                  Output JSON instead of formatted table
#   --export FILE.json      Append record to JSON file

# Manual data input (override scraper)
python -m backend.analytics model \
  --home-team Arsenal --away-team Chelsea \
  --home-results "2,1,3,2,1" --home-conceded "0,1,1,2,0" \
  --away-results "1,1,2,0,1" --away-conceded "1,2,0,1,2"

# Betika integration
python -m backend.analytics betika-fixtures [--limit 50] [--live]
python -m backend.analytics betika-search QUERY
python -m backend.analytics betika-model --home TEAM --away TEAM
```

---

## Confidence Score

```
confidence = round(75 × (1 - exp(-n / 8)))
```

Where `n = min(home_sample_size, away_sample_size)`. Capped at 75 — never reaches 100 regardless of data volume, reflecting irreducible uncertainty in football prediction.

| Matches | Confidence |
|---------|-----------|
| 3 | ~26 |
| 5 | ~37 |
| 8 | ~52 |
| 12 | ~61 |
| 20 | ~70 |

---

## Confidence Tier & Aggressive Staking (`intelligence.py`)

The system maps Confidence Scores + Market Edges into actionable tiers:
1. **`LOCK`**: Highly confident model, massive edge, zero data flags. Recommended stake up to 10% (Full-Kelly).
2. **`STRONG`**: Solid model agreement, clear edge. Recommended stake up to 5%.
3. **`VALUE`**: Lower confidence or marginal edge, but statistically positive expectation. Recommended stake up to 2%.
4. **`LEAN`**: No edge or model disagreement. Recommended stake: 0%.
5. **`NO_BET`**: Missing data or negative expectation.

`AggressiveStakeEngine` computes these stakes dynamically to maximize ROI.

---

## League Slugs Reference

| League | Slug |
|--------|------|
| Premier League | `eng.1` |
| La Liga | `esp.1` |
| Bundesliga | `ger.1` |
| Serie A | `ita.1` |
| Ligue 1 | `fra.1` |
| Champions League | `UEFA.CHAMPIONS_LEAGUE` |
| Kenyan Premier | `ken.1` |

---

## Automated Result Syncing (`pipeline.py`, `market_pipeline.py`)

`AutomatedPredictionPipeline` includes a `_sync_results_background()` method that calls `ResultsSyncer.sync()` for top competitions (PL, PD, SA, FL1, BL1, CL) after every pipeline run. A separate `start_result_sync()` thread is available for periodic background resolution. Both require `FOOTBALL_DATA_API_KEY` to be set; otherwise they skip gracefully.

---

## Session Export Format

The dashboard exports sessions as JSON arrays. Each record follows this schema:

```json
{
  "match_label": "Team A vs Team B",
  "match_date": "2025-08-15",
  "model": {
    "home_team": "Team A",
    "away_team": "Team B",
    "expected_home_goals": 1.72,
    "expected_away_goals": 1.18,
    "home_win_prob": 0.487,
    "draw_prob": 0.243,
    "away_win_prob": 0.270,
    "over_2_5_prob": 0.521,
    "btts_yes_prob": 0.468,
    "sample_size_home": 10,
    "sample_size_away": 10
  },
  "confidence_score": 62,
  "data_quality_note": "Reasonable sample size...",
  "market_comparison": {
    "bookmaker_overround_pct": 4.8,
    "home": { "model_prob_pct": 48.7, "market_implied_pct": 44.5, "edge_pct_points": 4.2 },
    "draw": { ... },
    "away": { ... }
  },
  "odds": { "home": 2.10, "draw": 3.40, "away": 3.60 },
  "sources": { "team_form": "ESPN", "odds": "Betika" },
  "financials": {
    "home": { "ticker": "MANU", "price": 14.22, "change_pct": -1.4, "market_cap": "2.4B" },
    "away": null
  }
}
```

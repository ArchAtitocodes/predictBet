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
- **Interactive Dashboard** — dark-mode web UI with Predictions Feed and Match Analyzer.
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

# Start the dashboard server
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

## Streamlit Pages

1. **Dashboard** — Top value bets ranked by EV, filters for league/day/tier
2. **Match Analyzer** — Deep-dive per match: markets, edges, momentum, model agreement, narrative, full report
3. **Today's Fixtures** — Betika upcoming/live matches with batch analyze
4. **Team Intel** — ELO, form, Wikipedia, yfinance per team
5. **Historical Performance** — Backtesting, Brier score, calibration ledger
6. **Betting Sites** — Line shopping across scraped sites, best odds comparison
7. **System Health** — Metrics, data source status, automation controls

---

## Automation

The app supports three levels of automation:

1. **Auto-analyze toggle** (sidebar) — When enabled, every fixture is automatically analyzed when the Fixtures page loads
2. **Auto-refresh interval** (sidebar) — Set 0–1800 seconds; app reruns automatically and re-fetches fixtures
3. **Auto-Scan All button** — One-click analysis of all fixtures in the current filtered view

Results are automatically sorted by confidence tier (LOCK → STRONG → VALUE → LEAN → NO_BET) and appear in the Match Analyzer dropdown.

---

## Key Design Principles

- **Never fabricate data** — reduce confidence when unavailable
- **Recommend bets only when positive EV exists**
- **Conservative bankroll management** (fractional Kelly with hard caps)
- **Insufficient evidence → NO BET**
- **Explicit uncertainty** — model disagreement widens reported uncertainty rather than hiding it
- **Auditable** — every prediction is logged with outcomes for Brier/calibration scoring
- **Fully automated** — from fixture ingestion to analysis to visualization with minimal user intervention

## License

Internal research tool. Not for production use without independent validation.

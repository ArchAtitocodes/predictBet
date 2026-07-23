# PredictBet AI — Institutional-Grade Football Betting Intelligence Engine

PredictBet is a Python-based football betting analytics platform that scrapes live team data from multiple sources, builds Poisson GLM models with attack/defense ratings, Bayesian shrinkage, and recency-weighted decay, then benchmarks model probabilities against bookmaker market odds to surface positive expected value (EV) opportunities.

## Features

- **Poisson GLM Modeling** — scipy MLE attack/defense optimizer, recency-weighted decay, Bayesian shrinkage
- **Ensemble Diversity** — Blends Shrinkage + Dixon-Coles, MLE GLM, statsmodels GLM, sklearn Poisson, and ELO priors
- **Multi-Market Analysis** — Over/Under 1.5/2.5/3.5, Double Chance, Team Totals, Correct Score matrices
- **Live Data Scraping** — Betika API, ESPN API, ClubELO, Wikipedia, Understat, football-data.org
- **ML Enhancement** — XGBoost/LightGBM gradient-boosted models on top of statistical base
- **Probability Calibration** — Platt scaling, isotonic regression, Bayesian binning
- **Market Edge Analysis** — de-vigged 1X2 comparison, overround calculation, edge quantification
- **Confidence Tiers & Staking** — LOCK/STRONG/VALUE/LEAN/NO_BET with fractional-Kelly stake sizing and hard ceilings
- **Audit Trail** — SQLite-backed PredictionLedger with Brier score, log loss, and calibration tables
- **Backtesting** — Historical ROI, flat-stake simulation, binomial significance tests, closing-line value
- **Monitoring** — Sliding-window metrics, alerting, data-source health tracking
- **Interactive Dashboard** — Streamlit UI with auto-refresh, dark/light theme, CSV/JSON export

## Architecture

```text
streamlit_app.py          — Primary UI (7 pages: Dashboard, Analyzer, Fixtures, Intel, History, Sites, Health)
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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit dashboard
streamlit run streamlit_app.py

# Or start the FastAPI server
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8080
```

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `PORT` | Server port | `8080` |
| `HOST` | Bind address | `0.0.0.0` |
| `FOOTBALL_DATA_API_KEY` | football-data.org API key | empty (optional) |
| `BETIKA_BASE_URL` | Betika endpoint override | `https://api.betika.com` |
| `CACHE_TTL_SECONDS` | Scraper cache TTL | `3600` |

## Data Sources

| Source | Coverage | Auth |
|--------|----------|------|
| Betika API | East Africa fixtures, live odds | None |
| ESPN API | Global team search, form, league averages | None |
| ClubELO | Live ELO ratings with empirical fallback | None |
| Understat | xG for/against per match | None |
| Wikipedia | Club overviews, managers, stadiums | None |
| yfinance | Public club stock prices | None |
| football-data.org | European fixtures, results | Free API key |

## Modelling Stack

### Statistical
- Poisson GLM via scipy MLE, statsmodels, and sklearn PoissonRegressor
- Bayesian shrinkage with exponential recency decay (`decay=0.92`, `shrinkage_k=6.0`)
- Dixon-Coles-style attack/defense ratings
- Joint Poisson PMF for 1X2, O/U, BTTS, correct scores

### ML
- XGBoost and LightGBM on 10 feature groups (form, ELO, xG, market, H2H, league context, temporal, news, financial, advanced stats)
- Versioned model persistence with schema validation
- Model confidence feeds back into ensemble blending

### Calibration
- Platt scaling, isotonic regression, Bayesian binning, temperature scaling
- Brier score and log loss tracked in PredictionLedger
- Calibration tables: predicted bucket vs actual hit rate

## Staking & Risk Management

The system uses **fractional Kelly** with hard ceilings per confidence tier:

| Tier | Edge Requirement | Min Confidence | Kelly Multiplier | Stake Ceiling |
|------|-----------------|----------------|------------------|---------------|
| `LOCK` | > 15pp | ≥ 60 | 100% | 10% |
| `STRONG` | > 8pp | ≥ 50 | 50% | 5% |
| `VALUE` | > 3pp | ≥ 35 | 25% | 3% |
| `LEAN` | > 0pp | any | 0% | 1% |
| `NO_BET` | ≤ 0pp | — | 0% | 0% |

Stake = min(raw_kelly × multiplier, ceiling). Loss-chasing/martingale is explicitly refused.

## Confidence Score

```
confidence = round(75 × (1 - exp(-n / 8)))
```

Capped at 75 — never reaches 100 regardless of data volume.

## Docker

```bash
docker-compose up --build
```

Container runs as non-root, exposes port 8080, includes healthcheck.

## Project Structure

```text
predictBet/
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
├── backend/
│   ├── server.py
│   ├── analytics.py
│   ├── scraper.py
│   ├── intelligence.py
│   ├── pipeline.py
│   ├── market_pipeline.py
│   ├── features.py
│   ├── ml_pipeline.py
│   ├── backtest.py
│   ├── calibration.py
│   ├── monitoring.py
│   ├── config.py
│   ├── migrations.py
│   ├── aiBetModel/
│   ├── engine/
│   ├── eqfbis_models/
│   └── frontend/
└── eqfbis_models/
```

## Streamlit Pages

1. **Dashboard** — Top value bets ranked by EV, filters for league/day/tier
2. **Match Analyzer** — Deep-dive per match: markets, edges, momentum, model agreement, narrative, full report
3. **Today's Fixtures** — Betika upcoming/live matches with batch analyze
4. **Team Intel** — ELO, form, Wikipedia, yfinance per team
5. **Historical Performance** — Backtesting, Brier score, calibration ledger
6. **Betting Sites** — Line shopping across scraped sites, best odds comparison
7. **System Health** — Metrics, data source status, automation controls

## Key Design Principles

- **Never fabricate data** — reduce confidence when unavailable
- **Recommend bets only when positive EV exists**
- **Conservative bankroll management** (fractional Kelly with hard caps)
- **Insufficient evidence → NO BET**
- **Explicit uncertainty** — model disagreement widens reported uncertainty rather than hiding it
- **Auditable** — every prediction is logged with outcomes for Brier/calibration scoring

## License

Internal research tool. Not for production use without independent validation.

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
- **Interactive Dashboard** — dark-mode web UI at `localhost:8080` with Predictions Feed and Match Analyzer.
- **CLI Tools** — headless model building, JSON export, fixture listing

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard server
python analytics.py serve --port 8080

# CLI: build a model for any fixture
python analytics.py model --home-team "Arsenal" --away-team "Chelsea"

# CLI: with manual odds for market comparison
python analytics.py model --home-team "Arsenal" --away-team "Chelsea" \
  --odds-home 2.10 --odds-draw 3.40 --odds-away 3.60

# CLI: JSON output for pipeline integration
python analytics.py model --home-team "Liverpool" --away-team "Man City" --json

# Betika fixture browser
python analytics.py betika-fixtures --limit 50
python analytics.py betika-model --home "Arsenal" --away "Chelsea"
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
analytics.py          — HTTP server, API endpoints, CLI entrypoint
scraper.py            — Data clients, statistical models, multi-market logic
intelligence.py       — Ensemble builder, Confidence Engine, Staking, Prediction Pipeline
market_pipeline.py    — Ledger tracking, odds movement tracking, results syncer
backtest.py           — Historical simulation, ROI/CLV metrics
dashboard.html        — Single-page web UI (vanilla JS + CSS)
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
```

---

## Project Structure

```text
PredictBet/
├── analytics.py        — Core engine: server & APIs
├── scraper.py          — Data collection, GLM modeling
├── intelligence.py     — Staking logic, confidence grading
├── market_pipeline.py  — Storage, tracking, versions
├── backtest.py         — ROI evaluation
├── dashboard.html      — Web interface
├── requirements.txt    — Python dependencies
├── README.md           — This file
└── DOCUMENTATION.md    — Technical deep-dive
```

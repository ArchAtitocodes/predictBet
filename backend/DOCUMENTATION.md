# PredictBet Technical Documentation

## Overview

PredictBet is a Python-based betting analytics platform with a modular architecture:

1. **Data Layer** (`scraper.py`) — multi-source scraping, validation, Head-to-Head fetcher, and empirical form momentum.
2. **Intelligence Layer** (`intelligence.py`) — Ensemble builder (Shrinkage + GLM + ELO), Confidence engine, Aggressive staking recommendations, and automated Prediction Pipelines.
3. **Market Layer** (`market_pipeline.py`, `backtest.py`) — Odds movement tracking, results syncing, model version comparison, and historical ROI backtesting.
4. **Presentation Layer** (`streamlit_app.py`, `dashboard.html`) — Streamlit UI, API endpoints, CLI, and interactive web UI with real-time predictions.

---

## Recent Changes

### Automation & UI
- **Auto-analyze toggle** in sidebar — automatically analyzes every fixture when the Fixtures page loads
- **Auto-refresh interval** — configurable 0–1800s slider for continuous automated updates
- **Auto-Scan All Fixtures** button — one-click analysis of all visible fixtures
- **Confidence-sorted feed** — all predictions ranked LOCK → STRONG → VALUE → LEAN → NO_BET
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

## Security: Safe Model Loading (`ml_pipeline.py`)

Model serialization uses a **restricted unpickler** (`_SafeModelUnpickler`) instead of raw `pickle.load()`. The unpickler maintains a whitelist of safe modules and classes (xgboost, lightgbm, sklearn, scipy, numpy, builtins, datetime, etc.) and rejects any payload from non-whitelisted modules. Loaded artifacts are also validated to ensure they are dicts with the required `model` and `version` keys.

This prevents arbitrary code execution if `.pkl` files on disk are tampered with.

---

## Data Layer (`scraper.py`)

### Caching

`ScraperCache` uses a SQLite-backed cache with a `threading.Lock` for thread-safe concurrent access from the HTTP server. Default TTL is 900 seconds. All HTTP responses from ESPN and Betika are cached automatically by URL key, preventing redundant requests during a session.

### Data Classes

```python
@dataclass
class TeamForm:
    team_name: str
    matches_played: int
    goals_scored: list[int]        # chronological, oldest first
    goals_conceded: list[int]      # chronological, oldest first
```

### Clients

#### `BetikaClient`
- `get_upcoming_fixtures(page, limit)` — paginated fixture list
- `get_all_fixtures()` — auto-paginates all pages
- `get_live_matches()` — in-play filter
- `get_match_odds(match_id)` — 1X2 odds for specific fixture
- `search_teams(query)` — fuzzy match on team names
- `get_competitions()` — available competition list

#### `ESPNScraperClient`
- `search_team(name)` — ESPN search API → short integer ID from `uid` field
- `fetch_recent_matches(team_id, league_slug)` → `TeamForm`
- `fetch_league_averages(league_slug)` → `(avg_home_goals, avg_away_goals)`
- `fetch_market_odds(home_id, away_id, league_slug)` → `(home, draw, away)`

**ID Resolution Note**: ESPN search returns UUID strings (`s:600~t:359`). The short integer ID (`359`) is parsed from the `uid` field using `split("~t:")[-1]`. Schedule/standings endpoints require the short ID.

#### `FootballDataClient`
- Requires `FOOTBALL_DATA_API_KEY` environment variable
- `find_team_id(name)` — fuzzy match on football-data.org teams
- `recent_results(team_id)` → `TeamForm`

#### `HeadToHeadFetcher`
- `get_h2h(home_id, away_id, league_slug)` — Fetches historical matchup records between two teams.

#### `FormMomentumCalculator`
- `calculate(team_form)` — Derives points-per-game trajectory, defensive solidity, and active streaks.

#### `EloRatingScraper`
- `get_club_elo(team_name, form)` — Fetches live ELO from ClubELO API, seamlessly falling back to a form-derived empirical ELO formula (`1500 + ppg * 160 + gd * 80`) if the API is unreachable.

### League Slug Resolution

ESPN does not always return a league for every team. The helper `config.resolve_league_slug(home_league, away_league)` returns the first non-empty league, or `None` if neither side resolved one. Callers treat `None` as a **data-quality failure** and abort the model build explicitly, rather than silently falling back to a hardcoded league like `eng.1`.

### Data Validation Pipeline

`validate_and_clean_match_data(form)` runs:
1. **pandas** — null removal, type coercion
2. **pandera** schema — enforces `goals_scored >= 0`, `goals_conceded >= 0`
3. **great_expectations** — expectation suite for value ranges
4. **polars** — final conversion for high-performance downstream ops

### JSON Site Scraper

`JSONLinkScraper` and `JSONSiteDataCleaner` extract odds from registered betting sites. Extraction uses a **three-tier strategy**:
1. **Structured first**: JSON-LD (`application/ld+json`) `SportsEvent`/`Event` blobs and HTML `data-odds`/`data-odd` attributes
2. **Regex fallback**: pattern matching for `1X2 | Match Odds | Odds` followed by three decimal odds
3. **Validation**: Pandas clipping, overround computation, de-vigging, and line-shopping summary

### Optional Integrations

```python
# soccerdata — FBref, WhoScored, etc.
import soccerdata as sd
ws = sd.WhoScored(leagues=["ENG-Premier League"])

# statsbombpy — StatsBomb open event data
from statsbombpy import sb
matches = sb.matches(competition_id=43, season_id=3)

# understat — xG data
import understat
# async client for expected goals

# yfinance — club financials
fetch_team_stock_data("Manchester United")  # → MANU ticker data
```

### Poisson Calculator

```python
def weighted_shrunk_rate(values, league_avg, decay=0.92, shrinkage_k=6.0):
    # Exponential recency decay + Bayesian shrinkage toward league average
```

---

## Model Layer (`scraper.py`, `analytics.py`)

### `MatchModelResult` Dataclass

```python
@dataclass
class MatchModelResult:
    home_team: str
    away_team: str
    expected_home_goals: float     # Poisson lambda for home
    expected_away_goals: float     # Poisson lambda for away
    home_win_prob: float           # P(H > A) over 0–8 goal grid
    draw_prob: float               # P(H == A)
    away_win_prob: float           # P(H < A)
    over_2_5_prob: float           # P(H + A > 2.5)
    under_2_5_prob: float
    btts_yes_prob: float           # P(H >= 1 AND A >= 1)
    btts_no_prob: float
    sample_size_home: int
    sample_size_away: int
    elo_home: Optional[float] = None
    elo_away: Optional[float] = None
    dixon_coles_applied: bool = False
```

### `build_model()`

```
home_attack  = shrunk_home_scored  / league_avg_home
home_defense = shrunk_home_conceded / league_avg_away
away_attack  = shrunk_away_scored  / league_avg_away
away_defense = shrunk_away_conceded / league_avg_home

exp_home = league_avg_home × home_attack × away_defense × home_advantage
exp_away = league_avg_away × away_attack × home_defense
```

Probabilities are computed by summing over the 9×9 joint Poisson PMF grid (goals 0–8).

### `MultiMarketPredictor` (in `scraper.py`)
Expands the core 1X2 model into alternative betting markets:
- Over/Under 1.5, 2.5, 3.5 Goals
- Both Teams to Score (BTTS)
- Double Chance (1X, 12, X2)
- Exact Score probability maps (e.g. 1-0, 2-1)

### `fit_poisson_glm_ratings()`

Uses `scipy.optimize.minimize` to fit attack/defense ratings via Maximum Likelihood Estimation (MLE) over observed match goal pairs. When available, replaces the simpler ratio-based calculation with proper GLM estimates.

```python
def neg_log_likelihood(params):
    # params = [home_attack, away_attack, home_defense, away_defense, intercept]
    # Returns negative log-likelihood over all match observations
```

### `generate_scoreline_heatmap()`

Renders a 6×6 scoreline probability matrix using matplotlib, saved as `scoreline_heatmap.png`. Served via `GET /api/chart`. Color intensity represents joint Poisson probability for each `(home_goals, away_goals)` combination.

### `update_model_probabilities()`

Recomputes all market probabilities from new expected goals, used after GLM refinement. Ensures the model, dashboard, and heatmap stay in sync.

### `compare_to_market()`

```python
# De-vig: raw implied probability / overround
fair_prob_home = (1/odds_home) / overround

# Edge in percentage points
edge = model_prob - fair_prob
```

---

## Staking Engine (`intelligence.py`, `aiBetModel/staking.py`)

All stake calculations flow through a **single contract**: `AggressiveStakeEngine.suggest(tier, edge_prob, odds)` where `edge_prob` is the absolute model-vs-market edge in probability units (e.g. `best_edge / 100` when edge is measured in percentage points). It is **not** the EV percentage.

```
kelly = edge_prob / (odds - 1)
stake = min(kelly, TIER_CAP)
```

Tier caps: LOCK=10%, STRONG=5%, VALUE=3%, LEAN=1%, NO_BET=0%.

The companion module `aiBetModel/staking.py` provides `recommend_stake()` with the same tier/cap logic plus a `KELLY_MULTIPLIER` per tier (LOCK=1.0, STRONG=0.5, VALUE=0.25). Both engines are kept in sync; the streamlit app and pipeline both call `AggressiveStakeEngine.suggest()` with the same `best_edge / 100` quantity.

---

## HTTP API Reference

### `GET /api/predictions`
Returns an aggressive auto-scanned prediction feed highlighting top betting signals (`LOCK`, `STRONG`, `VALUE`).

### `GET /api/h2h`
Returns Head-to-Head historical data. Requires `home_id` and `away_id`.

### `GET /api/form`
Returns Form momentum and PPG trajectory. Requires `team_id`.

### `GET /api/scrape`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `home_id` | str | required | ESPN team ID |
| `away_id` | str | required | ESPN team ID |
| `league_slug` | str | required | e.g. `eng.1` |
| `decay` | float | 0.92 | Recency decay factor |
| `shrinkage_k` | float | 6.0 | Bayesian shrinkage |
| `home_advantage` | float | 1.0 | Multiplier on home xG |

**Response:**
```json
{
  "match_label": "Arsenal vs Chelsea",
  "model": { "expected_home_goals": 1.72, "home_win_prob": 0.48, ... },
  "confidence_score": 62,
  "market_comparison": { "home": { "edge_pct_points": 4.2 }, ... },
  "financials": { "home": { "ticker": "MANU", "price": 14.22, ... }, "away": null }
}
```

### `GET /api/scrape_by_name`

Same as `/api/scrape` but accepts `home=Arsenal&away=Chelsea` strings and resolves IDs internally.

### `GET /api/betika/scrape`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `home_id` | str | required | Team name (Betika) |
| `away_id` | str | required | Team name (Betika) |
| `match_id` | str | optional | Betika match ID for odds |

### `GET /api/chart`

Returns `scoreline_heatmap.png` as `image/png`. Cache-busted by timestamp parameter in dashboard.

### `GET /api/monitoring`

Returns system metrics and active alerts.

### `GET /api/results/sync`

Triggers result syncing for pending predictions against finished fixtures. Requires `FOOTBALL_DATA_API_KEY`.

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

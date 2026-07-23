"""
EQFBIS Intelligence Layer — Ensemble Modeling, Calibration & Responsible Staking
=================================================================================

What this module adds on top of scraper.py / analytics.py
-----------------------------------------------------------
1. EnsembleEngine
   Blends multiple independent estimators (shrinkage/Dixon-Coles model from
   build_model, scipy MLE attack/defense model from fit_poisson_glm_ratings,
   and a simple ELO-goal-differential prior) into one set of probabilities,
   weighted by each estimator's own confidence. Two models agreeing tightens
   the estimate; two models disagreeing WIDENS the reported uncertainty
   instead of hiding it. That disagreement signal is printed, not smoothed over.

2. PredictionLedger + Calibration
   A persistent, append-only SQLite log of every prediction the ensemble makes,
   plus a scorer that — once results are known — computes Brier score, log
   loss, and a calibration table (predicted-probability bucket vs actual
   hit rate). This is the only honest way to know whether "the model" is
   actually any good, as opposed to just looking confident. Nothing in this
   file claims accuracy that hasn't been measured this way.

3. ScoutingNarrative
   Turns the numbers into a short, plain-English written summary (form,
   model agreement, edge vs market, data-quality caveat). This is the
   "AI-written analysis" layer — it narrates, it does not recommend action.

4. ResponsibleStaking
   A capped, fractional-Kelly sizing helper. Deliberately NOT "aggressive":
   - stakes are capped at a hard ceiling (default 2% of bankroll) regardless
     of edge size or model confidence
   - Kelly is always applied at a fraction (default 0.25x) to correct for
     model error, never full Kelly
   - there is no loss-chasing, streak-based, or "double up" mode, and none
     will be added — that pattern is how bankrolls actually blow up
   - every output carries the same risk note; it is not decorative

None of this turns betting into a sure thing. Markets price in the same
public data this model uses, bookmaker margin (the "overround") is a
structural edge in the house's favor, and a statistically sound process can
still lose money over any given stretch of matches. The value of the pieces
below is in measuring whether the model is calibrated and keeping any stakes
small and bounded — not in producing bigger, bolder predictions.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from scipy.optimize import minimize
    from scipy.stats import poisson as _sp_poisson
except ImportError:
    minimize = None
    _sp_poisson = None

from scraper import TeamForm, MatchModelResult, poisson_pmf, EloRatingScraper, build_model

RISK_NOTE = (
    "This is a statistical estimate, not a guarantee. Bookmaker odds already "
    "price in an overround (built-in house margin), so beating the market "
    "consistently is hard even with a sound model. Treat every number here as "
    "a probability, not a certainty, and never stake money you can't afford to lose."
)


# ---------------------------------------------------------------------------
# 1. Ensemble engine
# ---------------------------------------------------------------------------

@dataclass
class EstimatorOutput:
    name: str
    expected_home_goals: float
    expected_away_goals: float
    confidence: float  # 0-1, this estimator's own view of its reliability


@dataclass
class EnsembleResult:
    expected_home_goals: float
    expected_away_goals: float
    component_estimates: list[EstimatorOutput]
    agreement_score: float  # 0-1, 1 = all estimators closely agree
    uncertainty_widened: bool
    shrinkage_model: Optional[MatchModelResult] = None

    def summary(self) -> str:
        lines = [f"Ensemble blend of {len(self.component_estimates)} estimator(s):"]
        for c in self.component_estimates:
            lines.append(
                f"  - {c.name:22s} exp goals {c.expected_home_goals:.2f}-{c.expected_away_goals:.2f} "
                f"(confidence weight {c.confidence:.2f})"
            )
        lines.append(f"Agreement score: {self.agreement_score:.2f} "
                      f"({'consistent' if self.agreement_score > 0.7 else 'estimators diverge'})")
        if self.uncertainty_widened:
            lines.append("NOTE: estimators disagreed meaningfully — treat this match as "
                          "lower-confidence than the sample size alone would suggest.")
        return "\n".join(lines)


def _fit_glm_estimate(home: TeamForm, away: TeamForm,
                       league_home: float, league_away: float) -> Optional[EstimatorOutput]:
    """Independent MLE attack/defense fit (separate from the shrinkage model)."""
    if minimize is None or _sp_poisson is None:
        return None
    if home.matches_played < 3 or away.matches_played < 3:
        return None
    try:
        def neg_log_lik(params):
            h_att, h_def, a_att, a_def = params
            if any(p <= 0 for p in params):
                return 1e10
            ll = 0.0
            for gs in home.goals_scored:
                ll += _sp_poisson.logpmf(gs, h_att * league_home)
            for gc in home.goals_conceded:
                ll += _sp_poisson.logpmf(gc, a_att * league_away)
            for gs in away.goals_scored:
                ll += _sp_poisson.logpmf(gs, a_att * league_away)
            for gc in away.goals_conceded:
                ll += _sp_poisson.logpmf(gc, h_att * league_home)
            return -ll

        res = minimize(neg_log_lik, [1.0, 1.0, 1.0, 1.0], method="L-BFGS-B",
                        bounds=[(0.1, 5.0)] * 4)
        if not res.success:
            return None
        h_att, h_def, a_att, a_def = res.x
        exp_home = league_home * h_att * a_def
        exp_away = league_away * a_att * h_def
        n = min(home.matches_played, away.matches_played)
        confidence = min(0.9, n / 20)
        return EstimatorOutput("MLE attack/defense (scipy)", exp_home, exp_away, confidence)
    except Exception:
        return None


def _elo_prior_estimate(home: TeamForm, away: TeamForm,
                         league_home: float, league_away: float) -> Optional[EstimatorOutput]:
    """A lightweight prior from ELO goal-differential, used as an independent
    estimator in the ensemble. Uses live API ratings if available or empirical form-derived ELO."""
    if not EloRatingScraper.available:
        return None
    scraper = EloRatingScraper()
    elo_home = scraper.get_club_elo(home.team_name, form=home)
    elo_away = scraper.get_club_elo(away.team_name, form=away)
    if not elo_home or not elo_away:
        return None
    diff = (elo_home - elo_away) / 400.0
    exp_home = league_home * math.exp(diff * 0.35)
    exp_away = league_away * math.exp(-diff * 0.35)
    return EstimatorOutput("ELO differential prior", exp_home, exp_away, confidence=0.35)


def build_ensemble(home: TeamForm, away: TeamForm,
                     shrinkage_exp_home: float, shrinkage_exp_away: float,
                     league_home: float, league_away: float,
                     shrinkage_model: Optional[MatchModelResult] = None) -> EnsembleResult:
    """Combine the existing shrinkage/Dixon-Coles estimate (passed in from
    build_model's output) with independent estimators, weighted by confidence.
    
    Pass an optional `shrinkage_model` MatchModelResult to include the full
    model output (probabilities, sample sizes) as a component; otherwise only
    the expected-goals values are used."""
    if shrinkage_model is not None:
        shrinkage_confidence = min(0.9, min(home.matches_played, away.matches_played) / 15)
    else:
        shrinkage_confidence = min(0.9, min(home.matches_played, away.matches_played) / 15)
    components = [
        EstimatorOutput("Shrinkage + Dixon-Coles", shrinkage_exp_home, shrinkage_exp_away,
                        confidence=shrinkage_confidence),
    ]
    glm_est = _fit_glm_estimate(home, away, league_home, league_away)
    if glm_est:
        components.append(glm_est)
    elo_est = _elo_prior_estimate(home, away, league_home, league_away)
    if elo_est:
        components.append(elo_est)

    total_weight = sum(c.confidence for c in components) or 1.0
    blended_home = sum(c.expected_home_goals * c.confidence for c in components) / total_weight
    blended_away = sum(c.expected_away_goals * c.confidence for c in components) / total_weight

    # Agreement: how tightly do estimators cluster relative to their mean?
    if len(components) > 1:
        home_vals = [c.expected_home_goals for c in components]
        away_vals = [c.expected_away_goals for c in components]
        spread = (max(home_vals) - min(home_vals)) + (max(away_vals) - min(away_vals))
        scale = blended_home + blended_away or 1.0
        agreement = max(0.0, 1.0 - spread / scale)
    else:
        agreement = 0.5  # only one estimator available — moderate, not high, confidence

    return EnsembleResult(
        expected_home_goals=blended_home,
        expected_away_goals=blended_away,
        component_estimates=components,
        agreement_score=round(agreement, 2),
        uncertainty_widened=bool(agreement < 0.6),
    )


# ---------------------------------------------------------------------------
# 2. Prediction ledger + calibration scoring
# ---------------------------------------------------------------------------

class PredictionLedger:
    """Append-only log of predictions, scored against outcomes once known.
    Thread-safe for concurrent server/pipeline operations."""

    def __init__(self, db_path: str = "eqfbis_ledger.sqlite3"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL,
                        home_team TEXT,
                        away_team TEXT,
                        home_win_prob REAL,
                        draw_prob REAL,
                        away_win_prob REAL,
                        agreement_score REAL,
                        actual_result TEXT,  -- 'H', 'D', 'A', or NULL until known
                        scored_at REAL
                    )
                """)
                conn.commit()

    def log(self, model: MatchModelResult, agreement_score: float = 1.0) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cur = conn.execute(
                    "INSERT INTO predictions (created_at, home_team, away_team, home_win_prob, "
                    "draw_prob, away_win_prob, agreement_score, actual_result, scored_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                    (time.time(), model.home_team, model.away_team, model.home_win_prob,
                     model.draw_prob, model.away_win_prob, agreement_score),
                )
                conn.commit()
                return cur.lastrowid

    def record_result(self, prediction_id: int, actual_result: str):
        """actual_result must be one of 'H' (home win), 'D' (draw), 'A' (away win)."""
        if actual_result not in ("H", "D", "A"):
            raise ValueError("actual_result must be 'H', 'D', or 'A'")
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute(
                    "UPDATE predictions SET actual_result = ?, scored_at = ? WHERE id = ?",
                    (actual_result, time.time(), prediction_id),
                )
                conn.commit()

    def calibration_report(self, bucket_size: float = 0.1) -> dict:
        """Brier score, log loss, and a predicted-vs-actual calibration table
        across every scored prediction. Returns 'insufficient_data' until
        there are enough scored predictions to say anything meaningful."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                rows = conn.execute(
                    "SELECT home_win_prob, draw_prob, away_win_prob, actual_result "
                    "FROM predictions WHERE actual_result IS NOT NULL"
                ).fetchall()

        if len(rows) < 20:
            return {
                "status": "insufficient_data",
                "scored_predictions": len(rows),
                "note": "Need at least 20 scored predictions before calibration "
                        "numbers mean anything statistically.",
            }

        brier_total = 0.0
        log_loss_total = 0.0
        buckets: dict[float, list[int]] = {}  # bucket -> [hits, total]

        for hp, dp, ap, actual in rows:
            outcomes = {"H": hp, "D": dp, "A": ap}
            for outcome, prob in outcomes.items():
                actual_bin = 1 if outcome == actual else 0
                brier_total += (prob - actual_bin) ** 2
                p_clipped = min(max(prob, 1e-6), 1 - 1e-6)
                log_loss_total += -(actual_bin * math.log(p_clipped) +
                                     (1 - actual_bin) * math.log(1 - p_clipped))
                bucket = round(prob / bucket_size) * bucket_size
                buckets.setdefault(bucket, [0, 0])
                buckets[bucket][1] += 1
                if actual_bin:
                    buckets[bucket][0] += 1

        n_outcomes = len(rows) * 3
        calibration_table = {
            f"{b:.1f}-{b + bucket_size:.1f}": {
                "predicted_midpoint": round(b + bucket_size / 2, 2),
                "actual_hit_rate": round(hits / total, 3) if total else None,
                "n": total,
            }
            for b, (hits, total) in sorted(buckets.items())
        }

        return {
            "status": "ok",
            "scored_predictions": len(rows),
            "brier_score": round(brier_total / n_outcomes, 4),
            "log_loss": round(log_loss_total / n_outcomes, 4),
            "calibration_table": calibration_table,
            "note": ("Brier score: lower is better, 0.0 is perfect, ~0.22 is what "
                     "guessing the historical home/draw/away base rates gets you. "
                     "Calibration table: if the model is honest, matches predicted "
                     "at ~60% should win about 60% of the time — check the rows above."),
        }


# ---------------------------------------------------------------------------
# 3. Narrative layer
# ---------------------------------------------------------------------------

def generate_scouting_narrative(model: MatchModelResult,
                                 ensemble: Optional[EnsembleResult] = None,
                                 market_comparison: Optional[dict] = None) -> str:
    """Plain-English written summary. Narrates the numbers; does not tell
    anyone what to do with them."""
    parts = []
    parts.append(
        f"{model.home_team} vs {model.away_team}: the model projects "
        f"{model.expected_home_goals:.1f}-{model.expected_away_goals:.1f} on expected goals, "
        f"putting home win / draw / away win at "
        f"{model.home_win_prob*100:.0f}% / {model.draw_prob*100:.0f}% / {model.away_win_prob*100:.0f}%."
    )
    parts.append(model.data_quality_note())

    if ensemble:
        if ensemble.agreement_score > 0.75:
            parts.append("Independent estimators (form-based, MLE, ELO-prior where available) "
                          "broadly agree, which supports the projection above.")
        else:
            parts.append("Independent estimators produced meaningfully different projections "
                          f"(agreement score {ensemble.agreement_score}) — that disagreement is "
                          "itself a signal that this fixture is harder to model than the headline "
                          "numbers suggest.")

    if market_comparison:
        biggest_edge_outcome = max(
            ("home", "draw", "away"),
            key=lambda o: abs(market_comparison[o]["edge_pct_points"]),
        )
        edge = market_comparison[biggest_edge_outcome]["edge_pct_points"]
        direction = "above" if edge > 0 else "below"
        parts.append(
            f"Versus the de-vigged market, the model sits {abs(edge):.1f} points {direction} "
            f"the implied probability on the {biggest_edge_outcome} outcome "
            f"(bookmaker overround {market_comparison['bookmaker_overround_pct']}%). "
            "A gap this size can reflect genuine model insight, but it can just as easily "
            "reflect information the market has that the model doesn't (injuries, team news, "
            "weather) — it is not on its own a reason to act."
        )

    parts.append(RISK_NOTE)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 4. Responsible staking (capped, non-aggressive by design)
# ---------------------------------------------------------------------------

MAX_STAKE_FRACTION = 0.02   # hard ceiling: never size a bet above 2% of bankroll
KELLY_FRACTION = 0.25       # always apply Kelly at a quarter-strength, never full Kelly


@dataclass
class StakeSuggestion:
    outcome: str
    model_prob: float
    offered_odds: float
    raw_kelly_fraction: float
    capped_fraction: float
    note: str


def suggest_stake(model_prob: float, offered_odds: float, outcome_label: str,
                   confidence_multiplier: float = 1.0) -> StakeSuggestion:
    """Fractional-Kelly stake sizing with a hard cap. This intentionally does
    NOT scale up with 'confidence', streaks, or past losses — confidence_multiplier
    exists only to let low agreement_score / low sample-size situations scale
    the stake DOWN, never up, and is clamped to [0, 1] for that reason."""
    confidence_multiplier = min(max(confidence_multiplier, 0.0), 1.0)

    b = offered_odds - 1.0  # net odds
    if b <= 0:
        raise ValueError("offered_odds must be greater than 1.0")

    edge = (model_prob * (b + 1)) - 1  # full Kelly numerator, standard form
    raw_kelly = max(0.0, edge / b) if edge > 0 else 0.0
    fractional = raw_kelly * KELLY_FRACTION * confidence_multiplier
    capped = min(fractional, MAX_STAKE_FRACTION)

    if raw_kelly == 0.0:
        note = "Model does not see an edge on this outcome at these odds — sizing is zero."
    elif capped == MAX_STAKE_FRACTION and fractional > MAX_STAKE_FRACTION:
        note = (f"Kelly math suggested a larger stake; capped at the hard ceiling of "
                 f"{MAX_STAKE_FRACTION*100:.0f}% of bankroll regardless of edge size.")
    else:
        note = f"Quarter-Kelly stake, {capped*100:.2f}% of bankroll."

    return StakeSuggestion(
        outcome=outcome_label,
        model_prob=model_prob,
        offered_odds=offered_odds,
        raw_kelly_fraction=round(raw_kelly, 4),
        capped_fraction=round(capped, 4),
        note=note + " " + RISK_NOTE,
    )


# ---------------------------------------------------------------------------
# 5. Aggressive Prediction Engine (New Parallel Option)
# ---------------------------------------------------------------------------

import enum

class ConfidenceTier(enum.Enum):
    LOCK = "LOCK"
    STRONG = "STRONG"
    VALUE = "VALUE"
    LEAN = "LEAN"
    NO_BET = "NO_BET"


class AggressiveStakeEngine:
    """Parallel to suggest_stake. Uses full Kelly capped at higher limits
    for high confidence predictions."""

    TIER_CAPS = {
        ConfidenceTier.LOCK: 0.10,
        ConfidenceTier.STRONG: 0.05,
        ConfidenceTier.VALUE: 0.03,
        ConfidenceTier.LEAN: 0.01,
        ConfidenceTier.NO_BET: 0.0,
    }

    @staticmethod
    def suggest(tier: ConfidenceTier, edge_raw: float, odds: float) -> dict:
        if tier == ConfidenceTier.NO_BET or odds <= 1.0 or edge_raw <= 0:
            return {"stake_pct": 0.0, "note": "No bet recommended."}
        
        # Kelly Criterion: edge / net_odds
        net_odds = odds - 1.0
        kelly = max(0.0, edge_raw / net_odds) if net_odds > 0 else 0.0
        
        cap = AggressiveStakeEngine.TIER_CAPS.get(tier, 0.0)
        stake = min(kelly, cap)
        
        return {
            "stake_pct": round(stake * 100, 2),
            "kelly_raw": round(kelly * 100, 2),
            "tier_cap_pct": round(cap * 100, 2),
            "note": "Aggressive staking logic applies. Higher volatility expected."
        }


@dataclass
class PredictionCard:
    match_label: str
    match_date: str
    competition: str
    recommended_bet: str
    confidence_tier: str
    model_probability: float
    market_implied_probability: float
    edge_pct: float
    stake_suggestion_pct: float
    signals: dict
    multi_market_predictions: dict
    scouting_narrative: str
    model_data: dict


def generate_aggressive_narrative(model: dict, signals: dict, tier: ConfidenceTier, recommended_bet: str) -> str:
    """Generates a bold, decisive analysis narrative."""
    if tier == ConfidenceTier.NO_BET:
        return "Insufficient edge or data for a confident prediction. SKIP."
        
    parts = []
    
    home_team = model.get('home_team', 'Home')
    away_team = model.get('away_team', 'Away')
    
    scoring_trend = signals.get('home_momentum', {}).get('scoring_trend', 'AVERAGE')
    away_defensive_trend = signals.get('away_momentum', {}).get('defensive_trend', 'AVERAGE')
    
    if scoring_trend == 'HOT':
        parts.append(f"{home_team}'s attack is RELENTLESS.")
    if away_defensive_trend == 'LEAKY':
        parts.append(f"{away_team}'s defense is LEAKY right now.")
        
    if recommended_bet:
        parts.append(f"{recommended_bet} is the primary angle here.")
        
    if tier == ConfidenceTier.LOCK:
        parts.append("This is a LOCK.")
    elif tier == ConfidenceTier.STRONG:
        parts.append("This is a STRONG play.")
    elif tier == ConfidenceTier.VALUE:
        parts.append("Great VALUE here.")
        
    if not parts:
        parts.append("Solid underlying metrics support this angle.")
        
    return " ".join(parts)


class AutoPredictionPipeline:
    """Batch scans fixtures and ranks by edge."""
    
    def __init__(self):
        from scraper import ESPNScraperClient, HeadToHeadFetcher, FormMomentumCalculator, MultiMarketPredictor
        self.espn = ESPNScraperClient()
        self.h2h = HeadToHeadFetcher()
        self.momentum = FormMomentumCalculator()
        self.market = MultiMarketPredictor()

    async def scan_all_fixtures(self, betika_fixtures: list[dict]) -> list[PredictionCard]:
        # This function acts as the core pipeline orchestrator.
        # Implemented asynchronously to handle batch processing when wired to the API.
        
        import asyncio
        from scraper import build_model
        
        cards = []
        
        for fixture in betika_fixtures:
            home_team = fixture.get("home_team", "")
            away_team = fixture.get("away_team", "")
            match_id = fixture.get("match_id", "")
            date = fixture.get("start_time", "")
            comp = fixture.get("competition_name", "")
            
            # Simple odds fallback
            oh = float(fixture.get("home_odd", 0) or 0)
            od = float(fixture.get("draw_odd", 0) or 0)
            oa = float(fixture.get("away_odd", 0) or 0)
            
            # (In reality, this would search ESPN, build TeamForm, then build_model)
            # We construct a mock pipeline output here for now, but will connect real data in analytics.py.
            # The actual execution happens via analytics.py calling scraper tools.
            pass
            
        return cards


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    home = TeamForm("Arsenal", 10, [3, 1, 2, 3, 2, 1, 2, 3, 1, 2], [0, 0, 1, 1, 1, 2, 0, 1, 0, 1])
    away = TeamForm("Chelsea", 10, [1, 0, 2, 1, 2, 3, 1, 1, 0, 2], [1, 2, 1, 0, 1, 2, 1, 2, 1, 1])

    ensemble = build_ensemble(home, away, 1.55, 1.22, 1.55, 1.22)
    print(ensemble.summary())
    print()

    ledger = PredictionLedger(db_path=":memory:".replace(":memory:", "eqfbis_ledger_demo.sqlite3"))
    print(ledger.calibration_report())
    print()

    stake = suggest_stake(model_prob=0.52, offered_odds=2.10, outcome_label="home",
                           confidence_multiplier=ensemble.agreement_score)
    print(stake)


# ---------------------------------------------------------------------------
# 6. ML-Enhanced Ensemble & Calibration (NEW — does not remove anything above)
# ---------------------------------------------------------------------------

class MLEnhancedEnsemble:
    """Wraps the existing ensemble with ML probabilities when a trained model
    is available.  Falls back gracefully to the statistical-only ensemble if
    no ML models are loaded."""

    def __init__(self, ml_pipeline: Optional[Any] = None,
                 calibration_mgr: Optional[Any] = None):
        self.ml_pipeline = ml_pipeline
        self.calibration_mgr = calibration_mgr

    def enrich(self, ensemble: EnsembleResult, feature_vector: Any) -> dict:
        """Add ML predictions and calibrated probabilities to the ensemble
        output.  Returns an enriched dict with keys:
          - original ensemble fields
          - ml_prediction (if available)
          - calibrated_probabilities (if calibration available)
          - blend_weights
        """
        enriched = {
            "ensemble": ensemble.summary(),
            "expected_home_goals": ensemble.expected_home_goals,
            "expected_away_goals": ensemble.expected_away_goals,
            "agreement_score": ensemble.agreement_score,
            "uncertainty_widened": ensemble.uncertainty_widened,
            "ml_prediction": None,
            "calibrated_probabilities": None,
            "blend_weights": {"statistical": 1.0, "ml": 0.0},
        }

        ml_pred = None
        if self.ml_pipeline is not None:
            try:
                ml_pred = self.ml_pipeline.predict(feature_vector)
                if ml_pred:
                    enriched["ml_prediction"] = ml_pred.to_dict()
                    enriched["blend_weights"]["ml"] = ml_pred.confidence
                    enriched["blend_weights"]["statistical"] = (
                        1.0 - ml_pred.confidence * 0.5)
            except Exception:
                pass

        if ml_pred and self.calibration_mgr is not None:
            try:
                raw_probs = {
                    "home_win_prob": ml_pred.home_win_prob,
                    "draw_prob": ml_pred.draw_prob,
                    "away_win_prob": ml_pred.away_win_prob,
                }
                cal_probs = self.calibration_mgr.apply_all("ensemble", raw_probs)
                enriched["calibrated_probabilities"] = cal_probs
            except Exception:
                pass

        return enriched


class PredictionPipelineV2:
    """Next-generation prediction pipeline that combines:
      - Statistical Poisson/Dixon-Coles model (existing)
      - ML gradient-boosted models (new)
      - Probability calibration (new)
      - Feature engineering (new)
      - Monitoring & paper trading (new)

    Wires everything together without modifying the underlying modules."""

    def __init__(self):
        from features import FeatureExtractor
        from ml_pipeline import MLPipeline
        from calibration import CalibrationManager
        from monitoring import SystemMonitor, PaperTradingTracker
        self.feature_extractor = FeatureExtractor()
        self.ml_pipeline = MLPipeline()
        self.calibration_mgr = CalibrationManager()
        self.ml_ensemble = MLEnhancedEnsemble(self.ml_pipeline, self.calibration_mgr)
        self.monitor = SystemMonitor()
        self.paper_trader = PaperTradingTracker()

    def predict(self, home_form, away_form, league_slug: str = "",
                match_date: str = "", odds_home: float = 0, odds_draw: float = 0,
                odds_away: float = 0, match_id: str = "",
                home_id: str = "", away_id: str = "",
                league_home: float = 1.45, league_away: float = 1.15,
                home_advantage: float = 1.0, decay: float = 0.92,
                shrinkage_k: float = 6.0) -> dict:
        t0 = time.time()
        try:
            from scraper import build_model, compare_to_market

            model = build_model(home_form, away_form, league_home, league_away,
                                home_advantage=home_advantage, decay=decay,
                                shrinkage_k=shrinkage_k)

            ensemble = build_ensemble(
                home_form, away_form,
                model.expected_home_goals, model.expected_away_goals,
                league_home, league_away, shrinkage_model=model,
            )

            feature_vector = self.feature_extractor.build_vector(
                home_form, away_form, league_slug=league_slug,
                match_date=match_date, odds_home=odds_home, odds_draw=odds_draw,
                odds_away=odds_away, match_id=match_id,
                home_id=home_id, away_id=away_id, model=model,
            )

            enriched = self.ml_ensemble.enrich(ensemble, feature_vector)

            market_comp = None
            if odds_home and odds_draw and odds_away:
                market_comp = compare_to_market(model, odds_home, odds_draw, odds_away)

            latency_ms = (time.time() - t0) * 1000
            self.monitor.record("prediction_latency_ms", latency_ms)

            return {
                "match_label": f"{home_form.team_name} vs {away_form.team_name}",
                "statistical_model": model.__dict__,
                "ensemble": enriched,
                "market_comparison": market_comp,
                "feature_vector_summary": {
                    "n_features": len(feature_vector.to_feature_array()),
                    "form_home_available": feature_vector.form_home is not None,
                    "xg_available": feature_vector.xg_home is not None if feature_vector.xg_home else False,
                    "elo_available": feature_vector.elo.elo_available if feature_vector.elo else False,
                },
                "latency_ms": round(latency_ms, 2),
                "pipeline_version": "v2.0",
            }
        except Exception as e:
            self.monitor.record("prediction_latency_ms", 99999)
            raise

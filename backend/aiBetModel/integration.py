"""
aiBetModel integration helpers.

Wires the spec-driven modules together into a single callable pipeline:
  model -> market assessment -> stake recommendation -> comparison -> report.

No external dependencies beyond the project's own modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aiBetModel.market import (
    assess_market,
    classify_efficiency,
    expected_value,
    MarketAssessment,
)
from aiBetModel.staking import recommend_stake, determine_tier, STANDARD_DISCLAIMER
from aiBetModel.comparison import compare_numeric, build_comparison_table
from aiBetModel.quality import grade_data_quality, EvidenceChecklist
from aiBetModel.report import render_report


def build_market_assessments(
    model_home_prob: float,
    model_draw_prob: float,
    model_away_prob: float,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> List[Dict[str, Any]]:
    """Convert model probabilities + decimal odds into MarketAssessment dicts."""
    outcomes = [
        ("home", model_home_prob, odds_home),
        ("draw", model_draw_prob, odds_draw),
        ("away", model_away_prob, odds_away),
    ]
    full_market_odds = [odds_home, odds_draw, odds_away]
    assessments = []
    for name, true_prob, decimal_odds in outcomes:
        ma = assess_market(name, decimal_odds, true_prob, full_market_odds)
        assessments.append({
            "outcome": ma.outcome,
            "decimal_odds": ma.decimal_odds,
            "implied_prob": ma.implied_prob,
            "fair_prob": ma.fair_prob,
            "true_prob": ma.true_prob,
            "edge": ma.edge,
            "ev_per_unit": ma.ev_per_unit,
            "overround_pct": ma.overround_pct,
            "classification": ma.classification,
        })
    return assessments


def build_stake_recommendations(
    model_home_prob: float,
    model_draw_prob: float,
    model_away_prob: float,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    confidence: str = "medium",
    data_grade: str = "B",
    model_disagreement_pts: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Produce tiered stake recommendations for every market outcome."""
    outcomes = [
        ("home", model_home_prob, odds_home),
        ("draw", model_draw_prob, odds_draw),
        ("away", model_away_prob, odds_away),
    ]
    recs = {}
    for name, true_prob, decimal_odds in outcomes:
        edge = expected_value(true_prob, decimal_odds)
        rec = recommend_stake(
            true_prob=true_prob,
            decimal_odds=decimal_odds,
            confidence=confidence,
            data_grade=data_grade,
            model_disagreement_pts=model_disagreement_pts,
            edge=edge,
        )
        recs[name] = {
            "tier": rec.tier,
            "kelly_fraction_raw": rec.kelly_fraction_raw,
            "kelly_fraction_applied": rec.kelly_fraction_applied,
            "capped_stake_pct": rec.capped_stake_pct,
            "reason": rec.reason,
        }
    return recs


def build_data_quality_checklist(
    model_home_prob: float,
    model_draw_prob: float,
    model_away_prob: float,
    odds_home: Optional[float] = None,
    odds_draw: Optional[float] = None,
    odds_away: Optional[float] = None,
    xg_available: bool = False,
    team_strength_metrics_available: bool = False,
    historical_h2h_available: bool = False,
    lineups_confirmed: bool = False,
    injuries_verified: bool = False,
    conflicting_sources: bool = False,
    small_sample_size: bool = False,
) -> str:
    """Grade data quality from available evidence flags."""
    odds_verified = all(o is not None and o > 0 for o in [odds_home, odds_draw, odds_away])
    checklist = EvidenceChecklist(
        lineups_confirmed=lineups_confirmed,
        injuries_verified=injuries_verified,
        odds_verified=odds_verified,
        xg_data_available=xg_available,
        team_strength_metrics_available=team_strength_metrics_available,
        historical_h2h_available=historical_h2h_available,
        conflicting_sources=conflicting_sources,
        small_sample_size=small_sample_size,
    )
    return grade_data_quality(checklist)


def build_comparison_from_model(
    home_expected_goals: float,
    away_expected_goals: float,
    home_win_prob: float,
    away_win_prob: float,
    draw_prob: float,
    home_elo: Optional[float] = None,
    away_elo: Optional[float] = None,
) -> str:
    """Build a basic opponent comparison Markdown table from model outputs."""
    rows = [
        compare_numeric("Expected goals (home)", home_expected_goals, away_expected_goals),
        compare_numeric("Home win prob", home_win_prob, away_win_prob),
        compare_numeric("Draw prob", draw_prob, draw_prob, neutral_threshold=0.01),
        compare_numeric("Away win prob", away_win_prob, home_win_prob, higher_is_better=False),
    ]
    if home_elo is not None and away_elo is not None:
        rows.insert(0, compare_numeric("ELO", home_elo, away_elo))
    return build_comparison_table(rows)


def render_match_report(
    home_team: str,
    away_team: str,
    league: str,
    match_date: str,
    model_home_prob: float,
    model_draw_prob: float,
    model_away_prob: float,
    expected_home_goals: float,
    expected_away_goals: float,
    odds_home: Optional[float] = None,
    odds_draw: Optional[float] = None,
    odds_away: Optional[float] = None,
    confidence: str = "medium",
    data_grade: Optional[str] = None,
    model_disagreement_pts: float = 0.0,
    home_elo: Optional[float] = None,
    away_elo: Optional[float] = None,
    reasons_for: Optional[List[str]] = None,
    reasons_against: Optional[List[str]] = None,
    tactical_notes: Optional[List[str]] = None,
    statistical_notes: Optional[List[str]] = None,
    model_notes: Optional[List[str]] = None,
    risk_notes: Optional[List[str]] = None,
) -> str:
    """High-level helper: build every aiBetModel artifact for a single fixture
    and render it to the Markdown report format defined by the spec."""

    if data_grade is None:
        data_grade = build_data_quality_checklist(
            model_home_prob, model_draw_prob, model_away_prob,
            odds_home, odds_draw, odds_away,
        )

    market_assessments = []
    stake_recs = {}
    if odds_home and odds_draw and odds_away:
        market_assessments = build_market_assessments(
            model_home_prob, model_draw_prob, model_away_prob,
            odds_home, odds_draw, odds_away,
        )
        stake_recs = build_stake_recommendations(
            model_home_prob, model_draw_prob, model_away_prob,
            odds_home, odds_draw, odds_away,
            confidence=confidence,
            data_grade=data_grade,
            model_disagreement_pts=model_disagreement_pts,
        )

    comparison_md = build_comparison_from_model(
        expected_home_goals, expected_away_goals,
        model_home_prob, model_away_prob, model_draw_prob,
        home_elo=home_elo, away_elo=away_elo,
    )

    fixture = {
        "home": home_team,
        "away": away_team,
        "league": league,
        "date": match_date,
    }

    return render_report(
        fixture=fixture,
        comparison_table_md=comparison_md,
        tactical_notes=tactical_notes or [],
        statistical_notes=statistical_notes or [],
        market_assessments=[_dict_to_market_assessment(m) for m in market_assessments],
        model_notes=model_notes or [],
        risk_notes=risk_notes or [STANDARD_DISCLAIMER],
        stake_recs={k: _dict_to_stake_recommendation(v) for k, v in stake_recs.items()},
        confidence=confidence,
        data_grade=data_grade,
        reasons_for=reasons_for or [],
        reasons_against=reasons_against or [],
    )


def _dict_to_market_assessment(d: Dict[str, Any]) -> MarketAssessment:
    return MarketAssessment(
        outcome=d["outcome"],
        decimal_odds=d["decimal_odds"],
        implied_prob=d["implied_prob"],
        fair_prob=d["fair_prob"],
        true_prob=d["true_prob"],
        edge=d["edge"],
        ev_per_unit=d["ev_per_unit"],
        overround_pct=d["overround_pct"],
        classification=d["classification"],
    )


def _dict_to_stake_recommendation(d: Dict[str, Any]) -> Any:
    from aiBetModel.staking import StakeRecommendation
    return StakeRecommendation(
        tier=d["tier"],
        kelly_fraction_raw=d["kelly_fraction_raw"],
        kelly_fraction_applied=d["kelly_fraction_applied"],
        capped_stake_pct=d["capped_stake_pct"],
        reason=d["reason"],
    )

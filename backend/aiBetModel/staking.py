"""
Stake sizing discipline. This module is intentionally the most locked-down
part of the whole engine.

Hard rules enforced in code, not just in a prompt:
  1. Stake size is a pure function of (edge, confidence, data grade,
     model agreement). There is no parameter anywhere in this module for
     "previous results," "streak," or "conviction." That's not an
     oversight — it's the point.
  2. Every recommendation is capped by config.STAKE_CEILINGS regardless
     of how large the computed Kelly fraction is.
  3. No stake is ever recommended below config.MIN_GRADE_FOR_STAKE.
  4. request_loss_chasing_stake() exists ONLY to refuse. It is wired into
     main.py's CLI so that a user asking for "double up" / "martingale"
     sizing gets an explicit, explained refusal rather than the request
     silently falling through to the normal engine.
"""

from dataclasses import dataclass
from config import STAKE_CEILINGS, KELLY_MULTIPLIER, MIN_GRADE_FOR_STAKE, GRADE_ORDER


class LossChasingRefused(Exception):
    pass


def request_loss_chasing_stake(*_args, **_kwargs):
    """
    Deliberately refuses any request framed around recovering losses,
    doubling up, or "winning it back." Called from main.py whenever the
    CLI detects that kind of flag/intent.
    """
    raise LossChasingRefused(
        "Refused: loss-chasing / martingale / 'win it back' staking is not "
        "supported. Scaling a stake up because of a losing streak is the "
        "exact mechanism by which a statistically sound long-run process "
        "still produces a ruined bankroll on a short timeline — each stake "
        "must be sized only from that bet's own edge and your flat "
        "bankroll, independent of what happened on the last bet."
    )


def kelly_fraction(true_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction: f* = (b*p - q) / b
    where b = decimal_odds - 1 (net odds), p = true_prob, q = 1 - p.
    Returns 0 if the raw Kelly fraction is negative (i.e. no edge) rather
    than a negative "short" stake, since this engine doesn't lay bets.
    """
    b = decimal_odds - 1
    p = true_prob
    q = 1 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return max(f, 0.0)


def grade_meets_minimum(grade: str) -> bool:
    return GRADE_ORDER.index(grade) <= GRADE_ORDER.index(MIN_GRADE_FOR_STAKE)


@dataclass
class StakeRecommendation:
    tier: str
    kelly_fraction_raw: float
    kelly_fraction_applied: float   # after tier multiplier
    capped_stake_pct: float          # final number to show the user, as % of bankroll
    reason: str


def determine_tier(edge: float, confidence: str, data_grade: str,
                    model_disagreement_pts: float) -> str:
    """
    confidence: "high" | "medium" | "low"
    model_disagreement_pts: max pairwise spread between model outputs,
        in probability points (e.g. 0.04 = 4 points)

    This mirrors the spec's tier definitions:
      LOCK   — highly confident, massive edge, zero data flags
      STRONG — solid model agreement, clear edge
      VALUE  — lower confidence or marginal edge, still statistically +EV
      LEAN   — no edge or model disagreement
    """
    if not grade_meets_minimum(data_grade):
        return "NO BET"

    if edge <= 0:
        return "LEAN"

    high_disagreement = model_disagreement_pts >= 0.05  # 5+ point spread between models

    if edge >= 0.08 and confidence == "high" and data_grade == "A" and not high_disagreement:
        return "LOCK"
    if edge >= 0.04 and confidence in ("high", "medium") and data_grade in ("A", "B") and not high_disagreement:
        return "STRONG"
    if edge > 0 and not high_disagreement:
        return "VALUE"
    return "LEAN"  # positive edge but models disagree -> don't trust it


def recommend_stake(true_prob: float, decimal_odds: float, confidence: str,
                     data_grade: str, model_disagreement_pts: float,
                     edge: float) -> StakeRecommendation:
    tier = determine_tier(edge, confidence, data_grade, model_disagreement_pts)

    if tier in ("LEAN", "NO BET"):
        reason = (
            "No qualifying edge, model disagreement too high, or data grade "
            "below the minimum required to stake (Grade C)."
            if tier == "NO BET" or edge <= 0 or model_disagreement_pts >= 0.05
            else "No stake recommended."
        )
        return StakeRecommendation(
            tier=tier, kelly_fraction_raw=0.0, kelly_fraction_applied=0.0,
            capped_stake_pct=0.0, reason=reason,
        )

    raw_kelly = kelly_fraction(true_prob, decimal_odds)
    applied_kelly = raw_kelly * KELLY_MULTIPLIER[tier]
    ceiling = STAKE_CEILINGS[tier]
    capped = min(applied_kelly, ceiling)

    reason = (
        f"{tier} tier: raw Kelly {raw_kelly:.2%}, "
        f"{KELLY_MULTIPLIER[tier]:.0%}-Kelly applied = {applied_kelly:.2%}, "
        f"capped at tier ceiling {ceiling:.0%} -> final {capped:.2%} of bankroll."
    )

    return StakeRecommendation(
        tier=tier,
        kelly_fraction_raw=raw_kelly,
        kelly_fraction_applied=applied_kelly,
        capped_stake_pct=capped,
        reason=reason,
    )


STANDARD_DISCLAIMER = (
    "This is a statistical estimate, not a guarantee. Bookmaker overround is "
    "a structural, permanent edge against the bettor. A sound long-run "
    "process can still lose money over any given stretch of bets. Never "
    "stake money you cannot afford to lose."
)

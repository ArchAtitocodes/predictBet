"""
Market efficiency engine. Converts bookmaker decimal odds into implied and
fair (de-vigged) probabilities, and compares them against a model's true
probability to compute edge and expected value.
"""

from dataclasses import dataclass
from config import EFFICIENCY_THRESHOLDS


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1 / decimal_odds


def overround(decimal_odds_list: list[float]) -> float:
    """Sum of implied probabilities across a full market (e.g. all of 1X2)."""
    return sum(implied_probability(o) for o in decimal_odds_list)


def devig_proportional(decimal_odds_list: list[float]) -> list[float]:
    """
    Proportional (multiplicative) de-vig: scales each implied probability
    down by the market overround. Simple and standard; does not correct
    for favorite-longshot bias the way Shin's method does, which is worth
    noting explicitly rather than silently.
    """
    implied = [implied_probability(o) for o in decimal_odds_list]
    total = sum(implied)
    return [p / total for p in implied]


@dataclass
class MarketAssessment:
    outcome: str
    decimal_odds: float
    implied_prob: float
    fair_prob: float
    true_prob: float          # from your model
    edge: float                # true_prob - fair_prob
    ev_per_unit: float         # expected value per 1 unit staked
    overround_pct: float
    classification: str        # efficient / slightly inefficient / significantly inefficient


def expected_value(true_prob: float, decimal_odds: float) -> float:
    """
    EV per 1 unit staked: (true_prob * (odds - 1)) - (1 - true_prob) * 1
    Positive means the bet is +EV against the model's true probability.
    """
    return (true_prob * (decimal_odds - 1)) - ((1 - true_prob) * 1)


def classify_efficiency(edge: float) -> str:
    e = abs(edge)
    if e >= EFFICIENCY_THRESHOLDS["significantly_inefficient"]:
        return "significantly inefficient"
    if e >= EFFICIENCY_THRESHOLDS["slightly_inefficient"]:
        return "slightly inefficient"
    return "efficient"


def assess_market(outcome_name: str, decimal_odds: float, true_prob: float,
                   full_market_odds: list[float]) -> MarketAssessment:
    """
    full_market_odds: the decimal odds for every outcome in this market
    (e.g. [home, draw, away] for 1X2) — needed to compute the overround
    and de-vig correctly. Pass the outcome's own odds within that list.
    """
    implied = implied_probability(decimal_odds)
    fair_probs = devig_proportional(full_market_odds)
    idx = full_market_odds.index(decimal_odds)
    fair = fair_probs[idx]
    edge = true_prob - fair
    ev = expected_value(true_prob, decimal_odds)
    over = overround(full_market_odds)

    return MarketAssessment(
        outcome=outcome_name,
        decimal_odds=decimal_odds,
        implied_prob=implied,
        fair_prob=fair,
        true_prob=true_prob,
        edge=edge,
        ev_per_unit=ev,
        overround_pct=over - 1.0,
        classification=classify_efficiency(edge),
    )

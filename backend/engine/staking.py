"""
Engine staking module — thin re-export of aiBetModel.staking
to satisfy `from engine.staking import ...` imports.
"""

from aiBetModel.staking import (  # noqa: F401
    LossChasingRefused,
    request_loss_chasing_stake,
    kelly_fraction,
    grade_meets_minimum,
    StakeRecommendation,
    determine_tier,
    recommend_stake,
    STANDARD_DISCLAIMER,
)

__all__ = [
    "LossChasingRefused",
    "request_loss_chasing_stake",
    "kelly_fraction",
    "grade_meets_minimum",
    "StakeRecommendation",
    "determine_tier",
    "recommend_stake",
    "STANDARD_DISCLAIMER",
]

"""
Probabilistic models. Every function here requires real numeric inputs —
none of them fabricate a default xG, ELO, or rating if the caller doesn't
supply one. If an input is missing, the model is simply not run; see
main.py / report.py for how that omission gets surfaced to the user.
"""

import math
import random
from dataclasses import dataclass


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


@dataclass
class PoissonMarketProbabilities:
    home_win: float
    draw: float
    away_win: float
    btts_yes: float
    btts_no: float
    over_2_5: float
    under_2_5: float
    most_likely_score: tuple
    max_goals_modeled: int


def poisson_model(home_xg: float, away_xg: float, max_goals: int = 10) -> PoissonMarketProbabilities:
    """
    Independent-Poisson goal model. Requires expected-goals inputs for
    both sides (season/recent xG for, adjusted for opponent xGA, is the
    correct input here — that adjustment happens before this function,
    not inside it).
    """
    if home_xg <= 0 or away_xg <= 0:
        raise ValueError("Poisson model requires positive expected-goals values for both teams.")

    score_matrix = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_matrix[(hg, ag)] = poisson_pmf(hg, home_xg) * poisson_pmf(ag, away_xg)

    home_win = sum(p for (hg, ag), p in score_matrix.items() if hg > ag)
    draw = sum(p for (hg, ag), p in score_matrix.items() if hg == ag)
    away_win = sum(p for (hg, ag), p in score_matrix.items() if hg < ag)

    btts_yes = sum(p for (hg, ag), p in score_matrix.items() if hg > 0 and ag > 0)
    btts_no = 1 - btts_yes

    over_2_5 = sum(p for (hg, ag), p in score_matrix.items() if hg + ag > 2.5)
    under_2_5 = 1 - over_2_5

    most_likely = max(score_matrix.items(), key=lambda kv: kv[1])[0]

    return PoissonMarketProbabilities(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        btts_yes=btts_yes,
        btts_no=btts_no,
        over_2_5=over_2_5,
        under_2_5=under_2_5,
        most_likely_score=most_likely,
        max_goals_modeled=max_goals,
    )


def monte_carlo_simulation(home_xg: float, away_xg: float, n_sims: int = 100_000,
                            seed: int | None = None) -> dict:
    """
    Independent-Poisson Monte Carlo simulation, used as a cross-check
    against the closed-form Poisson model above. Disagreement between the
    two (beyond simulation noise) usually indicates a bug, not a real
    finding — they should converge for large n_sims since they share the
    same underlying assumption. True value-add comes when you extend this
    to correlated/Dixon-Coles adjustments later.
    """
    if home_xg <= 0 or away_xg <= 0:
        raise ValueError("Monte Carlo model requires positive expected-goals values for both teams.")

    rng = random.Random(seed)
    home_wins = draws = away_wins = btts = over_2_5 = 0

    for _ in range(n_sims):
        hg = _poisson_sample(rng, home_xg)
        ag = _poisson_sample(rng, away_xg)
        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1
        else:
            away_wins += 1
        if hg > 0 and ag > 0:
            btts += 1
        if hg + ag > 2.5:
            over_2_5 += 1

    return {
        "home_win": home_wins / n_sims,
        "draw": draws / n_sims,
        "away_win": away_wins / n_sims,
        "btts_yes": btts / n_sims,
        "over_2_5": over_2_5 / n_sims,
        "n_sims": n_sims,
    }


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm — avoids a numpy dependency."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def elo_win_probability(home_elo: float, away_elo: float, home_advantage: float = 100.0) -> dict:
    """
    Classic ELO expected-score formula, adapted to three-way football
    outcomes via a simple draw-margin heuristic. This is a coarse model —
    report it as such, and prefer Poisson/xG-based estimates when xG data
    is available.
    """
    diff = (home_elo + home_advantage) - away_elo
    expected_home = 1 / (1 + 10 ** (-diff / 400))

    # crude three-way split: shrink the binary expectation toward a draw
    # band proportional to how close the match is
    draw_weight = 0.25 * (1 - abs(2 * expected_home - 1))
    home_win = expected_home * (1 - draw_weight)
    away_win = (1 - expected_home) * (1 - draw_weight)
    draw = 1 - home_win - away_win

    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def bayesian_blend(model_estimates: list[dict], weights: list[float] | None = None) -> dict:
    """
    Blends multiple model outputs (e.g. Poisson, Monte Carlo, ELO) into a
    single estimate via weighted averaging. If weights are not supplied,
    equal weighting is used.

    IMPORTANT per spec: this function is for producing a single number to
    report ALONGSIDE the individual model outputs — never as a replacement
    for reporting them. Disagreement between inputs must still be surfaced
    by the caller (see report.py), not hidden by this blend.
    """
    if not model_estimates:
        raise ValueError("bayesian_blend requires at least one model estimate.")

    keys = model_estimates[0].keys()
    n = len(model_estimates)
    weights = weights or [1.0 / n] * n
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    blended = {}
    for k in keys:
        blended[k] = sum(est[k] * w for est, w in zip(model_estimates, weights))
    return blended


def model_disagreement(model_estimates: list[dict], key: str = "home_win") -> float:
    """
    Returns the max pairwise spread (in probability points) across models
    for a given outcome key. Used to flag disagreement per the spec's
    'disagreement is a signal, never smoothed over' rule.
    """
    values = [est[key] for est in model_estimates if key in est]
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)

"""
EQFBIS Accuracy Engine — Historical Performance, Odds & Market Backtesting
============================================================================

Purpose
-------
Everything in intelligence.py's PredictionLedger tells you if raw probabilities
are calibrated (does "60%" win 60% of the time). This module goes one step
further and asks the question that actually matters financially: if you had
bet the model's picks historically, at the odds actually on offer, would you
have made money — and is that result distinguishable from noise?

Three separate questions, kept separate on purpose
----------------------------------------------------
1. Is the model calibrated?            -> Brier score / log loss / calibration table
2. Would betting it have profited?     -> flat-stake ROI, capped-Kelly bankroll simulation
3. Is that profit real or just luck?   -> binomial significance test vs. the market's
                                           own implied win rate, plus closing-line value

A model can pass (1) and still fail (2) if it's calibrated but has no edge
over the market. It can even show positive ROI in (2) and still fail (3) if
the sample is too small to distinguish from a lucky streak — this module
will say so explicitly rather than let a small positive number look like
proven skill.

This is an audit tool. It does not recommend stakes going forward (see
intelligence.py's suggest_stake for that, which is capped by design) — it
only tells you, after the fact, whether the process is working.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None

from scraper import devig_1x2
from intelligence import PredictionLedger

RISK_NOTE = (
    "Past accuracy, even statistically significant past accuracy, does not "
    "guarantee future results. Markets adapt, team form changes, and a "
    "sample that clears a significance bar can still be a run of variance "
    "narrowly on the right side of the threshold."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SettledBet:
    """One historical, fully-resolved prediction. `actual_result` and
    `odds_taken` are required; `closing_odds` is optional and only needed
    for closing-line-value analysis."""
    date: str
    home_team: str
    away_team: str
    outcome_backed: str          # 'H', 'D', or 'A'
    model_prob: float            # model's probability for outcome_backed at bet time
    odds_taken: float            # decimal odds actually available when the bet was placed
    actual_result: str           # 'H', 'D', or 'A'
    market_prob_home: Optional[float] = None
    market_prob_draw: Optional[float] = None
    market_prob_away: Optional[float] = None
    closing_odds: Optional[float] = None   # odds just before kickoff, if tracked
    stake_fraction: Optional[float] = field(default=None)  # fraction of bankroll actually staked, if known

    def won(self) -> bool:
        return self.outcome_backed == self.actual_result

    def market_implied_prob(self) -> Optional[float]:
        """De-vigged market probability for the backed outcome, if all three
        market probabilities were supplied."""
        probs = {"H": self.market_prob_home, "D": self.market_prob_draw, "A": self.market_prob_away}
        return probs.get(self.outcome_backed)


def load_bets_from_csv(path: str) -> list[SettledBet]:
    """Expected columns: date,home_team,away_team,outcome_backed,model_prob,
    odds_taken,actual_result,market_prob_home,market_prob_draw,market_prob_away,
    closing_odds,stake_fraction. Optional columns may be blank."""
    bets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def flt(key):
                v = row.get(key, "")
                return float(v) if v not in (None, "",) else None

            bets.append(SettledBet(
                date=row["date"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                outcome_backed=row["outcome_backed"].strip().upper(),
                model_prob=float(row["model_prob"]),
                odds_taken=float(row["odds_taken"]),
                actual_result=row["actual_result"].strip().upper(),
                market_prob_home=flt("market_prob_home"),
                market_prob_draw=flt("market_prob_draw"),
                market_prob_away=flt("market_prob_away"),
                closing_odds=flt("closing_odds"),
                stake_fraction=flt("stake_fraction"),
            ))
    return bets


def load_bets_from_ledger(db_path: str = "eqfbis_ledger.sqlite3") -> list[SettledBet]:
    """Load scored predictions from intelligence.PredictionLedger and convert
    them into SettledBet objects for backtesting. Only predictions with a
    known actual_result are included.

    NOTE: The ledger does not store the odds that were available when the
    prediction was made, so odds_taken is set to None here. Financial
    backtests (ROI, bankroll simulation) require real odds and will return
    'insufficient_data' unless you attach odds separately via
    bets_with_devigged_market() or load_bets_from_csv().
    """
    ledger = PredictionLedger(db_path=db_path)
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        rows = conn.execute(
            "SELECT home_team, away_team, home_win_prob, draw_prob, away_win_prob, "
            "actual_result, created_at FROM predictions WHERE actual_result IS NOT NULL"
        ).fetchall()

    bets = []
    for home_team, away_team, hp, dp, ap, actual, created_at in rows:
        model_prob = {"H": hp, "D": dp, "A": ap}.get(actual, 0.0)
        bets.append(SettledBet(
            date=time.strftime("%Y-%m-%d", time.localtime(created_at)) if created_at else "",
            home_team=home_team, away_team=away_team,
            outcome_backed=actual, model_prob=model_prob,
            odds_taken=None,
            actual_result=actual,
            market_prob_home=None,
            market_prob_draw=None,
            market_prob_away=None,
        ))
    return bets


def load_bets_from_json(path: str) -> list[SettledBet]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [SettledBet(**r) for r in raw]


def bets_with_devigged_market(bets: list[SettledBet], odds_home: list[float],
                               odds_draw: list[float], odds_away: list[float]) -> list[SettledBet]:
    """Convenience: attach de-vigged market probabilities to a list of bets
    from parallel lists of raw 1X2 odds (same order/length as bets)."""
    out = []
    for bet, oh, od, oa in zip(bets, odds_home, odds_draw, odds_away):
        mh, md, ma = devig_1x2(oh, od, oa)
        bet.market_prob_home, bet.market_prob_draw, bet.market_prob_away = mh, md, ma
        out.append(bet)
    return out


# ---------------------------------------------------------------------------
# 1. Calibration / raw accuracy
# ---------------------------------------------------------------------------

def hit_rate(bets: list[SettledBet]) -> dict:
    if not bets:
        return {"status": "no_data"}
    wins = sum(1 for b in bets if b.won())
    return {
        "n": len(bets),
        "wins": wins,
        "hit_rate_pct": round(100 * wins / len(bets), 2),
        "avg_model_prob_pct": round(100 * sum(b.model_prob for b in bets) / len(bets), 2),
    }


def brier_and_log_loss(bets: list[SettledBet]) -> dict:
    if not bets:
        return {"status": "no_data"}
    brier_total = 0.0
    ll_total = 0.0
    for b in bets:
        actual_bin = 1
        p = min(max(b.model_prob, 1e-6), 1 - 1e-6)
        brier_total += (b.model_prob - actual_bin) ** 2 if b.won() else (b.model_prob - 0) ** 2
        actual_bin_for_ll = 1 if b.won() else 0
        ll_total += -(actual_bin_for_ll * math.log(p) + (1 - actual_bin_for_ll) * math.log(1 - p))
    n = len(bets)
    return {
        "n": n,
        "brier_score": round(brier_total / n, 4),
        "log_loss": round(ll_total / n, 4),
        "note": "Lower is better for both. This scores only the backed outcome's "
                "probability, not the full H/D/A distribution — use PredictionLedger "
                "in intelligence.py for full-distribution calibration.",
    }


def calibration_by_bucket(bets: list[SettledBet], bucket_size: float = 0.1) -> dict:
    buckets: dict[float, list[int]] = {}
    for b in bets:
        bucket = round(b.model_prob / bucket_size) * bucket_size
        buckets.setdefault(bucket, [0, 0])
        buckets[bucket][1] += 1
        if b.won():
            buckets[bucket][0] += 1
    table = {}
    for bucket, (wins, n) in sorted(buckets.items()):
        table[f"{bucket:.1f}-{bucket + bucket_size:.1f}"] = {
            "n": n,
            "predicted_midpoint": round(bucket + bucket_size / 2, 2),
            "actual_hit_rate": round(wins / n, 3) if n else None,
        }
    return table


# ---------------------------------------------------------------------------
# 2. Financial performance: flat stake and capped-Kelly bankroll simulation
# ---------------------------------------------------------------------------

def flat_stake_roi(bets: list[SettledBet], unit_stake: float = 1.0) -> dict:
    if not bets:
        return {"status": "no_data"}
    real_odds_bets = [b for b in bets if b.odds_taken is not None]
    if not real_odds_bets:
        return {
            "status": "insufficient_data",
            "note": "No bets have recorded odds_taken. Load bets from CSV or attach "
                    "real odds via bets_with_devigged_market() before computing ROI.",
        }
    total_staked = unit_stake * len(real_odds_bets)
    total_return = sum(unit_stake * b.odds_taken if b.won() else 0.0 for b in real_odds_bets)
    profit = total_return - total_staked
    return {
        "n": len(real_odds_bets),
        "total_staked": round(total_staked, 2),
        "total_return": round(total_return, 2),
        "profit": round(profit, 2),
        "roi_pct": round(100 * profit / total_staked, 2) if total_staked else None,
    }


def capped_kelly_bankroll_simulation(bets: list[SettledBet], starting_bankroll: float = 100.0,
                                      kelly_fraction: float = 0.25, max_stake_fraction: float = 0.02) -> dict:
    """Replays bets in order using the same capped-Kelly logic as
    intelligence.suggest_stake, tracking the bankroll curve and drawdown.
    Bets are assumed to be in chronological order."""
    if not bets:
        return {"status": "no_data"}

    real_odds_bets = [b for b in bets if b.odds_taken is not None]
    if not real_odds_bets:
        return {
            "status": "insufficient_data",
            "note": "No bets have recorded odds_taken. Load bets from CSV or attach "
                    "real odds via bets_with_devigged_market() before simulating bankroll.",
        }

    bankroll = starting_bankroll
    peak = starting_bankroll
    max_drawdown_pct = 0.0
    curve = [bankroll]

    for b in real_odds_bets:
        net_odds = b.odds_taken - 1.0
        edge = (b.model_prob * b.odds_taken) - 1
        raw_kelly = max(0.0, edge / net_odds) if edge > 0 and net_odds > 0 else 0.0
        fraction = min(raw_kelly * kelly_fraction, max_stake_fraction)
        stake = bankroll * fraction

        bankroll = bankroll - stake + (stake * b.odds_taken if b.won() else 0.0)
        curve.append(round(bankroll, 2))

        peak = max(peak, bankroll)
        if peak > 0:
            drawdown_pct = 100 * (peak - bankroll) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

    return {
        "n_bets": len(real_odds_bets),
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": round(bankroll, 2),
        "total_return_pct": round(100 * (bankroll - starting_bankroll) / starting_bankroll, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "bankroll_curve": curve,
        "note": "Simulation only — assumes bets are independent and stakes were actually "
                "available at the recorded odds. Real execution (line movement, stake limits, "
                "correlated fixtures) will differ from this.",
    }


# ---------------------------------------------------------------------------
# 3. Is any of it real? Significance vs. the market, and closing-line value
# ---------------------------------------------------------------------------

def significance_vs_market(bets: list[SettledBet]) -> dict:
    """Binomial test: is the observed hit rate distinguishable from what the
    de-vigged market itself implied, given this many bets? Requires market
    probabilities to be attached (see bets_with_devigged_market)."""
    usable = [b for b in bets if b.market_implied_prob() is not None]
    if len(usable) < 20:
        return {
            "status": "insufficient_data",
            "usable_bets": len(usable),
            "note": "Need at least 20 bets with market probabilities attached "
                    "for this test to mean anything.",
        }

    wins = sum(1 for b in usable if b.won())
    n = len(usable)
    expected_p = sum(b.market_implied_prob() for b in usable) / n

    result = {
        "n": n,
        "observed_hit_rate_pct": round(100 * wins / n, 2),
        "market_implied_hit_rate_pct": round(100 * expected_p, 2),
    }

    if binomtest is not None:
        test = binomtest(wins, n, expected_p, alternative="greater")
        result["p_value_outperformance"] = round(test.pvalue, 4)
        result["statistically_significant_at_5pct"] = test.pvalue < 0.05
        result["interpretation"] = (
            "p < 0.05 means the outperformance over the market's own implied rate "
            "is unlikely to be pure chance at this sample size. It does NOT mean "
            "the edge will persist."
            if test.pvalue < 0.05 else
            "Not statistically significant — the observed hit rate is consistent "
            "with the market simply being right and this being normal variance."
        )
    else:
        result["note"] = "scipy not available — install scipy for a formal significance test."

    return result


def closing_line_value(bets: list[SettledBet]) -> dict:
    """CLV compares the odds you actually bet against the closing odds just
    before kickoff. Beating the closing line consistently is one of the more
    respected proxies for genuine predictive skill in betting markets,
    because it doesn't require waiting on results (and their variance) at all —
    it only requires the odds to have moved in the direction you bet."""
    usable = [b for b in bets if b.closing_odds]
    if not usable:
        return {"status": "no_data", "note": "No closing_odds recorded on any bet."}

    clv_pcts = []
    beat_close = 0
    for b in usable:
        implied_taken = 1 / b.odds_taken
        implied_close = 1 / b.closing_odds
        clv_pct = 100 * (implied_close - implied_taken) / implied_taken
        clv_pcts.append(clv_pct)
        if b.odds_taken > b.closing_odds:  # got better (higher) odds than the close
            beat_close += 1

    return {
        "n": len(usable),
        "avg_clv_pct": round(sum(clv_pcts) / len(clv_pcts), 2),
        "pct_bets_beating_close": round(100 * beat_close / len(usable), 2),
        "note": "Positive avg_clv_pct means you were, on average, getting better odds "
                "than the market settled on — a sign of timing/information edge "
                "independent of whether individual bets won or lost.",
    }


# ---------------------------------------------------------------------------
# 4. Baseline comparison — is the model beating trivial strategies?
# ---------------------------------------------------------------------------

def baseline_comparisons(bets: list[SettledBet]) -> dict:
    """Compares the model's actual ROI against two naive baselines that any
    real model needs to beat to be worth using at all."""
    if not bets:
        return {"status": "no_data"}

    model_roi = flat_stake_roi(bets)["roi_pct"]

    # Baseline A: always back the market favorite (lowest odds / highest implied prob)
    # Only computable for bets where all three market probs are attached.
    usable = [b for b in bets if b.market_implied_prob() is not None]
    favorite_bets = []
    for b in usable:
        probs = {"H": b.market_prob_home, "D": b.market_prob_draw, "A": b.market_prob_away}
        favorite_outcome = max(probs, key=probs.get)
        favorite_bets.append(SettledBet(
            date=b.date, home_team=b.home_team, away_team=b.away_team,
            outcome_backed=favorite_outcome, model_prob=probs[favorite_outcome],
            odds_taken=b.odds_taken if b.outcome_backed == favorite_outcome else b.odds_taken,
            actual_result=b.actual_result,
        ))
    favorite_roi = flat_stake_roi(favorite_bets)["roi_pct"] if favorite_bets else None

    return {
        "model_flat_stake_roi_pct": model_roi,
        "always_back_favorite_roi_pct": favorite_roi,
        "note": "If the model's ROI doesn't clear the 'always back the favorite' baseline "
                "by a meaningful margin, the extra modeling complexity isn't earning its keep. "
                "Note the favorite baseline here reuses recorded odds_taken as an approximation; "
                "for an exact comparison, favorite-outcome odds should be sourced separately.",
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def full_accuracy_report(bets: list[SettledBet]) -> dict:
    return {
        "hit_rate": hit_rate(bets),
        "brier_and_log_loss": brier_and_log_loss(bets),
        "calibration_by_bucket": calibration_by_bucket(bets),
        "flat_stake_roi": flat_stake_roi(bets),
        "capped_kelly_simulation": capped_kelly_bankroll_simulation(bets),
        "significance_vs_market": significance_vs_market(bets),
        "closing_line_value": closing_line_value(bets),
        "baseline_comparisons": baseline_comparisons(bets),
        "risk_note": RISK_NOTE,
    }


def print_report(bets: list[SettledBet]):
    report = full_accuracy_report(bets)
    print("=" * 78)
    print("  EQFBIS ACCURACY REPORT")
    print("=" * 78)
    for section, data in report.items():
        print(f"\n[{section}]")
        print(json.dumps(data, indent=2, default=str))
    print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# CLI smoke test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(7)

    demo_bets = []
    for i in range(40):
        true_p = random.uniform(0.35, 0.65)
        odds = round(1 / (true_p * 0.93), 2)  # market with ~7% overround, roughly fair
        won = random.random() < true_p
        demo_bets.append(SettledBet(
            date=f"2026-0{(i % 9) + 1}-01",
            home_team="Team A", away_team="Team B",
            outcome_backed="H", model_prob=round(true_p + random.uniform(-0.03, 0.03), 3),
            odds_taken=odds,
            actual_result="H" if won else "A",
            market_prob_home=round(true_p, 3),
            market_prob_draw=0.25, market_prob_away=round(1 - true_p - 0.25, 3),
            closing_odds=round(odds * random.uniform(0.95, 1.05), 2),
        ))

    print_report(demo_bets)

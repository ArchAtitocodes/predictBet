"""
Assembles the full fixture report in the output format required by the
spec: Executive Summary, Opponent Comparison, Tactical/Statistical/Market/
Model/Risk Assessment, Recommended Market(s), Capped Stake Suggestion,
Confidence, Data Quality Grade, Reasons For/Against, plus the summary
table.
"""

from datetime import datetime, timezone
from engine.staking import STANDARD_DISCLAIMER


def render_report(
    fixture: dict,               # {"home": str, "away": str, "league": str, "date": str}
    comparison_table_md: str,
    tactical_notes: list,
    statistical_notes: list,
    market_assessments: list,     # list of engine.market.MarketAssessment
    model_notes: list,
    risk_notes: list,
    stake_recs: dict,             # {market_outcome: StakeRecommendation}
    confidence: str,
    data_grade: str,
    reasons_for: list,
    reasons_against: list,
    track_record_note: str = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append(f"# PredictBet Report — {fixture['home']} vs {fixture['away']}")
    lines.append(f"*{fixture.get('league', 'Unknown league')} — {fixture.get('date', 'date TBC')} — generated {now}*")
    lines.append("")

    # Any market with no positive-EV, sufficiently-graded stake is NO BET.
    any_bet_recommended = any(
        rec.tier not in ("LEAN", "NO BET") for rec in stake_recs.values()
    )

    lines.append("## Executive Summary")
    if not any_bet_recommended:
        lines.append("")
        lines.append("> **NO BET** — no market on this fixture cleared the full "
                      "filter (positive EV, statistical support, tactical support, "
                      "genuine inefficiency, reliable data, acceptable variance, "
                      "sufficient liquidity). See Reasons Against below.")
    else:
        qualifying = [f"{m} ({rec.tier})" for m, rec in stake_recs.items()
                      if rec.tier not in ("LEAN", "NO BET")]
        lines.append(f"Qualifying market(s): {', '.join(qualifying)}. "
                     f"Data quality grade **{data_grade}**, overall confidence **{confidence}**.")
    lines.append("")

    lines.append("## Opponent Comparison")
    lines.append(comparison_table_md)
    lines.append("")

    lines.append("## Tactical Assessment")
    lines.extend(f"- {n}" for n in tactical_notes) if tactical_notes else lines.append("- No verified tactical inputs provided.")
    lines.append("")

    lines.append("## Statistical Assessment")
    lines.extend(f"- {n}" for n in statistical_notes) if statistical_notes else lines.append("- No verified statistical inputs provided.")
    lines.append("")

    lines.append("## Market Assessment")
    if market_assessments:
        lines.append("| Market | Odds | Implied | Fair | True (model) | Edge | EV/unit | Overround | Classification |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for m in market_assessments:
            lines.append(
                f"| {m.outcome} | {m.decimal_odds:.2f} | {m.implied_prob:.1%} | "
                f"{m.fair_prob:.1%} | {m.true_prob:.1%} | {m.edge:+.1%} | "
                f"{m.ev_per_unit:+.3f} | {m.overround_pct:.1%} | {m.classification} |"
            )
    else:
        lines.append("- No market data available/verified for this fixture.")
    lines.append("")

    lines.append("## Model Assessment")
    lines.extend(f"- {n}" for n in model_notes) if model_notes else lines.append("- No models could be run (insufficient verified inputs).")
    lines.append("")

    lines.append("## Risk Assessment")
    lines.extend(f"- {n}" for n in risk_notes) if risk_notes else lines.append("- No specific risk flags noted.")
    lines.append("")

    lines.append("## Recommended Market(s) & Capped Stake Suggestion")
    if any_bet_recommended:
        for market_name, rec in stake_recs.items():
            if rec.tier in ("LEAN", "NO BET"):
                continue
            lines.append(f"- **{market_name}** — Tier: {rec.tier}, "
                         f"Capped stake: **{rec.capped_stake_pct:.1%} of bankroll**. {rec.reason}")
    else:
        lines.append("- **NO BET** on every market considered.")
    lines.append("")
    lines.append(f"> {STANDARD_DISCLAIMER}")
    lines.append("")

    lines.append(f"## Confidence: {confidence.upper()}")
    lines.append(f"## Data Quality Grade: {data_grade}")
    lines.append("")

    lines.append("## Reasons For")
    lines.extend(f"- {r}" for r in reasons_for) if reasons_for else lines.append("- None identified.")
    lines.append("")

    lines.append("## Reasons Against")
    lines.extend(f"- {r}" for r in reasons_against) if reasons_against else lines.append("- None identified.")
    lines.append("")

    lines.append("## Track Record")
    lines.append(track_record_note or (
        "This tool does not maintain a scored history of past predictions in "
        "this run. No hit rate, ROI, or accuracy figure is claimed. If you "
        "want a real track record, log outcomes yourself and wire that log "
        "into this section — see README.md."
    ))
    lines.append("")

    lines.append("## Summary Table")
    lines.append("| Match | Market | Odds | Implied Prob | Fair Prob | Edge | EV | Capped Stake | Confidence | Risk | Data Grade |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    match_str = f"{fixture['home']} vs {fixture['away']}"
    if market_assessments:
        for m in market_assessments:
            rec = stake_recs.get(m.outcome)
            stake_str = f"{rec.capped_stake_pct:.1%}" if rec and rec.tier not in ("LEAN", "NO BET") else "0% (NO BET)"
            risk_str = "Standard" if rec and rec.tier not in ("LEAN", "NO BET") else "N/A"
            lines.append(
                f"| {match_str} | {m.outcome} | {m.decimal_odds:.2f} | {m.implied_prob:.1%} | "
                f"{m.fair_prob:.1%} | {m.edge:+.1%} | {m.ev_per_unit:+.3f} | {stake_str} | "
                f"{confidence} | {risk_str} | {data_grade} |"
            )
    else:
        lines.append(f"| {match_str} | — | — | — | — | — | — | NO BET | {confidence} | — | {data_grade} |")

    return "\n".join(lines)

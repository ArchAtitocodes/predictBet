"""
Opponent comparison engine (mandatory, per spec — no team is ever
analyzed in isolation). Builds a row-by-row home-vs-away comparison and
concludes each row with Edge -> Home / Away / Neutral.

Each comparison function takes the two raw values plus an optional
"higher_is_better" flag and returns a structured row. Rows where a value
is missing on either side get flagged as "not verified" rather than
silently skipped or guessed.
"""

from dataclasses import dataclass


@dataclass
class ComparisonRow:
    dimension: str
    home_value: object
    away_value: object
    edge: str          # "Home" | "Away" | "Neutral" | "Unverified"
    note: str = ""


def compare_numeric(dimension: str, home_value, away_value, higher_is_better: bool = True,
                     neutral_threshold: float = 0.0) -> ComparisonRow:
    if home_value is None or away_value is None:
        return ComparisonRow(dimension, home_value, away_value, "Unverified",
                              "One or both values not available from verified sources.")

    diff = home_value - away_value
    if not higher_is_better:
        diff = -diff

    if abs(diff) <= neutral_threshold:
        edge = "Neutral"
    elif diff > 0:
        edge = "Home"
    else:
        edge = "Away"

    return ComparisonRow(dimension, home_value, away_value, edge)


def compare_qualitative(dimension: str, home_note: str, away_note: str, edge: str) -> ComparisonRow:
    """For dimensions that aren't purely numeric (tactical matchup, motivation, etc.)."""
    if edge not in ("Home", "Away", "Neutral", "Unverified"):
        raise ValueError("edge must be Home / Away / Neutral / Unverified")
    return ComparisonRow(dimension, home_note, away_note, edge)


STANDARD_DIMENSIONS = [
    "Squad quality", "Tactical matchup", "ELO", "xG", "xGA",
    "Home/away form", "Offensive efficiency", "Defensive efficiency",
    "Goalkeeper quality", "Press resistance", "Set pieces",
    "Transition play", "Squad depth", "Rotation risk", "Rest days",
    "Travel fatigue", "Motivation",
]


def build_comparison_table(rows: list[ComparisonRow]) -> str:
    """Renders rows as a Markdown table."""
    lines = ["| Dimension | Home | Away | Edge |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.dimension} | {r.home_value} | {r.away_value} | {r.edge} |")
    return "\n".join(lines)

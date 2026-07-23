"""
Data quality grading (A-F), per the evidence hierarchy in the spec.
Grade is computed from which evidence tiers were actually verified for
this fixture — never assumed.
"""

from dataclasses import dataclass, field


@dataclass
class EvidenceChecklist:
    lineups_confirmed: bool = False
    injuries_verified: bool = False
    odds_verified: bool = False
    xg_data_available: bool = False
    team_strength_metrics_available: bool = False   # ELO / squad value / etc.
    historical_h2h_available: bool = False
    conflicting_sources: bool = False
    small_sample_size: bool = False   # e.g. new manager, early season, friendly
    notes: list = field(default_factory=list)


def grade_data_quality(checklist: EvidenceChecklist) -> str:
    """
    A — Fully verified: lineups + injuries + odds + xG + strength metrics,
        no conflicts, no small-sample flags.
    B — Minor uncertainty: Tier-1/2 mostly present, one minor gap.
    C — Missing advanced metrics: Tier-1 present (odds/injuries) but no
        xG/advanced metrics.
    D — Significant uncertainty: missing Tier-1 verification (e.g. no
        confirmed lineups/injuries) or conflicting sources.
    F — Insufficient evidence: odds not even verified, or almost nothing
        confirmed.
    """
    if not checklist.odds_verified:
        return "F"

    tier1_score = sum([
        checklist.lineups_confirmed,
        checklist.injuries_verified,
        checklist.odds_verified,
    ])

    if checklist.conflicting_sources or tier1_score <= 1:
        return "D"

    if not checklist.xg_data_available and not checklist.team_strength_metrics_available:
        return "C"

    if checklist.small_sample_size or not checklist.historical_h2h_available:
        return "B"

    if (checklist.lineups_confirmed and checklist.injuries_verified and
            checklist.odds_verified and checklist.xg_data_available and
            checklist.team_strength_metrics_available and
            not checklist.conflicting_sources and not checklist.small_sample_size):
        return "A"

    return "B"

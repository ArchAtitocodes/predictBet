import os

STAKE_CEILINGS = {
    "LOCK": 0.10,
    "STRONG": 0.05,
    "VALUE": 0.03,
    "LEAN": 0.01,
    "NO_BET": 0.0,
}

KELLY_MULTIPLIER = {
    "LOCK": 1.0,
    "STRONG": 0.5,
    "VALUE": 0.25,
    "LEAN": 0.0,
    "NO_BET": 0.0,
}

MIN_GRADE_FOR_STAKE = "C"

GRADE_ORDER = ["A", "B", "C", "D", "F"]

EFFICIENCY_THRESHOLDS = {
    "significantly_inefficient": 0.05,
    "slightly_inefficient": 0.02,
}

# Default league slug when ESPN does not return one for either team.
# Using None instead of a hardcoded league makes failures visible instead of
# silently producing wrong-data models.
DEFAULT_LEAGUE_SLUG = None

# Environment & Server Config
PORT = int(os.environ.get("PORT", 8080))
HOST = os.environ.get("HOST", "0.0.0.0")
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
BETIKA_BASE_URL = os.environ.get("BETIKA_BASE_URL", "https://api.betika.com")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 3600))


def resolve_league_slug(home_league: Optional[str], away_league: Optional[str]) -> Optional[str]:
    """Return a league slug if at least one side resolved one, otherwise None.
    Callers should treat None as a data-quality failure rather than silently
    falling back to a different league."""
    return home_league or away_league or DEFAULT_LEAGUE_SLUG


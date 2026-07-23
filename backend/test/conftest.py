"""
Shared pytest fixtures and configuration for PredictBet backend tests.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import resolve_league_slug, STAKE_CEILINGS, KELLY_MULTIPLIER  # noqa: E402


@pytest.fixture(autouse=True)
def _suppress_warnings():
    warnings.simplefilter("ignore")
    yield


@pytest.fixture
def sample_team_form_home():
    from scraper import TeamForm
    return TeamForm(
        team_name="Arsenal",
        matches_played=5,
        goals_scored=[2, 3, 1, 2, 1],
        goals_conceded=[0, 1, 0, 1, 1],
    )


@pytest.fixture
def sample_team_form_away():
    from scraper import TeamForm
    return TeamForm(
        team_name="Chelsea",
        matches_played=5,
        goals_scored=[1, 2, 0, 1, 2],
        goals_conceded=[2, 1, 1, 0, 2],
    )


@pytest.fixture
def sample_league_averages():
    return 1.45, 1.15


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def registry(tmp_path):
    from scraper import BettingSiteRegistry
    json_path = tmp_path / "sites.json"
    json_path.write_text(
        '[{"name":"Site A","url":"https://sitea.com"},{"name":"Site B","url":"https://siteb.com"}]',
        encoding="utf-8",
    )
    return BettingSiteRegistry(json_path=str(json_path))

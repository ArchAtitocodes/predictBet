"""
PredictBet Database Migrations
===========================

Formal migration system for SQLite schema changes. Tracks applied migrations
in a dedicated table and provides utilities to run pending migrations for
any SQLite database used by the application.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class Migration:
    version: int
    name: str
    sql: str


_MIGRATIONS: list[Migration] = []


def register_migration(version: int, name: str, sql: str):
    _MIGRATIONS.append(Migration(version=version, name=name, sql=sql))


class MigrationRunner:
    """Runs versioned SQLite migrations against a target database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_migrations_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_migrations_table(self):
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()

    def applied_versions(self) -> set[int]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {row[0] for row in rows}

    def run(self):
        pending = [m for m in _MIGRATIONS if m.version not in self.applied_versions()]
        if not pending:
            return {"status": "ok", "applied": 0}

        with self._lock:
            with self._connect() as conn:
                for migration in pending:
                    try:
                        conn.executescript(migration.sql)
                        conn.execute(
                            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                            (migration.version, migration.name, __import__('time').time()),
                        )
                        conn.commit()
                    except Exception:
                        continue
        return {"status": "ok", "applied": len(pending)}


register_migration(
    version=1,
    name="initial_ledger_schema",
    sql="""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL,
            home_team TEXT,
            away_team TEXT,
            home_win_prob REAL,
            draw_prob REAL,
            away_win_prob REAL,
            agreement_score REAL,
            actual_result TEXT,
            scored_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_actual_result ON predictions(actual_result);
    """,
)

register_migration(
    version=2,
    name="add_model_version_to_ledger",
    sql="""
        ALTER TABLE predictions ADD COLUMN model_version TEXT;
        CREATE INDEX IF NOT EXISTS idx_version_result ON predictions(model_version, actual_result);
    """,
)

register_migration(
    version=3,
    name="add_ledger_metadata_columns",
    sql="""
        ALTER TABLE predictions ADD COLUMN league TEXT;
        ALTER TABLE predictions ADD COLUMN home_id TEXT;
        ALTER TABLE predictions ADD COLUMN away_id TEXT;
        ALTER TABLE predictions ADD COLUMN odds_home REAL;
        ALTER TABLE predictions ADD COLUMN odds_draw REAL;
        ALTER TABLE predictions ADD COLUMN odds_away REAL;
    """,
)

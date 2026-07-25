"""
fault_tolerance.py
===================
Fault tolerance toolkit for a Streamlit betting-analysis app.

Provides:
  1. Retry w/ exponential backoff + jitter          -> retry decorator
  2. Circuit breaker (per data source)               -> CircuitBreaker
  3. TTL cache for stale-data fallback                -> TTLCache
  4. Multi-provider fallback chain (odds/data APIs)   -> DataSourceManager
  5. Data quality validation for odds records         -> validate_odds_record
  6. High-performance indexed store (hash map + tree) -> IndexedDataStore
  7. Streamlit-safe error boundaries                  -> safe_ui / st_safe_call

Design note on "fall back to scraping other sites, including google.com":
  Scraping google.com directly is against Google's Terms of Service, gets
  IP-blocked/CAPTCHA'd almost immediately, and is not a stable data source
  for anything you'd want to bet real analysis on. Instead of hardcoding
  that, this module gives you a generic, pluggable "provider" interface —
  register as many real odds APIs / scrapers as you want, in priority
  order, and the manager will retry each and fall through automatically.
  A scraper template (BeautifulSoup-based) is included at the bottom so
  you can point it at sites that actually allow scraping (check
  robots.txt / ToS first).

Dependencies: standard library only. `requests` + `beautifulsoup4` are
optional and only needed if you use the example scraper provider.
"""

from __future__ import annotations

import bisect
import functools
import logging
import random
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
)

logger = logging.getLogger("betting_app.fault_tolerance")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

T = TypeVar("T")


# ============================================================================
# 1. EXCEPTIONS
# ============================================================================

class FaultToleranceError(Exception):
    """Base class for all errors raised by this module."""


class CircuitBreakerOpenError(FaultToleranceError):
    """Raised when a call is attempted while a circuit breaker is OPEN."""


class DataValidationError(FaultToleranceError):
    """Raised when incoming data fails quality/shape checks."""


class AllProvidersFailedError(FaultToleranceError):
    """Raised when every registered data provider (and cache) failed."""


# ============================================================================
# 2. RETRY WITH EXPONENTIAL BACKOFF + JITTER
# ============================================================================

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5          # seconds
    max_delay: float = 8.0           # seconds
    backoff_factor: float = 2.0
    jitter: float = 0.25             # +/- fraction of delay
    retry_on: Tuple[type, ...] = (Exception,)


def retry(config: Optional[RetryConfig] = None, **overrides) -> Callable:
    """
    Decorator: retries a function with exponential backoff + jitter.

    Usage:
        @retry(RetryConfig(max_attempts=4, base_delay=1.0))
        def fetch_odds(...): ...

        @retry(max_attempts=5, retry_on=(requests.RequestException,))
        def fetch_odds(...): ...
    """
    cfg = config or RetryConfig(**overrides)

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            delay = cfg.base_delay
            last_exc: Optional[Exception] = None

            while attempt < cfg.max_attempts:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except cfg.retry_on as exc:
                    last_exc = exc
                    logger.warning(
                        "retry: %s attempt %d/%d failed: %s",
                        fn.__name__, attempt, cfg.max_attempts, exc,
                    )
                    if attempt >= cfg.max_attempts:
                        break
                    sleep_for = min(delay, cfg.max_delay)
                    jitter_amt = sleep_for * cfg.jitter
                    sleep_for += random.uniform(-jitter_amt, jitter_amt)
                    time.sleep(max(0.0, sleep_for))
                    delay *= cfg.backoff_factor

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


# ============================================================================
# 3. CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"             # failing fast, calls blocked
    HALF_OPEN = "half_open"   # trial call allowed to test recovery


class CircuitBreaker:
    """
    Thread-safe circuit breaker. Wrap any flaky call (API, scraper) so that
    after repeated failures it "opens" and fails fast for `reset_timeout`
    seconds instead of hammering a dead endpoint, then allows one trial
    call (HALF_OPEN) before fully closing again.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (time.time() - self._opened_at) >= self.reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info("circuit[%s]: OPEN -> HALF_OPEN (testing recovery)", self.name)

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"circuit '{self.name}' is OPEN — failing fast"
                )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"circuit '{self.name}' is HALF_OPEN — trial call in flight"
                    )
                self._half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                logger.info("circuit[%s]: recovered -> CLOSED", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._half_open_calls = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._trip()
            elif self._failure_count >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        logger.warning(
            "circuit[%s]: tripped OPEN after %d failures (cooling down %.0fs)",
            self.name, self._failure_count, self.reset_timeout,
        )


# ============================================================================
# 4. TTL CACHE (for stale-data fallback)
# ============================================================================

class TTLCache:
    """Thread-safe cache with per-key expiry. Used to serve last-known-good
    data when every live provider fails."""

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._store: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        """Fresh (non-expired) value, or None."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                return None
            return value

    def get_stale(self, key: str) -> Optional[Any]:
        """Value regardless of expiry — last resort fallback."""
        with self._lock:
            entry = self._store.get(key)
            return entry[0] if entry else None

    def has_fresh(self, key: str) -> bool:
        return self.get(key) is not None


# ============================================================================
# 5. MULTI-PROVIDER FALLBACK CHAIN
# ============================================================================

@dataclass
class DataProvider:
    """One data source (a real odds API, a scraper, a backup mirror, ...)."""
    name: str
    fetch: Callable[..., Any]                 # fetch(**query) -> raw data
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: Optional[CircuitBreaker] = None

    def __post_init__(self):
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreaker(name=self.name)


class DataSourceManager:
    """
    Orchestrates: retry -> circuit breaker -> next provider -> cache fallback.

    Register providers in priority order (e.g. primary odds API first,
    secondary API next, scraper last). On each request it tries them in
    order; the first one to succeed wins and its result is cached. If all
    providers fail, it serves the last cached value (marking it stale)
    instead of crashing the app.
    """

    def __init__(self, cache: Optional[TTLCache] = None):
        self.providers: List[DataProvider] = []
        self.cache = cache or TTLCache()

    def register(
        self,
        name: str,
        fetch: Callable[..., Any],
        retry_config: Optional[RetryConfig] = None,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> "DataSourceManager":
        provider = DataProvider(
            name=name,
            fetch=fetch,
            retry_config=retry_config or RetryConfig(),
            circuit_breaker=CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
            ),
        )
        self.providers.append(provider)
        return self

    def fetch(self, cache_key: str, cache_ttl: Optional[float] = None, **query) -> Dict[str, Any]:
        """
        Returns:
            {
                "data": <result>,
                "source": "<provider name>" | "cache",
                "stale": bool,
            }
        Raises AllProvidersFailedError only if there is also no cached
        value at all (fresh or stale) to fall back on.
        """
        errors: List[str] = []

        for provider in self.providers:
            retried_fetch = retry(provider.retry_config)(provider.fetch)
            try:
                result = provider.circuit_breaker.call(retried_fetch, **query)
            except CircuitBreakerOpenError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            else:
                self.cache.set(cache_key, result, ttl=cache_ttl)
                return {"data": result, "source": provider.name, "stale": False}

        # every live provider failed -> fall back to cache
        fresh = self.cache.get(cache_key)
        if fresh is not None:
            return {"data": fresh, "source": "cache", "stale": False}

        stale = self.cache.get_stale(cache_key)
        if stale is not None:
            logger.warning(
                "all providers failed for '%s'; serving STALE cached data. errors=%s",
                cache_key, errors,
            )
            return {"data": stale, "source": "cache", "stale": True}

        raise AllProvidersFailedError(
            f"no provider succeeded and no cache available for '{cache_key}'. "
            f"errors: {errors}"
        )


# ============================================================================
# 6. DATA QUALITY VALIDATION
# ============================================================================

@dataclass
class OddsRecord:
    event_id: str
    home_team: str
    away_team: str
    market: str            # e.g. "1X2", "over_under"
    selection: str          # e.g. "home", "over_2.5"
    odds: float
    timestamp: float
    bookmaker: str = "unknown"


def validate_odds_record(raw: Dict[str, Any]) -> OddsRecord:
    """Raises DataValidationError on any structural or range problem."""
    required = ("event_id", "home_team", "away_team", "market", "selection", "odds", "timestamp")
    missing = [f for f in required if raw.get(f) in (None, "")]
    if missing:
        raise DataValidationError(f"missing/empty fields: {missing}")

    try:
        odds = float(raw["odds"])
    except (TypeError, ValueError):
        raise DataValidationError(f"odds not numeric: {raw.get('odds')!r}")

    if not (1.0 <= odds <= 1000.0):
        raise DataValidationError(f"odds out of plausible range: {odds}")

    try:
        timestamp = float(raw["timestamp"])
    except (TypeError, ValueError):
        raise DataValidationError(f"timestamp not numeric: {raw.get('timestamp')!r}")

    if timestamp > time.time() + 3600:
        raise DataValidationError("timestamp is implausibly far in the future")

    return OddsRecord(
        event_id=str(raw["event_id"]),
        home_team=str(raw["home_team"]),
        away_team=str(raw["away_team"]),
        market=str(raw["market"]),
        selection=str(raw["selection"]),
        odds=odds,
        timestamp=timestamp,
        bookmaker=str(raw.get("bookmaker", "unknown")),
    )


def validate_batch(
    raw_records: Iterable[Dict[str, Any]],
) -> Tuple[List[OddsRecord], List[Tuple[Dict[str, Any], str]]]:
    """Splits a batch into (valid_records, [(bad_record, reason), ...])."""
    good: List[OddsRecord] = []
    bad: List[Tuple[Dict[str, Any], str]] = []
    for raw in raw_records:
        try:
            good.append(validate_odds_record(raw))
        except DataValidationError as exc:
            bad.append((raw, str(exc)))
    return good, bad


# ============================================================================
# 7. HIGH-PERFORMANCE INDEXED STORE (hash map + sorted tree-index)
# ============================================================================
#
# - Primary lookup by id:      O(1)   via a dict (hash map)
# - Range / ordered queries:   O(log n + k) via a sorted index maintained
#   with `bisect` (kept sorted like a balanced tree walk; no external deps)
# - Multiple secondary indexes can be built on any field (e.g. "odds",
#   "timestamp") so you can do things like "all bets with odds between
#   1.5 and 2.0" or "all events after time T" fast, without scanning
#   every record.

class IndexedDataStore:
    """
    High-performance store combining:
      - hash map  : id -> record                (O(1) get/update/delete)
      - sorted index per field : [(value, id), ...] kept sorted with bisect
                                  (O(log n) insert/delete, O(log n + k) range)

    Usage:
        store = IndexedDataStore(indexed_fields=["odds", "timestamp"])
        store.add("bet_1", {"odds": 1.85, "timestamp": 1720000000, ...})
        store.get("bet_1")
        store.range_query("odds", 1.5, 2.0)         # -> list of records
        store.top_n("odds", n=5, reverse=True)       # best 5 odds
    """

    def __init__(self, indexed_fields: Optional[List[str]] = None):
        self._records: Dict[str, Dict[str, Any]] = {}          # hash map
        self._indexed_fields = indexed_fields or []
        # each secondary index: field_name -> sorted list of (value, id)
        self._indexes: Dict[str, List[Tuple[Any, str]]] = {
            f: [] for f in self._indexed_fields
        }
        self._lock = threading.RLock()

    # -- core hash-map ops --------------------------------------------------

    def add(self, record_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            if record_id in self._records:
                self.delete(record_id)
            self._records[record_id] = record
            for field_name in self._indexed_fields:
                if field_name in record:
                    entry = (record[field_name], record_id)
                    bisect.insort(self._indexes[field_name], entry)

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        with self._lock:
            record = self._records.pop(record_id, None)
            if record is None:
                return False
            for field_name in self._indexed_fields:
                if field_name in record:
                    entry = (record[field_name], record_id)
                    idx_list = self._indexes[field_name]
                    pos = bisect.bisect_left(idx_list, entry)
                    if pos < len(idx_list) and idx_list[pos] == entry:
                        idx_list.pop(pos)
            return True

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._records

    # -- tree-style ordered / range queries ---------------------------------

    def range_query(
        self, field_name: str, low: Any, high: Any
    ) -> List[Dict[str, Any]]:
        """All records where low <= record[field_name] <= high, O(log n + k)."""
        with self._lock:
            idx_list = self._indexes.get(field_name)
            if idx_list is None:
                raise KeyError(f"'{field_name}' is not an indexed field")
            lo_pos = bisect.bisect_left(idx_list, (low,))
            hi_pos = bisect.bisect_right(idx_list, (high, chr(0x10FFFF)))
            return [self._records[rid] for _, rid in idx_list[lo_pos:hi_pos]]

    def top_n(
        self, field_name: str, n: int = 5, reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """Smallest (or largest, if reverse=True) n values on an indexed field."""
        with self._lock:
            idx_list = self._indexes.get(field_name)
            if idx_list is None:
                raise KeyError(f"'{field_name}' is not an indexed field")
            chosen = idx_list[-n:][::-1] if reverse else idx_list[:n]
            return [self._records[rid] for _, rid in chosen]

    def all_sorted_by(self, field_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            idx_list = self._indexes.get(field_name)
            if idx_list is None:
                raise KeyError(f"'{field_name}' is not an indexed field")
            return [self._records[rid] for _, rid in idx_list]


# ============================================================================
# 8. STREAMLIT-SAFE ERROR BOUNDARIES
# ============================================================================
# These work even if streamlit isn't installed (falls back to logging),
# so the module is testable outside a Streamlit runtime.

try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _ui_warn(msg: str) -> None:
    if _HAS_STREAMLIT:
        st.warning(msg)
    else:
        logger.warning(msg)


def _ui_error(msg: str) -> None:
    if _HAS_STREAMLIT:
        st.error(msg)
    else:
        logger.error(msg)


@contextmanager
def safe_ui(section_name: str, fallback_message: Optional[str] = None):
    """
    Context manager: wraps a chunk of Streamlit rendering code so that if
    it throws, the rest of the page still renders instead of a full crash.

        with safe_ui("Odds table"):
            render_odds_table(df)
    """
    try:
        yield
    except Exception as exc:
        logger.exception("safe_ui: '%s' failed", section_name)
        _ui_error(
            fallback_message
            or f"⚠️ '{section_name}' couldn't be displayed right now ({exc.__class__.__name__}). "
               f"The rest of the page is unaffected."
        )


def st_safe_call(
    fallback_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """Decorator version of safe_ui for whole render functions."""

    def decorator(fn: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Optional[T]:
            with safe_ui(fn.__name__, fallback_message):
                return fn(*args, **kwargs)
            return None

        return wrapper

    return decorator


def render_data_freshness_badge(result: Dict[str, Any]) -> None:
    """Call after DataSourceManager.fetch(...) to show the user where the
    data came from and whether it's stale, instead of silently swapping data."""
    source = result.get("source")
    stale = result.get("stale")
    if not _HAS_STREAMLIT:
        logger.info("data source: %s (stale=%s)", source, stale)
        return
    if stale:
        st.warning(f"⚠️ Showing cached data (source unavailable): last known from '{source}'.")
    elif source == "cache":
        st.caption("📦 Served from cache.")
    else:
        st.caption(f"✅ Live data from {source}.")


# ============================================================================
# 9. EXAMPLE USAGE (safe to delete — shows how the pieces fit together)
# ============================================================================

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # --- 9a. Define a couple of providers (replace with real odds APIs) ---
    import urllib.request
    import json as _json

    def fetch_primary_api(date: str) -> Any:
        # e.g. a real odds API such as api-football.com, the-odds-api.com, etc.
        # Placeholder that simulates an unreliable endpoint.
        if random.random() < 0.5:
            raise ConnectionError("primary API timed out")
        return [{"event_id": "e1", "home_team": "A", "away_team": "B",
                  "market": "1X2", "selection": "home", "odds": 1.9,
                  "timestamp": time.time(), "bookmaker": "PrimaryAPI"}]

    def fetch_backup_api(date: str) -> Any:
        return [{"event_id": "e1", "home_team": "A", "away_team": "B",
                  "market": "1X2", "selection": "home", "odds": 1.95,
                  "timestamp": time.time(), "bookmaker": "BackupAPI"}]

    def fetch_via_scraper_template(date: str) -> Any:
        """
        Generic scraper fallback. Swap the URL for a source that actually
        permits scraping (check robots.txt / ToS). Do NOT point this at
        google.com search results pages — Google blocks/HTML-changes on
        automated scraping almost immediately and it violates their ToS,
        so it isn't a usable "API" in practice.
        """
        raise NotImplementedError("plug in a real, scraping-permitted source here")

    manager = DataSourceManager(cache=TTLCache(default_ttl=120))
    manager.register("primary_odds_api", fetch_primary_api,
                      retry_config=RetryConfig(max_attempts=3, base_delay=0.2))
    manager.register("backup_odds_api", fetch_backup_api,
                      retry_config=RetryConfig(max_attempts=2, base_delay=0.2))
    # manager.register("scraper_fallback", fetch_via_scraper_template)

    result = manager.fetch(cache_key="odds:2026-07-24", date="2026-07-24")
    print("fetch result:", result["source"], "stale=", result["stale"])

    # --- 9b. Validate what came back ------------------------------------
    good, bad = validate_batch(result["data"])
    print(f"valid records: {len(good)}, invalid: {len(bad)}")

    # --- 9c. Load into the indexed store ---------------------------------
    store = IndexedDataStore(indexed_fields=["odds", "timestamp"])
    for i, rec in enumerate(good):
        store.add(rec.event_id + f"_{i}", rec.__dict__)

    print("best odds (top 3):", [r["odds"] for r in store.top_n("odds", 3, reverse=True)])
    print("odds between 1.5-2.0:", [r["odds"] for r in store.range_query("odds", 1.5, 2.0)])

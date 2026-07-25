"""
scraper_engine.py
===================
A clean, generic multi-site scraping engine that plugs directly into
fault_tolerance.DataSourceManager as a fallback DataProvider.

It replaces the placeholder `fetch_via_scraper_template()` stub from
fault_tolerance.py with a real, configurable implementation.

Design goals:
  - Config-driven: add a new source by writing a SiteConfig, not new code.
  - Respectful: checks robots.txt before hitting a page, rate-limits per
    domain, uses a real User-Agent, and times out instead of hanging.
  - Composable: each site scraper is just a callable, so it drops straight
    into fault_tolerance.DataSourceManager.register(...) and inherits
    retry + circuit-breaker + cache-fallback behavior for free.
  - Fail loud: a page redesign that breaks selectors should raise
    DataValidationError immediately, not silently return rows full of None.

What this deliberately does NOT do:
  - It will not scrape google.com (or any search engine results page).
    That's against Google's ToS, gets blocked/CAPTCHA'd almost instantly,
    and search-result HTML is not a stable data contract — it will break
    your pipeline randomly. Point SiteConfig at actual football data
    sites/APIs instead.
  - It does not bypass robots.txt, paywalls, auth, rate limits, or
    anti-bot protections. If a site disallows scraping, this engine will
    refuse to fetch it (see RobotsChecker below) rather than work around it.

Dependencies: requests, beautifulsoup4, lxml
  (pip install requests beautifulsoup4 lxml)
  If lxml isn't available, the engine falls back to Python's built-in
  html.parser automatically (slower, but zero extra dependency).
"""

from __future__ import annotations

import logging
import re
import string
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from fault_tolerance import (
    DataSourceManager,
    RetryConfig,
    DataValidationError,
)

logger = logging.getLogger("betting_app.scraper_engine")

DEFAULT_USER_AGENT = (
    "BettingAnalysisBot/1.0 (+https://example.com/bot-info; "
    "respects robots.txt; contact: you@example.com)"
)

# Pick the best available HTML parser once, at import time, instead of
# hard-failing deep inside a request if lxml isn't installed.
try:
    import lxml  # noqa: F401
    _HTML_PARSER = "lxml"
except ImportError:
    logger.warning("lxml not installed — falling back to html.parser (slower). "
                    "Run `pip install lxml` for better performance/robustness.")
    _HTML_PARSER = "html.parser"

# If more than this fraction of rows come back with every field None,
# treat it as a structural change (redesigned page) rather than valid
# "no data today" output.
DEFAULT_MAX_EMPTY_ROW_FRACTION = 0.5


# ============================================================================
# ROBOTS.TXT + RATE LIMITING
# ============================================================================

class RobotsChecker:
    """Caches and consults robots.txt per domain so the engine never
    fetches a path a site has disallowed for bots.

    Cache entries expire after `ttl_seconds` so a robots.txt change on a
    long-running process eventually gets picked up instead of being
    trusted forever from process start.
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, ttl_seconds: float = 3600.0):
        self.user_agent = user_agent
        self.ttl_seconds = ttl_seconds
        self._parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._fetched_at: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _get_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        domain = urlparse(url).netloc
        with self._lock:
            stale = (
                domain not in self._parsers
                or (time.time() - self._fetched_at.get(domain, 0.0)) > self.ttl_seconds
            )
            if stale:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(urljoin(f"https://{domain}", "/robots.txt"))
                try:
                    rp.read()
                except Exception as exc:
                    logger.warning("robots.txt unreachable for %s (%s) — treating as disallow-all", domain, exc)
                    rp.disallow_all = True  # fail closed, not open
                self._parsers[domain] = rp
                self._fetched_at[domain] = time.time()
            return self._parsers[domain]

    def allowed(self, url: str) -> bool:
        try:
            return self._get_parser(url).can_fetch(self.user_agent, url)
        except Exception:
            return False


class RateLimiter:
    """Simple per-domain minimum-interval rate limiter (thread-safe)."""

    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval
        self._last_call: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        with self._lock:
            last = self._last_call.get(domain, 0.0)
            now = time.time()
            elapsed = now - last
            sleep_for = self.min_interval - elapsed
            self._last_call[domain] = max(now, last) + max(0.0, sleep_for)
        if sleep_for > 0:
            time.sleep(sleep_for)


# ============================================================================
# SITE CONFIG — add a new scrapeable source by declaring one of these
# ============================================================================

@dataclass
class SiteConfig:
    name: str
    search_url_template: str        # e.g. "https://example.com/fixtures?date={date}"
    row_selector: str                # CSS selector for each record's container
    field_selectors: Dict[str, str]  # field_name -> CSS selector, relative to row
    field_attr: Dict[str, str] = field(default_factory=dict)  # field_name -> attr (default: text)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    rate_limit_seconds: float = 2.0
    max_empty_row_fraction: float = DEFAULT_MAX_EMPTY_ROW_FRACTION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SiteConfig.name must be non-empty")
        if not self.field_selectors:
            raise ValueError(f"{self.name}: field_selectors must declare at least one field")
        try:
            parsed = urlparse(self.search_url_template.split("{", 1)[0])
        except Exception as exc:
            raise ValueError(f"{self.name}: search_url_template is not a usable URL: {exc}")
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"{self.name}: search_url_template must be an absolute http(s) URL")
        # sanity-check the template's placeholders are well-formed, so a
        # typo like "{date" fails at config time, not at fetch time.
        try:
            self.template_fields = {
                fname for _, fname, _, _ in string.Formatter().parse(self.search_url_template)
                if fname
            }
        except ValueError as exc:
            raise ValueError(f"{self.name}: malformed search_url_template: {exc}")


# ============================================================================
# SINGLE-SITE SCRAPER
# ============================================================================

class SiteScraper:
    """Fetches + parses one configured site. Callable so it can be
    registered directly as a fault_tolerance DataProvider.

    Note: if you share one `requests.Session` across many SiteScrapers
    (as ScraperEngine does) and call them concurrently from multiple
    threads, be aware requests.Session is not guaranteed thread-safe for
    all connection-pool edge cases. It's fine for typical light
    concurrency, but for heavy parallel scraping give each scraper (or
    each thread) its own Session.
    """

    def __init__(
        self,
        config: SiteConfig,
        session: Optional[requests.Session] = None,
        robots: Optional[RobotsChecker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.config = config
        self.session = session or requests.Session()
        self.robots = robots or RobotsChecker(user_agent)
        self.rate_limiter = rate_limiter or RateLimiter(config.rate_limit_seconds)
        self.user_agent = user_agent

    def _build_url(self, query: Dict[str, Any]) -> str:
        missing = self.config.template_fields - query.keys()
        if missing:
            raise ValueError(
                f"{self.config.name}: missing required query param(s) {sorted(missing)} "
                f"for template {self.config.search_url_template!r}"
            )
        # URL-encode every value so spaces/&/# etc. in a query param can't
        # break the URL structure or smuggle in extra query params.
        safe_query = {k: quote(str(v), safe="") for k, v in query.items()}
        return self.config.search_url_template.format(**safe_query)

    def __call__(self, **query) -> List[Dict[str, Any]]:
        """query must supply whatever placeholders search_url_template needs,
        e.g. date='2026-07-24' for a template containing '{date}'."""
        url = self._build_url(query)

        if not self.robots.allowed(url):
            raise PermissionError(
                f"{self.config.name}: robots.txt disallows fetching {url} — skipping"
            )

        self.rate_limiter.wait(url)

        headers = {"User-Agent": self.user_agent, **self.config.headers}
        resp = self.session.get(url, headers=headers, timeout=self.config.timeout)
        resp.raise_for_status()

        records = self._parse(resp.text)
        logger.info("%s: parsed %d record(s) from %s", self.config.name, len(records), url)
        return records

    def _parse(self, html: str) -> List[Dict[str, Any]]:
        soup = BeautifulSoup(html, _HTML_PARSER)
        rows = soup.select(self.config.row_selector)
        if not rows:
            raise DataValidationError(
                f"{self.config.name}: 0 rows matched selector "
                f"'{self.config.row_selector}' — page structure may have changed"
            )

        records: List[Dict[str, Any]] = []
        empty_rows = 0
        for row in rows:
            record: Dict[str, Any] = {"source_site": self.config.name}
            for field_name, selector in self.config.field_selectors.items():
                el = row.select_one(selector)
                if el is None:
                    record[field_name] = None
                    continue
                attr = self.config.field_attr.get(field_name)
                record[field_name] = el.get(attr) if attr else el.get_text(strip=True)
            if all(record.get(f) is None for f in self.config.field_selectors):
                empty_rows += 1
            records.append(record)

        empty_fraction = empty_rows / len(records)
        if empty_fraction > self.config.max_empty_row_fraction:
            raise DataValidationError(
                f"{self.config.name}: {empty_rows}/{len(records)} rows had every field "
                f"selector miss ({empty_fraction:.0%} > {self.config.max_empty_row_fraction:.0%} "
                f"threshold) — page structure has likely changed, refusing to return bad data"
            )
        return records


# ============================================================================
# MULTI-SITE ENGINE — wires several SiteScrapers into one fallback chain
# ============================================================================

class ScraperEngine:
    """
    Convenience wrapper: builds a DataSourceManager pre-loaded with one
    provider per configured site, in priority order, sharing a single
    robots-checker/rate-limiter/session across all of them.

    Usage:
        engine = ScraperEngine([site_a_config, site_b_config])
        manager = engine.as_data_source_manager()
        result = manager.fetch(cache_key="odds:2026-07-24", date="2026-07-24")

    Use as a context manager (or call close()) to release the shared
    requests.Session's connections when you're done:

        with ScraperEngine([site_a_config]) as engine:
            manager = engine.as_data_source_manager()
            ...
    """

    def __init__(
        self,
        sites: List[SiteConfig],
        user_agent: str = DEFAULT_USER_AGENT,
        max_attempts_per_site: int = 2,
        robots_ttl_seconds: float = 3600.0,
    ):
        if not sites:
            raise ValueError("ScraperEngine requires at least one SiteConfig")
        names = [s.name for s in sites]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate SiteConfig name(s): {sorted(dupes)}")

        self.robots = RobotsChecker(user_agent, ttl_seconds=robots_ttl_seconds)
        self.session = requests.Session()
        self.scrapers: List[SiteScraper] = [
            SiteScraper(
                cfg,
                session=self.session,
                robots=self.robots,
                rate_limiter=RateLimiter(cfg.rate_limit_seconds),
                user_agent=user_agent,
            )
            for cfg in sites
        ]
        self.max_attempts_per_site = max_attempts_per_site

    def as_data_source_manager(self, manager: Optional[DataSourceManager] = None) -> DataSourceManager:
        mgr = manager or DataSourceManager()
        for scraper in self.scrapers:
            mgr.register(
                name=f"scrape:{scraper.config.name}",
                fetch=scraper,
                retry_config=RetryConfig(
                    max_attempts=self.max_attempts_per_site,
                    base_delay=1.0,
                    # don't retry on robots.txt disallow / bad selectors —
                    # retrying won't fix a policy block or a page redesign
                    retry_on=(requests.RequestException,),
                ),
            )
        return mgr

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "ScraperEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ============================================================================
# EXAMPLE (replace selectors/URLs with a real, scraping all football source)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example_config = SiteConfig(
        name="example_fixtures_site",
        search_url_template="https://example.com/fixtures/{date}",
        row_selector="table.fixtures tr",
        field_selectors={
            "home_team": "td.home",
            "away_team": "td.away",
            "odds": "td.odds-home",
        },
        rate_limit_seconds=3.0,
    )

    with ScraperEngine([example_config]) as engine:
        manager = engine.as_data_source_manager()
        try:
            result = manager.fetch(cache_key="fixtures:2026-07-24", date="2026-07-24")
            print(result)
        except Exception as exc:
            # Expected here since example.com isn't a real fixtures source —
            # this just demonstrates the call shape and error path.
            print("expected failure against placeholder URL:", exc)
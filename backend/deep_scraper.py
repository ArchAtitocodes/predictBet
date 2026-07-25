"""
scraper_engine.py
===================
A clean, generic, MULTI-BACKEND scraping engine that plugs directly into
fault_tolerance.DataSourceManager as a fallback DataProvider.

It replaces the placeholder `fetch_via_scraper_template()` stub from
fault_tolerance.py with a real, configurable implementation.

Supported fetch backends (pick per-source via SiteConfig.fetch_mode):
  - "requests"    plain sync HTTP, the default, zero extra setup.
  - "httpx"       sync OR async HTTP (optional HTTP/2). Async mode enables
                  real concurrency via ScraperEngine.fetch_all_async().
  - "playwright"  headless-browser rendering for JS-heavy pages.
  - "selenium"    alternative headless-browser rendering.
  - "feed"        RSS/Atom feeds via feedparser (news, fixture feeds, etc).
Plus a separate, non-HTML backend:
  - UnderstatConfig / UnderstatScraper  wraps the `understat` library for
    football analytics (xG, xGA, shot maps, league/team/player stats).

Design goals (unchanged from the original, just applied more broadly now):
  - Config-driven: add a new source by writing a SiteConfig/UnderstatConfig,
    not new code.
  - Respectful: checks robots.txt before hitting a page, rate-limits per
    domain, uses a real User-Agent, and times out instead of hanging —
    for every backend, not just plain requests.
  - Composable: each scraper is just a callable, so it drops straight into
    fault_tolerance.DataSourceManager.register(...) and inherits retry +
    circuit-breaker + cache-fallback behavior for free.
  - Fail loud: a page redesign that breaks selectors, or an unavailable
    backend library, raises immediately — not silently returns rows full
    of None or fails mysteriously three layers down.

What this deliberately does NOT do:
  - It will not scrape google.com (or any search engine results page).
    That's against Google's ToS, gets blocked/CAPTCHA'd almost instantly,
    and search-result HTML is not a stable data contract — it will break
    your pipeline randomly. Point SiteConfig at actual football data
    sites/APIs instead.
  - It does not bypass robots.txt, paywalls, auth, rate limits, or
    anti-bot protections — for ANY backend, including the headless
    browsers. Playwright/Selenium are here to render JavaScript on pages
    that already allow bots, not to defeat CAPTCHAs or fingerprinting
    defenses. If a site disallows scraping, this engine refuses to fetch
    it (see RobotsChecker below) regardless of which backend you pick.

Dependencies (all optional except requests/beautifulsoup4/lxml, which are
the default path — everything else is imported lazily and only required
if you actually configure a source that uses it):
  pip install requests httpx beautifulsoup4 feedparser lxml
  pip install playwright && playwright install     # for fetch_mode="playwright"
  pip install selenium                             # for fetch_mode="selenium"
  pip install understat aiohttp                    # for UnderstatScraper
"""

from __future__ import annotations

import asyncio
import logging
import string
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
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

# ---------------------------------------------------------------------------
# Optional backends — imported lazily/guarded so the module still loads
# (and requests-only sources still work) even if these aren't installed.
# ---------------------------------------------------------------------------
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from playwright.sync_api import sync_playwright, Error as _PWError, TimeoutError as _PWTimeoutError
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import WebDriverException as _SeleniumError
    _HAS_SELENIUM = True
except ImportError:
    _HAS_SELENIUM = False

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    import aiohttp
    from understat import Understat
    _HAS_UNDERSTAT = True
except ImportError:
    _HAS_UNDERSTAT = False

# If more than this fraction of rows come back with every field None,
# treat it as a structural change (redesigned page) rather than valid
# "no data today" output.
DEFAULT_MAX_EMPTY_ROW_FRACTION = 0.5

_VALID_FETCH_MODES = {"requests", "httpx", "playwright", "selenium", "feed"}

# Whitelist of understat.Understat methods we're willing to call. This is a
# deliberate allowlist (not "whatever string you pass") so a config file
# can't be used to invoke arbitrary attributes on the client.
ALLOWED_UNDERSTAT_METHODS = {
    "get_league_players",
    "get_league_teams",
    "get_league_table",
    "get_league_results",
    "get_league_fixtures",
    "get_team_stats",
    "get_team_results",
    "get_team_fixtures",
    "get_player_stats",
    "get_player_shots",
    "get_player_matches",
    "get_match_shots",
    "get_match_players",
    "get_match_info",
}


# ============================================================================
# ROBOTS.TXT + RATE LIMITING
# ============================================================================

class RobotsChecker:
    """Caches and consults robots.txt per domain so the engine never
    fetches a path a site has disallowed for bots — regardless of which
    backend (requests/httpx/playwright/selenium/feed) ends up doing the
    actual fetch.

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

    async def allowed_async(self, url: str) -> bool:
        """Same check, off the event loop thread (robots.txt fetch/parse is
        blocking I/O — this keeps an async fetch_all_async() run from
        stalling on it)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.allowed, url)


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


class AsyncRateLimiter:
    """Async equivalent of RateLimiter, for the httpx-async concurrent path.
    One asyncio.Lock guards the whole table; per-domain waits still happen
    independently since we compute-then-release before sleeping."""

    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval
        self._last_call: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        async with self._lock:
            last = self._last_call.get(domain, 0.0)
            now = time.time()
            sleep_for = self.min_interval - (now - last)
            self._last_call[domain] = max(now, last) + max(0.0, sleep_for)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


# ============================================================================
# SITE CONFIG — add a new scrapeable HTML/feed source by declaring one of these
# ============================================================================

@dataclass
class SiteConfig:
    name: str
    search_url_template: str        # e.g. "https://example.com/fixtures?date={date}"
    row_selector: str                # CSS selector for each record's container
    field_selectors: Dict[str, str]  # field_name -> CSS selector (or feed attr name), relative to row
    field_attr: Dict[str, str] = field(default_factory=dict)  # field_name -> HTML attr (default: text)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    rate_limit_seconds: float = 2.0
    max_empty_row_fraction: float = DEFAULT_MAX_EMPTY_ROW_FRACTION

    # --- backend selection ---
    fetch_mode: str = "requests"          # "requests" | "httpx" | "playwright" | "selenium" | "feed"
    use_http2: bool = False               # httpx only; falls back to HTTP/1.1 with a warning if `h2` isn't installed
    browser: str = "chromium"             # playwright: chromium|firefox|webkit ; selenium: chrome|firefox
    headless: bool = True                 # playwright/selenium
    wait_selector: Optional[str] = None   # playwright/selenium: CSS selector to wait for before reading the DOM
    wait_timeout: float = 15.0            # playwright/selenium: seconds to wait for wait_selector

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SiteConfig.name must be non-empty")
        if not self.field_selectors:
            raise ValueError(f"{self.name}: field_selectors must declare at least one field")
        if self.fetch_mode not in _VALID_FETCH_MODES:
            raise ValueError(
                f"{self.name}: fetch_mode must be one of {sorted(_VALID_FETCH_MODES)}, "
                f"got {self.fetch_mode!r}"
            )
        if self.fetch_mode == "httpx" and not _HAS_HTTPX:
            raise ImportError(f"{self.name}: fetch_mode='httpx' requires `pip install httpx`")
        if self.fetch_mode == "playwright" and not _HAS_PLAYWRIGHT:
            raise ImportError(
                f"{self.name}: fetch_mode='playwright' requires "
                f"`pip install playwright` and `playwright install`"
            )
        if self.fetch_mode == "selenium" and not _HAS_SELENIUM:
            raise ImportError(f"{self.name}: fetch_mode='selenium' requires `pip install selenium`")
        if self.fetch_mode == "feed" and not _HAS_FEEDPARSER:
            raise ImportError(f"{self.name}: fetch_mode='feed' requires `pip install feedparser`")
        if self.fetch_mode != "feed" and not self.row_selector:
            raise ValueError(f"{self.name}: row_selector is required for fetch_mode={self.fetch_mode!r}")

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


@dataclass
class UnderstatConfig:
    """Config for a football-analytics source served via the `understat`
    library (xG/xGA, shot maps, league/team/player season stats)."""
    name: str
    endpoint: str  # must be one of ALLOWED_UNDERSTAT_METHODS
    rate_limit_seconds: float = 2.0
    timeout: float = 15.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UnderstatConfig.name must be non-empty")
        if not _HAS_UNDERSTAT:
            raise ImportError(
                f"{self.name}: UnderstatConfig requires `pip install understat aiohttp`"
            )
        if self.endpoint not in ALLOWED_UNDERSTAT_METHODS:
            raise ValueError(
                f"{self.name}: endpoint {self.endpoint!r} is not in the allowed set "
                f"{sorted(ALLOWED_UNDERSTAT_METHODS)}"
            )


# ============================================================================
# SINGLE-SITE SCRAPER (requests / httpx / playwright / selenium / feed)
# ============================================================================

class SiteScraper:
    """Fetches + parses one configured HTML or feed site. Callable so it can
    be registered directly as a fault_tolerance DataProvider. Which backend
    actually does the fetching is chosen entirely by `config.fetch_mode`;
    everything downstream (robots check, rate limit, empty-row validation)
    behaves identically no matter which one runs.

    Note: if you share one `requests.Session` across many SiteScrapers
    (as ScraperEngine does) and call them concurrently from multiple
    threads, be aware requests.Session is not guaranteed thread-safe for
    all connection-pool edge cases. It's fine for typical light
    concurrency, but for heavy parallel scraping give each scraper (or
    each thread) its own Session. The same caveat applies to a shared
    httpx.Client.

    Playwright/Selenium note: for simplicity and correctness this launches
    a fresh browser per call and tears it down afterward. That's the right
    default for occasional JS-rendered fetches; if you're scraping a
    playwright/selenium source at high frequency, keep a persistent
    browser instance yourself and swap it in via a thin subclass — the
    per-call launch cost otherwise dominates.
    """

    def __init__(
        self,
        config: SiteConfig,
        session: Optional[requests.Session] = None,
        httpx_client: Optional["httpx.Client"] = None,
        robots: Optional[RobotsChecker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        async_rate_limiter: Optional[AsyncRateLimiter] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.config = config
        self.session = session or requests.Session()
        self.robots = robots or RobotsChecker(user_agent)
        self.rate_limiter = rate_limiter or RateLimiter(config.rate_limit_seconds)
        self.async_rate_limiter = async_rate_limiter or AsyncRateLimiter(config.rate_limit_seconds)
        self.user_agent = user_agent

        self._httpx_client = httpx_client
        if config.fetch_mode == "httpx" and self._httpx_client is None:
            self._httpx_client = self._make_httpx_client()

    def _make_httpx_client(self) -> "httpx.Client":
        try:
            return httpx.Client(http2=self.config.use_http2)
        except ImportError:
            # `h2` package missing for HTTP/2 — degrade gracefully rather than blow up.
            if self.config.use_http2:
                logger.warning(
                    "%s: HTTP/2 requested but `h2` isn't installed (`pip install h2`); "
                    "falling back to HTTP/1.1", self.config.name
                )
            return httpx.Client(http2=False)

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

    # -- sync entrypoint (used by ScraperEngine / DataSourceManager) --------

    def __call__(self, **query) -> List[Dict[str, Any]]:
        """query must supply whatever placeholders search_url_template needs,
        e.g. date='2026-07-24' for a template containing '{date}'."""
        url = self._build_url(query)

        if not self.robots.allowed(url):
            raise PermissionError(
                f"{self.config.name}: robots.txt disallows fetching {url} — skipping"
            )

        self.rate_limiter.wait(url)

        dispatch: Dict[str, Callable[[str], str]] = {
            "requests": self._fetch_requests_html,
            "httpx": self._fetch_httpx_html,
            "playwright": self._fetch_playwright_html,
            "selenium": self._fetch_selenium_html,
        }

        if self.config.fetch_mode == "feed":
            records = self._fetch_and_parse_feed(url)
        else:
            html = dispatch[self.config.fetch_mode](url)
            records = self._parse_html(html)

        logger.info("%s: parsed %d record(s) from %s via %s",
                    self.config.name, len(records), url, self.config.fetch_mode)
        return records

    # -- async entrypoint (httpx-mode only; used by fetch_all_async) --------

    async def afetch(self, **query) -> List[Dict[str, Any]]:
        if self.config.fetch_mode != "httpx":
            raise ValueError(
                f"{self.config.name}: afetch() only supports fetch_mode='httpx' "
                f"(got {self.config.fetch_mode!r}) — sync backends don't have an async path here"
            )
        url = self._build_url(query)
        if not await self.robots.allowed_async(url):
            raise PermissionError(f"{self.config.name}: robots.txt disallows fetching {url} — skipping")
        await self.async_rate_limiter.wait(url)

        headers = {"User-Agent": self.user_agent, **self.config.headers}
        async with httpx.AsyncClient(http2=self.config.use_http2) as client:
            resp = await client.get(url, headers=headers, timeout=self.config.timeout)
            resp.raise_for_status()
            html = resp.text
        records = self._parse_html(html)
        logger.info("%s: parsed %d record(s) from %s via async httpx",
                    self.config.name, len(records), url)
        return records

    # -- per-backend fetchers: each returns raw HTML text -------------------

    def _fetch_requests_html(self, url: str) -> str:
        headers = {"User-Agent": self.user_agent, **self.config.headers}
        resp = self.session.get(url, headers=headers, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.text

    def _fetch_httpx_html(self, url: str) -> str:
        headers = {"User-Agent": self.user_agent, **self.config.headers}
        resp = self._httpx_client.get(url, headers=headers, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.text

    def _fetch_playwright_html(self, url: str) -> str:
        cfg = self.config
        try:
            with sync_playwright() as p:
                browser_type = getattr(p, cfg.browser)
                browser = browser_type.launch(headless=cfg.headless)
                try:
                    page = browser.new_page(user_agent=self.user_agent)
                    page.goto(url, timeout=cfg.timeout * 1000)
                    if cfg.wait_selector:
                        page.wait_for_selector(cfg.wait_selector, timeout=cfg.wait_timeout * 1000)
                    return page.content()
                finally:
                    browser.close()
        except (_PWError, _PWTimeoutError) as exc:
            raise DataValidationError(f"{cfg.name}: playwright render failed for {url}: {exc}") from exc

    def _fetch_selenium_html(self, url: str) -> str:
        cfg = self.config
        options_cls = {
            "chrome": webdriver.ChromeOptions,
            "firefox": webdriver.FirefoxOptions,
        }.get(cfg.browser, webdriver.ChromeOptions)
        driver_cls = {
            "chrome": webdriver.Chrome,
            "firefox": webdriver.Firefox,
        }.get(cfg.browser, webdriver.Chrome)

        options = options_cls()
        if cfg.headless:
            options.add_argument("--headless=new")
        try:
            options.add_argument(f"user-agent={self.user_agent}")
        except Exception:
            pass  # Firefox options handle UA differently; not fatal.

        driver = driver_cls(options=options)  # selenium 4.15+ auto-resolves the driver binary
        try:
            driver.set_page_load_timeout(cfg.timeout)
            driver.get(url)
            if cfg.wait_selector:
                WebDriverWait(driver, cfg.wait_timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, cfg.wait_selector))
                )
            return driver.page_source
        except _SeleniumError as exc:
            raise DataValidationError(f"{cfg.name}: selenium render failed for {url}: {exc}") from exc
        finally:
            driver.quit()

    # -- parsing --------------------------------------------------------------

    def _parse_html(self, html: str) -> List[Dict[str, Any]]:
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

        self._check_empty_fraction(records, empty_rows)
        return records

    def _fetch_and_parse_feed(self, url: str) -> List[Dict[str, Any]]:
        """feed mode: field_selectors maps field_name -> feedparser entry
        attribute (e.g. {"headline": "title", "url": "link", "published": "published"})
        instead of a CSS selector, since there's no DOM to query."""
        headers = {"User-Agent": self.user_agent, **self.config.headers}
        resp = self.session.get(url, headers=headers, timeout=self.config.timeout)
        resp.raise_for_status()

        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            raise DataValidationError(
                f"{self.config.name}: feed at {url} failed to parse "
                f"({getattr(parsed, 'bozo_exception', 'unknown error')})"
            )
        if not parsed.entries:
            raise DataValidationError(f"{self.config.name}: feed at {url} returned 0 entries")

        records: List[Dict[str, Any]] = []
        empty_rows = 0
        for entry in parsed.entries:
            record: Dict[str, Any] = {"source_site": self.config.name}
            for field_name, attr_path in self.config.field_selectors.items():
                record[field_name] = entry.get(attr_path)
            if all(record.get(f) is None for f in self.config.field_selectors):
                empty_rows += 1
            records.append(record)

        self._check_empty_fraction(records, empty_rows)
        return records

    def _check_empty_fraction(self, records: List[Dict[str, Any]], empty_rows: int) -> None:
        empty_fraction = empty_rows / len(records)
        if empty_fraction > self.config.max_empty_row_fraction:
            raise DataValidationError(
                f"{self.config.name}: {empty_rows}/{len(records)} rows had every field "
                f"selector miss ({empty_fraction:.0%} > {self.config.max_empty_row_fraction:.0%} "
                f"threshold) — source structure has likely changed, refusing to return bad data"
            )

    def close(self) -> None:
        if self._httpx_client is not None:
            self._httpx_client.close()


# ============================================================================
# UNDERSTAT SCRAPER — football analytics (xG/xGA, shots, season stats)
# ============================================================================

class UnderstatScraper:
    """Wraps the `understat` library (async, aiohttp-based) to fetch
    football analytics from understat.com behind the same robots.txt +
    rate-limiting discipline as everything else in this module.

    Only methods in ALLOWED_UNDERSTAT_METHODS can be invoked — the
    endpoint is validated once at UnderstatConfig construction time, so
    there's no path from config data to arbitrary attribute access.

    Note on robots.txt here: understat's public data endpoints aren't
    plain HTML pages, so we check permission against the site root as a
    good-faith gate rather than against the exact internal URL the
    library hits. If understat.com's robots.txt disallows the root for
    bots, we refuse to proceed at all.
    """

    BASE_URL = "https://understat.com"

    def __init__(
        self,
        config: UnderstatConfig,
        robots: Optional[RobotsChecker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.config = config
        self.robots = robots or RobotsChecker(user_agent)
        self.rate_limiter = rate_limiter or RateLimiter(config.rate_limit_seconds)

    async def _afetch(self, **query) -> Any:
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            method = getattr(understat, self.config.endpoint)
            return await asyncio.wait_for(method(**query), timeout=self.config.timeout)

    def __call__(self, **query) -> Any:
        root = f"{self.BASE_URL}/"
        if not self.robots.allowed(root):
            raise PermissionError(f"{self.config.name}: robots.txt disallows {root} — skipping understat fetch")

        self.rate_limiter.wait(self.BASE_URL)

        result = asyncio.run(self._afetch(**query))
        if not result:
            raise DataValidationError(
                f"{self.config.name}: understat.{self.config.endpoint}({query}) returned no data — "
                f"check the season/league/id args, or understat may have changed its response shape"
            )
        logger.info("%s: fetched understat.%s(%s)", self.config.name, self.config.endpoint, query)
        return result


# ============================================================================
# MULTI-SOURCE ENGINE — wires HTML/feed sites + understat sources into one
# fallback chain
# ============================================================================

def _retry_exceptions_for(fetch_mode: str) -> Tuple[type, ...]:
    """Which exception types are worth retrying for a given backend.
    Policy/validation failures (robots disallow, bad selectors) are
    deliberately excluded — retrying won't fix those."""
    excs: List[type] = [requests.RequestException]
    if fetch_mode == "httpx" and _HAS_HTTPX:
        excs.append(httpx.HTTPError)
    if fetch_mode == "playwright" and _HAS_PLAYWRIGHT:
        excs.extend([_PWError, _PWTimeoutError])
    if fetch_mode == "selenium" and _HAS_SELENIUM:
        excs.append(_SeleniumError)
    return tuple(excs)


class ScraperEngine:
    """
    Convenience wrapper: builds a DataSourceManager pre-loaded with one
    provider per configured HTML/feed site (any mix of requests / httpx /
    playwright / selenium / feed) plus one provider per understat source,
    in priority order, sharing robots-checker/rate-limiter/session state
    across all of them.

    Usage:
        engine = ScraperEngine(
            sites=[site_a_config, feed_config],
            understat_sources=[understat_config],
        )
        manager = engine.as_data_source_manager()
        result = manager.fetch(cache_key="odds:2026-07-24", date="2026-07-24")

    Concurrent async fetch across all httpx-mode sites:
        results = asyncio.run(engine.fetch_all_async({
            "site_a": {"date": "2026-07-24"},
            "site_b": {"date": "2026-07-24"},
        }))

    Use as a context manager (or call close()) to release the shared
    requests.Session/httpx.Client connections when you're done:

        with ScraperEngine([site_a_config]) as engine:
            manager = engine.as_data_source_manager()
            ...
    """

    def __init__(
        self,
        sites: Optional[List[SiteConfig]] = None,
        understat_sources: Optional[List[UnderstatConfig]] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        max_attempts_per_site: int = 2,
        robots_ttl_seconds: float = 3600.0,
    ):
        sites = sites or []
        understat_sources = understat_sources or []
        if not sites and not understat_sources:
            raise ValueError("ScraperEngine requires at least one SiteConfig or UnderstatConfig")

        names = [s.name for s in sites] + [u.name for u in understat_sources]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate source name(s) across sites/understat_sources: {sorted(dupes)}")

        self.robots = RobotsChecker(user_agent, ttl_seconds=robots_ttl_seconds)
        self.session = requests.Session()

        self.scrapers: List[SiteScraper] = [
            SiteScraper(
                cfg,
                session=self.session,
                robots=self.robots,
                rate_limiter=RateLimiter(cfg.rate_limit_seconds),
                async_rate_limiter=AsyncRateLimiter(cfg.rate_limit_seconds),
                user_agent=user_agent,
            )
            for cfg in sites
        ]
        self.understat_scrapers: List[UnderstatScraper] = [
            UnderstatScraper(
                cfg,
                robots=self.robots,
                rate_limiter=RateLimiter(cfg.rate_limit_seconds),
                user_agent=user_agent,
            )
            for cfg in understat_sources
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
                    retry_on=_retry_exceptions_for(scraper.config.fetch_mode),
                ),
            )
        for u_scraper in self.understat_scrapers:
            mgr.register(
                name=f"understat:{u_scraper.config.name}",
                fetch=u_scraper,
                retry_config=RetryConfig(
                    max_attempts=self.max_attempts_per_site,
                    base_delay=1.0,
                    # understat's own HTTP layer raises aiohttp errors; keep this
                    # generic instead of importing aiohttp's whole exception tree.
                    retry_on=(Exception,) if _HAS_UNDERSTAT else (),
                ),
            )
        return mgr

    async def fetch_all_async(self, queries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Concurrently fetch multiple httpx-mode sites at once. `queries`
        maps site name -> kwargs for that site's URL template. Sites not in
        httpx mode, or not present in `queries`, are skipped. Returns a dict
        of name -> records (or the Exception raised, so one bad source
        doesn't sink the whole batch)."""
        httpx_scrapers = {s.config.name: s for s in self.scrapers if s.config.fetch_mode == "httpx"}
        targets = {name: kwargs for name, kwargs in queries.items() if name in httpx_scrapers}
        if not targets:
            logger.warning("fetch_all_async: none of %s match a configured httpx-mode site", list(queries))
            return {}

        async def _run(name: str, kwargs: Dict[str, Any]):
            try:
                return name, await httpx_scrapers[name].afetch(**kwargs)
            except Exception as exc:  # noqa: BLE001 — deliberately captured per-source
                logger.warning("fetch_all_async: %s failed: %s", name, exc)
                return name, exc

        results = await asyncio.gather(*(_run(name, kwargs) for name, kwargs in targets.items()))
        return dict(results)

    def close(self) -> None:
        self.session.close()
        for scraper in self.scrapers:
            scraper.close()

    def __enter__(self) -> "ScraperEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ============================================================================
# EXAMPLES (replace selectors/URLs with real sources before using for real)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1) Plain requests-backed fixtures page (unchanged from before).
    fixtures_site = SiteConfig(
        name="example_fixtures_site",
        search_url_template="https://example.com/fixtures/{date}",
        row_selector="table.fixtures tr",
        field_selectors={"home_team": "td.home", "away_team": "td.away", "odds": "td.odds-home"},
        rate_limit_seconds=3.0,
        fetch_mode="requests",
    )

    # 2) RSS/Atom news feed via feedparser — no row_selector, field_selectors
    #    map to feedparser entry attributes instead of CSS.
    news_feed = SiteConfig(
        name="example_football_news_feed",
        search_url_template="https://example.com/football/rss",
        row_selector="",
        field_selectors={"headline": "title", "url": "link", "published": "published"},
        fetch_mode="feed",
        rate_limit_seconds=5.0,
    )

    # 3) JS-rendered odds board via Playwright, waiting for the odds table
    #    to actually paint before reading the DOM.
    js_odds_site = SiteConfig(
        name="example_js_odds_board",
        search_url_template="https://example.com/live-odds/{date}",
        row_selector="div.odds-row",
        field_selectors={"match": "span.match-name", "odds": "span.odds-value"},
        fetch_mode="playwright",
        wait_selector="div.odds-row",
        wait_timeout=10.0,
        rate_limit_seconds=5.0,
    )

    # 4) Understat: league-wide xG stats for a season.
    xg_source = UnderstatConfig(
        name="epl_xg_players",
        endpoint="get_league_players",
        rate_limit_seconds=5.0,
    )

    with ScraperEngine(
        sites=[fixtures_site, news_feed, js_odds_site],
        understat_sources=[xg_source],
    ) as engine:
        manager = engine.as_data_source_manager()

        try:
            result = manager.fetch(cache_key="fixtures:2026-07-24", date="2026-07-24")
            print(result)
        except Exception as exc:
            # Expected here since example.com isn't a real fixtures source —
            # this just demonstrates the call shape and error path.
            print("expected failure against placeholder URL:", exc)

        try:
            xg = manager.fetch(cache_key="epl_xg:2025-2026", league="epl", season="2025")
            print(xg)
        except Exception as exc:
            print("understat call shape demo (needs real args to succeed):", exc)

        # Concurrent async fetch demo, across whichever configured sites use fetch_mode="httpx".
        # (None of the three sites above are httpx-mode, so this returns {} here —
        #  it's here to show the call shape for when you add one.)
        async_results = asyncio.run(engine.fetch_all_async({"example_fixtures_site": {"date": "2026-07-24"}}))
        print("async batch results:", async_results)
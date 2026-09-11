"""
app/web_search.py
─────────────────────────────────────────────────────────────────────────────
Web search with SerpApi (Google) as the PRIMARY provider and DuckDuckGo as
the FALLBACK provider.

Public interface — unchanged from the previous implementation:

    search_web(query: str, num_results: int = 5) -> list[dict]

Each result dict:
    {"title": str, "url": str, "snippet": str}

Provider priority
─────────────────
    1. SerpApi Google Search  — requires SERPAPI_API_KEY in environment
    2. DuckDuckGo HTML scrape — no key required, always available as fallback

Fallback is triggered when SerpApi:
    • API key is missing or empty
    • request raises any exception (network error, timeout, auth failure)
    • returns an API-level error field
    • returns zero organic results after parsing

No other part of the project needs to change.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
# Override with SERPAPI_TIMEOUT_SECONDS in .env if your network is slow.
_SERPAPI_TIMEOUT    = (5, int(os.getenv("SERPAPI_TIMEOUT_SECONDS", "25")))
_DUCKDUCKGO_TIMEOUT = (5, 12)

# ── SerpApi endpoint ──────────────────────────────────────────────────────────
_SERPAPI_URL = "https://serpapi.com/search"


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web and return up to *num_results* results.

    Tries SerpApi Google Search first.  Falls back to DuckDuckGo automatically
    on any failure or when SerpApi returns no usable results.

    Parameters
    ----------
    query       : search query (1–2000 characters)
    num_results : how many results to return (1–10)

    Returns
    -------
    list of {"title": str, "url": str, "snippet": str}
    Empty list only when BOTH providers return nothing.

    Raises
    ------
    ValueError  — bad arguments (caller's fault, not retried)
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("Invalid search query")
    if not isinstance(num_results, int) or not 1 <= num_results <= 10:
        raise ValueError("num_results must be an integer between 1 and 10")

    query = query.strip()

    # Guard: if the query looks like an article title (very long, contains em-dash
    # or specific formatting) it was probably constructed from a search result.
    # Truncate to 120 chars to avoid sending useless long strings to the APIs.
    if len(query) > 120:
        # Cut at last word boundary before 120 chars
        query = query[:120].rsplit(" ", 1)[0].rstrip(".,;:-–—")

    # ── 1. Try SerpApi ────────────────────────────────────────────────────────
    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not serpapi_key:
        logger.warning(
            "SERPAPI_API_KEY not set — skipping SerpApi, using DuckDuckGo"
        )
    else:
        results = _serpapi_search(query, num_results, serpapi_key)
        if results is not None:          # None means "failed/no results"
            return results
        logger.warning("SerpApi failed or returned no results — falling back to DuckDuckGo")

    # ── 2. Fallback: DuckDuckGo ───────────────────────────────────────────────
    logger.info("Using DuckDuckGo as search provider")
    return _duckduckgo_search(query, num_results)


# ═════════════════════════════════════════════════════════════════════════════
# SerpApi provider
# ═════════════════════════════════════════════════════════════════════════════

def _serpapi_search(
    query: str,
    num_results: int,
    api_key: str,
) -> list[dict] | None:
    """
    Call the SerpApi Google Search endpoint.

    Returns a list of result dicts on success, or None if the call
    should be considered failed/empty (triggering DuckDuckGo fallback).
    Never raises — all exceptions are caught and logged.
    """
    logger.info("Using SerpApi Google Search")
    try:
        params = {
            "engine":  "google",
            "q":       query,
            "api_key": api_key,
            "num":     num_results,
            "hl":      "en",
            "gl":      "us",
        }
        response = requests.get(
            _SERPAPI_URL,
            params=params,
            timeout=_SERPAPI_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        # SerpApi signals errors via a top-level "error" key
        if "error" in data:
            logger.warning("SerpApi API error: %s", data["error"])
            return None

        organic = data.get("organic_results") or []
        if not organic:
            logger.info("SerpApi returned no organic results")
            return None

        results: list[dict] = []
        for item in organic:
            title   = str(item.get("title")   or "").strip()
            url     = str(item.get("link")    or "").strip()
            snippet = str(item.get("snippet") or "").strip()

            # Skip items without a usable URL
            if not url or not url.startswith(("http://", "https://")):
                continue

            results.append({
                "title":   title[:300],
                "url":     url,
                "snippet": snippet[:1200],
            })
            if len(results) == num_results:
                break

        if not results:
            logger.info("SerpApi organic results had no usable URLs")
            return None

        logger.info(
            "SerpApi returned %d result(s) for query: %.80r",
            len(results), query,
        )
        return results

    except requests.Timeout:
        logger.warning("SerpApi request timed out")
        return None
    except requests.HTTPError as exc:
        logger.warning("SerpApi HTTP error: %s", exc)
        return None
    except requests.RequestException as exc:
        logger.warning("SerpApi request failed: %s", type(exc).__name__)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("SerpApi unexpected error: %s", type(exc).__name__)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# DuckDuckGo provider  (unchanged from original implementation)
# ═════════════════════════════════════════════════════════════════════════════

def _source_url(value: str) -> str:
    """Decode a DDG redirect URL to the real destination URL."""
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        value = parse_qs(parsed.query).get("uddg", [value])[0]
    parsed = urlparse(value)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
    ):
        return ""
    return value


class _DDGResultsParser(HTMLParser):
    """Minimal HTML parser for the DuckDuckGo HTML endpoint."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({
                "title":   "",
                "url":     _source_url(attrs_dict.get("href", "")),
                "snippet": "",
            })
            self._field = "title"
        elif tag == "a" and "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field and self.results:
            self.results[-1][self._field] += data


def _duckduckgo_search(query: str, num_results: int) -> list[dict]:
    """
    Scrape DuckDuckGo HTML results.

    Returns an empty list on failure — never raises so the agent
    can handle "no results" gracefully.
    """
    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=_DUCKDUCKGO_TIMEOUT,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("DuckDuckGo request failed: %s", type(exc).__name__)
        return []

    parser = _DDGResultsParser()
    try:
        parser.feed(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo HTML parse error: %s", type(exc).__name__)
        return []

    results: list[dict] = []
    seen: set[str] = set()
    for row in parser.results:
        url = row["url"]
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({
            "title":   " ".join(row["title"].split())[:300],
            "url":     url,
            "snippet": " ".join(row["snippet"].split())[:1200],
        })
        if len(results) == num_results:
            break

    if results:
        logger.info(
            "DuckDuckGo returned %d result(s) for query: %.80r",
            len(results), query,
        )
    else:
        logger.warning("DuckDuckGo returned no results for query: %.80r", query)

    return results

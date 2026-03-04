"""LangChain agent tools — Wikiloc trail fetching and SerpAPI web search.

These are the two ``@tool``-decorated entry points exposed directly to the
LLM agent. All other agent-chain functions live in focused modules:

- :mod:`geocoding` — centroid computation and reverse geocoding
- :mod:`database` — municipality code lookup and mountain zone containment
- :mod:`aemet` — AEMET municipal and mountain weather fetchers
- :mod:`weather_alerts` — CAP alert fetching and polygon containment
- :mod:`weather_mountains` — mountain forecast scraping
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from langchain.tools import tool
from playwright.sync_api import sync_playwright

load_dotenv()

_WIKILOC_JS = """
() => {
    if (!window.trailMap) return null;
    var collection = {'type':'FeatureCollection','features':[]};
    trailMap.eachLayer(function (layer) {
        if (typeof(layer.toGeoJSON) === 'function')
            collection.features.push(layer.toGeoJSON());
    });
    return collection;
}
"""


@tool('Trail downloader', description='Based on the URL parameter it fetches it with playwright in headless mode and \
using a js extraction code it dumps the geojson to a file. Use it when the input is a summit from somewhere in text.')
def get_wikiloc_trail(url: str) -> dict | None:
    """Extract trail GeoJSON from a Wikiloc web page via headless Chromium.

    Args:
        url: Full Wikiloc trail page URL.

    Returns:
        GeoJSON FeatureCollection dict, or *None* if ``window.trailMap`` is
        not available on the page.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("window.trailMap !== undefined", timeout=15000)
        geojson = page.evaluate(_WIKILOC_JS)
        browser.close()

    return geojson


@tool('Web search with SerpAPI', description='Uses SerpAPI to perform web search on `site:wikiloc.com` to get specific\
                                             trail url based on the input')
def search_wikiloc_trail(
    where: str,
    site: str = "wikiloc.com",
    engine: str = "google",
    serpapi_key: str = os.getenv("SERPAPI_API_KEY"),
) -> str:
    """Search Wikiloc for a trail matching the given location description.

    Args:
        where: Free-text location or trail description to search for.
        site: Domain to restrict the search to. Defaults to ``"wikiloc.com"``.
        engine: Search engine identifier for SerpAPI. Defaults to ``"google"``.
        serpapi_key: SerpAPI key. Reads from ``SERPAPI_API_KEY`` env var by
            default.

    Returns:
        URL of the first organic result, or ``"No trails found"`` on failure.
    """
    url = f"https://serpapi.com/search.json?engine={engine}&api_key={serpapi_key}&q=site:{site}+{where}"
    response = httpx.get(url).json()
    if response["search_metadata"]["status"] == "Success":
        organic_result = response["organic_results"][0]
        return organic_result["link"]
    return "No trails found"

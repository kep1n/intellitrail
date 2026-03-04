"""Wikiloc scraper — Playwright-based metadata and GeoJSON extraction.

Standalone module. No LangGraph imports.

Entry points:
    is_wikiloc_route_url(url: str) -> bool
    scrape_metadata(url: str, timeout_ms: int = 15000) -> dict
    scrape_geojson(url: str, timeout_ms: int = 25000) -> dict

All scraping operations use a per-call Playwright context (sync API).
Browser lifecycle is scoped to each function call via context manager.
"""

import logging
import re
from playwright.sync_api import sync_playwright
from langsmith import traceable

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIKILOC_ROUTE_PATTERN = re.compile(
    r"wikiloc\.com/[\w-]+/[\w-]+-\d+$"
)

_NON_ROUTE_SEGMENTS = ("/user/", "/discover/", "/search", "/wikiloc/")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ScraperError(Exception):
    """Raised when Wikiloc scraping fails (blocked, unexpected structure, timeout)."""

    def __init__(self, message: str, url: str = ""):
        super().__init__(message)
        self.url = url


# ---------------------------------------------------------------------------
# URL filter
# ---------------------------------------------------------------------------

def is_wikiloc_route_url(url: str) -> bool:
    """Return True if *url* points to a Wikiloc route page.

    Accepts: https://www.wikiloc.com/{activity}-trails/{slug}-{numeric_id}
    Rejects: profile pages, search pages, discover pages, non-Wikiloc domains.

    Query strings and fragments are stripped before matching.
    """
    # Strip query string and fragment
    clean = url.split("?")[0].split("#")[0]

    # Reject non-route path segments
    for seg in _NON_ROUTE_SEGMENTS:
        if seg in clean:
            return False

    return bool(_WIKILOC_ROUTE_PATTERN.search(clean))


# ---------------------------------------------------------------------------
# Metadata scraper
# ---------------------------------------------------------------------------

def scrape_metadata(url: str, timeout_ms: int = 15000) -> dict:
    """Scrape metadata from a Wikiloc route page.

    Returns:
        {"name": str, "distance_km": float, "elevation_gain_m": float, "difficulty": str}

    Raises:
        ScraperError: If name or distance cannot be extracted, or on any
                      browser / navigation error.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=_USER_AGENT)
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            try:
                page.goto(url, wait_until="domcontentloaded")
            except Exception as exc:
                raise ScraperError(
                    f"Navigation failed: {exc}", url=url
                ) from exc

            # --- Strategy 1: window.__INITIAL_STATE__ ---
            trail = None
            try:
                trail = page.evaluate("() => window.__INITIAL_STATE__?.trail")
            except Exception:
                pass

            metadata = {}

            if trail:
                metadata = _extract_metadata_from_trail_obj(trail)

            # --- Strategy 2: CSS selector fallback ---
            if not _metadata_complete(metadata):
                metadata = _extract_metadata_from_dom(page, metadata)

            # Validate required fields
            if not metadata.get("name") or metadata.get("distance_km") is None:
                raise ScraperError(
                    "Could not extract required metadata fields (name, distance_km) "
                    "from Wikiloc page — page structure may have changed.",
                    url=url,
                )

            return metadata
        finally:
            browser.close()


def _extract_metadata_from_trail_obj(trail: dict) -> dict:
    """Extract metadata fields from window.__INITIAL_STATE__.trail object."""
    metadata: dict = {}

    # Name
    name = trail.get("name") or trail.get("title")
    if name:
        metadata["name"] = str(name).strip()

    # Distance
    stats = trail.get("stats") or {}
    total_distance = stats.get("totalDistance")
    if total_distance is not None:
        # Wikiloc stores distances in metres internally
        val = float(total_distance)
        metadata["distance_km"] = val / 1000.0 if val > 1000 else val

    # Elevation gain
    elev_gain = stats.get("elevationGain") or stats.get("positiveElevation")
    if elev_gain is not None:
        metadata["elevation_gain_m"] = float(elev_gain)

    # Difficulty
    difficulty = trail.get("difficulty") or trail.get("difficultyLevel")
    if difficulty is not None:
        metadata["difficulty"] = str(difficulty)

    return metadata


def _metadata_complete(metadata: dict) -> bool:
    """Return True if all four required metadata fields are present."""
    return bool(
        metadata.get("name")
        and metadata.get("distance_km") is not None
        and metadata.get("elevation_gain_m") is not None
        and metadata.get("difficulty") is not None
    )


def _extract_metadata_from_dom(page, existing: dict) -> dict:
    """Fallback: extract metadata from DOM elements using CSS selectors."""
    metadata = dict(existing)

    # Name fallback
    if not metadata.get("name"):
        for selector in ["h1", '[data-testid*="title"]', ".trail-name"]:
            try:
                el = page.query_selector(selector)
                if el:
                    text = el.text_content().strip()
                    if text:
                        metadata["name"] = text
                        break
            except Exception:
                pass

    # Stats fallback
    stat_selectors = page.query_selector_all('[data-testid], [class*="stat"]')
    for el in stat_selectors:
        try:
            text = el.text_content().strip()
            test_id = el.get_attribute("data-testid") or ""

            if "distance" in test_id.lower() and "distance_km" not in metadata:
                val = _parse_float(text)
                if val is not None:
                    metadata["distance_km"] = val

            elif "elevation" in test_id.lower() and "elevation_gain_m" not in metadata:
                val = _parse_float(text)
                if val is not None:
                    metadata["elevation_gain_m"] = val

        except Exception:
            pass

    # Difficulty fallback
    if "difficulty" not in metadata:
        for selector in ['[data-testid*="difficulty"]', '[class*="difficulty"]']:
            try:
                el = page.query_selector(selector)
                if el:
                    text = el.text_content().strip()
                    if text:
                        metadata["difficulty"] = text
                        break
            except Exception:
                pass

    # Provide defaults for optional fields so the dict is always consistent
    metadata.setdefault("elevation_gain_m", 0.0)
    metadata.setdefault("difficulty", "unknown")

    return metadata


def _parse_float(text: str) -> float | None:
    """Parse a float from a string that may contain units like 'km', 'm', ','."""
    import re as _re
    cleaned = _re.sub(r"[^\d.,]", "", text).replace(",", ".")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_distance_km(value: str) -> float | None:
    """Parse a distance string and return kilometres, converting mi → km if needed."""
    num = _parse_trail_number(value)
    if num is None:
        return None
    t = value.lower().replace("\xa0", " ")
    if " mi" in t or t.rstrip().endswith("mi"):
        return round(num * 1.60934, 3)
    return num  # assume km


def _parse_elevation_m(value: str) -> float | None:
    """Parse an elevation string and return metres, converting ft → m if needed."""
    num = _parse_trail_number(value)
    if num is None:
        return None
    t = value.lower().replace("\xa0", " ")
    if " ft" in t or t.rstrip().endswith("ft"):
        return round(num * 0.3048, 1)
    return num  # assume metres


def _parse_trail_number(text: str) -> float | None:
    """Parse a Wikiloc numeric value in either EN or ES locale.

    English: '17.59 km'  '1,481 m'   (period=decimal, comma=thousands)
    Spanish: '17,59 km'  '1.481 m'   (comma=decimal, period=thousands)

    Rule: a separator followed by exactly 3 digits is a thousands separator;
          any other digit count → decimal separator.
    """
    import re as _re
    cleaned = _re.sub(r"[^\d.,]", "", text.replace("\xa0", ""))
    if not cleaned:
        return None
    has_comma = "," in cleaned
    has_period = "." in cleaned
    if has_comma and has_period:
        # Both present: the last one is the decimal separator
        if cleaned.rfind(",") > cleaned.rfind("."):  # e.g. "1.234,56"
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:                                          # e.g. "1,234.56"
            cleaned = cleaned.replace(",", "")
    elif has_comma:
        after = cleaned.split(",", 1)[-1]
        if len(after) == 3:   # "1,481" → thousands separator
            cleaned = cleaned.replace(",", "")
        else:                  # "17,59" → decimal separator
            cleaned = cleaned.replace(",", ".")
    elif has_period:
        after = cleaned.split(".", 1)[-1]
        if len(after) == 3:   # "1.481" → thousands separator
            cleaned = cleaned.replace(".", "")
        # else: "17.59" → already valid float
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_trail_data_section(page) -> dict:
    """Extract stats from <section id='trail-data'> dl.data-items dt/dd pairs.

    Handles both ES and EN page locales. Skips dl.more-data (upload dates, etc.)
    and the TrailRank item (complex nested HTML).

    Returns dict with optional keys: distance_km, elevation_gain_m, elevation_loss_m,
    max_elevation_m, min_elevation_m, difficulty, trail_type.
    """
    try:
        # Wait for section to be in DOM (it's static HTML, should be fast)
        page.wait_for_selector('#trail-data', timeout=5000)
        raw = page.evaluate("""
() => {
    const section = document.querySelector('#trail-data');
    if (!section) return [];
    const result = [];

    // Primary stats from dl.data-items
    const dl = section.querySelector('dl.data-items');
    if (dl) {
        dl.querySelectorAll('.d-item').forEach(item => {
            const dt = item.querySelector('dt');
            const dd = item.querySelector('dd');
            if (dt && dd)
                result.push([dt.textContent.trim(), dd.textContent.trim()]);
        });
    }

    // All pairs from dl.more-data — label filtering done in Python
    const moreDl = section.querySelector('dl.more-data') || document.querySelector('dl.more-data');
    if (moreDl) {
        moreDl.querySelectorAll('.d-item').forEach(item => {
            const dt = item.querySelector('dt');
            const dd = item.querySelector('dd');
            if (dt && dd)
                result.push([dt.textContent.trim(), dd.textContent.trim()]);
        });
    }

    return result;
}
""")
    except Exception as exc:
        logger.warning("_extract_trail_data_section failed: %s", exc)
        return {}

    if not raw:
        logger.debug("_extract_trail_data_section: JS returned empty list")
        return {}

    logger.debug("_extract_trail_data_section raw pairs: %s", raw)

    stats: dict = {}
    for label, value in raw:
        label_l = label.lower().replace("\u00a0", " ")
        # Max/min must be checked before the generic 'altitud' substring
        if "altitud máxima" in label_l or "altitud maxima" in label_l or "max elevation" in label_l:
            v = _parse_elevation_m(value)
            if v is not None:
                stats["max_elevation_m"] = v
        elif "altitud mínima" in label_l or "altitud minima" in label_l or "min elevation" in label_l:
            v = _parse_elevation_m(value)
            if v is not None:
                stats["min_elevation_m"] = v
        elif "distancia" in label_l or label_l.startswith("distance"):
            v = _parse_distance_km(value)
            if v is not None:
                stats["distance_km"] = v
        elif "desnivel positivo" in label_l or "elevation gain" in label_l:
            v = _parse_elevation_m(value)
            if v is not None:
                stats["elevation_gain_m"] = v
        elif "desnivel negativo" in label_l or "elevation loss" in label_l:
            v = _parse_elevation_m(value)
            if v is not None:
                stats["elevation_loss_m"] = v
        elif "dificultad" in label_l or "difficulty" in label_l:
            stats["difficulty"] = value
        elif "tipo de ruta" in label_l or "trail type" in label_l:
            stats["trail_type"] = value
        elif "moving time" in label_l or "tiempo en movimiento" in label_l:
            stats["moving_time"] = value.strip()
    return stats


def _extract_moving_time(page) -> str | None:
    """Scan every dt element on the page for a Moving time label and return its dd text.

    Independent of DOM structure — works regardless of which section contains the element.
    Normalises non-breaking spaces before matching.
    """
    try:
        result = page.evaluate("""
() => {
    const targets = ['moving time', 'tiempo en movimiento'];
    for (const dt of document.querySelectorAll('dt')) {
        const label = dt.textContent.replace(/\u00a0/g, ' ').trim().toLowerCase();
        if (targets.includes(label)) {
            const dd = dt.closest('.d-item')?.querySelector('dd')
                     || dt.nextElementSibling;
            if (dd) return dd.textContent.replace(/\u00a0/g, ' ').trim();
        }
    }
    return null;
}
""")
        if result:
            logger.debug("_extract_moving_time found: %s", result)
        else:
            logger.debug("_extract_moving_time: label not found on page")
        return result or None
    except Exception as exc:
        logger.warning("_extract_moving_time failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GeoJSON scraper
# ---------------------------------------------------------------------------

_ANTI_BOT_INIT_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.chrome = { runtime: {} };
"""

# window.trailMap is a Leaflet map instance — iterate layers to collect GeoJSON.
# Only collect LineString / MultiLineString features (the actual track).
# Skip FeatureCollection layers (LayerGroups) because their children are
# iterated separately by eachLayer, which would otherwise produce duplicates.
# Deduplicate by coordinate count to guard against redundant track layers.
_TRAIL_MAP_JS = """
() => {
    if (!window.trailMap) return null;
    var collection = {'type':'FeatureCollection','features':[]};
    var seenSizes = new Set();
    trailMap.eachLayer(function (layer) {
        if (typeof(layer.toGeoJSON) !== 'function') return;
        var g = layer.toGeoJSON();
        // Skip LayerGroups — their children are visited individually by eachLayer
        if (g.type === 'FeatureCollection') return;
        // Only keep line geometries (the actual trail track)
        var geomType = g.type === 'Feature' ? (g.geometry && g.geometry.type) : g.type;
        if (geomType !== 'LineString' && geomType !== 'MultiLineString') return;
        // Deduplicate: skip if a line with the same number of coords was already added
        var coords = g.type === 'Feature' ? g.geometry.coordinates : g.coordinates;
        var n = coords ? coords.length : 0;
        if (n > 10 && seenSizes.has(n)) return;
        seenSizes.add(n);
        collection.features.push(g);
    });
    return collection.features.length ? collection : null;
}
"""


@traceable
def scrape_geojson(url: str, timeout_ms: int = 25000) -> tuple[dict, dict]:
    """Scrape GeoJSON track data and trail stats from a Wikiloc route page.

    Uses three strategies in priority order for the GeoJSON:
        1. window.trailMap (requires anti-bot init script + Chrome channel)
        2. window.__INITIAL_STATE__.trail.geojson
        3. Leaflet layer extraction via JavaScript

    Also extracts <section id="trail-data"> stats in the same browser session.

    Returns:
        (geojson, trail_stats) where:
        - geojson: GeoJSON dict (FeatureCollection, Feature, or bare geometry)
        - trail_stats: dict with optional keys distance_km, elevation_gain_m,
          elevation_loss_m, max_elevation_m, min_elevation_m, difficulty, trail_type

    Raises:
        ScraperError: If no GeoJSON strategy succeeds.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        try:
            context = browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            # Spoof headless detection so Wikiloc delivers full JS bundle
            page.add_init_script(_ANTI_BOT_INIT_SCRIPT)

            try:
                page.goto(url, wait_until="networkidle")
            except Exception as exc:
                raise ScraperError(
                    f"Navigation failed: {exc}", url=url
                ) from exc

            # --- Strategy 1: window.trailMap (Leaflet map instance) ---
            try:
                page.wait_for_function(
                    "window.trailMap !== undefined", timeout=timeout_ms
                )
                geojson = page.evaluate(_TRAIL_MAP_JS)
                if geojson and _is_valid_geojson(geojson):
                    trail_stats = _extract_trail_data_section(page)
                    mt = _extract_moving_time(page)
                    if mt:
                        trail_stats["moving_time"] = mt
                    return geojson, trail_stats
            except Exception:
                pass

            # --- Strategy 2: window.__INITIAL_STATE__ ---
            try:
                geojson = page.evaluate(
                    "() => window.__INITIAL_STATE__?.trail?.geojson"
                )
                if geojson and _is_valid_geojson(geojson):
                    trail_stats = _extract_trail_data_section(page)
                    mt = _extract_moving_time(page)
                    if mt:
                        trail_stats["moving_time"] = mt
                    return geojson, trail_stats
            except Exception:
                pass

            # --- Strategy 3: Leaflet layer extraction ---
            try:
                geojson = page.evaluate("""
() => {
    try {
        const layers = Object.values(window.L?._layers || {});
        for (const layer of layers) {
            if (layer?.feature?.geometry?.coordinates) return layer.feature;
            if (layer?.getLatLngs) {
                const latlngs = layer.getLatLngs().flat(Infinity);
                if (latlngs.length > 0) {
                    return {
                        type: 'Feature',
                        geometry: {
                            type: 'LineString',
                            coordinates: latlngs.map(ll => [ll.lng, ll.lat])
                        },
                        properties: {}
                    };
                }
            }
        }
    } catch(e) {}
    return null;
}
""")
                if geojson and _is_valid_geojson(geojson):
                    trail_stats = _extract_trail_data_section(page)
                    mt = _extract_moving_time(page)
                    if mt:
                        trail_stats["moving_time"] = mt
                    return geojson, trail_stats
            except Exception:
                pass

            raise ScraperError(
                "Could not extract GeoJSON from Wikiloc page — "
                "structure may have changed",
                url=url,
            )
        finally:
            browser.close()


def _is_valid_geojson(data: dict) -> bool:
    """Return True if *data* has the minimum required GeoJSON structure."""
    if not isinstance(data, dict):
        return False
    if "type" not in data:
        return False
    # Accept: FeatureCollection, Feature, bare geometry
    if "features" in data:
        return True
    if "geometry" in data:
        return True
    if "coordinates" in data:
        return True
    return False

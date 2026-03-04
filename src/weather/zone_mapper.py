"""Geographic zone mapper for AEMET weather integration.

Maps (lat, lon) coordinates to:
  - Nearest Spanish municipality INE code (via Nominatim reverse geocoding + SQLite INE table)
  - AEMET mountain area code (via bounding box or polygon containment)
  - Nivologica (avalanche bulletin) zone code (via NIVOLOGICA_ZONE_MAP)

CRITICAL FINDING (verified 2026-02-21):
  /api/maestro/municipios returns HTTP 404 — this endpoint does NOT exist in the current
  AEMET OpenData API. municipality_zone() now uses Nominatim reverse geocoding instead,
  with name lookup against the SQLite projectdb.db (mun_codes table, 8132 entries).

  mountain_zone_url() uses exact polygon containment from the urls_alerts SQLite table.
  mountain_zone() uses bounding boxes as a fallback (backward-compatible).

  Nominatim ToS: 1 req/s rate limit, User-Agent required, no bulk use.
  Acceptable for single-track safety queries.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from src.weather.client import AEMETClient

# SQLite database with full INE municipality table and mountain zone polygons.
# mun_codes: CPRO, CMUN, NOMBRE — 8132 Spanish municipalities
# urls_alerts: region_montana, url, region_en, coords (lon,lat WKT polygons)
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "projectdb.db"


# ---------------------------------------------------------------------------
# Mountain zone bounding boxes
# ---------------------------------------------------------------------------
# Each entry: {code, name, min_lat, max_lat, min_lon, max_lon}
# Approximate bounding boxes for Spain's nine AEMET mountain zone codes.
# Source: geographic knowledge of zone names from AEMET.es mountain zone pages.
# NOTE: These boxes are approximate starting points — validate with known test
#       coordinates per zone before production use (RESEARCH.md open question #2).
#
# Coordinate system: WGS84 decimal degrees (lat positive = N, lon negative = W)

MOUNTAIN_ZONES: list[dict] = [
    # Pirineo Aragonés — central Pyrenees (Aragón side)
    {"code": "arn1", "name": "Pirineo Aragonés", "min_lat": 42.3, "max_lat": 42.9, "min_lon": -1.0, "max_lon": 0.6},
    # Pirineo Catalán — eastern Pyrenees (Catalonia side)
    {"code": "cat1", "name": "Pirineo Catalán", "min_lat": 42.3, "max_lat": 42.9, "min_lon": 0.6, "max_lon": 3.3},
    # Pirineo Navarro — western Pyrenees (Navarre)
    {"code": "nav1", "name": "Pirineo Navarro", "min_lat": 42.6, "max_lat": 43.1, "min_lon": -2.0, "max_lon": -1.0},
    # Pirineu (Lleida) — southern Pyrenean slopes, Lleida province
    {"code": "peu1", "name": "Pirineu (Lleida)", "min_lat": 42.0, "max_lat": 42.8, "min_lon": 0.0, "max_lon": 1.5},
    # Sierra Nevada — Andalucía, highest peak Mulhacén 3479m
    {"code": "nev1", "name": "Sierra Nevada", "min_lat": 36.9, "max_lat": 37.3, "min_lon": -3.6, "max_lon": -2.9},
    # Sistema Central (Guadarrama/Somosierra) — near Madrid
    {"code": "mad2", "name": "Sistema Central (Guadarrama)", "min_lat": 40.7, "max_lat": 41.1, "min_lon": -4.2, "max_lon": -3.5},
    # Sierra de Gredos — Sistema Central, Ávila province
    {"code": "gre1", "name": "Sierra de Gredos", "min_lat": 40.1, "max_lat": 40.5, "min_lon": -5.6, "max_lon": -4.7},
    # Sistema Ibérico Aragonés — Iberian range, Aragón south
    {"code": "arn2", "name": "Sistema Ibérico Aragonés", "min_lat": 40.8, "max_lat": 41.5, "min_lon": -2.0, "max_lon": 0.2},
    # Sistema Ibérico Riojano/Soriano — Iberian range, La Rioja/Soria
    {"code": "rio1", "name": "Sistema Ibérico (Rioja/Soria)", "min_lat": 41.7, "max_lat": 42.3, "min_lon": -3.5, "max_lon": -2.0},
]

# ---------------------------------------------------------------------------
# Nivologica zone mapping
# ---------------------------------------------------------------------------
# Maps mountain area codes to AEMET nivologica (avalanche bulletin) zone codes.
# Only Pyrenean zones are covered by the nivologica endpoint.
# All other mountain zones → None (zone_not_covered).
#
# Source: AEMET OpenAPI spec + jblanco89 docs (RESEARCH.md)
# Confirmed: "0" = Pirineo Catalán, "1" = Pirineo Navarro/Aragonés

NIVOLOGICA_ZONE_MAP: dict[str, str] = {
    "cat1": "0",   # Pirineo Catalán → zone code "0"
    "peu1": "0",   # Pirineu (Lleida) → zone code "0" (Catalán bulletin)
    "arn1": "1",   # Pirineo Aragonés → zone code "1"
    "nav1": "1",   # Pirineo Navarro → zone code "1"
    # arn2 (Sistema Ibérico Aragonés) removed — not a Pyrenean snow zone.
    # All other zones (nev1, mad2, gre1, rio1) → not covered by nivologica.
    # NOTE: fetcher.py now uses direct XML URLs for cat1/arn1/nav1 instead
    # of calling AEMETClient.fetch_avalanche_bulletin() via this map.
}


# ---------------------------------------------------------------------------
# Name normalisation helper (diacritic-insensitive matching)
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase and strip diacritics for fuzzy matching.

    Example: "Güéjar Sierra" → "guejar sierra"
    """
    nfkd = unicodedata.normalize("NFKD", name.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Haversine distance (stdlib only — no external dependency)
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two WGS84 points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _parse_coord(value: object) -> float:
    """Parse an AEMET coordinate value to float.

    AEMET maestro/municipios may return coordinates as:
      - float: 42.153
      - str with period decimal: "42.153"
      - str with comma decimal (Spanish locale): "42,153"

    Confirmed field format is determined at Task 3 checkpoint and documented
    in the module-level docstring above.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Replace comma decimal separator with period
        return float(value.replace(",", "."))
    raise ValueError(f"Unexpected coordinate type: {type(value).__name__!r} value={value!r}")


# ---------------------------------------------------------------------------
# ZoneMapper
# ---------------------------------------------------------------------------

class ZoneMapper:
    """Maps geographic coordinates to AEMET forecast zones.

    Requires an AEMETClient for optional forward-compatibility, but
    municipality_zone() now uses Nominatim reverse geocoding + SQLite INE
    table instead of the broken AEMET endpoint.

    ENDPOINT STATUS (verified 2026-02-21):
      MUNICIPIOS_ENDPOINT_STATUS = "404_unavailable"
      /api/maestro/municipios returns HTTP 404. municipality_zone() uses
      Nominatim + SQLite mun_codes table (8132 entries). See module docstring.
    """

    # Status marker for /api/maestro/municipios endpoint.
    # Verified 2026-02-21: endpoint returns HTTP 404 — does not exist in current AEMET API.
    MUNICIPIOS_ENDPOINT_STATUS: str = "404_unavailable"

    # Class-level cache for the bundled INE municipios list.
    # Loaded once per process lifetime; shared across all ZoneMapper instances.
    _MUNICIPIOS_CACHE: list[dict] | None = None

    def __init__(self, client: "AEMETClient") -> None:
        self._client = client

    # -------------------------------------------------------------------------
    # Static INE municipios loader
    # -------------------------------------------------------------------------

    @classmethod
    def _load_static_municipios(cls) -> list[dict]:
        """Load full INE municipios list from SQLite. Cached after first call.

        Reads data/projectdb.db mun_codes table — 8132 Spanish municipalities
        (all official INE municipalities).

        Returns:
            List of {"id": "NNNNN", "name": "Municipality Name"} dicts.

        Raises:
            WeatherError: if the database is missing or query fails.
        """
        if cls._MUNICIPIOS_CACHE is None:
            from src.weather.client import WeatherError
            try:
                with sqlite3.connect(_DB_PATH) as conn:
                    rows = conn.execute(
                        "SELECT CPRO || CMUN, NOMBRE FROM mun_codes"
                    ).fetchall()
                cls._MUNICIPIOS_CACHE = [{"id": row[0], "name": row[1]} for row in rows]
            except Exception as exc:
                raise WeatherError(
                    f"Failed to load INE municipios from database: {exc}"
                ) from exc
        return cls._MUNICIPIOS_CACHE

    def _load_municipios(self) -> list[dict]:
        """Stub for the old broken AEMET endpoint method.

        The /api/maestro/municipios endpoint returns HTTP 404 (verified 2026-02-21).
        Use _load_static_municipios() instead.

        Raises:
            WeatherError: always — endpoint unavailable.
        """
        from src.weather.client import WeatherError
        raise WeatherError(
            "AEMET /api/maestro/municipios endpoint unavailable (HTTP 404) — "
            "use _load_static_municipios() instead"
        )

    # -------------------------------------------------------------------------
    # municipality_zone() — Nominatim + static INE lookup
    # -------------------------------------------------------------------------

    def municipality_zone(self, lat: float, lon: float) -> tuple[str, str]:
        """Reverse geocode via Nominatim to get municipality name, then look up INE code.

        Returns (ine_code, municipality_name).

        Algorithm:
          1. Call Nominatim /reverse with (lat, lon) to get municipality name.
          2. Exact-match the name against bundled INE table (diacritic-insensitive).
          3. Fall back to substring match if no exact match.

        Uses Nominatim because /api/maestro/municipios returns HTTP 404
        (verified 2026-02-21).

        Nominatim ToS: 1 req/s, User-Agent required, no bulk use.
        Acceptable for single-track safety queries.

        Args:
            lat: Track centroid latitude in WGS84 decimal degrees.
            lon: Track centroid longitude in WGS84 decimal degrees.

        Returns:
            Tuple of (ine_code, municipality_name).

        Raises:
            WeatherError: if Nominatim call fails or municipality not found in INE table.
        """
        from src.weather.client import WeatherError

        # Step 1: Nominatim reverse geocoding (jsonv2 — top-level "name" is the place name)
        try:
            response = httpx.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": str(lat),
                    "lon": str(lon),
                    "format": "jsonv2",
                    "zoom": 10,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "mountain-safety-agent/1.0 (safety research tool)"},
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise WeatherError(f"Nominatim reverse geocoding failed: {exc}") from exc

        if "error" in data:
            raise WeatherError(
                f"Nominatim returned error for ({lat}, {lon}): {data['error']}"
            )

        # jsonv2: top-level "name" is the primary place name (most reliable for municipalities).
        # Fall back to address sub-fields if name is blank.
        address = data.get("address", {})
        muni_name = (
            data.get("name")
            or address.get("municipality")
            or address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
        )
        if not muni_name:
            raise WeatherError(
                f"Nominatim returned no municipality for ({lat}, {lon}): {address}"
            )

        # Step 2: Look up INE code directly in SQLite (mirrors auxiliary_code/database.py)
        # Query 1: exact match on NOMBRE
        # Query 2: LIKE '%name%' (same strategy as working auxiliary code)
        try:
            with sqlite3.connect(_DB_PATH) as conn:
                row = conn.execute(
                    "SELECT CPRO || CMUN, NOMBRE FROM mun_codes WHERE NOMBRE = ?",
                    (muni_name,),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        "SELECT CPRO || CMUN, NOMBRE FROM mun_codes WHERE NOMBRE LIKE ?",
                        (f"%{muni_name}%",),
                    ).fetchone()
        except Exception as exc:
            raise WeatherError(f"INE database lookup failed: {exc}") from exc

        if row is None:
            raise WeatherError(
                f"Municipality '{muni_name}' from Nominatim not found in INE database"
            )
        return row[0], row[1]

    # -------------------------------------------------------------------------
    # mountain_zone() — bounding box lookup
    # -------------------------------------------------------------------------

    def mountain_zone(self, lat: float, lon: float) -> str | None:
        """Return AEMET area code if (lat, lon) falls within any mountain zone bounding box.

        Uses simple bounding box containment (not polygon). If the centroid falls
        within multiple zones, the first match in MOUNTAIN_ZONES order is returned
        (centroid wins — no multi-zone merging, per CONTEXT.md).

        Args:
            lat: Latitude in WGS84 decimal degrees.
            lon: Longitude in WGS84 decimal degrees.

        Returns:
            AEMET area code string (e.g., "mad2") or None if outside all zones.
        """
        for zone in MOUNTAIN_ZONES:
            if (
                zone["min_lat"] <= lat <= zone["max_lat"]
                and zone["min_lon"] <= lon <= zone["max_lon"]
            ):
                return zone["code"]
        return None

    # -------------------------------------------------------------------------
    # mountain_zone_url() — polygon-exact zone + URL lookup
    # -------------------------------------------------------------------------

    def mountain_zone_url(self, lat: float, lon: float) -> tuple[str, str] | tuple[None, None]:
        """Return (zone_code, url_template) if (lat, lon) is inside a mountain zone polygon.

        Uses exact polygon containment from the urls_alerts SQLite table.
        URL template uses {} as a placeholder for the date in YYYYMMDD format.
        Zone code is extracted from the URL pattern predmmon_{zone}.xml.

        Args:
            lat: Latitude in WGS84 decimal degrees.
            lon: Longitude in WGS84 decimal degrees.

        Returns:
            (zone_code, url_template) tuple, or (None, None) if outside all zones.
            Never raises — errors return (None, None).
        """
        from src.weather.geometry import parse_polygon, point_in_polygon

        try:
            with sqlite3.connect(_DB_PATH) as conn:
                rows = conn.execute("SELECT url, coords FROM urls_alerts").fetchall()
        except Exception:
            return None, None

        for url_template, coords in rows:
            try:
                polygon = parse_polygon(coords, lon_lat=True)
                if point_in_polygon(lat, lon, polygon):
                    m = re.search(r"predmmon_(\w+)\.xml", url_template)
                    if m:
                        return m.group(1), url_template
            except Exception:
                continue

        return None, None

    # -------------------------------------------------------------------------
    # nivologica_zone() — avalanche bulletin zone lookup
    # -------------------------------------------------------------------------

    def nivologica_zone(self, area_code: str) -> str | None:
        """Return nivologica zone code ("0" or "1") or None if zone not covered.

        Looks up NIVOLOGICA_ZONE_MAP. Only Pyrenean mountain zones have
        corresponding avalanche bulletins. All other zones return None.

        Args:
            area_code: AEMET mountain area code (e.g., "cat1", "mad2").

        Returns:
            "0", "1", or None. Never raises.
        """
        return NIVOLOGICA_ZONE_MAP.get(area_code)

    # -------------------------------------------------------------------------
    # is_avalanche_season() — month-based seasonal gate
    # -------------------------------------------------------------------------

    def is_avalanche_season(self) -> bool:
        """Return True if today falls within avalanche season.

        Season definition: December (month 12) through May (month 5) inclusive.
        This matches AEMET's typical daily bulletin publication period.

        Outside season (June–November), the nivologica endpoint may return
        weekly or empty bulletins — AvalancheBulletin.unavailable_reason
        should be set to 'out_of_season'.

        Returns:
            True if current month is >= 12 or <= 5, False otherwise.
        """
        today = date.today()
        return today.month >= 12 or today.month <= 5

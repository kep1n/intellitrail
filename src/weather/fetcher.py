"""Weather data orchestration for the mountain safety agent.

fetch_weather_data() is the single entry point consumed by the LangGraph fetch_weather node.
It coordinates all AEMET calls for a given track geometry:
  1. Municipality zone lookup (INE code for municipal forecast)
  2. Municipal forecast (today + tomorrow)
  3. Mountain zone forecast (optional — raw narrative text)
  4. Avalanche/nivologica bulletin (optional — Pyrenean zones only)

The function always returns a WeatherData instance:
  - data_complete=False if the municipal forecast step fails (SAFE-02 gate)
  - data_complete=True with partial optional fields otherwise
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

from langsmith import traceable
from src.weather.alerts import check_alerts_for_centroid
from src.weather.client import WeatherError
from src.weather.models import (
    AvalancheBulletin,
    MountainForecast,
    MunicipalForecast,
    PeriodForecast,
    WeatherData,
)
from src.weather.mountain_forecast import (
    format_mountain_forecast_text,
    scrape_mountain_forecasts,
)

# ---------------------------------------------------------------------------
# Pyrenean snow bulletin constants
# ---------------------------------------------------------------------------

# Snow bulletins are available only for two Pyrenean zones.
# These are an addition on top of the mountain zone forecast — they provide
# detailed nivological (snow/avalanche) information for mountaineers.
# nav1 and all non-Pyrenean zones have no snow bulletin; avalanche=None for them.
_PYRENEAN_SNOW_ZONES: frozenset[str] = frozenset({"cat1", "arn1"})

# Direct AEMET XML URLs — no API key required, 24-hour ahead forecast.
_PYRENEAN_SNOW_XML: dict[str, str] = {
    "cat1": "https://www.aemet.es/xml/montana/p18tcat1.xml",  # eastern Pyrenees
    "arn1": "https://www.aemet.es/xml/montana/p18tarn1.xml",  # western Pyrenees
}

if TYPE_CHECKING:
    from src.models.geometry import ResolvedGeometry
    from src.weather.client import AEMETClient
    from src.weather.zone_mapper import ZoneMapper


@traceable
def fetch_weather_data(
    geometry: "ResolvedGeometry",
    client: "AEMETClient",
    zone_mapper: "ZoneMapper",
) -> WeatherData:
    """Orchestrates all AEMET calls for a given track geometry.

    Derives centroid from geometry.coordinates (mean lat, mean lon of all points).
    Calls municipality_zone → fetch_municipal_forecast → parse into MunicipalForecast list.
    Calls mountain_zone → if area code found, fetch_mountain_forecast (dia=0 and dia=1).
    Calls nivologica_zone + is_avalanche_season → if both True, fetch_avalanche_bulletin.
    Returns WeatherData with data_complete=False if municipal forecast fails.

    Args:
        geometry: Parsed GPX track with coordinates list.
        client: Authenticated AEMETClient instance.
        zone_mapper: ZoneMapper instance bound to the same client.

    Returns:
        WeatherData with all available fields populated.
        data_complete=False if the municipal forecast step fails.
    """
    # -------------------------------------------------------------------------
    # 1. CENTROID — mean lat/lon of all track coordinates
    # -------------------------------------------------------------------------
    try:
        coords = geometry.coordinates
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)
    except Exception as _geo_exc:
        return WeatherData(data_complete=False, incomplete_reason=f"geometry error: {_geo_exc}")

    # -------------------------------------------------------------------------
    # 2. MUNICIPAL FORECAST (required — failure triggers data_complete=False)
    # -------------------------------------------------------------------------
    try:
        ine_code, muni_name = zone_mapper.municipality_zone(centroid_lat, centroid_lon)
        raw_days = client.fetch_municipal_forecast(ine_code)
        municipal = _parse_municipal_forecast(raw_days, ine_code, muni_name)
    except Exception as _exc:
        logger.exception("Municipal forecast failed")
        return WeatherData(
            data_complete=False,
            incomplete_reason=f"{type(_exc).__name__}: {_exc}",
        )

    # -------------------------------------------------------------------------
    # 3. MOUNTAIN FORECAST (optional — failure sets missing_mountain_data=True)
    # -------------------------------------------------------------------------
    mountain: MountainForecast | None = None
    area_code: str | None = None
    missing_mountain_data: bool = False

    try:
        # Polygon-exact zone detection from SQLite; bounding box as fallback
        area_code, _ = zone_mapper.mountain_zone_url(centroid_lat, centroid_lon)
        if area_code is None:
            area_code = zone_mapper.mountain_zone(centroid_lat, centroid_lon)
        if area_code:
            forecasts = scrape_mountain_forecasts(area_code, num_days=1)
            forecast_text = format_mountain_forecast_text(forecasts)
            freezing_level, wind_1500m = _extract_free_atmosphere(forecasts)
            mountain = MountainForecast(
                area_code=area_code,
                area_name=area_code,
                forecast_text=forecast_text,
                freezing_level=freezing_level,
                wind_1500m=wind_1500m,
            )
        # else: mountain is None — centroid outside all mountain zones
    except Exception:
        mountain = None
        missing_mountain_data = True

    # -------------------------------------------------------------------------
    # 4. SNOW BULLETIN (optional — Pyrenean zones only)
    #    Uses direct AEMET XML endpoints; no API key required.
    #    Only cat1, arn1, nav1 have meaningful snow/avalanche data.
    #    All other mountain zones (nev1, mad2, gre1, etc.) → avalanche=None
    #    so the LLM never reasons about snow for non-Pyrenean hikes.
    # -------------------------------------------------------------------------
    avalanche: AvalancheBulletin | None = None

    if area_code in _PYRENEAN_SNOW_ZONES:
        if zone_mapper.is_avalanche_season():
            avalanche = _fetch_pyrenean_snow_bulletin(area_code)
        else:
            avalanche = AvalancheBulletin(
                zone_code=area_code,
                raw_text="",
                unavailable_reason="out_of_season",
            )

    # -------------------------------------------------------------------------
    # 5. CAP WEATHER ALERTS (optional — failure returns empty list)
    # -------------------------------------------------------------------------
    alerts: list[dict] = []
    try:
        alerts = check_alerts_for_centroid(centroid_lat, centroid_lon)
    except Exception:
        alerts = []

    # -------------------------------------------------------------------------
    # 6. ASSEMBLE WeatherData
    # -------------------------------------------------------------------------
    return WeatherData(
        municipal=municipal,
        mountain=mountain,
        avalanche=avalanche,
        alerts=alerts,
        missing_mountain_data=missing_mountain_data,
        data_complete=True,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_pyrenean_snow_bulletin(area_code: str) -> AvalancheBulletin:
    """Fetch the 24-hour snow/avalanche bulletin from the AEMET direct XML endpoint.

    Parses <ROOT><TXT_PREDICCION>forecast text</TXT_PREDICCION></ROOT>.
    Extracts the first European-scale risk level (1–5) found in parentheses
    in the text — e.g. "Débil (1)" → risk_level=1.

    Returns AvalancheBulletin with raw_text=forecast_text on success,
    or unavailable_reason='fetch_failed' on any network/parse error.
    """
    url = _PYRENEAN_SNOW_XML.get(area_code, "")
    if not url:
        return AvalancheBulletin(zone_code=area_code, raw_text="", unavailable_reason="fetch_failed")

    try:
        r = httpx.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        # Pass raw bytes so ET reads the <?xml encoding="..."> declaration
        # (AEMET XMLs are ISO-8859-1, not UTF-8)
        root = ET.fromstring(r.content)
        txt_el = root.find("TXT_PREDICCION")
        forecast_text = (txt_el.text or "").strip() if txt_el is not None else ""

        return AvalancheBulletin(
            zone_code=area_code,
            raw_text=forecast_text,
        )
    except Exception as exc:
        logger.warning("Pyrenean snow bulletin fetch failed for %s: %s", area_code, exc)
        return AvalancheBulletin(
            zone_code=area_code,
            raw_text="",
            unavailable_reason="fetch_failed",
        )


def _extract_free_atmosphere(forecasts: dict) -> tuple[str | None, str | None]:
    """Extract isocero (freezing level) and v1500 (wind at 1500m) from atmosferalibre sections.

    Returns (freezing_level, wind_1500m) strings, or (None, None) if not found.
    Uses the first day in the forecast dict that has free_atmosphere data.
    """
    for day_fc in forecasts.values():
        if day_fc is None:
            continue
        sections = getattr(day_fc, "sections", {})
        fa_items = sections.get("free_atmosphere", [])
        freezing_level = None
        wind_1500m = None
        for item in fa_items:
            h = item.header.lower()
            if "isocero" in h or "freezing" in h:
                freezing_level = item.text
            elif "v1500" in h or "1500" in h:
                wind_1500m = item.text
        if freezing_level or wind_1500m:
            return freezing_level, wind_1500m
    return None, None


def _parse_municipal_forecast(
    raw_days: list[dict],
    ine_code: str,
    muni_name: str,
) -> list[MunicipalForecast]:
    """Parse today + tomorrow from AEMET raw day list.

    Slices to first 2 entries. Extracts PeriodForecast for each period in
    estadoCielo (sky state). Uses .get() defensively — missing fields become
    None and never raise on missing AEMET field.

    AEMET daily forecast structure (abbreviated):
      [
        {
          "fecha": "2026-02-21T00:00:00",
          "estadoCielo": [{"periodo": "00-24", "valor": "11", "descripcion": "Despejado"}],
          "probPrecipitacion": [{"periodo": "00-24", "valor": "5"}],
          "precipitacion": [{"periodo": "00-24", "valor": "0.0"}],
          "vientoAndRachaMax": [...],
          "temperatura": {"minima": "5", "maxima": "15"},
          ...
        },
        ...
      ]

    Args:
        raw_days: Raw list of day dicts from AEMETClient.fetch_municipal_forecast().
        ine_code: 5-digit INE municipality code.
        muni_name: Municipality name string.

    Returns:
        List of MunicipalForecast, one per day (today + tomorrow, max 2 entries).
    """
    result: list[MunicipalForecast] = []

    # Only today + tomorrow
    for day in raw_days[:2]:
        date_str = str(day.get("fecha", ""))
        # AEMET fecha format: "2026-02-21T00:00:00" — extract date part only
        if "T" in date_str:
            date_str = date_str.split("T")[0]

        # Build period forecasts from estadoCielo entries
        # Each period entry: {"periodo": "00-24", "valor": "11", "descripcion": "..."}
        sky_entries: list[dict] = day.get("estadoCielo") or []
        prob_precip_entries: list[dict] = day.get("probPrecipitacion") or []
        precip_entries: list[dict] = day.get("precipitacion") or []
        temperatura: dict = day.get("temperatura") or {}

        # Build period lookup dicts keyed by periodo string
        prob_precip_by_period: dict[str, int | None] = {}
        for entry in prob_precip_entries:
            periodo = str(entry.get("periodo", ""))
            valor = entry.get("valor")
            try:
                prob_precip_by_period[periodo] = int(valor) if valor not in (None, "") else None
            except (ValueError, TypeError):
                prob_precip_by_period[periodo] = None

        precip_by_period: dict[str, float | None] = {}
        for entry in precip_entries:
            periodo = str(entry.get("periodo", ""))
            valor = entry.get("valor")
            try:
                precip_by_period[periodo] = float(valor) if valor not in (None, "") else None
            except (ValueError, TypeError):
                precip_by_period[periodo] = None

        # Wind: AEMET uses "viento" (not "vientoAndRachaMax") with velocidad/direccion per period
        viento_entries: list[dict] = day.get("viento") or []
        wind_by_period: dict[str, tuple[float | None, str | None]] = {}
        for entry in viento_entries:
            periodo = str(entry.get("periodo", ""))
            velocidad = entry.get("velocidad")
            direccion = entry.get("direccion") or None
            try:
                speed = float(velocidad) if velocidad not in (None, "") else None
            except (ValueError, TypeError):
                speed = None
            wind_by_period[periodo] = (speed, direccion if direccion else None)

        periods: list[PeriodForecast] = []
        if sky_entries:
            for sky_entry in sky_entries:
                periodo = str(sky_entry.get("periodo", ""))
                if not periodo:
                    continue

                sky_state = sky_entry.get("descripcion") or sky_entry.get("valor")
                if sky_state is not None:
                    sky_state = str(sky_state)

                wind_speed, wind_dir = wind_by_period.get(periodo, (None, None))
                periods.append(
                    PeriodForecast(
                        period=periodo,
                        sky_state=sky_state,
                        precipitation_probability=prob_precip_by_period.get(periodo),
                        precipitation_amount_mm=precip_by_period.get(periodo),
                        wind_speed_kmh=wind_speed,
                        wind_direction=wind_dir,
                        temperature_min_c=_parse_optional_float(temperatura.get("minima")),
                        temperature_max_c=_parse_optional_float(temperatura.get("maxima")),
                    )
                )
        else:
            # No sky state entries — create a single minimal period entry
            periods = [
                PeriodForecast(
                    period="00-24",
                    temperature_min_c=_parse_optional_float(temperatura.get("minima")),
                    temperature_max_c=_parse_optional_float(temperatura.get("maxima")),
                )
            ]

        result.append(
            MunicipalForecast(
                municipality_id=ine_code,
                municipality_name=muni_name,
                date=date_str,
                periods=periods,
            )
        )

    return result


def _parse_optional_float(value: object) -> float | None:
    """Parse an optional numeric value to float, returning None on failure."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _parse_avalanche_bulletin(raw: dict, zone_code: str) -> AvalancheBulletin:
    """Parse numeric risk level from AEMET bulletin dict.

    Tries to extract an integer risk level (1–5 European scale) from known
    risk field names. If not found or not parseable, risk_level=None.
    Stores the full raw dict as a JSON string in raw_text for transparency.

    AEMET nivologica bulletin fields are not fully documented — tries several
    candidate field names defensively.

    Args:
        raw: Raw bulletin dict from AEMETClient.fetch_avalanche_bulletin().
        zone_code: Nivologica zone code ("0" or "1").

    Returns:
        AvalancheBulletin with risk_level parsed if available.
    """
    risk_level: int | None = None

    # Try candidate field names for the numeric risk level
    risk_field_candidates = (
        "nivelAvalanchas",
        "riesgoAvalanchas",
        "riesgo",
        "nivel",
        "risk_level",
        "risk",
    )
    for field in risk_field_candidates:
        value = raw.get(field)
        if value is not None:
            try:
                risk_level = int(value)
                break
            except (ValueError, TypeError):
                continue

    # Serialize full raw dict to JSON string for transparency
    try:
        raw_text = json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        raw_text = str(raw)

    return AvalancheBulletin(
        zone_code=zone_code,
        risk_level=risk_level,
        raw_text=raw_text,
    )

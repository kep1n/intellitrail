"""AEMET mountain forecast XML scraping (snow, wind, avalanche).

Fetches zone-specific mountain forecasts from the AEMET XML endpoint and
extracts the prediccion and atmosferalibre sections.

URL pattern: https://www.aemet.es/xml/montana/{YYYYMMDD}_predmmon_{zone}.xml
Zone codes: arn1, cat1, nav1, peu1, nev1, mad2, gre1, arn2, rio1
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.aemet.es/xml/montana/{date}_predmmon_{zone}.xml"

TARGET_SECTIONS: frozenset[str] = frozenset({"prediccion", "atmosferalibre"})

_SECTION_NAMES: dict[str, str] = {
    "nubosidad": "cloudiness",
    "pcp": "precipitation",
    "temperatura": "temperature",
    "viento": "wind",
    "tormentas": "storms",
    "cota_nieve": "snow_level",
    "estado_cielo": "sky_condition",
    "atmosferalibre": "free_atmosphere",
    "prediccion": "forecast",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastSection:
    """A single subsection (apartado) within a forecast section.

    Attributes:
        header: English name of this subsection (e.g. "wind").
        text: Forecast text (Spanish — the LLM handles translation).
    """

    header: str
    text: str


@dataclass
class DayForecast:
    """Parsed mountain forecast for a single day.

    Attributes:
        date: ISO date string, e.g. "2026-02-25".
        url: Source URL used to fetch this forecast.
        sections: Mapping of section name to ForecastSection entries.
    """

    date: str
    url: str
    sections: dict[str, list[ForecastSection]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_url(forecast_date: date, zone: str) -> str:
    return BASE_URL.format(date=forecast_date.strftime("%Y%m%d"), zone=zone)


def _fetch_mountain_xml(url: str, client: httpx.Client) -> ET.Element | None:
    logger.debug("Fetching: %s", url)
    try:
        r = client.get(url, follow_redirects=True)
        r.raise_for_status()
        content = r.content.decode("iso-8859-15", errors="replace")
        return ET.fromstring(content)
    except (httpx.HTTPStatusError, httpx.HTTPError) as e:
        logger.warning("HTTP error fetching %s: %s", url, e)
    except ET.ParseError as e:
        logger.warning("XML parse error for %s: %s", url, e)
    return None


def _parse_section(section: ET.Element) -> list[ForecastSection] | None:
    apartados = []
    for apartado in section.findall("apartado"):
        texto_el = apartado.find("texto")
        texto = texto_el.text.strip() if texto_el is not None and texto_el.text else None
        if not texto:
            continue
        apartados.append(ForecastSection(
            header=_SECTION_NAMES.get(
                apartado.attrib.get("nombre", ""),
                apartado.attrib.get("nombre", ""),
            ),
            text=texto,
        ))
    return apartados if apartados else None


def _parse_forecast(root: ET.Element) -> dict[str, list[ForecastSection]]:
    result: dict[str, list[ForecastSection]] = {}
    for section in root.iter():
        local_tag = section.tag.split("}")[-1].lower()
        if local_tag != "seccion":
            continue
        name = section.attrib.get("nombre", "").strip().lower()
        if name in TARGET_SECTIONS:
            parsed = _parse_section(section)
            if parsed:
                translated_name = _SECTION_NAMES.get(name, name)
                result[translated_name] = parsed
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_mountain_forecasts(
    zone: str,
    start_date: date | None = None,
    num_days: int = 3,
) -> dict[str, DayForecast | None]:
    """Scrape AEMET mountain forecasts for consecutive days.

    Args:
        zone: Zone code from the URL (e.g. "peu1", "gre1").
        start_date: First date to fetch. Defaults to today.
        num_days: Number of consecutive days to fetch. Default is 3.

    Returns:
        Dict keyed by ISO date string. Each value is a DayForecast with
        the parsed sections, or None if the fetch failed for that date.
    """
    if start_date is None:
        start_date = date.today()

    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    output: dict[str, DayForecast | None] = {}

    with httpx.Client(timeout=30) as client:
        for forecast_date in dates:
            date_key = forecast_date.isoformat()
            url = _build_url(forecast_date, zone)
            logger.info("Fetching mountain forecast for %s — %s", date_key, url)

            root = _fetch_mountain_xml(url, client)
            if root is None:
                logger.warning("No data for %s, skipping.", date_key)
                output[date_key] = None
                continue

            sections = _parse_forecast(root)
            output[date_key] = DayForecast(date=date_key, url=url, sections=sections)

    return output


def format_mountain_forecast_text(forecasts: dict[str, DayForecast | None]) -> str:
    """Format a DayForecast dict into a readable text string for the LLM.

    Args:
        forecasts: Output of scrape_mountain_forecasts().

    Returns:
        Multi-line string with dated sections. Empty string if no data.
    """
    lines = []
    for date_str, forecast in forecasts.items():
        if forecast is None:
            lines.append(f"{date_str}: No data available")
            continue
        lines.append(f"{date_str}:")
        for section_name, items in forecast.sections.items():
            lines.append(f"  [{section_name}]")
            for item in items:
                lines.append(f"    {item.header}: {item.text}")
    return "\n".join(lines) if lines else ""

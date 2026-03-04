"""AEMET mountain forecast XML scraping (snow, wind, avalanche).

Fetches zone-specific mountain forecasts from the AEMET XML endpoint and
extracts the ``prediccion`` and ``atmosferalibre`` sections with optional
Spanish-to-English translation via ``deep_translator``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from xml.etree import ElementTree as ET

import httpx

from _http import fetch_xml

try:
    from deep_translator import GoogleTranslator
    _translator = GoogleTranslator(source="es", target="en")
    def _translate(text: str) -> str:
        return _translator.translate(text) if text else text
except ImportError:
    logging.getLogger(__name__).warning(
        "deep_translator not installed — text will not be translated. "
        "Run: pip install deep-translator"
    )
    def _translate(text: str) -> str:
        return text

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

logger = logging.getLogger(__name__)

BASE_URL = "https://www.aemet.es/xml/montana/{date}_predmmon_{zone}.xml"
TARGET_SECTIONS: frozenset[str] = frozenset({"prediccion", "atmosferalibre"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastSection:
    """A single translated subsection (apartado) within a forecast section.

    Attributes:
        header: English name of this subsection (e.g. ``"wind"``).
        text: Translated forecast text.
    """

    header: str
    text: str


@dataclass
class DayForecast:
    """Parsed mountain forecast for a single day.

    Attributes:
        date: ISO date string, e.g. ``"2026-02-25"``.
        url: Source URL used to fetch this forecast.
        sections: Mapping of translated section name to its
            :class:`ForecastSection` entries.
    """

    date: str
    url: str
    sections: dict[str, list[ForecastSection]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_url(forecast_date: date, zone: str) -> str:
    """Build the AEMET mountain forecast XML URL for a given date and zone.

    Args:
        forecast_date: The forecast date.
        zone: Zone code used in the URL (e.g. ``"peu1"``).

    Returns:
        Fully-formed URL string.
    """
    return BASE_URL.format(date=forecast_date.strftime("%Y%m%d"), zone=zone)


def _fetch_mountain_xml(url: str, client: httpx.Client) -> ET.Element | None:
    """Fetch and parse a mountain forecast XML document.

    Wraps :func:`~_http.fetch_xml` with ``iso-8859-15`` encoding and
    catches HTTP and parse errors gracefully.

    Args:
        url: URL of the AEMET mountain forecast XML.
        client: Active ``httpx.Client`` to use.

    Returns:
        Parsed XML root element, or *None* on error.
    """
    logger.debug("Fetching: %s", url)
    try:
        return fetch_xml(url, client, encoding="iso-8859-15")
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error fetching %s: %s", url, e)
    except ET.ParseError as e:
        logger.warning("XML parse error for %s: %s", url, e)
    return None


def _parse_section(section: ET.Element) -> list[ForecastSection] | None:
    """Extract and translate all ``<apartado>`` entries from a ``<seccion>`` element.

    Args:
        section: A ``<seccion>`` XML element from the AEMET mountain forecast.

    Returns:
        List of :class:`ForecastSection` instances, or *None* if the section
        contains no non-empty text.
    """
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
            text=_translate(texto),
        ))

    return apartados if apartados else None


def _parse_forecast(root: ET.Element) -> dict[str, list[ForecastSection]]:
    """Parse all target sections from a forecast XML root.

    Args:
        root: Root element of the AEMET mountain forecast XML.

    Returns:
        Dict mapping translated section names to their
        :class:`ForecastSection` lists.
    """
    result: dict[str, list[ForecastSection]] = {}

    # Log the full tag tree at DEBUG to diagnose structure mismatches
    for el in root.iter():
        logger.debug("TAG: %r | ATTRIB: %r | TEXT: %r", el.tag, el.attrib, (el.text or "").strip()[:80])

    for section in root.iter():
        # Match regardless of namespace and tag casing
        local_tag = section.tag.split("}")[-1].lower()
        if local_tag != "seccion":
            continue

        # Normalize attribute value: strip whitespace and lowercase
        name = section.attrib.get("nombre", "").strip().lower()
        translated_name = _SECTION_NAMES.get(name, name)
        logger.debug("Found <seccion nombre=%r>", name)

        if name in TARGET_SECTIONS:
            parsed = _parse_section(section)
            if parsed:
                result[translated_name] = parsed
                logger.info("Parsed section: %s → %s", name, translated_name)
            else:
                logger.warning("Section %r found but no content extracted", name)

    if not result:
        logger.warning(
            "No target sections found. Available: %r",
            [s.attrib.get("nombre") for s in root.iter() if s.tag.split("}")[-1].lower() == "seccion"],
        )
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
        zone: Zone code from the URL (e.g. ``"peu1"``).
        start_date: First date to fetch. Defaults to today.
        num_days: Number of consecutive days to fetch. Default is ``3``.

    Returns:
        Dict keyed by ISO date string. Each value is a :class:`DayForecast`
        with the parsed sections, or *None* if the fetch failed.
    """
    if start_date is None:
        start_date = date.today()

    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    output: dict[str, DayForecast | None] = {}

    with httpx.Client(timeout=30) as client:
        for forecast_date in dates:
            date_key = forecast_date.isoformat()
            url = _build_url(forecast_date, zone)
            logger.info("Fetching forecast for %s — %s", date_key, url)

            root = _fetch_mountain_xml(url, client)
            if root is None:
                logger.warning("No data for %s, skipping.", date_key)
                output[date_key] = None
                continue

            sections = _parse_forecast(root)
            output[date_key] = DayForecast(date=date_key, url=url, sections=sections)
            logger.info("Parsed %d section(s) for %s", len(sections), date_key)

    return output


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    from datetime import date as _date
    result = scrape_mountain_forecasts(
        zone="peu1",
        start_date=_date(2026, 2, 25),
        num_days=3,
    )
    print(json.dumps(
        {k: dataclasses.asdict(v) if v is not None else None for k, v in result.items()},
        indent=2,
        ensure_ascii=False,
    ))

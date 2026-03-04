"""AEMET CAP XML alert parsing with ray-casting polygon containment.

Fetches the AEMET ATOM alert feed, iterates over each CAP alert XML, and
determines whether a given (lat, lon) centroid falls inside any alert polygon
using the ray-casting algorithm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from xml.etree import ElementTree as ET

from _http import fetch_xml
from geometry import parse_polygon, point_in_polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
NS_ATOM = "http://www.w3.org/2005/Atom"
NS_CAP = "urn:oasis:names:tc:emergency:cap:1.2"

ATOM_INDEX_URL = "https://www.aemet.es/documentos_d/eltiempo/prediccion/avisos/rss/CAP_AFAE_ATOM.xml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AlertInfo:
    """Parsed representation of a single CAP alert.

    Attributes:
        event: Human-readable event type (e.g. ``"Wind"``).
        severity: CAP severity level (e.g. ``"Moderate"``, ``"Severe"``).
        certainty: CAP certainty level (e.g. ``"Likely"``).
        event_code: AEMET internal event code value.
        effective: Start time of the alert, or *None* if missing/unparseable.
        expires: End time of the alert, or *None* if missing/unparseable.
        description: Free-text alert description.
        area_desc: Name of the affected geographic area.
        parameters: Additional CAP ``<parameter>`` key-value pairs.
    """

    event: str
    severity: str
    certainty: str
    event_code: str
    effective: datetime | None
    expires: datetime | None
    description: str
    area_desc: str
    parameters: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _tag(ns: str, local: str) -> str:
    """Build a Clark-notation qualified tag string.

    Args:
        ns: XML namespace URI.
        local: Local element name.

    Returns:
        String in the form ``"{namespace}local"``.
    """
    return f"{{{ns}}}{local}"


def _find_text(el: ET.Element, ns: str, local: str, default: str = "") -> str:
    """Find a child element and return its stripped text content.

    Args:
        el: Parent XML element.
        ns: Namespace URI of the child element.
        local: Local name of the child element.
        default: Value to return when the element is absent or has no text.

    Returns:
        Stripped text of the found element, or *default*.
    """
    found = el.find(_tag(ns, local))
    return (found.text or "").strip() if found is not None else default


def _parse_dt(value: str) -> datetime | None:
    """Parse a CAP ISO-8601 datetime string.

    Args:
        value: Datetime string, e.g. ``"2024-01-15T12:00:00+01:00"``.

    Returns:
        Parsed :class:`datetime` object, or *None* on failure.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _extract_cap_urls(atom_root: ET.Element) -> list[str]:
    """Extract all ``.xml`` href links from ATOM feed entries.

    Args:
        atom_root: Root element of the ATOM index feed.

    Returns:
        List of CAP XML URLs found in the feed.
    """
    urls = [
        href
        for entry in atom_root.findall(_tag(NS_ATOM, "entry"))
        for link in entry.findall(_tag(NS_ATOM, "link"))
        if (href := link.attrib.get("href", "")).endswith(".xml")
    ]
    logger.debug("Found %d CAP URLs in ATOM feed", len(urls))
    return urls


def _parse_en_info(cap_root: ET.Element) -> list[tuple[ET.Element, list[ET.Element]]]:
    """Return ``(info_el, area_elements)`` for all English ``<info>`` blocks.

    Args:
        cap_root: Root element of a CAP alert XML document.

    Returns:
        List of ``(info_element, [area_element, ...])`` tuples for each
        English-language ``<info>`` block.
    """
    return [
        (info, info.findall(_tag(NS_CAP, "area")))
        for info in cap_root.findall(_tag(NS_CAP, "info"))
        if _find_text(info, NS_CAP, "language").lower().startswith("en")
    ]


def _parse_alert_info(info: ET.Element, area_desc: str) -> AlertInfo:
    """Build an :class:`AlertInfo` from a CAP ``<info>`` element.

    Args:
        info: A CAP ``<info>`` XML element.
        area_desc: Pre-extracted area description string.

    Returns:
        A populated :class:`AlertInfo` dataclass instance.
    """
    event_code = next(
        (
            _find_text(ec, NS_CAP, "value")
            for ec in info.findall(_tag(NS_CAP, "eventCode"))
            if _find_text(ec, NS_CAP, "value")
        ),
        "",
    )

    params: dict[str, str] = {
        k: _find_text(param, NS_CAP, "value")
        for param in info.findall(_tag(NS_CAP, "parameter"))
        if (k := _find_text(param, NS_CAP, "valueName"))
    }

    return AlertInfo(
        event=_find_text(info, NS_CAP, "event"),
        severity=_find_text(info, NS_CAP, "severity"),
        certainty=_find_text(info, NS_CAP, "certainty"),
        event_code=event_code,
        effective=_parse_dt(_find_text(info, NS_CAP, "effective")),
        expires=_parse_dt(_find_text(info, NS_CAP, "expires")),
        description=_find_text(info, NS_CAP, "description"),
        area_desc=area_desc,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_alerts_for_centroid(
    lat: float,
    lon: float,
    atom_url: str = ATOM_INDEX_URL,
) -> list[dict[str, Any]]:
    """Fetch AEMET CAP alerts and return those whose polygon contains (lat, lon).

    Args:
        lat: Latitude of the centroid to check (decimal degrees).
        lon: Longitude of the centroid to check (decimal degrees).
        atom_url: URL of the AEMET ATOM alert index feed.

    Returns:
        List of matching alert dicts with fields: ``source_url``, ``area_desc``,
        ``event``, ``severity``, ``certainty``, ``event_code``, ``effective``,
        ``expires``, ``description``, ``parameters``.
    """
    results = []

    with httpx.Client(timeout=30) as client:
        logger.info("Checking alerts for centroid (lat=%.4f, lon=%.4f)", lat, lon)
        atom_root = fetch_xml(atom_url, client)
        cap_urls = _extract_cap_urls(atom_root)

        for url in cap_urls:
            try:
                cap_root = fetch_xml(url, client)
            except Exception as e:
                logger.warning("Could not fetch %s: %s", url, e)
                continue

            for info, areas in _parse_en_info(cap_root):
                for area in areas:
                    polygon_el = area.find(_tag(NS_CAP, "polygon"))
                    if polygon_el is None or not polygon_el.text:
                        continue

                    polygon = parse_polygon(polygon_el.text)
                    if not point_in_polygon(lat, lon, polygon):
                        continue

                    area_desc = _find_text(area, NS_CAP, "areaDesc")
                    alert = _parse_alert_info(info, area_desc)
                    logger.info(
                        "Alert match — event: %s | severity: %s | area: %s",
                        alert.event, alert.severity, area_desc,
                    )

                    results.append({
                        "source_url": url,
                        "area_desc": alert.area_desc,
                        "event": alert.event,
                        "severity": alert.severity,
                        "certainty": alert.certainty,
                        "event_code": alert.event_code,
                        "effective": alert.effective.isoformat() if alert.effective else None,
                        "expires": alert.expires.isoformat() if alert.expires else None,
                        "description": alert.description,
                        "parameters": alert.parameters,
                    })

    logger.info(
        "Found %d matching alert(s) for centroid (lat=%.4f, lon=%.4f)",
        len(results), lat, lon,
    )
    return results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Tenerife centroid
    lat, lon = 43.33078463426459, -8.625642126271716

    alerts = check_alerts_for_centroid(lat, lon)

    if not alerts:
        print("No active alerts for this location.")
    else:
        print(json.dumps(alerts, indent=2, ensure_ascii=False))

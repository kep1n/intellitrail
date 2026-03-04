"""AEMET CAP XML alert parsing with ray-casting polygon containment.

Fetches the AEMET ATOM alert feed, iterates over each CAP alert XML, and
determines whether a given (lat, lon) centroid falls inside any alert polygon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from src.weather.geometry import parse_polygon, point_in_polygon

logger = logging.getLogger(__name__)

NS_ATOM = "http://www.w3.org/2005/Atom"
NS_CAP = "urn:oasis:names:tc:emergency:cap:1.2"

ATOM_INDEX_URL = "https://www.aemet.es/documentos_d/eltiempo/prediccion/avisos/rss/CAP_AFAE_ATOM.xml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _find_text(el: ET.Element, ns: str, local: str, default: str = "") -> str:
    found = el.find(_tag(ns, local))
    return (found.text or "").strip() if found is not None else default


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _fetch_xml(url: str, client: httpx.Client) -> ET.Element:
    r = client.get(url, follow_redirects=True)
    r.raise_for_status()
    return ET.fromstring(r.content)


def _extract_cap_urls(atom_root: ET.Element) -> list[str]:
    return [
        href
        for entry in atom_root.findall(_tag(NS_ATOM, "entry"))
        for link in entry.findall(_tag(NS_ATOM, "link"))
        if (href := link.attrib.get("href", "")).endswith(".xml")
    ]


def _parse_en_info(cap_root: ET.Element) -> list[tuple[ET.Element, list[ET.Element]]]:
    return [
        (info, info.findall(_tag(NS_CAP, "area")))
        for info in cap_root.findall(_tag(NS_CAP, "info"))
        if _find_text(info, NS_CAP, "language").lower().startswith("en")
    ]


def _build_alert_dict(info: ET.Element, area_desc: str, source_url: str) -> dict[str, Any]:
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
    effective = _parse_dt(_find_text(info, NS_CAP, "effective"))
    expires = _parse_dt(_find_text(info, NS_CAP, "expires"))

    return {
        "source_url": source_url,
        "area_desc": area_desc,
        "event": _find_text(info, NS_CAP, "event"),
        "severity": _find_text(info, NS_CAP, "severity"),
        "certainty": _find_text(info, NS_CAP, "certainty"),
        "event_code": event_code,
        "effective": effective.isoformat() if effective else None,
        "expires": expires.isoformat() if expires else None,
        "description": _find_text(info, NS_CAP, "description"),
        "parameters": params,
    }


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
        List of matching alert dicts. Empty list if no alerts match or on error.
    """
    results: list[dict[str, Any]] = []

    try:
        with httpx.Client(timeout=30) as client:
            atom_root = _fetch_xml(atom_url, client)
            cap_urls = _extract_cap_urls(atom_root)
            logger.info(
                "Checking %d CAP alerts for centroid (lat=%.4f, lon=%.4f)",
                len(cap_urls), lat, lon,
            )

            for url in cap_urls:
                try:
                    cap_root = _fetch_xml(url, client)
                except Exception as e:
                    logger.warning("Could not fetch %s: %s", url, e)
                    continue

                for info, areas in _parse_en_info(cap_root):
                    for area in areas:
                        polygon_el = area.find(_tag(NS_CAP, "polygon"))
                        if polygon_el is None or not polygon_el.text:
                            continue
                        try:
                            polygon = parse_polygon(polygon_el.text)
                        except (ValueError, IndexError):
                            continue
                        if not point_in_polygon(lat, lon, polygon):
                            continue

                        area_desc = _find_text(area, NS_CAP, "areaDesc")
                        logger.debug(
                            "Polygon match — centroid (%.4f, %.4f) inside area '%s' "
                            "from %s | polygon bbox: lat=[%.2f–%.2f] lon=[%.2f–%.2f]",
                            lat, lon, area_desc, url,
                            min(p[0] for p in polygon), max(p[0] for p in polygon),
                            min(p[1] for p in polygon), max(p[1] for p in polygon),
                        )
                        results.append(_build_alert_dict(info, area_desc, url))
                        logger.info(
                            "Alert match — event: %s | severity: %s | area: %s",
                            results[-1]["event"], results[-1]["severity"], area_desc,
                        )
    except Exception as e:
        logger.warning("Alert fetch failed (non-fatal): %s", e)

    return results

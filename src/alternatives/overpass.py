import math
import httpx
from typing import Any

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY_TEMPLATE = """
[out:json][timeout:25];
relation[route=hiking](around:{radius},{lat},{lon});
out center tags;
"""


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km between two WGS84 coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def search_overpass(lat: float, lon: float, radius_m: int = 10000) -> list[dict]:
    """Find hiking route relations within radius_m meters of (lat, lon) using Overpass API.

    Overpass QL: relation[route=hiking](around:{radius},{lat},{lon}); out center tags;

    Returns up to 3 results. Each result dict:
    {
        "name": str,           # from OSM name tag, or "Unnamed route {osm_id}"
        "centroid_lat": float, # from center.lat in Overpass response
        "centroid_lon": float, # from center.lon in Overpass response
        "distance_km": float,  # straight-line from original (lat, lon) to route centroid
    }

    Returns empty list if:
    - No hiking relations found within radius
    - Overpass API returns an error (logs nothing, returns empty — caller handles "none found")

    Uses httpx.get() with 30-second timeout. No retry — transient failures surface as
    empty results (the report will show "none found" per user decision).
    """
    query = OVERPASS_QUERY_TEMPLATE.format(radius=radius_m, lat=lat, lon=lon)
    try:
        response = httpx.get(
            OVERPASS_URL,
            params={"data": query},
            timeout=30.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except Exception:
        return []

    elements = data.get("elements", [])
    results = []
    for elem in elements:
        if elem.get("type") != "relation":
            continue
        center = elem.get("center", {})
        clat = center.get("lat")
        clon = center.get("lon")
        if clat is None or clon is None:
            continue
        tags = elem.get("tags", {})
        name = (
            tags.get("name")
            or tags.get("name:en")
            or tags.get("name:es")
            or f"Unnamed route {elem.get('id', '?')}"
        )
        results.append({
            "name": name,
            "centroid_lat": clat,
            "centroid_lon": clon,
            "distance_km": round(_haversine_km(lat, lon, clat, clon), 2),
        })

    # Sort by distance, return top 3 (per user decision: up to 3 alternatives)
    results.sort(key=lambda r: r["distance_km"])
    return results[:3]

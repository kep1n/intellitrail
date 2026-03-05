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


_REFUGE_QUERY_TEMPLATE = """
[out:json][timeout:25];
(
  node[tourism=alpine_hut](around:{radius},{lat},{lon});
  node[tourism=wilderness_hut](around:{radius},{lat},{lon});
  node[amenity=shelter](around:{radius},{lat},{lon});
);
out body;
"""


def search_refuges(lat: float, lon: float, radius_m: int = 15000) -> list[dict]:
    """Find alpine huts and shelters within radius_m meters of (lat, lon).

    Queries OSM nodes tagged tourism=alpine_hut, tourism=wilderness_hut, or
    amenity=shelter. Returns up to 5 results sorted by distance.

    Each result dict:
    {
        "name":         str,   # OSM name tag or fallback
        "distance_km":  float,
        "lat":          float,
        "lon":          float,
        "type":         str,   # "alpine_hut" | "wilderness_hut" | "shelter"
    }

    Returns empty list on any exception — caller handles "none found".
    """
    query = _REFUGE_QUERY_TEMPLATE.format(radius=radius_m, lat=lat, lon=lon)
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

    results = []
    for elem in data.get("elements", []):
        rlat = elem.get("lat")
        rlon = elem.get("lon")
        if rlat is None or rlon is None:
            continue
        tags = elem.get("tags", {})
        osm_type = (
            tags.get("tourism") or tags.get("amenity") or "shelter"
        )
        name = (
            tags.get("name")
            or tags.get("name:en")
            or tags.get("name:es")
            or f"Unnamed {osm_type.replace('_', ' ')} {elem.get('id', '?')}"
        )
        results.append({
            "name": name,
            "distance_km": round(_haversine_km(lat, lon, rlat, rlon), 2),
            "lat": rlat,
            "lon": rlon,
            "type": osm_type,
        })

    results.sort(key=lambda r: r["distance_km"])
    return results[:5]

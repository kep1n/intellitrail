"""Geospatial geometry utilities — polygon parsing and point-in-polygon testing.

Used by both ``weather_alerts`` (CAP alert zone matching) and ``database``
(mountain zone polygon containment checks).
"""

from __future__ import annotations


def parse_polygon(polygon_str: str, lon_lat: bool = False) -> list[tuple[float, float]]:
    """Parse a space-separated polygon string into a list of ``(lat, lon)`` tuples.

    Args:
        polygon_str: Space-separated coordinate pairs. Each pair is
            ``"lat,lon"`` by default, or ``"lon,lat"`` when *lon_lat* is *True*.
        lon_lat: Set to *True* if the input is in ``"lon,lat"`` order (e.g.
            GeoJSON). Default is *False* (CAP standard ``"lat,lon"``).

    Returns:
        List of ``(lat, lon)`` float tuples forming the polygon boundary.
    """
    return [
        (float(b), float(a)) if lon_lat else (float(a), float(b))
        for pair in polygon_str.strip().split()
        for a, b in [pair.split(",")]
    ]


def point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test.

    Casts a horizontal ray along the longitude axis and counts how many
    polygon edges it crosses. An odd count means the point is inside.

    Args:
        lat: Latitude of the test point (decimal degrees).
        lon: Longitude of the test point (decimal degrees).
        polygon: Ordered list of ``(lat, lon)`` vertices.

    Returns:
        *True* if the point is inside the polygon, *False* otherwise.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if (lat_i > lat) != (lat_j > lat):
            lon_intercept = lon_i + (lat - lat_i) * (lon_j - lon_i) / (lat_j - lat_i)
            if lon < lon_intercept:
                inside = not inside
        j = i
    return inside

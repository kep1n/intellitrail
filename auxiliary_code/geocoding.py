"""Geocoding and geospatial helpers — centroid computation and reverse geocoding.

Provides:
- :func:`get_centroid` — compute the centroid of a GPX or GeoJSON track file.
- :func:`localise_coordinates` — reverse-geocode a coordinate via OSM Nominatim.
"""

from __future__ import annotations

import geopandas as gpd
import httpx


def get_centroid(track_file: str) -> tuple[float, float]:
    """Compute the centroid of a GPX or GeoJSON track file.

    Args:
        track_file: Path to the track file (``*.gpx`` or ``*.geojson``).

    Returns:
        ``(longitude, latitude)`` of the centroid in WGS-84 decimal degrees.
    """
    layers = "tracks" if track_file.endswith(".gpx") else None
    gdf = gpd.read_file(track_file, layer=layers)
    centroid = gdf.to_crs(3857).centroid
    centroid_wgs = centroid.to_crs(4326)
    longitude, latitude = float(centroid_wgs.x[0]), float(centroid_wgs.y[0])
    return longitude, latitude


def localise_coordinates(
    longitude: float,
    latitude: float,
    zoom: int = 10,
    polygon_geojson: int = 0,
    fmt: str = "jsonv2",
) -> tuple[str, str, str] | None:
    """Reverse-geocode a coordinate pair using OSM Nominatim.

    Args:
        longitude: Longitude in decimal degrees.
        latitude: Latitude in decimal degrees.
        zoom: Nominatim zoom level (controls result granularity).
        polygon_geojson: Pass ``1`` to include the area polygon in the response.
        fmt: Nominatim response format. Defaults to ``"jsonv2"``.

    Returns:
        ``(municipality, province, address_type)`` tuple, or *None* on error.
    """
    base_url = "https://nominatim.openstreetmap.org/reverse"
    url = (
        f"{base_url}?lat={latitude}&lon={longitude}"
        f"&zoom={zoom}&polygon_geojson={polygon_geojson}&format={fmt}"
    )
    headers = {"User-Agent": "IronhackFinalProject/1.0 (sinozuke@gmail.com)"}
    response = httpx.get(url, headers=headers).json()
    if "error" in response:
        return None
    municipality = response["name"]
    province = response["address"]["province"]
    address_type = response["address_type"]
    return municipality, province, address_type

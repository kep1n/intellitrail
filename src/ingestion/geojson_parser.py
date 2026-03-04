"""GeoJSON parser — converts a Wikiloc GeoJSON dict into a ResolvedGeometry.

Entry points:
    geojson_centroid(geojson: dict) -> tuple[float, float]
    parse_geojson_to_geometry(geojson: dict, track_name: str | None = None) -> ResolvedGeometry

All elevation fields in the returned ResolvedGeometry are None because
Wikiloc GeoJSON tracks do not include elevation data.
"""

import math

import geopandas as gpd
from shapely.geometry import Point

from src.ingestion.errors import IngestionError
from src.models.geometry import ResolvedGeometry


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def geojson_centroid(geojson: dict) -> tuple[float, float]:
    """Extract the geographic centroid from a GeoJSON structure.

    Handles FeatureCollection, Feature, and bare geometry types.
    Uses CRS EPSG:4326 (WGS84 — GeoJSON default).

    Returns:
        (lat, lon) float tuple (note: y=lat, x=lon in GeoJSON convention).
    """
    geo_type = geojson.get("type")

    if geo_type == "FeatureCollection":
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
        centroid = gdf.geometry.union_all().centroid
    elif geo_type == "Feature":
        gdf = gpd.GeoDataFrame.from_features([geojson], crs="EPSG:4326")
        centroid = gdf.geometry.union_all().centroid
    else:
        # Bare geometry (Point, LineString, MultiLineString, etc.)
        from shapely.geometry import shape
        geom = shape(geojson)
        centroid = geom.centroid

    return float(centroid.y), float(centroid.x)  # (lat, lon)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_coordinates(geojson: dict) -> list[tuple[float, float]]:
    """Extract all coordinate pairs from a GeoJSON structure as (lat, lon) tuples.

    GeoJSON coordinates are in [lon, lat] order; this function converts them
    to (lat, lon) tuples for consistency with the rest of the codebase.

    Handles: FeatureCollection, Feature, LineString, MultiLineString.
    """
    geo_type = geojson.get("type")

    if geo_type == "FeatureCollection":
        coords: list[tuple[float, float]] = []
        for feature in geojson.get("features", []):
            coords.extend(_extract_coordinates(feature))
        return coords

    if geo_type == "Feature":
        geometry = geojson.get("geometry") or {}
        return _extract_coordinates(geometry)

    # Bare geometry
    geom_type = geo_type
    raw_coords = geojson.get("coordinates", [])

    if geom_type == "LineString":
        return [(float(pt[1]), float(pt[0])) for pt in raw_coords]

    if geom_type == "MultiLineString":
        result: list[tuple[float, float]] = []
        for line in raw_coords:
            result.extend((float(pt[1]), float(pt[0])) for pt in line)
        return result

    if geom_type == "Point":
        if raw_coords:
            return [(float(raw_coords[1]), float(raw_coords[0]))]
        return []

    # Fallback: try to flatten anything coordinate-shaped
    flat: list[tuple[float, float]] = []
    _flatten_coords(raw_coords, flat)
    return flat


def _flatten_coords(obj, result: list) -> None:
    """Recursively flatten nested coordinate arrays into (lat, lon) tuples."""
    if not obj:
        return
    if isinstance(obj[0], (int, float)):
        # Leaf node: [lon, lat, ?ele]
        if len(obj) >= 2:
            result.append((float(obj[1]), float(obj[0])))
        return
    for item in obj:
        _flatten_coords(item, result)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two WGS84 points in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_geojson_to_geometry(
    geojson: dict,
    track_name: str | None = None,
    trail_stats: dict | None = None,
) -> ResolvedGeometry:
    """Convert a Wikiloc GeoJSON dict into a ResolvedGeometry.

    Elevation fields are populated from trail_stats when provided (scraped from
    the <section id="trail-data"> section), otherwise set to None.

    Args:
        geojson:     A GeoJSON dict (FeatureCollection, Feature, or bare geometry).
        track_name:  Optional human-readable name for the track.
        trail_stats: Optional dict from _extract_trail_data_section with keys
                     elevation_gain_m, elevation_loss_m, min_elevation_m,
                     max_elevation_m, difficulty, trail_type.

    Returns:
        ResolvedGeometry with coordinates, distances, bbox, utm_crs, and
        elevation/difficulty/trail_type fields where trail_stats provides them.

    Raises:
        IngestionError: If the GeoJSON contains no coordinates (EMPTY_TRACK).
    """
    # 1. Extract coordinates
    coordinates = _extract_coordinates(geojson)

    # 2. Guard: empty track
    if not coordinates:
        raise IngestionError("GeoJSON contains no coordinates", "EMPTY_TRACK")

    # 3. Bounding box
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    bbox_min_lat = float(min(lats))
    bbox_max_lat = float(max(lats))
    bbox_min_lon = float(min(lons))
    bbox_max_lon = float(max(lons))

    # 4. 2D distance via haversine (no elevation from Wikiloc)
    distance_2d_km = 0.0
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        distance_2d_km += _haversine_km(lat1, lon1, lat2, lon2)

    # 5. 3D distance equals 2D (no elevation data)
    distance_3d_km = distance_2d_km

    # 6. Estimate UTM CRS from sample points
    sample_geom = [Point(lon, lat) for lat, lon in coordinates]
    gdf_sample = gpd.GeoDataFrame(geometry=sample_geom, crs="EPSG:4326")
    utm_crs_obj = gdf_sample.estimate_utm_crs()
    utm_crs = f"EPSG:{utm_crs_obj.to_epsg()}"

    # 7. Build ResolvedGeometry — elevation from trail_stats if available
    ts = trail_stats or {}
    return ResolvedGeometry(
        track_name=track_name,
        coordinates=coordinates,
        elevation_raw=None,
        elevation_smoothed=None,
        elevation_gain_m=ts.get("elevation_gain_m"),
        elevation_loss_m=ts.get("elevation_loss_m"),
        min_elevation_m=ts.get("min_elevation_m"),
        max_elevation_m=ts.get("max_elevation_m"),
        distance_2d_km=distance_2d_km,
        distance_3d_km=distance_3d_km,
        bbox_min_lat=bbox_min_lat,
        bbox_max_lat=bbox_max_lat,
        bbox_min_lon=bbox_min_lon,
        bbox_max_lon=bbox_max_lon,
        utm_crs=utm_crs,
        difficulty=ts.get("difficulty"),
        trail_type=ts.get("trail_type"),
        moving_time=ts.get("moving_time"),
    )

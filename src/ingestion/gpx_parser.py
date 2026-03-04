"""GPX parser — converts raw GPX bytes into a ResolvedGeometry.

Entry point: parse_gpx_bytes(data: bytes) -> ResolvedGeometry

Algorithm:
  1. Parse XML bytes with stdlib ElementTree
  2. Detect GPX namespace (GPX 1.0 uses no namespace; GPX 1.1 uses
     http://www.topografix.com/GPX/1/1)
  3. Collect all <trkpt> elements across ALL <trk>/<trkseg> blocks
     (document order = merge order)
  4. Raise IngestionError("EMPTY_TRACK") when zero trackpoints found
  5. Extract (lat, lon) coordinates and optional <ele> values
  6. Compute distances (2D and 3D) in UTM projection via GeoPandas
  7. Compute elevation stats + smoothing via NumPy
  8. Return ResolvedGeometry(**fields)
"""
import math
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.ingestion.errors import IngestionError
from src.models.geometry import ResolvedGeometry


def _detect_namespace(root: ET.Element) -> str:
    """Return the XML namespace URI from the root element tag, or '' if none."""
    if root.tag.startswith("{"):
        return root.tag.split("}")[0].lstrip("{")
    return ""


def _tag(ns: str, local: str) -> str:
    """Build a Clark-notation tag, e.g. '{http://...}trkpt'."""
    return f"{{{ns}}}{local}" if ns else local


def _smooth_elevation(raw: list[float]) -> list[float]:
    """Apply a 5-point centered moving average to raw elevation data."""
    arr = np.array(raw, dtype=float)
    smoothed = np.convolve(arr, np.ones(5) / 5, mode="same")
    return smoothed.tolist()


def _compute_distances(
    coords: list[tuple[float, float]],
    elevation_raw: Optional[list[float]],
    utm_crs: str,
) -> tuple[float, float]:
    """Compute 2D and 3D track distances in kilometres.

    Args:
        coords: list of (lat, lon) tuples in WGS84
        elevation_raw: optional list of elevation values (metres)
        utm_crs: EPSG string for UTM projection

    Returns:
        (distance_2d_km, distance_3d_km)
    """
    if len(coords) < 2:
        return 0.0, 0.0

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    geom = [Point(lon, lat) for lat, lon in coords]

    gdf = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")
    gdf_utm = gdf.to_crs(utm_crs)

    # Compute segment horizontal distances
    geom_shifted = gdf_utm.geometry.shift()
    horiz_dists = gdf_utm.geometry.distance(geom_shifted).fillna(0.0).tolist()

    distance_2d_m = sum(horiz_dists)
    distance_2d_km = distance_2d_m / 1000.0

    # 3D distance: sqrt(horiz² + elev_diff²) per segment
    if elevation_raw is not None:
        elev_diffs = [
            elevation_raw[i] - elevation_raw[i - 1]
            for i in range(1, len(elevation_raw))
        ]
    else:
        elev_diffs = [0.0] * (len(coords) - 1)

    distance_3d_m = 0.0
    # horiz_dists[0] is NaN (shifted from nothing); segments start at index 1
    for i in range(1, len(horiz_dists)):
        h = horiz_dists[i]
        e = elev_diffs[i - 1]
        distance_3d_m += math.sqrt(h * h + e * e)

    distance_3d_km = distance_3d_m / 1000.0
    return distance_2d_km, distance_3d_km


def parse_gpx_bytes(data: bytes) -> ResolvedGeometry:
    """Parse raw GPX bytes and return a ResolvedGeometry.

    Args:
        data: Raw bytes of a GPX file (UTF-8 or UTF-16 with BOM).

    Returns:
        ResolvedGeometry with coordinates, distances, elevation, bbox, UTM CRS.

    Raises:
        IngestionError: If the GPX contains zero trackpoints (EMPTY_TRACK) or
                        cannot be parsed (PARSE_ERROR).
    """
    # --- 1. Parse XML ---
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise IngestionError(
            f"GPX file could not be parsed: {exc}", "PARSE_ERROR"
        ) from exc

    # --- 2. Detect namespace ---
    ns = _detect_namespace(root)

    # --- 3. Collect all trackpoints (across all <trk> and <trkseg> elements) ---
    trkpt_tag = _tag(ns, "trkpt")
    ele_tag = _tag(ns, "ele")
    trk_tag = _tag(ns, "trk")
    trkseg_tag = _tag(ns, "trkseg")
    name_tag = _tag(ns, "name")

    # Extract track name from first <trk><name>
    track_name: Optional[str] = None
    first_trk = root.find(f".//{trk_tag}")
    if first_trk is not None:
        name_el = first_trk.find(name_tag)
        if name_el is not None and name_el.text:
            track_name = name_el.text.strip()

    # Collect all trkpt elements in document order
    trkpts = list(root.iter(trkpt_tag))

    # --- 4. Guard: empty track ---
    if not trkpts:
        raise IngestionError("GPX file contains no track data", "EMPTY_TRACK")

    # --- 5. Extract coordinates and elevation ---
    coordinates: list[tuple[float, float]] = []
    elevations: list[Optional[float]] = []
    has_elevation = True

    for trkpt in trkpts:
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        coordinates.append((lat, lon))

        ele_el = trkpt.find(ele_tag)
        if ele_el is not None and ele_el.text is not None:
            elevations.append(float(ele_el.text))
        else:
            has_elevation = False
            elevations.append(None)

    # Treat elevation as all-or-nothing: if any trkpt lacks <ele>, discard all
    elevation_raw: Optional[list[float]] = (
        [e for e in elevations if e is not None]  # type: ignore[misc]
        if has_elevation
        else None
    )

    # --- 6. Compute UTM CRS ---
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    sample_geom = [Point(lon, lat) for lat, lon in coordinates]
    gdf_sample = gpd.GeoDataFrame(geometry=sample_geom, crs="EPSG:4326")
    utm_crs_obj = gdf_sample.estimate_utm_crs()
    utm_crs = utm_crs_obj.to_epsg()
    utm_crs_str = f"EPSG:{utm_crs}"

    # --- 7. Compute distances ---
    distance_2d_km, distance_3d_km = _compute_distances(
        coordinates, elevation_raw, utm_crs_str
    )

    # --- 8. Compute elevation stats ---
    elevation_smoothed: Optional[list[float]] = None
    elevation_gain_m: Optional[float] = None
    elevation_loss_m: Optional[float] = None
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None

    if elevation_raw is not None:
        elevation_smoothed = _smooth_elevation(elevation_raw)

        # Gain/loss from smoothed values
        gain = 0.0
        loss = 0.0
        for i in range(1, len(elevation_smoothed)):
            diff = elevation_smoothed[i] - elevation_smoothed[i - 1]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)
        elevation_gain_m = gain
        elevation_loss_m = loss

        min_elevation_m = float(min(elevation_raw))
        max_elevation_m = float(max(elevation_raw))

    # --- 9. Compute bounding box ---
    bbox_min_lat = float(min(lats))
    bbox_max_lat = float(max(lats))
    bbox_min_lon = float(min(lons))
    bbox_max_lon = float(max(lons))

    # --- 10. Return ResolvedGeometry ---
    return ResolvedGeometry(
        track_name=track_name,
        coordinates=coordinates,
        elevation_raw=elevation_raw,
        elevation_smoothed=elevation_smoothed,
        elevation_gain_m=elevation_gain_m,
        elevation_loss_m=elevation_loss_m,
        min_elevation_m=min_elevation_m,
        max_elevation_m=max_elevation_m,
        distance_2d_km=distance_2d_km,
        distance_3d_km=distance_3d_km,
        bbox_min_lat=bbox_min_lat,
        bbox_max_lat=bbox_max_lat,
        bbox_min_lon=bbox_min_lon,
        bbox_max_lon=bbox_max_lon,
        utm_crs=utm_crs_str,
    )

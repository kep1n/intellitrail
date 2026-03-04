from typing import Optional
from pydantic import BaseModel


class ResolvedGeometry(BaseModel):
    """Parsed GPX track, normalized for use by all pipeline stages.

    Both 2D and 3D distances are always present. Elevation fields are Optional
    because some GPX sources (e.g., Wikiloc tracks) do not include <ele> tags.
    """

    # Track identity
    track_name: Optional[str] = None

    # Coordinates — list of (lat, lon) tuples, all segments merged
    coordinates: list[tuple[float, float]]

    # Elevation — both raw GPS readings and smoothed (5-point centered moving avg).
    # Both are None when source GPX has no <ele> tags (e.g., Wikiloc tracks).
    # Rationale: GPS elevation has ±5–15m noise. AEMET altitude-band analysis in
    # Phase 3 needs smoothed values for reliable zone mapping. Raw values preserved
    # for transparency.
    elevation_raw: Optional[list[float]] = None
    elevation_smoothed: Optional[list[float]] = None
    # Derived elevation stats (None when elevation_raw is None)
    elevation_gain_m: Optional[float] = None   # cumulative ascent in metres
    elevation_loss_m: Optional[float] = None   # cumulative descent in metres
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None


    # Distance
    distance_2d_km: float   # horizontal-only distance
    distance_3d_km: float   # accounts for elevation change

    # Bounding box (WGS84)
    bbox_min_lat: float
    bbox_max_lat: float
    bbox_min_lon: float
    bbox_max_lon: float

    # UTM zone used for spatial ops (e.g., "EPSG:32630")
    utm_crs: str

    # Wikiloc trail metadata from <section id="trail-data"> (None for GPX sources)
    difficulty: Optional[str] = None   # e.g. "Moderate", "Moderado", "Hard"
    trail_type: Optional[str] = None   # e.g. "Loop", "Circular", "One way"
    moving_time: Optional[str] = None  # e.g. "2 hours 16 minutes" from dl.more-data

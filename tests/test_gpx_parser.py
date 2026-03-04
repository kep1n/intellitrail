"""Tests for parse_gpx_bytes() — GPX parser for the mountain safety agent.

Covers:
  1. simple_track.gpx  — single track, elevation present
  2. no_elevation.gpx  — single track, no <ele> tags
  3. multi_segment.gpx — two <trk> elements, merged in document order
  4. empty_track.gpx   — zero trackpoints, must raise IngestionError
"""
import math
import pathlib
import pytest

from src.ingestion.gpx_parser import parse_gpx_bytes
from src.ingestion.errors import IngestionError
from src.models.geometry import ResolvedGeometry

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# ---------------------------------------------------------------------------
# Fixture 1: simple_track.gpx
# ---------------------------------------------------------------------------

class TestSimpleTrack:
    def setup_method(self):
        self.result = parse_gpx_bytes(_load("simple_track.gpx"))

    def test_returns_resolved_geometry(self):
        assert isinstance(self.result, ResolvedGeometry)

    def test_coordinate_count(self):
        # 5 trackpoints in simple_track.gpx
        assert len(self.result.coordinates) == 5

    def test_coordinates_are_lat_lon_tuples(self):
        for lat, lon in self.result.coordinates:
            assert isinstance(lat, float)
            assert isinstance(lon, float)

    def test_elevation_raw_present_and_correct_length(self):
        assert self.result.elevation_raw is not None
        assert len(self.result.elevation_raw) == 5

    def test_elevation_raw_values(self):
        # simple_track.gpx has ele: 1000, 1010, 1005, 1020, 1015
        assert self.result.elevation_raw[0] == pytest.approx(1000.0)
        assert self.result.elevation_raw[1] == pytest.approx(1010.0)

    def test_elevation_smoothed_present(self):
        assert self.result.elevation_smoothed is not None
        assert len(self.result.elevation_smoothed) == 5

    def test_elevation_smoothed_is_list_of_floats(self):
        for v in self.result.elevation_smoothed:
            assert isinstance(v, float)

    def test_distance_2d_positive(self):
        assert self.result.distance_2d_km > 0.0

    def test_distance_3d_positive(self):
        assert self.result.distance_3d_km > 0.0

    def test_distances_are_floats(self):
        assert isinstance(self.result.distance_2d_km, float)
        assert isinstance(self.result.distance_3d_km, float)

    def test_bbox_min_lat(self):
        assert self.result.bbox_min_lat == pytest.approx(42.000)

    def test_bbox_max_lat(self):
        assert self.result.bbox_max_lat == pytest.approx(42.004)

    def test_bbox_min_lon(self):
        assert self.result.bbox_min_lon == pytest.approx(-3.004)

    def test_bbox_max_lon(self):
        assert self.result.bbox_max_lon == pytest.approx(-3.000)

    def test_utm_crs_is_epsg_string(self):
        assert self.result.utm_crs.startswith("EPSG:")

    def test_track_name(self):
        assert self.result.track_name == "Test Track"

    def test_elevation_gain_populated(self):
        # Track goes up from 1000 → 1010 → 1005 → 1020 → 1015
        assert self.result.elevation_gain_m is not None
        assert self.result.elevation_gain_m >= 0.0

    def test_elevation_loss_populated(self):
        assert self.result.elevation_loss_m is not None
        assert self.result.elevation_loss_m >= 0.0

    def test_min_max_elevation(self):
        assert self.result.min_elevation_m is not None
        assert self.result.max_elevation_m is not None
        assert self.result.min_elevation_m <= self.result.max_elevation_m


# ---------------------------------------------------------------------------
# Fixture 2: no_elevation.gpx
# ---------------------------------------------------------------------------

class TestNoElevation:
    def setup_method(self):
        self.result = parse_gpx_bytes(_load("no_elevation.gpx"))

    def test_returns_resolved_geometry(self):
        assert isinstance(self.result, ResolvedGeometry)

    def test_elevation_raw_is_none(self):
        assert self.result.elevation_raw is None

    def test_elevation_smoothed_is_none(self):
        assert self.result.elevation_smoothed is None

    def test_elevation_gain_is_none(self):
        assert self.result.elevation_gain_m is None

    def test_elevation_loss_is_none(self):
        assert self.result.elevation_loss_m is None

    def test_distance_2d_still_populated(self):
        assert self.result.distance_2d_km > 0.0

    def test_distance_3d_still_populated(self):
        assert self.result.distance_3d_km > 0.0

    def test_coordinate_count(self):
        assert len(self.result.coordinates) == 5

    def test_bbox_populated(self):
        assert self.result.bbox_min_lat is not None
        assert self.result.bbox_max_lat is not None


# ---------------------------------------------------------------------------
# Fixture 3: multi_segment.gpx
# ---------------------------------------------------------------------------

class TestMultiSegment:
    def setup_method(self):
        self.result = parse_gpx_bytes(_load("multi_segment.gpx"))

    def test_returns_resolved_geometry(self):
        assert isinstance(self.result, ResolvedGeometry)

    def test_coordinate_count_is_sum_of_all_tracks(self):
        # Two <trk> elements, 3 trkpt each = 6 total
        assert len(self.result.coordinates) == 6

    def test_elevation_raw_covers_all_tracks(self):
        assert self.result.elevation_raw is not None
        assert len(self.result.elevation_raw) == 6

    def test_first_coordinate_from_first_track(self):
        lat, lon = self.result.coordinates[0]
        assert lat == pytest.approx(42.000)
        assert lon == pytest.approx(-3.000)

    def test_last_coordinate_from_second_track(self):
        lat, lon = self.result.coordinates[-1]
        assert lat == pytest.approx(42.012)
        assert lon == pytest.approx(-3.012)


# ---------------------------------------------------------------------------
# Fixture 4: empty_track.gpx
# ---------------------------------------------------------------------------

class TestEmptyTrack:
    def test_raises_ingestion_error(self):
        with pytest.raises(IngestionError) as exc_info:
            parse_gpx_bytes(_load("empty_track.gpx"))
        err = exc_info.value
        assert err.error_code == "EMPTY_TRACK"

    def test_error_message_describes_issue(self):
        with pytest.raises(IngestionError) as exc_info:
            parse_gpx_bytes(_load("empty_track.gpx"))
        assert "no track data" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Smoothing variance test
# ---------------------------------------------------------------------------

class TestSmoothingVariance:
    """Verify elevation_smoothed differs from elevation_raw when variance exists."""

    def test_smoothed_differs_from_raw_for_varied_elevation(self):
        # Use multi_segment which spans different elevation values
        result = parse_gpx_bytes(_load("multi_segment.gpx"))
        assert result.elevation_raw is not None
        assert result.elevation_smoothed is not None
        # For a 6-point track with variance, smoothed should not equal raw exactly
        # (convolution with window=5 will blend values)
        # At minimum, confirm lengths match
        assert len(result.elevation_smoothed) == len(result.elevation_raw)
        # At least one value differs (except for perfectly flat tracks)
        raw = result.elevation_raw
        smoothed = result.elevation_smoothed
        # The tracks span 100m+ variance so smoothed middle values will differ
        diffs = [abs(s - r) for s, r in zip(smoothed, raw)]
        assert max(diffs) > 0.0 or all(r == raw[0] for r in raw), (
            "Smoothed should differ from raw when elevation varies"
        )

"""Tests for ZoneMapper — coordinate-to-zone resolution.

Covered behaviors:
  - mountain_zone(): returns correct area code for 7 known zone midpoints
  - mountain_zone(): returns None for coordinates outside all zones (2 cases)
  - nivologica_zone(): correct codes for cat1/arn1/nav1/peu1/arn2 and None for others
  - is_avalanche_season(): tested for months 1,2,3,4,5 (True) and 6-11 (False), 12 (True)
  - municipality_zone(): uses Nominatim + static INE lookup (not broken AEMET endpoint)
    - success: Nominatim returns "Jaca" → ("22130", "Jaca")
    - Nominatim network failure → WeatherError
    - Nominatim returns unknown municipality → WeatherError (not in INE table)

All tests use mocked httpx.get — no live HTTP calls.
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch, Mock

from src.weather.client import WeatherError
from src.weather.zone_mapper import ZoneMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Mock AEMETClient — no real HTTP calls."""
    return MagicMock()


@pytest.fixture
def mapper(mock_client):
    """ZoneMapper backed by a mock client. Also resets class-level cache."""
    # Reset the class-level municipios cache before each test so tests are isolated
    ZoneMapper._MUNICIPIOS_CACHE = None
    return ZoneMapper(mock_client)


# ---------------------------------------------------------------------------
# mountain_zone() — known coordinates per zone
# ---------------------------------------------------------------------------

class TestMountainZone:
    def test_pirineo_aragones_arn1(self, mapper):
        """(42.5, 0.5) should match arn1 or peu1 — both valid (overlap zone)."""
        result = mapper.mountain_zone(42.5, 0.5)
        assert result in ("arn1", "peu1"), f"Expected arn1 or peu1, got {result!r}"

    def test_pirineo_catalan_cat1(self, mapper):
        """(42.5, 2.0) → cat1 (Pirineo Catalán)."""
        result = mapper.mountain_zone(42.5, 2.0)
        assert result == "cat1", f"Expected cat1, got {result!r}"

    def test_pirineo_navarro_nav1(self, mapper):
        """(42.9, -1.5) → nav1 (Pirineo Navarro)."""
        result = mapper.mountain_zone(42.9, -1.5)
        assert result == "nav1", f"Expected nav1, got {result!r}"

    def test_sierra_nevada_nev1(self, mapper):
        """(36.95, -3.3) → nev1 (Sierra Nevada)."""
        result = mapper.mountain_zone(36.95, -3.3)
        assert result == "nev1", f"Expected nev1, got {result!r}"

    def test_sistema_central_mad2(self, mapper):
        """(40.75, -4.0) → mad2 (Sistema Central)."""
        result = mapper.mountain_zone(40.75, -4.0)
        assert result == "mad2", f"Expected mad2, got {result!r}"

    def test_gredos_gre1(self, mapper):
        """(40.3, -5.1) → gre1 (Sierra de Gredos)."""
        result = mapper.mountain_zone(40.3, -5.1)
        assert result == "gre1", f"Expected gre1, got {result!r}"

    def test_sistema_iberico_rio1(self, mapper):
        """(42.0, -3.0) → rio1 (Sistema Ibérico Riojano)."""
        result = mapper.mountain_zone(42.0, -3.0)
        assert result == "rio1", f"Expected rio1, got {result!r}"

    def test_canary_islands_outside_all_zones(self, mapper):
        """(28.0, -15.0) → None (Canary Islands — no mainland mountain zones)."""
        result = mapper.mountain_zone(28.0, -15.0)
        assert result is None

    def test_london_outside_all_zones(self, mapper):
        """(51.5, -0.1) → None (London — outside all zones)."""
        result = mapper.mountain_zone(51.5, -0.1)
        assert result is None


# ---------------------------------------------------------------------------
# nivologica_zone() — mapping mountain codes to bulletin zones
# ---------------------------------------------------------------------------

class TestNivologicaZone:
    def test_cat1_maps_to_zone_0(self, mapper):
        assert mapper.nivologica_zone("cat1") == "0"

    def test_peu1_maps_to_zone_0(self, mapper):
        assert mapper.nivologica_zone("peu1") == "0"

    def test_arn1_maps_to_zone_1(self, mapper):
        assert mapper.nivologica_zone("arn1") == "1"

    def test_nav1_maps_to_zone_1(self, mapper):
        assert mapper.nivologica_zone("nav1") == "1"

    def test_arn2_returns_none(self, mapper):
        """arn2 (Sistema Ibérico Aragonés) is not a Pyrenean snow zone."""
        assert mapper.nivologica_zone("arn2") is None

    def test_nev1_returns_none(self, mapper):
        assert mapper.nivologica_zone("nev1") is None

    def test_mad2_returns_none(self, mapper):
        assert mapper.nivologica_zone("mad2") is None

    def test_gre1_returns_none(self, mapper):
        assert mapper.nivologica_zone("gre1") is None

    def test_rio1_returns_none(self, mapper):
        assert mapper.nivologica_zone("rio1") is None

    def test_unknown_code_returns_none(self, mapper):
        assert mapper.nivologica_zone("xyz9") is None


# ---------------------------------------------------------------------------
# is_avalanche_season() — month-based seasonal gate
# ---------------------------------------------------------------------------

class TestIsAvalancheSeason:
    @pytest.mark.parametrize("month", [1, 2, 3, 4, 5, 12])
    def test_winter_spring_months_are_in_season(self, mapper, month):
        """Months 1-5 and 12 are avalanche season."""
        with patch("src.weather.zone_mapper.date") as mock_date:
            mock_date.today.return_value = date(2026, month, 15)
            assert mapper.is_avalanche_season() is True

    @pytest.mark.parametrize("month", [6, 7, 8, 9, 10, 11])
    def test_summer_autumn_months_are_out_of_season(self, mapper, month):
        """Months 6-11 are outside avalanche season."""
        with patch("src.weather.zone_mapper.date") as mock_date:
            mock_date.today.return_value = date(2026, month, 15)
            assert mapper.is_avalanche_season() is False


# ---------------------------------------------------------------------------
# municipality_zone() — Nominatim + static INE lookup
# ---------------------------------------------------------------------------

def _make_nominatim_response(city: str | None = None, municipality: str | None = None,
                              town: str | None = None, village: str | None = None) -> Mock:
    """Build a mock httpx response that mimics a Nominatim /reverse JSON reply."""
    address: dict = {}
    if municipality is not None:
        address["municipality"] = municipality
    if city is not None:
        address["city"] = city
    if town is not None:
        address["town"] = town
    if village is not None:
        address["village"] = village

    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()  # no-op
    mock_resp.json.return_value = {
        "place_id": 12345,
        "lat": "42.5661",
        "lon": "-0.5504",
        "address": address,
    }
    return mock_resp


class TestMunicipalityZone:
    def test_municipality_zone_nominatim_success_jaca(self, mapper):
        """Nominatim returns city='Jaca' → exact match in INE table → ('22130', 'Jaca')."""
        mock_resp = _make_nominatim_response(city="Jaca")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp) as mock_get:
            ine_code, name = mapper.municipality_zone(42.5661, -0.5504)

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        # Verify Nominatim URL was called
        assert "nominatim.openstreetmap.org/reverse" in call_kwargs.args[0]
        # Verify correct User-Agent (ToS compliance)
        assert "User-Agent" in call_kwargs.kwargs.get("headers", {})

        assert ine_code == "22130"
        assert name == "Jaca"

    def test_municipality_zone_nominatim_success_madrid(self, mapper):
        """Nominatim returns city='Madrid' → ('28079', 'Madrid')."""
        mock_resp = _make_nominatim_response(city="Madrid")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            ine_code, name = mapper.municipality_zone(40.4168, -3.7038)

        assert ine_code == "28079"
        assert name == "Madrid"

    def test_municipality_zone_nominatim_success_diacritic(self, mapper):
        """Nominatim returns city with diacritics — matched via normalisation."""
        # Nominatim may return "Benasque" — test exact match with diacritic stripping
        mock_resp = _make_nominatim_response(city="Benasque")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            ine_code, name = mapper.municipality_zone(42.6048, 0.5207)

        assert ine_code == "22054"
        assert name == "Benasque"

    def test_municipality_zone_uses_municipality_key_first(self, mapper):
        """address.municipality is preferred over address.city (Nominatim key priority)."""
        # Nominatim can return both 'municipality' and 'city'; 'municipality' wins
        mock_resp = _make_nominatim_response(municipality="Jaca", city="SomeOtherCity")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            ine_code, name = mapper.municipality_zone(42.5661, -0.5504)

        assert ine_code == "22130"
        assert name == "Jaca"

    def test_municipality_zone_falls_back_to_town(self, mapper):
        """If city is absent, falls back to address.town."""
        mock_resp = _make_nominatim_response(town="Canfranc")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            ine_code, name = mapper.municipality_zone(42.745, -0.515)

        assert ine_code == "22078"
        assert name == "Canfranc"

    def test_municipality_zone_nominatim_connect_error(self, mapper):
        """httpx.ConnectError during Nominatim call → WeatherError raised."""
        import httpx as httpx_module

        with patch(
            "src.weather.zone_mapper.httpx.get",
            side_effect=httpx_module.ConnectError("connection refused"),
        ):
            with pytest.raises(WeatherError) as exc_info:
                mapper.municipality_zone(42.5661, -0.5504)

        assert exc_info.value.forces_caution is True
        assert "Nominatim" in str(exc_info.value)

    def test_municipality_zone_nominatim_http_status_error(self, mapper):
        """httpx.HTTPStatusError (e.g. 429 rate limit) → WeatherError raised."""
        import httpx as httpx_module

        mock_resp = Mock()
        mock_resp.raise_for_status.side_effect = httpx_module.HTTPStatusError(
            "429 Too Many Requests",
            request=Mock(),
            response=Mock(),
        )

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            with pytest.raises(WeatherError) as exc_info:
                mapper.municipality_zone(42.5661, -0.5504)

        assert exc_info.value.forces_caution is True

    def test_municipality_zone_no_address_keys(self, mapper):
        """Nominatim returns empty address dict → WeatherError (no municipality name)."""
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        mock_resp.json.return_value = {"address": {}}

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            with pytest.raises(WeatherError) as exc_info:
                mapper.municipality_zone(0.0, 0.0)

        assert exc_info.value.forces_caution is True
        assert "no municipality" in str(exc_info.value).lower()

    def test_municipality_zone_not_in_ine_table(self, mapper):
        """Nominatim returns a valid name not in INE table → WeatherError."""
        mock_resp = _make_nominatim_response(city="XyzUnknownPlace99")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            with pytest.raises(WeatherError) as exc_info:
                mapper.municipality_zone(1.0, 1.0)

        assert exc_info.value.forces_caution is True
        assert "XyzUnknownPlace99" in str(exc_info.value)

    def test_municipality_zone_substring_match_fallback(self, mapper):
        """Nominatim returns 'Torla' which substring-matches 'Torla-Ordesa' in INE table."""
        # "torla" is a substring of "torla-ordesa" (after normalisation)
        mock_resp = _make_nominatim_response(city="Torla")

        with patch("src.weather.zone_mapper.httpx.get", return_value=mock_resp):
            ine_code, name = mapper.municipality_zone(42.621, -0.082)

        assert ine_code == "22230"
        assert name == "Torla-Ordesa"

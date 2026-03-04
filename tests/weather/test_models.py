"""Tests for Pydantic weather data models.

Covered behaviors:
  - WeatherData() defaults: municipal=None, mountain=None, avalanche=None,
                             missing_mountain_data=False, data_complete=True,
                             incomplete_reason=None
  - WeatherData with data_complete=False and incomplete_reason set
  - AvalancheBulletin with risk_level set (unavailable_reason=None)
  - AvalancheBulletin with unavailable_reason set (risk_level=None)
  - MunicipalForecast with periods list accessible as list[PeriodForecast]
"""

import pytest

from src.weather.models import (
    AvalancheBulletin,
    MunicipalForecast,
    MountainForecast,
    PeriodForecast,
    WeatherData,
)


# ---------------------------------------------------------------------------
# WeatherData defaults
# ---------------------------------------------------------------------------

class TestWeatherDataDefaults:
    def test_all_optional_fields_default_to_none(self):
        """WeatherData() with no args has None for all optional fields."""
        data = WeatherData()
        assert data.municipal is None
        assert data.mountain is None
        assert data.avalanche is None
        assert data.incomplete_reason is None

    def test_missing_mountain_data_defaults_to_false(self):
        """missing_mountain_data defaults to False."""
        data = WeatherData()
        assert data.missing_mountain_data is False

    def test_data_complete_defaults_to_true(self):
        """data_complete defaults to True."""
        data = WeatherData()
        assert data.data_complete is True


class TestWeatherDataIncomplete:
    def test_data_complete_false_with_reason(self):
        """data_complete=False and incomplete_reason can be set together."""
        data = WeatherData(data_complete=False, incomplete_reason="weather data unavailable")
        assert data.data_complete is False
        assert data.incomplete_reason == "weather data unavailable"

    def test_missing_mountain_data_can_be_set_independently(self):
        """missing_mountain_data=True does not affect data_complete."""
        data = WeatherData(missing_mountain_data=True)
        assert data.missing_mountain_data is True
        assert data.data_complete is True  # still complete


# ---------------------------------------------------------------------------
# AvalancheBulletin
# ---------------------------------------------------------------------------

class TestAvalancheBulletin:
    def test_bulletin_with_risk_level(self):
        """Bulletin with risk_level=3 has unavailable_reason=None."""
        bulletin = AvalancheBulletin(zone_code="0", raw_text="Riesgo considerable", risk_level=3)
        assert bulletin.zone_code == "0"
        assert bulletin.risk_level == 3
        assert bulletin.unavailable_reason is None

    def test_bulletin_with_unavailable_reason(self):
        """Bulletin with unavailable_reason has risk_level=None."""
        bulletin = AvalancheBulletin(
            zone_code="0",
            raw_text="",
            unavailable_reason="out_of_season",
        )
        assert bulletin.risk_level is None
        assert bulletin.unavailable_reason == "out_of_season"

    def test_bulletin_zone_1_accepted(self):
        """zone_code='1' is valid."""
        bulletin = AvalancheBulletin(zone_code="1", raw_text="Low risk", risk_level=1)
        assert bulletin.zone_code == "1"
        assert bulletin.risk_level == 1

    def test_bulletin_fetch_failed_reason(self):
        """unavailable_reason='fetch_failed' is a valid state."""
        bulletin = AvalancheBulletin(
            zone_code="0",
            raw_text="",
            unavailable_reason="fetch_failed",
        )
        assert bulletin.unavailable_reason == "fetch_failed"

    def test_bulletin_zone_not_covered_reason(self):
        """unavailable_reason='zone_not_covered' is a valid state."""
        bulletin = AvalancheBulletin(
            zone_code="0",
            raw_text="",
            unavailable_reason="zone_not_covered",
        )
        assert bulletin.unavailable_reason == "zone_not_covered"


# ---------------------------------------------------------------------------
# MunicipalForecast with periods list
# ---------------------------------------------------------------------------

class TestMunicipalForecast:
    def test_periods_accessible_as_list(self):
        """MunicipalForecast.periods is a list of PeriodForecast instances."""
        period1 = PeriodForecast(period="00-12", sky_state="despejado", wind_speed_kmh=20)
        period2 = PeriodForecast(period="12-24", sky_state="nuboso", precipitation_probability=40)

        forecast = MunicipalForecast(
            municipality_id="28079",
            municipality_name="Madrid",
            date="2026-02-21",
            periods=[period1, period2],
        )

        assert isinstance(forecast.periods, list)
        assert len(forecast.periods) == 2
        assert forecast.periods[0].period == "00-12"
        assert forecast.periods[0].sky_state == "despejado"
        assert forecast.periods[1].precipitation_probability == 40

    def test_period_forecast_optional_fields_are_none_by_default(self):
        """PeriodForecast optional fields default to None."""
        period = PeriodForecast(period="00-24")
        assert period.sky_state is None
        assert period.precipitation_probability is None
        assert period.precipitation_amount_mm is None
        assert period.wind_direction is None
        assert period.wind_speed_kmh is None
        assert period.temperature_min_c is None
        assert period.temperature_max_c is None


# ---------------------------------------------------------------------------
# WeatherData with full nested models
# ---------------------------------------------------------------------------

class TestWeatherDataWithNestedModels:
    def test_weather_data_accepts_avalanche_bulletin(self):
        """WeatherData.avalanche can hold an AvalancheBulletin."""
        bulletin = AvalancheBulletin(zone_code="1", raw_text="Moderate risk", risk_level=2)
        data = WeatherData(avalanche=bulletin)
        assert data.avalanche is not None
        assert data.avalanche.risk_level == 2

    def test_weather_data_accepts_mountain_forecast(self):
        """WeatherData.mountain can hold a MountainForecast."""
        mountain = MountainForecast(
            area_code="mad2",
            area_name="Sistema Central",
            forecast_text="Vientos moderados en cotas altas.",
        )
        data = WeatherData(mountain=mountain)
        assert data.mountain is not None
        assert data.mountain.area_code == "mad2"

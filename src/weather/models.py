"""Pydantic V2 data models for AEMET weather integration.

These models form the data contract consumed by Phase 3 (graph wiring).
All Optional fields have explicit = None defaults as required by Pydantic V2.
"""

from typing import Optional

from pydantic import BaseModel


class PeriodForecast(BaseModel):
    """One morning/afternoon/night forecast bucket.

    The 'period' field holds an AEMET period string such as "00-24", "00-12", or "12-24".
    All weather variables are optional because AEMET omits fields when data is unavailable.
    """

    period: str
    sky_state: Optional[str] = None
    precipitation_probability: Optional[int] = None  # 0-100
    precipitation_amount_mm: Optional[float] = None
    wind_direction: Optional[str] = None
    wind_speed_kmh: Optional[int] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None


class MunicipalForecast(BaseModel):
    """Forecast for one day at the nearest municipality.

    municipality_id is the 5-digit INE code (e.g., "28079" for Madrid).
    date is an ISO date string "YYYY-MM-DD".
    periods holds one entry per AEMET forecast period (morning/afternoon/night).
    """

    municipality_id: str  # 5-digit INE code
    municipality_name: str
    date: str  # ISO date "YYYY-MM-DD"
    periods: list[PeriodForecast]


class MountainForecast(BaseModel):
    """Raw mountain zone forecast narrative returned by AEMET.

    AEMET returns a text narrative for mountain zones — it is NOT parsed into
    structured altitude bands. The raw text is stored as-is for display.
    area_code corresponds to one of: arn1, cat1, nav1, peu1, nev1, mad2, gre1, arn2, rio1.
    """

    area_code: str  # e.g., "mad2"
    area_name: str
    forecast_text: str  # raw narrative from AEMET — NOT parsed


class AvalancheBulletin(BaseModel):
    """Snow/avalanche bulletin from AEMET direct XML (Pyrenean zones only).

    Only fetched when the track centroid falls inside cat1 (Pirineo Catalán),
    arn1 (Pirineo Aragonés), or nav1 (Pirineo Navarro). All other mountain
    zones leave avalanche=None — snow reasoning is not applicable there.

    zone_code holds the AEMET mountain area code ('cat1', 'arn1', 'nav1').
    raw_text holds the full forecast narrative from <TXT_PREDICCION>.
    risk_level is the European 1–5 avalanche risk scale parsed from the text;
    it may be None if no parseable level was found.

    unavailable_reason values:
      - 'out_of_season': outside avalanche season (June–November)
      - 'fetch_failed': network or parse error when fetching the XML
    """

    zone_code: str  # 'cat1' or 'arn1'
    risk_level: Optional[int] = None  # reserved; not populated by XML bulletin
    raw_text: str  # full TXT_PREDICCION narrative from AEMET XML
    unavailable_reason: Optional[str] = None  # 'out_of_season' | 'fetch_failed'


class WeatherData(BaseModel):
    """Top-level weather data contract consumed by Phase 3 LangGraph routing.

    municipal holds today + tomorrow forecasts for the nearest municipality.
    mountain holds the raw mountain zone narrative (None if centroid outside all zones).
    avalanche holds the avalanche bulletin (None if out of season or zone not covered).

    data_complete=False triggers the CAUTION gate in Phase 3 routing.
    missing_mountain_data=True means municipal succeeded but mountain fetch failed —
    this is non-fatal and does not set data_complete=False by itself.
    """

    municipal: Optional[list[MunicipalForecast]] = None  # today + tomorrow
    mountain: Optional[MountainForecast] = None
    avalanche: Optional[AvalancheBulletin] = None
    alerts: list[dict] = []  # Active AEMET CAP weather alerts for this location
    missing_mountain_data: bool = False  # True when municipal succeeded, mountain failed
    data_complete: bool = True  # False triggers CAUTION gate in Phase 3
    incomplete_reason: Optional[str] = None  # set when data_complete=False

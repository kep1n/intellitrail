"""AEMET municipal daily forecast normalisation.

Parses the JSON returned by the AEMET ``/prediccion/especifica/municipio``
endpoint and converts each day's raw dict into a typed :class:`DayReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Preferred period granularity order: 6h > 12h > 24h
_PERIOD_PRIORITY = ["00-06", "06-12", "12-18", "18-24", "00-12", "12-24", "00-24"]

_WIND_DIR = {
    "C": "calm", "N": "N", "NE": "NE", "E": "E", "SE": "SE",
    "S": "S", "SO": "SW", "O": "W", "NO": "NW",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HourlyMeasurement:
    """A single hourly reading.

    Attributes:
        hour: Hour of day (0–23).
        value: Measured value (unit depends on context, e.g. °C or %).
    """

    hour: int
    value: float


@dataclass
class RangeWithHourly:
    """Daily min/max range with an optional hourly breakdown.

    Attributes:
        min: Minimum daily value, or *None* if unavailable.
        max: Maximum daily value, or *None* if unavailable.
        hourly: Ordered list of hourly readings throughout the day.
    """

    min: float | None
    max: float | None
    hourly: list[HourlyMeasurement] = field(default_factory=list)


@dataclass
class SkyEntry:
    """Sky condition description for a forecast period.

    Attributes:
        period: Period string, e.g. ``"00-06"``.
        description: Human-readable sky description.
    """

    period: str
    description: str


@dataclass
class PrecipEntry:
    """Precipitation probability for a forecast period.

    Attributes:
        period: Period string, e.g. ``"00-12"``.
        value_pct: Probability as an integer percentage (or a string if the
            API returns a non-numeric sentinel).
    """

    period: str
    value_pct: int | str


@dataclass
class WindEntry:
    """Wind speed and direction for a forecast period.

    Attributes:
        period: Period string, e.g. ``"00-24"``.
        direction: Compass direction in English (e.g. ``"NW"``), or
            ``"calm"`` when the wind speed is zero.
        speed_kmh: Wind speed in km/h.
    """

    period: str
    direction: str
    speed_kmh: int


@dataclass
class DayReport:
    """Normalised weather report for a single forecast day.

    Attributes:
        date: Human-readable date, e.g. ``"Wednesday, 26 February 2026"``.
        sky: Ordered sky-condition entries from highest to lowest granularity.
        precipitation_prob: Precipitation probability entries.
        temperature: Min/max temperature with optional hourly breakdown (°C).
        feels_like: Min/max apparent temperature (°C); no hourly data.
        humidity: Min/max relative humidity with optional hourly breakdown (%).
        wind: Wind entries per forecast period.
        uv_index: Maximum UV index, or *None* if unavailable.
    """

    date: str
    sky: list[SkyEntry]
    precipitation_prob: list[PrecipEntry]
    temperature: RangeWithHourly
    feels_like: RangeWithHourly
    humidity: RangeWithHourly
    wind: list[WindEntry]
    uv_index: Any | None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _by_period(entries: list[dict]) -> dict:
    """Index a list of period-keyed entries by their ``'periodo'`` value.

    Args:
        entries: List of dicts, each potentially containing a ``"periodo"`` key.

    Returns:
        Dict mapping period strings to their corresponding entry dicts.
    """
    return {e["periodo"]: e for e in entries if "periodo" in e}


def _best_periods(indexed: dict) -> list[tuple[str, dict]]:
    """Return entries ordered by preferred granularity, skipping duplicates.

    Args:
        indexed: Dict of period string → entry dict, as returned by
            :func:`_by_period`.

    Returns:
        Ordered list of ``(period, entry)`` tuples from finest to coarsest
        granularity, with duplicate time ranges omitted.
    """
    seen_ranges: set[tuple[str, str]] = set()
    result = []
    for period in _PERIOD_PRIORITY:
        entry = indexed.get(period)
        if entry is None:
            continue
        # Skip if a finer-grained period already covers this range
        start, end = period.split("-")
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        result.append((period, entry))
    return result


def _sky_summary(estadoCielo: list[dict]) -> str:
    """Format sky condition as a human-readable summary string.

    Args:
        estadoCielo: Raw ``estadoCielo`` list from the AEMET API response.

    Returns:
        Semicolon-separated period descriptions, or ``"No data"``.
    """
    indexed = _by_period(estadoCielo)
    parts = []
    for period, entry in _best_periods(indexed):
        desc = entry.get("descripcion", "").strip()
        if desc:
            parts.append(f"{period}h: {desc}")
    return "; ".join(parts) if parts else "No data"


def _precip_summary(probPrecipitacion: list[dict]) -> str:
    """Format precipitation probability as a human-readable summary string.

    Args:
        probPrecipitacion: Raw ``probPrecipitacion`` list from the AEMET API response.

    Returns:
        Semicolon-separated period probabilities, or ``"No data"``.
    """
    indexed = _by_period(probPrecipitacion)
    parts = []
    for period, entry in _best_periods(indexed):
        v = entry.get("value", "")
        if v != "":
            parts.append(f"{period}h: {v}%")
    return "; ".join(parts) if parts else "No data"


def _wind_summary(viento: list[dict]) -> str:
    """Format wind data as a human-readable summary string.

    Args:
        viento: Raw ``viento`` list from the AEMET API response.

    Returns:
        Semicolon-separated period wind descriptions, or ``"No data"``.
    """
    indexed = _by_period(viento)
    parts = []
    for period, entry in _best_periods(indexed):
        dir_raw = entry.get("direccion", "")
        speed = entry.get("velocidad", 0)
        dir_str = _WIND_DIR.get(dir_raw, dir_raw)
        if dir_str == "calm" or speed == 0:
            parts.append(f"{period}h: calm")
        elif dir_str:
            parts.append(f"{period}h: {dir_str} {speed} km/h")
    return "; ".join(parts) if parts else "No data"


def _temp_summary(temperatura: dict) -> str:
    """Format temperature data as a human-readable summary string.

    Args:
        temperatura: Raw ``temperatura`` dict from the AEMET API response.

    Returns:
        String in the form ``"min X°C / max Y°C (hourly breakdown...)"``.
    """
    mx, mn = temperatura.get("maxima"), temperatura.get("minima")
    dato = temperatura.get("dato", [])
    base = f"min {mn}°C / max {mx}°C"
    if dato:
        hourly = ", ".join(f"{d['hora']:02d}h={d['value']}°C" for d in dato)
        return f"{base} ({hourly})"
    return base


def _humidity_summary(humedadRelativa: dict) -> str:
    """Format relative humidity as a human-readable summary string.

    Args:
        humedadRelativa: Raw ``humedadRelativa`` dict from the AEMET API response.

    Returns:
        String in the form ``"min X% / max Y% (hourly breakdown...)"``.
    """
    mx, mn = humedadRelativa.get("maxima"), humedadRelativa.get("minima")
    dato = humedadRelativa.get("dato", [])
    base = f"min {mn}% / max {mx}%"
    if dato:
        hourly = ", ".join(f"{d['hora']:02d}h={d['value']}%" for d in dato)
        return f"{base} ({hourly})"
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_day_report(day: dict) -> DayReport:
    """Parse a single day's forecast dict from the AEMET API into a :class:`DayReport`.

    Args:
        day: A single element from ``prediccion.dia`` in the AEMET municipality
            forecast response.

    Returns:
        A typed :class:`DayReport` instance.
    """
    date = datetime.fromisoformat(day["fecha"]).strftime("%A, %d %B %Y")
    temp = day["temperatura"]
    feels = day["sensTermica"]
    humidity = day["humedadRelativa"]

    return DayReport(
        date=date,
        sky=[
            SkyEntry(period=p, description=e.get("descripcion", "").strip())
            for p, e in _best_periods(_by_period(day["estadoCielo"]))
            if e.get("descripcion", "").strip()
        ],
        precipitation_prob=[
            PrecipEntry(period=p, value_pct=e["value"])
            for p, e in _best_periods(_by_period(day["probPrecipitacion"]))
            if e.get("value", "") != ""
        ],
        temperature=RangeWithHourly(
            min=temp.get("minima"),
            max=temp.get("maxima"),
            hourly=[
                HourlyMeasurement(hour=d["hora"], value=d["value"])
                for d in temp.get("dato", [])
            ],
        ),
        feels_like=RangeWithHourly(
            min=feels.get("minima"),
            max=feels.get("maxima"),
        ),
        humidity=RangeWithHourly(
            min=humidity.get("minima"),
            max=humidity.get("maxima"),
            hourly=[
                HourlyMeasurement(hour=d["hora"], value=d["value"])
                for d in humidity.get("dato", [])
            ],
        ),
        wind=[
            WindEntry(
                period=p,
                direction=_WIND_DIR.get(e.get("direccion", ""), e.get("direccion", "")),
                speed_kmh=e.get("velocidad", 0),
            )
            for p, e in _best_periods(_by_period(day["viento"]))
        ],
        uv_index=day.get("uvMax"),
    )

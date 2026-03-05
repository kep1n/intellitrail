"""Naismith's Rule hiking time estimator with sunset warning.

Estimates trail completion time using Naismith's Rule and fetches local sunset
time from the sunrise-sunset.org API to compute a latest-safe-start time.

Entry point:
    estimate_hiking_time(geometry: ResolvedGeometry) -> dict
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from src.models.geometry import ResolvedGeometry

_SPAIN_TZ = ZoneInfo("Europe/Madrid")
_SUNRISE_SUNSET_URL = "https://api.sunrise-sunset.org/json"
_SAFETY_BUFFER_MINUTES = 30


def estimate_hiking_time(geometry: ResolvedGeometry) -> dict:
    """Estimate hiking time via Naismith's Rule and compute latest safe start.

    Naismith's Rule:
      - 5 km/h base speed on flat ground
      - +1 minute per 10 m of ascent

    Sunset is fetched from sunrise-sunset.org (free, no key) using the route
    centroid. Times are localised to Europe/Madrid (all supported ranges are
    in Spain). A 30-minute safety buffer is applied to the latest start time.

    Returns dict:
        estimated_time_str: str  — e.g. "5h 30m"
        estimated_minutes:  int  — total minutes
        sunset_time:        str | None  — "HH:MM" local, None if API unavailable
        latest_start_time:  str | None  — "HH:MM" local, None if API unavailable

    Returns all-None dict if geometry lacks distance data.
    """
    distance_km = geometry.distance_3d_km
    gain_m = geometry.elevation_gain_m or 0.0

    if not distance_km:
        return {
            "estimated_time_str": None,
            "estimated_minutes": None,
            "sunset_time": None,
            "latest_start_time": None,
        }

    total_minutes = round((distance_km / 5.0) * 60 + gain_m / 10.0)
    h, m = divmod(total_minutes, 60)
    time_str = f"{h}h {m:02d}m"

    sunset_time, latest_start = _fetch_sunset(geometry, total_minutes)

    return {
        "estimated_time_str": time_str,
        "estimated_minutes": total_minutes,
        "sunset_time": sunset_time,
        "latest_start_time": latest_start,
    }


def _fetch_sunset(geometry: ResolvedGeometry, duration_minutes: int) -> tuple[str | None, str | None]:
    """Fetch sunset time and compute latest start. Returns (sunset_str, latest_start_str)."""
    lat = (geometry.bbox_min_lat + geometry.bbox_max_lat) / 2.0
    lon = (geometry.bbox_min_lon + geometry.bbox_max_lon) / 2.0
    today = date.today().isoformat()

    try:
        resp = httpx.get(
            _SUNRISE_SUNSET_URL,
            params={"lat": lat, "lng": lon, "date": today, "formatted": 0},
            timeout=5.0,
        )
        resp.raise_for_status()
        sunset_utc_str = resp.json()["results"]["sunset"]
        sunset_utc = datetime.fromisoformat(sunset_utc_str.replace("Z", "+00:00"))
        sunset_local = sunset_utc.astimezone(_SPAIN_TZ)

        latest_dt = sunset_local - timedelta(minutes=duration_minutes + _SAFETY_BUFFER_MINUTES)

        return sunset_local.strftime("%H:%M"), latest_dt.strftime("%H:%M")
    except Exception:
        return None, None

"""AEMET weather API fetchers — municipal forecasts and mountain snow/avalanche data.

Provides high-level functions that query the AEMET OpenData API and return
typed results using the dataclasses defined in ``weather.py``.
"""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

from weather import DayReport, parse_day_report

load_dotenv()


def get_municipality_intermediate_weather(mun_code: str) -> list[DayReport]:
    """Fetch and parse the AEMET daily municipal forecast for a given code.

    Args:
        mun_code: AEMET municipality code (``CPRO + CMUN``).

    Returns:
        List of :class:`~weather.DayReport` instances, one per forecast day.
    """
    aemet_api_key = os.getenv("AEMET_API_KEY")
    url = (
        f"https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/"
        f"{mun_code}?api_key={aemet_api_key}"
    )
    intermediate_url = httpx.get(url).json()["datos"]
    return _get_weather_info(intermediate_url)


def get_mountain_snow_weather(code: str) -> bytes | None:
    """Fetch the AEMET nivological (snow/avalanche) forecast for a zone.

    Args:
        code: AEMET nivological zone code.

    Returns:
        Raw response bytes from the data endpoint, or *None* on key error.
    """
    try:
        aemet_api_key = os.getenv("AEMET_API_KEY")
        url = (
            f"https://opendata.aemet.es/opendata/api/prediccion/especifica/nivologica/"
            f"{code}?api_key={aemet_api_key}"
        )
        intermediate_url = httpx.get(url).json()["datos"]
        return httpx.get(intermediate_url).content
    except KeyError:
        return None


def _get_weather_info(intermediate_url: str) -> list[DayReport]:
    """Fetch and parse daily weather data from an AEMET intermediate data URL.

    Args:
        intermediate_url: The ``datos`` URL returned by the AEMET API
            intermediate response.

    Returns:
        List of :class:`~weather.DayReport` instances, one per forecast day.
    """
    response = httpx.get(intermediate_url)
    data_parsed = json.loads(response.content.decode(response.encoding))
    return [parse_day_report(day) for day in data_parsed[0]["prediccion"]["dia"]]

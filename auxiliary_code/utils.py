"""Date helpers for forecast window calculation."""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastDates:
    """Immutable container for the three-day forecast window.

    Attributes:
        today: Current date in ``YYYYMMDD`` format.
        tomorrow: Next calendar day in ``YYYYMMDD`` format.
        day_after_tomorrow: Two days from now in ``YYYYMMDD`` format.
    """

    today: str
    tomorrow: str
    day_after_tomorrow: str


def calculate_forecast_dates(
    reference: datetime.datetime | None = None,
) -> ForecastDates:
    """Return a :class:`ForecastDates` for today, tomorrow, and the day after.

    Args:
        reference: Datetime to treat as "now". Defaults to
            :func:`datetime.datetime.now` when *None*.

    Returns:
        A frozen :class:`ForecastDates` instance with dates as ``YYYYMMDD``
        strings.
    """
    now = reference or datetime.datetime.now()
    return ForecastDates(
        today=now.strftime("%Y%m%d"),
        tomorrow=(now + datetime.timedelta(days=1)).strftime("%Y%m%d"),
        day_after_tomorrow=(now + datetime.timedelta(days=2)).strftime("%Y%m%d"),
    )


def calculate_days_for_mountain_predition() -> tuple[str, str, str]:
    """Deprecated: use :func:`calculate_forecast_dates` instead.

    Returns:
        Tuple of ``(today, tomorrow, day_after_tomorrow)`` as ``YYYYMMDD`` strings.
    """
    fd = calculate_forecast_dates()
    return fd.today, fd.tomorrow, fd.day_after_tomorrow

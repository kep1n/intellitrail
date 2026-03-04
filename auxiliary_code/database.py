"""SQLite database access for Spanish municipality codes and mountain zone polygons.

The database is expected at ``<repo_root>/data/projectdb.db`` and must contain:
- ``mun_codes`` — columns ``CPRO``, ``CMUN``, ``nombre``
- ``url_alerts`` — columns ``url``, ``coords`` (space-separated ``"lon,lat"`` polygon)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from geometry import parse_polygon, point_in_polygon

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "projectdb.db"


def get_municipality_code(municipality: str) -> str:
    """Look up the AEMET municipality code for a Spanish place name.

    Args:
        municipality: Full or partial Spanish municipality name.

    Returns:
        Concatenated ``CPRO + CMUN`` code string, or an error message if the
        municipality is not found in the database.
    """
    try:
        with sqlite3.connect(_DB_PATH) as connection:
            result = connection.execute(
                "SELECT concat(CPRO, CMUN) FROM mun_codes WHERE nombre LIKE ?",
                (f"%{municipality}%",),
            )
            return result.fetchone()[0]
    except (IndexError, TypeError):
        return f"{municipality} couldn't be found in the database"


def is_inside_mountains(lat: float, lon: float) -> str | bool:
    """Check whether a coordinate falls inside any mountain zone polygon.

    Queries all rows from ``url_alerts`` and tests each polygon via ray-casting.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        The URL associated with the first matching mountain zone, or *False* if
        the point is not inside any zone.
    """
    with sqlite3.connect(_DB_PATH) as connection:
        result = connection.execute("SELECT url, coords FROM url_alerts")
        for url, coords in result:
            polygon = parse_polygon(coords, lon_lat=True)
            if point_in_polygon(lat, lon, polygon):
                return url
    return False

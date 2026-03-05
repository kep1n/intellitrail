"""Physical difficulty scorer for hiking routes.

Scores a route on a Low / Moderate / High / Extreme scale based on distance
and elevation gain independently, then takes the maximum of the two scores.
This ensures a short-but-steep route and a long-but-flat route are both
rated correctly without either dimension masking the other.

Entry point:
    compute_physical_difficulty(geometry: ResolvedGeometry) -> dict
"""

from src.models.geometry import ResolvedGeometry

_LABELS = {1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"}

_DESCRIPTIONS = {
    1: "Suitable for most fitness levels.",
    2: "Moderate fitness required.",
    3: "Good fitness and mountain experience recommended.",
    4: "High fitness and prior alpine experience required.",
}


def compute_physical_difficulty(geometry: ResolvedGeometry) -> dict:
    """Return a physical difficulty rating for the route.

    Scoring grid (max of distance score and gain score):

    Score | Distance         | Elevation gain
    ------|------------------|---------------
    1     | < 10 km          | < 400 m
    2     | 10 – 18 km       | 400 – 800 m
    3     | 18 – 25 km       | 800 – 1 400 m
    4     | > 25 km          | > 1 400 m

    Returns dict:
        level:       str | None  — "Low" / "Moderate" / "High" / "Extreme"
        score:       int | None  — 1–4
        description: str | None  — one-sentence fitness note
    """
    distance_km = geometry.distance_3d_km
    gain_m = geometry.elevation_gain_m

    if not distance_km:
        return {"level": None, "score": None, "description": None}

    dist_score = _score_distance(distance_km)
    gain_score = _score_gain(gain_m or 0.0)
    score = max(dist_score, gain_score)

    return {
        "level": _LABELS[score],
        "score": score,
        "description": _DESCRIPTIONS[score],
    }


def _score_distance(km: float) -> int:
    if km < 10:
        return 1
    if km < 18:
        return 2
    if km < 25:
        return 3
    return 4


def _score_gain(m: float) -> int:
    if m < 400:
        return 1
    if m < 800:
        return 2
    if m < 1400:
        return 3
    return 4

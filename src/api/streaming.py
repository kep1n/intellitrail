import json
from typing import Any


def format_sse(event: str, data: dict) -> str:
    """Format a single SSE message. Blank line terminates the event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def extract_final_state(state_snapshot) -> dict[str, Any]:
    """Extract serializable fields from a LangGraph state snapshot for the result SSE event.

    Returns verdict, report, and trail_info (geometry summary).
    Only populated after analyze_route has been called in the session.
    """
    if hasattr(state_snapshot, "values"):
        values = state_snapshot.values
    elif isinstance(state_snapshot, dict):
        values = state_snapshot
    else:
        values = {}

    trail_info: dict | None = None
    geometry = values.get("geometry")
    if geometry is not None:
        g = geometry.model_dump() if hasattr(geometry, "model_dump") else {}
        trail_info = {
            "track_name": g.get("track_name"),
            "trail_type": g.get("trail_type"),
            "difficulty": g.get("difficulty"),
            "distance_km": g.get("distance_2d_km"),
            "elevation_gain_m": g.get("elevation_gain_m"),
            "elevation_loss_m": g.get("elevation_loss_m"),
            "max_elevation_m": g.get("max_elevation_m"),
            "min_elevation_m": g.get("min_elevation_m"),
            "moving_time": g.get("moving_time"),
        }

    return {
        "verdict": values.get("verdict"),
        "report": values.get("report"),
        "trail_info": trail_info,
    }

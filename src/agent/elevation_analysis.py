from datetime import datetime

from src.models.geometry import ResolvedGeometry


def analyze_elevation(geometry: ResolvedGeometry) -> str:
    """Return a narrative string summarising the track elevation profile
    for inclusion in the LLM prompt. Never raises — returns placeholder if
    no elevation data.
    """
    if geometry.elevation_smoothed is None:
        # No per-point profile (e.g. Wikiloc GeoJSON has no <ele> tags),
        # but scalar stats from trail-data scraping may still be present.
        if geometry.max_elevation_m is None:
            return "No elevation data available for this track."

        lines = []
        max_elev = geometry.max_elevation_m
        if geometry.min_elevation_m is not None and geometry.elevation_gain_m is not None:
            lines.append(
                f"Elevation range: {geometry.min_elevation_m:.0f}m – {max_elev:.0f}m | "
                f"Total gain: {geometry.elevation_gain_m:.0f}m"
            )
        else:
            lines.append(f"Maximum elevation: {max_elev:.0f}m")
        lines.append("No per-point elevation profile available — ridge segment inference not possible.")

        month = datetime.now().month
        snow_line = 1400 if month in (12, 1, 2) else (1800 if month in (3, 4, 10, 11) else 2500)
        if max_elev > snow_line:
            lines.append(
                f"Track max elevation ({max_elev:.0f}m) exceeds estimated seasonal "
                f"snow line ({snow_line}m)."
            )
        else:
            lines.append(f"Track remains below estimated seasonal snow line ({snow_line}m).")
        return "\n".join(lines)

    min_elev = geometry.min_elevation_m
    max_elev = geometry.max_elevation_m
    gain = geometry.elevation_gain_m

    lines = []

    # Basic elevation stats
    lines.append(
        f"Elevation range: {min_elev:.0f}m – {max_elev:.0f}m | "
        f"Total gain: {gain:.0f}m"
    )

    # Ridge inference heuristic
    elev_range = max_elev - min_elev
    if elev_range > 0:
        threshold = max_elev - 0.15 * elev_range
        total_points = len(geometry.elevation_smoothed)
        min_segment_length = max(1, int(0.05 * total_points))

        ridge_segments = 0
        in_ridge = False
        current_run = 0

        for elev in geometry.elevation_smoothed:
            if elev >= threshold:
                current_run += 1
                if not in_ridge and current_run >= min_segment_length:
                    in_ridge = True
                    ridge_segments += 1
            else:
                if in_ridge:
                    in_ridge = False
                current_run = 0

        if ridge_segments == 0:
            lines.append("No exposed ridge segments inferred.")
        else:
            lines.append(
                f"~{ridge_segments} exposed ridge segment(s) inferred "
                f"(elevation consistently above {threshold:.0f}m)."
            )
    else:
        lines.append("No exposed ridge segments inferred (flat track).")

    # Snow line by season
    month = datetime.now().month
    if month in (12, 1, 2):
        snow_line = 1400
    elif month in (3, 4, 10, 11):
        snow_line = 1800
    else:
        snow_line = 2500

    if max_elev > snow_line:
        lines.append(
            f"Track max elevation ({max_elev:.0f}m) exceeds estimated seasonal "
            f"snow line ({snow_line}m)."
        )
    else:
        lines.append(
            f"Track remains below estimated seasonal snow line ({snow_line}m)."
        )

    return "\n".join(lines)

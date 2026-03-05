from langchain_core.prompts import ChatPromptTemplate

from src.models.geometry import ResolvedGeometry
from src.weather.models import WeatherData

SYSTEM_PROMPT = """You are a mountain safety expert. Your task is to assess whether it is safe to hike the described route today and tomorrow given the weather and avalanche conditions. You must produce a structured GO/CAUTION/NO-GO safety verdict.

Rules:
- GO: conditions are acceptable across all risk dimensions.
- CAUTION: one or more risk factors require attention but the route remains feasible with care.
- NO-GO: one or more risk factors make the route unacceptably dangerous.

Severity thresholds (apply to the route's maximum elevation):
- Wind: CAUTION >50 km/h | NO-GO >80 km/h
- Avalanche risk level (European 1-5 scale): CAUTION Level 3 | NO-GO Level 4 or 5
- Active weather alerts (CAP severity → minimum verdict):
  IMPORTANT: only apply these rules if the weather data contains an "## Active weather alerts" section. If that section is absent or says "No active alerts", there are NO formal alerts — do not infer or hallucinate alerts from wind speed, precipitation, or any other forecast values.
  - Minor (yellow): CAUTION minimum regardless of other conditions
  - Moderate (orange): CAUTION; upgrade to NO-GO if description indicates heavy rain accumulation (>30 mm/24 h), damaging wind, or dangerous temperature extremes
  - Severe or Extreme (red): NO-GO always
- Precipitation probability alone does not trigger CAUTION — use it as supporting context only when an active alert is present

Conflict handling: if signals conflict (e.g. morning GO, afternoon NO-GO), weight by route-relevance (higher elevation = stronger weight). Explicitly name the conflict and the dominant signal in your reasoning.

Missing data rule: if weather data is incomplete, set verdict to CAUTION and note it in data_completeness and reasoning.

Source data is in Spanish — translate and interpret Spanish weather terms correctly.

Output language: ALWAYS --> English.

For time_windows: express as named time bands (morning/afternoon/evening/night) for today and tomorrow. If no safe window exists for a day, suggest trying the next day.
For elevation_context: use the elevation analysis provided; describe conditions at each altitude band on the route.
For risk_factors: list each risk as a single string: "Factor: value — implication" (e.g. "Wind: 85 km/h at summit — exceeds NO-GO threshold").
"""

HUMAN_TEMPLATE = """## Track Information
{geometry_summary}

## Elevation Analysis
{elevation_analysis}

## Weather Data
{weather_summary}

Produce the safety verdict for this route."""


def _geometry_summary(geometry: ResolvedGeometry) -> str:
    lines = []
    if geometry.track_name:
        lines.append(f"Track: {geometry.track_name}")
    if geometry.trail_type:
        lines.append(f"Trail type: {geometry.trail_type}")
    if geometry.difficulty:
        lines.append(f"Difficulty: {geometry.difficulty}")
    lines.append(f"Distance: {geometry.distance_3d_km:.1f} km (3D)")
    if geometry.min_elevation_m is not None:
        lines.append(f"Elevation range: {geometry.min_elevation_m:.0f}m – {geometry.max_elevation_m:.0f}m")
    if geometry.elevation_gain_m is not None:
        loss_str = f" | Loss: {geometry.elevation_loss_m:.0f}m" if geometry.elevation_loss_m is not None else ""
        lines.append(f"Total gain: {geometry.elevation_gain_m:.0f}m{loss_str}")
    lines.append(f"Bounding box: lat {geometry.bbox_min_lat:.3f}–{geometry.bbox_max_lat:.3f}, "
                 f"lon {geometry.bbox_min_lon:.3f}–{geometry.bbox_max_lon:.3f}")
    return "\n".join(lines)


def _weather_summary(weather_data: dict | None) -> str:
    """Convert raw weather_data dict (from AgentState) to a readable string for the LLM.

    Uses WeatherData.model_validate() per the established Phase 2 pattern.
    Returns a placeholder string when weather_data is None or data_complete=False.
    """
    if weather_data is None:
        return "No weather data available — apply missing-data CAUTION rule."
    wd = WeatherData.model_validate(weather_data)
    if not wd.data_complete:
        reason = wd.incomplete_reason or "unknown reason"
        return f"Weather data incomplete: {reason} — apply missing-data CAUTION rule."

    parts = []
    if wd.municipal:
        for day in wd.municipal:
            parts.append(f"\n### Municipal forecast — {day.municipality_name} ({day.date})")
            for p in day.periods:
                fields = [f"period={p.period}"]
                if p.sky_state:
                    fields.append(f"sky={p.sky_state}")
                if p.precipitation_probability is not None:
                    fields.append(f"precip_prob={p.precipitation_probability}%")
                if p.wind_speed_kmh is not None:
                    fields.append(f"wind={p.wind_speed_kmh}km/h {p.wind_direction or ''}")
                if p.temperature_max_c is not None:
                    fields.append(f"temp={p.temperature_min_c}–{p.temperature_max_c}°C")
                parts.append("  " + " | ".join(fields))
    if wd.mountain:
        parts.append(f"\n### Mountain forecast — {wd.mountain.area_name}")
        if wd.mountain.freezing_level or wd.mountain.wind_1500m:
            fa_lines = []
            if wd.mountain.freezing_level:
                fa_lines.append(f"Freezing level (isocero): {wd.mountain.freezing_level}")
            if wd.mountain.wind_1500m:
                fa_lines.append(f"Wind at 1500m (v1500): {wd.mountain.wind_1500m}")
            parts.append("  Free atmosphere: " + " | ".join(fa_lines))
        parts.append(wd.mountain.forecast_text[:2000])  # cap very long narratives
    if wd.avalanche:
        av = wd.avalanche
        if av.unavailable_reason:
            parts.append(f"\n### Snow bulletin: unavailable ({av.unavailable_reason})")
        else:
            parts.append(f"\n### Snow bulletin — zone {av.zone_code}")
            parts.append(av.raw_text[:1500])
    if wd.alerts:
        parts.append(f"\n### Active weather alerts ({len(wd.alerts)})")
        for alert in wd.alerts[:5]:  # cap at 5 alerts
            severity = alert.get("severity", "")
            event = alert.get("event", "")
            area = alert.get("area_desc", "")
            expires = alert.get("expires", "")
            description = alert.get("description", "")
            parameters = alert.get("parameters") or {}
            parts.append(f"  - [{severity}] {event} — {area} (expires {expires})")
            if description:
                parts.append(f"    {description[:400]}")
            if parameters:
                params_str = " | ".join(f"{k}={v}" for k, v in parameters.items())
                parts.append(f"    Parameters: {params_str}")
    else:
        parts.append("\n### Active weather alerts\nNo active CAP alerts for this location.")
    if wd.missing_mountain_data:
        parts.append("\n### Mountain forecast: unavailable (non-fatal — municipal data present)")
    return "\n".join(parts) if parts else "Weather data returned empty fields."


def build_verdict_prompt(
    geometry: ResolvedGeometry,
    weather_data: dict | None,
    elevation_analysis: str,
) -> list:
    """Return a list of message dicts suitable for ChatOpenAI.invoke().

    Returns the rendered messages from a ChatPromptTemplate so the caller
    (generate_verdict node) only needs to pass the result directly to the LLM.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])
    rendered = template.format_messages(
        geometry_summary=_geometry_summary(geometry),
        elevation_analysis=elevation_analysis,
        weather_summary=_weather_summary(weather_data),
    )
    return rendered


GEAR_SYSTEM_PROMPT = """You are a mountain safety equipment expert. Given a safety verdict, weather conditions, and relevant passages from alpine safety documents, produce a concise gear and equipment recommendation list.

Format your response as a markdown bullet list. Each bullet must include:
- The item name (bold)
- One sentence explaining why it is needed for the specific conditions described

Label the section header based on the verdict:
- GO verdict → "## Standard kit"
- CAUTION verdict → "## Recommended extras"
- NO-GO verdict → "## Essential if you proceed"

Keep the list focused: 5–8 items maximum. Prioritize items directly relevant to the specific conditions (wind speed, precipitation, temperature, avalanche risk). Do not list generic items that apply to all hikes.

Source data may be in Spanish — translate and interpret correctly. Output language: English.
"""

GEAR_HUMAN_TEMPLATE = """## Verdict
{verdict}

## Conditions Summary
{conditions_summary}

## Relevant Safety Document Passages
{rag_context}

Produce the gear recommendation list for this specific outing."""


def build_gear_prompt(
    verdict: str,
    weather_data: dict | None,
    rag_context: str,
) -> list:
    """Build gear recommendation prompt messages.

    verdict: "GO", "CAUTION", or "NO-GO"
    weather_data: raw AgentState.weather_data dict (or None)
    rag_context: retrieved passages string from retrieve_gear_context()
    """
    conditions_summary = _weather_summary(weather_data)
    template = ChatPromptTemplate.from_messages([
        ("system", GEAR_SYSTEM_PROMPT),
        ("human", GEAR_HUMAN_TEMPLATE),
    ])
    return template.format_messages(
        verdict=verdict,
        conditions_summary=conditions_summary,
        rag_context=rag_context or "No safety document passages available.",
    )

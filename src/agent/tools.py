"""LangChain tools for the conversational mountaineering safety agent.

Three tools:
- search_routes: SerpAPI + Wikiloc metadata scrape → candidate list
- analyze_route: full pipeline (geometry → weather → verdict → alternatives)
- rag_query: RAG retrieval over alpine safety documents
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import serpapi
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.agent.elevation_analysis import analyze_elevation
from src.agent.prompts import build_verdict_prompt
from src.agent.wikiloc_scraper import ScraperError, is_wikiloc_route_url, scrape_geojson, scrape_metadata
from src.alternatives.overpass import search_overpass
from src.config import Settings
from src.ingestion.geojson_parser import parse_geojson_to_geometry
from src.ingestion.gpx_parser import parse_gpx_bytes
from src.models.state import AgentState
from src.models.verdict import VerdictEnum, VerdictReport
from src.rag.retriever import retrieve_gear_context
from src.weather.client import AEMETClient
from src.weather.fetcher import fetch_weather_data
from src.weather.models import WeatherData
from src.weather.zone_mapper import ZoneMapper


@tool
def search_routes(query: str) -> str:
    """Search for Wikiloc hiking routes matching a query string.

    Use this when the user mentions a route, mountain, or trail by name
    but has not uploaded a GPX file. Returns a JSON list of up to 3
    candidate routes with name, distance_km, elevation_gain_m, difficulty,
    and URL. Present the results to the user and ask them to pick one
    before calling analyze_route.
    """
    settings = Settings()
    try:
        client = serpapi.Client(api_key=settings.serp_api_key)
        results = client.search({"engine": "google", "q": f"site:wikiloc.com {query}"})
        raw_results = [
            {"url": r["link"], "title": r.get("title", "")}
            for r in results.get("organic_results", [])
            if is_wikiloc_route_url(r.get("link", ""))
        ][:3]
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})

    if not raw_results:
        return json.dumps({"error": f"No Wikiloc routes found for '{query}'."})

    candidates = []
    for raw in raw_results:
        try:
            meta = scrape_metadata(raw["url"])
            candidates.append({**meta, "url": raw["url"]})
        except ScraperError:
            candidates.append({
                "url": raw["url"],
                "name": raw["title"],
                "distance_km": None,
                "elevation_gain_m": None,
                "difficulty": None,
            })

    return json.dumps(candidates)


@tool
def analyze_route(
    wikiloc_url: Annotated[
        str | None,
        "Wikiloc route URL selected by the user. Omit if the user uploaded a GPX file.",
    ] = None,
    state: Annotated[AgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Analyze a hiking route for safety: weather, mountain conditions, snow/avalanche risk, and elevation.

    Use this when the user asks whether a route is safe, requests a GO/CAUTION/NO-GO verdict,
    or has uploaded a GPX file. On CAUTION or NO-GO also finds alternative routes within 10 km.

    - If the user uploaded a GPX file, call without wikiloc_url.
    - If the user selected a candidate from search_routes, pass that URL as wikiloc_url.
    """
    settings = Settings()

    # --- 1. Resolve geometry ---
    gpx_bytes = state.gpx_input if state else None

    if gpx_bytes:
        try:
            geometry = parse_gpx_bytes(gpx_bytes)
        except Exception as e:
            return _tool_error(tool_call_id, f"Failed to parse GPX file: {e}")
    elif wikiloc_url:
        try:
            geojson, trail_stats = scrape_geojson(wikiloc_url)
            geometry = parse_geojson_to_geometry(geojson, trail_stats=trail_stats)
        except Exception as e:
            return _tool_error(tool_call_id, f"Failed to extract route data from Wikiloc: {e}")
    else:
        return _tool_error(
            tool_call_id,
            "No route provided. Please upload a GPX file or specify a Wikiloc URL.",
        )

    # --- 2. Fetch weather ---
    try:
        aemet = AEMETClient(api_key=settings.aemet_api_key)
        zone_mapper = ZoneMapper(client=aemet)
        weather = fetch_weather_data(geometry, aemet, zone_mapper)
    except Exception as e:
        return _tool_error(tool_call_id, f"Weather fetch failed: {e}")

    weather_dict = weather.model_dump()

    # --- 3. Deterministic snow-line CAUTION rule ---
    forced_caution_reason: str | None = None
    if geometry.max_elevation_m is not None:
        month = date.today().month
        snow_line = 1400 if month in (12, 1, 2) else (1800 if month in (3, 4, 10, 11) else 2500)
        if geometry.max_elevation_m > snow_line:
            wd = WeatherData.model_validate(weather_dict)
            if wd.avalanche and wd.avalanche.unavailable_reason == "fetch_failed":
                forced_caution_reason = (
                    f"Track max elevation ({geometry.max_elevation_m:.0f}m) exceeds "
                    f"seasonal snow line ({snow_line}m) and snow bulletin could not be retrieved."
                )

    # --- 4. Generate verdict via LLM ---
    try:
        elev_text = analyze_elevation(geometry)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key,
        ).with_structured_output(VerdictReport, method="json_schema", strict=True)
        report: VerdictReport = llm.invoke(
            build_verdict_prompt(geometry, weather_dict, elev_text)
        )
    except Exception as e:
        return _tool_error(tool_call_id, f"Verdict generation failed: {e}")

    if forced_caution_reason and report.verdict == VerdictEnum.GO:
        report = report.model_copy(update={
            "verdict": VerdictEnum.CAUTION,
            "summary": f"CAUTION (safety rule override) — {forced_caution_reason}",
            "reasoning": f"{forced_caution_reason}\n\n{report.reasoning}",
            "risk_factors": [forced_caution_reason] + report.risk_factors,
        })

    verdict_str = report.verdict.value

    # --- 5. Alternatives on CAUTION / NO-GO ---
    alternatives: list[dict] = []
    if verdict_str in ("CAUTION", "NO-GO"):
        centroid_lat = (geometry.bbox_min_lat + geometry.bbox_max_lat) / 2.0
        centroid_lon = (geometry.bbox_min_lon + geometry.bbox_max_lon) / 2.0
        try:
            alternatives = search_overpass(centroid_lat, centroid_lon, radius_m=10000)
        except Exception:
            pass

        if len(alternatives) < 2:
            try:
                serp = serpapi.Client(api_key=settings.serp_api_key)
                res = serp.search({
                    "engine": "google",
                    "q": f"site:wikiloc.com hiking trails near {centroid_lat:.3f},{centroid_lon:.3f}",
                })
                wikiloc_alts = [
                    {"name": r.get("title", ""), "url": r["link"]}
                    for r in res.get("organic_results", [])
                    if is_wikiloc_route_url(r.get("link", ""))
                ]
                existing_names = {a.get("name", "") for a in alternatives}
                for alt in wikiloc_alts:
                    if alt["name"] not in existing_names:
                        alternatives.append(alt)
                        existing_names.add(alt["name"])
            except Exception:
                pass

        alternatives = alternatives[:3]

    # --- 6. Assemble report and ToolMessage summary ---
    report_dict = report.model_dump(mode="json")
    report_dict["alternatives"] = alternatives

    summary_lines = [f"**Verdict: {verdict_str}**", f"{report.summary}"]

    wd = WeatherData.model_validate(weather_dict)
    if wd.municipal:
        day = wd.municipal[0]
        if day.periods:
            p = day.periods[0]
            cond_parts = []
            if p.wind_speed_kmh is not None:
                cond_parts.append(f"wind {p.wind_speed_kmh} km/h")
            if p.sky_state:
                cond_parts.append(p.sky_state)
            if p.temperature_min_c is not None and p.temperature_max_c is not None:
                cond_parts.append(f"{p.temperature_min_c:.0f}–{p.temperature_max_c:.0f}°C")
            if cond_parts:
                summary_lines.append(f"Conditions ({day.municipality_name}): {', '.join(cond_parts)}")

    if wd.mountain:
        summary_lines.append(f"Mountain forecast: {wd.mountain.forecast_text[:200]}")

    if wd.avalanche and not wd.avalanche.unavailable_reason:
        summary_lines.append(f"Snow bulletin: {wd.avalanche.raw_text[:200]}")

    if report.risk_factors:
        summary_lines.append(f"Risk factors: {'; '.join(report.risk_factors[:3])}")

    if report.time_windows:
        summary_lines.append(f"Safe windows: {report.time_windows}")

    if alternatives:
        alt_names = [a.get("name", "unknown") for a in alternatives]
        summary_lines.append(f"Alternatives within 10 km: {', '.join(alt_names)}")

    return Command(
        update={
            "geometry": geometry,
            "weather_data": weather_dict,
            "verdict": verdict_str,
            "report": report_dict,
            "messages": [ToolMessage(content="\n".join(summary_lines), tool_call_id=tool_call_id)],
        }
    )


@tool
def rag_query(question: str) -> str:
    """Search the alpine safety knowledge base for mountaineering information.

    Use this when the user asks about gear, equipment, techniques, safety practices,
    acclimatisation, navigation, weather interpretation, or general mountaineering advice.
    Do NOT use this for route safety verdicts — use analyze_route for that.
    """
    try:
        return retrieve_gear_context(question)
    except FileNotFoundError:
        return (
            "The alpine safety knowledge base is not available. "
            "To enable it, add PDF documents to 'data/pdfs/' and run the indexing script."
        )
    except Exception as e:
        return f"Knowledge base query failed: {e}"


def _tool_error(tool_call_id: str, message: str) -> Command:
    """Return a Command that adds an error ToolMessage without modifying analysis state."""
    return Command(
        update={"messages": [ToolMessage(content=f"Error: {message}", tool_call_id=tool_call_id)]}
    )

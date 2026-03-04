import json
from langchain_core.tools import tool

from src.alternatives.overpass import search_overpass


@tool
def run_overpass_search(coordinates: str) -> str:
    """Search OpenStreetMap via Overpass API for hiking routes near a location.

    Input: coordinates as "lat,lon" string (e.g. "42.5,1.8").
    Returns: JSON string with list of up to 3 hiking routes. Each route has:
      - name: route name from OSM
      - centroid_lat, centroid_lon: route centroid coordinates
      - distance_km: straight-line distance from the query location
    Returns JSON with empty list if no routes found within 10 km.
    """
    try:
        lat_str, lon_str = coordinates.strip().split(",")
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid coordinates format '{coordinates}': {e}"})

    results = search_overpass(lat, lon, radius_m=10000)
    return json.dumps(results)


@tool
def run_wikiloc_search(coordinates: str) -> str:
    """Search Wikiloc for documented hiking alternatives near a location using SerpAPI.

    Input: coordinates as "lat,lon" string (e.g. "42.5,1.8").
    The tool constructs a location-aware Wikiloc search query and returns
    up to 3 results with URL and title.

    Returns: JSON string with list of {url, title} dicts.
    Returns JSON with empty list if no results found or SerpAPI unavailable.

    Note: This tool searches by general location area. The LLM should use it
    when Overpass results are insufficient or when documented/popular routes are preferred.
    """
    # Import here to avoid circular import (nodes.py imports from this module in Plan 04)
    import serpapi
    from src.config import Settings

    try:
        lat_str, lon_str = coordinates.strip().split(",")
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid coordinates format '{coordinates}': {e}"})

    try:
        settings = Settings()
        client = serpapi.Client(api_key=settings.serp_api_key)
        # Search for hiking near this location — area-based query
        query = f"site:wikiloc.com hiking trails near {lat:.3f},{lon:.3f}"
        results_raw = client.search({"engine": "google", "q": query})

        from src.agent.wikiloc_scraper import is_wikiloc_route_url
        results = [
            {"url": r["link"], "title": r.get("title", "Wikiloc Route")}
            for r in results_raw.get("organic_results", [])
            if is_wikiloc_route_url(r.get("link", ""))
        ]
        return json.dumps(results[:3])
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})

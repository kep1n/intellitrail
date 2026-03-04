# Mountain Safety Agent

![Logo Intellitrail](./imgs/intellitrail.png)

An AI agent that gives hikers in Spain a GO/NO-GO/CAUTION safety verdict before a mountain trip. Users submit their destination via text, voice, or GPX file; the agent resolves the location, queries AEMET weather and avalanche conditions, and responds with a structured report including verdict, reasoning, gear recommendations, time windows, elevation thresholds, and alternative tracks when conditions are unfavorable.

## Setup

1. Install dependencies using uv:

```bash
uv sync
```

2. Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:
- `AEMET_API_KEY` — obtain from https://opendata.aemet.es/
- `SERP_API_KEY` — obtain from https://serpapi.com/

## Usage

```bash
python main.py path/to/track.gpx
```

## Tech Stack

- LangGraph — graph-based agent orchestration
- Pydantic — data validation and settings management
- GeoPandas + Shapely + PyProj — geospatial processing
- AEMET API — Spanish meteorological data
- SerpAPI — web search for location resolution

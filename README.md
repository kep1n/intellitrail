# IntelliTrail

> **Proof of concept** — built to fiddle with the concepts covered during the Ironhack AI Engineering bootcamp: LLM agents, RAG, tool use, streaming APIs, web scraping, geospatial processing, and containerised deployment. Not production-ready.

IntelliTrail is a conversational mountaineering safety assistant that gives hikers a **GO / CAUTION / NO-GO** verdict before heading into the Spanish mountains. Users can describe a route by name, paste a Wikiloc URL, upload a GPX file, or speak — the agent resolves the location, pulls live weather and snow data from AEMET, and responds with a structured safety report.

---

## How it works

The user interacts through a chat UI served by a FastAPI backend. Each message is routed through a LangGraph ReAct agent that decides which tools to call, streams tokens back via SSE, and keeps full conversation memory across turns.

```
User message / GPX / voice
        |
        v
  intent_router  ──────────────────────────────────────┐
  (gpt-4o-mini)                                         |
        |                                               |
        v                                               |
   chat_agent  (gpt-4o, tool-bound ReAct loop)         |
    /    |    \                                         |
   /     |     \                                        |
search  analyze  rag_query                              |
routes  route    (Pinecone)                             |
(SERP + (AEMET +                                        |
Wikiloc) LLM verdict)                                   |
        |                                               |
        v                                               |
  SSE stream → browser  ◄──────────────────────────────┘
```

### Intent routing

Before each agent step a lightweight `gpt-4o-mini` classifier stores one of three intents in the graph state:

| Intent | Trigger |
|---|---|
| `route_analysis` | user asks for a verdict, has a GPX or URL, or is selecting from a list |
| `route_search` | user names a mountain/trail, no GPX or URL present |
| `gear_question` | gear, techniques, safety practices, or off-topic messages |

The intent is injected into the system prompt as a hint to guide the ReAct loop.

---

## Source layout

```
src/
├── api/
│   ├── server.py          # FastAPI app — /api/chat (SSE), /api/transcribe, /api/clear-gpx
│   └── streaming.py       # SSE formatting and final-state extraction
│
├── agent/
│   ├── graph.py           # LangGraph StateGraph — intent_router → chat_agent ⇄ tools
│   ├── nodes.py           # chat_agent node + system prompt
│   ├── tools.py           # search_routes, analyze_route, rag_query LangChain tools
│   ├── prompts.py         # Verdict prompt builder
│   ├── wikiloc_scraper.py # Playwright scraper — extracts GeoJSON + metadata from Wikiloc
│   ├── elevation_analysis.py  # Elevation profile summary
│   ├── hiking_time.py     # Naismith's rule estimate + sunset check
│   ├── difficulty.py      # Physical difficulty scoring
│   └── voice_recorder.py  # Whisper transcription validation
│
├── weather/
│   ├── client.py          # AEMET REST API client
│   ├── fetcher.py         # Orchestrates municipal + mountain + avalanche fetches
│   ├── zone_mapper.py     # Maps route bbox → AEMET mountain zones
│   ├── mountain_forecast.py  # Mountain-specific forecast parsing
│   ├── alerts.py          # CAP alert polygon containment check
│   ├── geometry.py        # Spatial helpers
│   └── models.py          # Pydantic weather data models
│
├── rag/
│   ├── loader.py          # PDF ingestion from data/pdfs/
│   ├── index.py           # Pinecone index creation + upsert
│   └── retriever.py       # Semantic retrieval over alpine safety documents
│
├── alternatives/
│   ├── overpass.py        # OpenStreetMap Overpass API — nearby trails + refuges
│   └── tools.py           # Alternative route helpers
│
├── ingestion/
│   ├── gpx_parser.py      # GPX → RouteGeometry
│   └── geojson_parser.py  # GeoJSON (from Wikiloc) → RouteGeometry
│
├── models/
│   ├── state.py           # LangGraph AgentState + Intent
│   ├── verdict.py         # VerdictReport + VerdictEnum (GO/CAUTION/NO-GO)
│   └── geometry.py        # RouteGeometry Pydantic model
│
└── config.py              # Pydantic-settings — loads all API keys from .env
```

---

## Key features

- **Three input modes** — free-text route name, Wikiloc URL, GPX file upload, or voice (Whisper)
- **Live weather** — AEMET municipal forecast + mountain bulletin + snow/avalanche bulletin
- **LLM verdict** — structured GO / CAUTION / NO-GO with reasoning, risk factors, gear recommendations, and safe time windows
- **Deterministic safety override** — if max elevation exceeds the seasonal snow line and the avalanche bulletin is unavailable, the verdict is forced to CAUTION regardless of the LLM output
- **Alternatives** — on CAUTION or NO-GO, searches OSM Overpass + Wikiloc for nearby trails within 10 km
- **RAG knowledge base** — answers gear and technique questions from alpine safety PDFs indexed in Pinecone
- **Streaming UI** — tokens streamed via SSE; tool invocations surfaced in the UI in real time
- **Session memory** — LangGraph MemorySaver keeps the full conversation across turns

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

```env
# Required
OPENAI_API_KEY=...          # GPT-4o agent + GPT-4o-mini intent router + Whisper
AEMET_API_KEY=...           # https://opendata.aemet.es/
SERP_API_KEY=...            # https://serpapi.com/ (route search)
PINECONE_API_KEY=...        # https://www.pinecone.io/ (RAG index)

# Optional — LangSmith tracing
LANGSMITH_API_KEY=...
LANGSMITH_ENDPOINT=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=intellitrail
```

---

## Local development

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Install Playwright browser (needed for Wikiloc scraping)
uv run playwright install chromium --with-deps

# Start the server
uv run uvicorn src.api.server:app --reload --port 8000
```

Open `http://localhost:8000`.

### RAG index (optional)

Drop alpine safety PDF documents into `data/pdfs/`, then run the indexing script:

```bash
uv run python -m src.rag.index
```

This chunks, embeds, and upserts the documents into Pinecone. The `rag_query` tool will silently degrade if the index is absent.

---

## Docker deployment

The project ships with a `Dockerfile` and a `docker-compose.yml`.

**Build and run locally:**

```bash
docker compose up --build
```

The app is exposed on port `9100` (`http://localhost:9100`).

**Deploy on a Synology NAS (or any remote host):**

1. Build the image and export it:

```bash
docker build -t intellitrail:latest .
docker save intellitrail:latest | gzip > intellitrail.tar.gz
```

2. Copy the archive and your `.env` file to the NAS, then load and start:

```bash
docker load < intellitrail.tar.gz
docker compose up -d
```

The Playwright Chromium binary is baked into the image at build time (`playwright install chromium --with-deps`), so no browser download is needed at runtime — this is what makes it work on NAS deployments where outbound traffic or write permissions may be restricted.

---

## Tech stack

| Layer | Library |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Speech-to-text | OpenAI Whisper |
| Web framework | FastAPI |
| Weather data | AEMET OpenData API |
| Web scraping | Playwright (Chromium, headless) |
| Geospatial | GeoPandas, Shapely, PyProj |
| Vector store | Pinecone |
| Data validation | Pydantic v2 + pydantic-settings |
| Package manager | uv |
| Containerisation | Docker + Docker Compose |

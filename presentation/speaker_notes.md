# Speaker Notes — IntelliTrail (~10 min)

---

## Slide 1 — Title `~1 min`

My project is called IntelliTrail — a conversational AI agent that tells mountain hikers whether they should go out or stay home.

Before anything else: this is a proof of concept. The goal was to plug together as many of the things we covered during the bootcamp as possible — agents, RAG, tool use, streaming, scraping, deployment. It's not a polished product, but it works end to end."

---

## Slide 2 — The Problem `~2 min`

"The problem is simple: before a mountain trip you need to check the weather, the snow conditions, the route profile, maybe the avalanche bulletin — all from different sources. It's scattered and time-consuming, and people skip it.

IntelliTrail collapses all of that into one conversation. You describe a route — by name, by URL, by uploading a GPX file, or just speaking — and the agent comes back with a structured verdict: GO, CAUTION, or NO-GO.

There's a deterministic safety rule that overrides the LLM. If the track goes above the seasonal snow line and the avalanche bulletin can't be fetched, the verdict is forced to CAUTION regardless of what the model says. I didn't want to rely purely on the LLM for something safety-critical.

On CAUTION or NO-GO it also finds alternative trails nearby using OpenStreetMap."

---

## Slide 3 — Architecture `~2.5 min`

"Under the hood it's a LangGraph ReAct agent. Every message goes through a lightweight intent classifier first — GPT-4o-mini — that buckets the request into one of three types: the interaction is pretty straightforward --> the user wants a route analysed, they want to search for a route by name, or they have a gear or technique question.

That intent gets injected as a hint into the main agent's system prompt, which is GPT-4o running a standard ReAct loop with three tools:

- **search_routes** — hits SerpAPI to find Wikiloc results, then Playwright scrapes each page for distance, elevation, and difficulty.
- **analyze_route** — the main pipeline: parses the geometry, calls AEMET for weather and snow data, runs the LLM verdict, and fetches alternatives.
- **rag_query** — semantic search over alpine safety PDFs I indexed in Pinecone. Answers gear and technique questions.

The whole thing streams back to the browser via SSE — so you see tokens as they arrive and the UI shows which tool is being called in real time."

---

## Slide 4 — Tech Stack `~1.5 min`

"Quick tour of what's in the box:

LangGraph for the agent graph and memory — it keeps the full conversation across turns so you can say 'actually, show me that route instead' and it knows what you mean.

AEMET is Spain's meteorological agency and has a free open API — municipal forecasts, mountain bulletins, and avalanche bulletins by zone.

Playwright runs a headless Chrome to scrape Wikiloc, which doesn't have a public API. That was one of the trickier parts — Wikiloc has anti-bot detection so I had to spoof the browser fingerprint.

Pinecone for the RAG index, FastAPI for the backend, and the whole thing is containerised with Docker and running on a Synology NAS at home."

---

## SHORT DEMO

## Slide 5 — Who built what `~1.5 min`

"I want to be honest about the AI contribution because it's relevant to the course.

LangGraph wiring — almost entirely AI-generated. I described what I wanted and iterated on the output.

Functionality and integration — that's mostly me. Connecting AEMET, parsing the responses, handling the scraping edge cases, writing the safety override logic, wiring the tools together, encoding issues, non standard responses... That took real debugging time.

Frontend — 100% AI. I described the design and it produced the HTML and CSS.

Deployment — fifty-fifty. The Docker setup was collaborative. For example, just yesterday I was debugging why the scraper worked locally but failed on the NAS — turned out Playwright was installed with the wrong browser channel in the image.

The honest takeaway: AI is fast at structure and boilerplate. The domain knowledge, the debugging, and the decisions about what to build — that's still on you."

---

## Slide 6 — Takeaways & Improvements `~1.5 min`

"Three things I'd take away from this project.

First: AI is great at structure, terrible at judgment depending on the System Prompt (still to be refined). The graph wiring, the HTML, the boilerplate — fast. But deciding *what* data matters, *how* to interpret an avalanche bulletin, *when* to distrust the model — that's still a human call.

Second: for anything safety-related, don't rely purely on the LLM. I added a deterministic override that ignores the model output if certain conditions are met. That felt like the right call and I'd do it again.

Third: deployment is where everything gets real. Running a headless browser inside a Docker container on a NAS taught me more about systems than any tutorial.

On improvements — the obvious one is multi-day planning with date-aware forecasts, since right now it only looks at today. Longer term, expanding beyond Spain and building a mobile app for actual trailhead use would make it genuinely useful."

---

## Slide 7 — Thanks `~30 sec`

"That's it. The code is on GitHub if you want to look at it.

Happy to take any questions — about the agent architecture, the AEMET integration, the scraping, or how I used Claude to build this."

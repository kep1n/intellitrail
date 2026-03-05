"""LangGraph nodes for the conversational mountaineering safety agent."""

import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.tools import analyze_route, rag_query, search_routes
from src.agent.wikiloc_scraper import is_wikiloc_route_url
from src.config import Settings
from src.models.state import AgentState

_TOOLS = [analyze_route, search_routes, rag_query]

_URL_PATTERN = re.compile(r"https?://\S+")

AGENT_SYSTEM_PROMPT = """You are IntelliTrail, an expert mountaineering safety assistant. \
You help hikers and mountaineers assess route safety, understand mountain conditions, \
and make informed decisions about their outings in Spanish mountain ranges.

You have access to three tools:

1. search_routes(query) — Search Wikiloc for routes by name or mountain. Use this when the user
   mentions a route or mountain by name but has not uploaded a GPX file. Present the results
   to the user and ask them to select one before calling analyze_route.

2. analyze_route(wikiloc_url?) — Run a full safety analysis: weather, mountain forecast,
   snow/avalanche bulletin (where available), elevation profile, and GO/CAUTION/NO-GO verdict.
   Also finds alternative routes on CAUTION or NO-GO verdicts. Call this when:
   - The user asks if a route is safe or requests a verdict
   - The user has uploaded a GPX file (omit wikiloc_url — it is read from state automatically)
   - The user has selected a route from search_routes results (pass that URL)

3. rag_query(question) — Search the alpine safety knowledge base. Use this for questions about
   gear, equipment, techniques, acclimatisation, navigation, avalanche safety, or general
   mountaineering best practices. Do NOT use it for route verdicts.

Conversation guidelines:
- Always be concise and safety-focused.
- When presenting search_routes results, list them clearly and ask the user to choose.
- After analyze_route returns, summarise the verdict and key conditions in plain language.
  Reference the specific conditions (wind speed, temperature, snow risk) from the tool output.
- If the verdict is CAUTION or NO-GO, proactively mention the alternatives found.
- For gear or technique questions, use rag_query to ground your answer in safety documents.
- If uncertain whether the user wants a full analysis or just a quick answer, ask.
- Always respond in English, regardless of the language of the user's message or the source data.
"""

_INTENT_HINTS = {
    "route_analysis": "The user wants a route safety analysis. Call analyze_route.",
    "route_search": "The user wants to find routes by name. Call search_routes first, then present results and ask them to choose.",
    "gear_question": "The user has a gear, technique or safety question related to mountaineering. Call rag_query. If the message is off-topic, politely redirect them to route safety or gear questions.",
}


def chat_agent(state: AgentState) -> dict:
    """Run one LLM step of the ReAct agent loop.

    Prepends the system prompt (with intent hint if available) and invokes
    the LLM with all tools bound. Returns the AI message.
    The graph routes to tools if the response contains tool calls, or END otherwise.
    """
    settings = Settings()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=settings.openai_api_key,
    ).bind_tools(_TOOLS)

    hint = _INTENT_HINTS.get(state.intent or "", "")
    system_content = AGENT_SYSTEM_PROMPT
    if hint:
        system_content += f"\n\nCurrent request type: {hint}"

    response = llm.invoke([SystemMessage(content=system_content)] + state.messages)
    return {"messages": [response]}


def extract_url_from_messages(messages) -> str | None:
    """Scan message history newest-first for a valid Wikiloc route URL."""
    for message in reversed(messages):
        text = message.content
        if not isinstance(text, str):
            continue
        for url in _URL_PATTERN.findall(text):
            url = url.rstrip(".,)")
            if is_wikiloc_route_url(url):
                return url
    return None

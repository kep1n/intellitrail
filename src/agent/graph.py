from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import chat_agent, extract_url_from_messages
from src.agent.tools import analyze_route, rag_query, search_routes
from src.models.state import AgentState, Intent

_TOOLS = [analyze_route, search_routes, rag_query]


def _should_continue(state: AgentState) -> str:
    """Route to tools if the last AI message contains tool calls, otherwise end."""
    last = state.messages[-1] if state.messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _route_by_intent(state: AgentState) -> str:
    """Read the intent already stored in state by intent_router."""
    return state.intent or "chitchat"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", _classify_and_store_intent)
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("tools", ToolNode(tools=_TOOLS))

    graph.set_entry_point("intent_router")

    # intent_router classifies and stores intent, then routes to chat_agent
    # (all intents use the same ReAct loop — intent hint guides the LLM)
    graph.add_conditional_edges(
        "intent_router",
        _route_by_intent,
        {
            "route_analysis": "chat_agent",
            "route_search": "chat_agent",
            "gear_question": "chat_agent",
        },
    )

    # Original ReAct loop: chat_agent -> tools -> chat_agent -> END
    graph.add_conditional_edges(
        "chat_agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "chat_agent")

    return graph


def _classify_and_store_intent(state: AgentState) -> dict:
    """Classify the user's intent and store it in state for chat_agent to use."""
    has_gpx = state.gpx_input is not None
    has_url = extract_url_from_messages(state.messages) is not None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Intent)
    result = llm.invoke([
        SystemMessage(f"""Classify the user intent. Context:
- GPX file uploaded: {has_gpx}
- Wikiloc URL in conversation: {has_url}

Rules:
- route_analysis: user wants a safety verdict, OR has GPX/URL, OR is selecting a route from a previous list
- route_search: user mentions a mountain/route by name, no GPX/URL, not selecting from a list
- gear_question: user asks about gear, techniques, safety practices, OR the message is off-topic/unrelated
"""),
        state.messages[-1],
    ])
    return {"intent": result.intent}


# Compiled with MemorySaver for full session memory across chat turns
web_app = build_graph().compile(checkpointer=MemorySaver())

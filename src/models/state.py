from typing import Annotated, Any, Optional, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from src.models.geometry import ResolvedGeometry


class AgentState(BaseModel):
    """Conversational agent state.

    messages holds the full conversation history (human, AI, tool messages).
    The add_messages reducer appends new messages rather than overwriting.

    geometry / weather_data / verdict / report are populated by the
    analyze_route tool and used by the frontend side panel.
    gpx_input is set when the user attaches a GPX file in the chat.
    intent is set by intent_router and used to guide chat_agent.
    """

    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages] = []

    # GPX bytes if user uploaded a file in the chat
    gpx_input: Optional[bytes] = None

    # Populated by analyze_route tool — drives the frontend side panel
    geometry: Optional[ResolvedGeometry] = None
    weather_data: Optional[dict[str, Any]] = None
    verdict: Optional[str] = None
    report: Optional[dict[str, Any]] = None

    # Set by intent_router — used to guide chat_agent tool selection
    intent: Optional[str] = None


class Intent(BaseModel):
    intent: Literal["route_analysis", "gear_question", "route_search"]

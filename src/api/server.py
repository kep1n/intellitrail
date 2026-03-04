from dotenv import load_dotenv
_ = load_dotenv()

import io
import logging
import logging.handlers
from pathlib import Path

_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "app.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB per file, 3 backups
    backupCount=3,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
# Root logger stays at WARNING — suppress third-party noise
logging.getLogger().setLevel(logging.WARNING)
# Our modules log at DEBUG
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("src").addHandler(_handler)
import uuid
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from openai import OpenAI

from src.agent.graph import web_app
from src.api.streaming import extract_final_state, format_sse
from src.config import Settings

app = FastAPI(title="Intellitrail")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat(
    request: Request,
    message: Annotated[str, Form()],
    thread_id: Annotated[Optional[str], Form()] = None,
    gpx_file: Annotated[Optional[UploadFile], File()] = None,
):
    """Conversational chat endpoint. Returns an SSE stream.

    Each call represents one user turn. The thread_id ties turns together
    so the agent remembers the full conversation. If thread_id is absent a
    new conversation is started and the generated ID is emitted as the first
    SSE event so the frontend can store it.

    Form fields:
    - message:   user text (required)
    - thread_id: existing conversation ID (optional; omit to start fresh)
    - gpx_file:  GPX file attachment (optional; stored in state for analyze_route)

    SSE events emitted:
    - thread_id:  {"thread_id": "..."}  — always first
    - tool_start: {"tool": "..."}       — when the agent invokes a tool
    - token:      {"text": "..."}       — streaming tokens of the agent reply
    - result:     {verdict, report, trail_info} — after stream, if a verdict exists
    - error:      {"message": "..."}    — on unhandled exception
    - done:       {}                    — always last
    """
    active_thread = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": active_thread}}

    gpx_bytes = await gpx_file.read() if gpx_file and gpx_file.filename else None

    # Always include the new human message; only overwrite gpx_input when a file is attached
    input_data: dict = {"messages": [HumanMessage(content=message)]}
    if gpx_bytes:
        input_data["gpx_input"] = gpx_bytes

    async def generate():
        yield format_sse("thread_id", {"thread_id": active_thread})

        emitted_tools: set[str] = set()
        try:
            async for mode, chunk in web_app.astream(
                input_data,
                config=config,
                stream_mode=["messages"],
            ):
                if await request.is_disconnected():
                    break

                if mode == "messages":
                    msg, meta = chunk
                    node = meta.get("langgraph_node", "")

                    if node == "chat_agent":
                        # Stream agent reply tokens
                        content = getattr(msg, "content", "")
                        if content:
                            yield format_sse("token", {"text": content})

                        # Detect tool calls and emit tool_start once per tool name
                        tool_call_chunks = getattr(msg, "tool_call_chunks", []) or []
                        for tc in tool_call_chunks:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                            if name and name not in emitted_tools:
                                emitted_tools.add(name)
                                yield format_sse("tool_start", {"tool": name})

            if not await request.is_disconnected():
                final = web_app.get_state(config)
                yield format_sse("result", extract_final_state(final))

        except Exception as exc:
            yield format_sse("error", {"message": str(exc)})
        finally:
            yield format_sse("done", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/transcribe")
async def transcribe(audio: Annotated[UploadFile, File()]):
    """Transcribe browser audio (webm blob) to text using OpenAI Whisper.

    Browser sends audio/webm from MediaRecorder. Server returns {"text": "..."}.
    """
    from src.agent.voice_recorder import VoiceInputError, validate_transcription
    from fastapi import HTTPException

    audio_bytes = await audio.read()
    buf = io.BytesIO(audio_bytes)
    buf.name = "recording.webm"

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
            response_format="text",
        )
        text = str(transcript).strip()
        validated = validate_transcription(text)
        return {"text": validated}
    except VoiceInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

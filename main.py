#!/usr/bin/env python3
"""CLI: python main.py (--gpx PATH | --query TEXT | --voice)"""

import sys
import json
import argparse
from src.agent.graph import app
from src.models.state import AgentState


def _run_pipeline(initial_state: AgentState) -> AgentState:
    """Invoke the LangGraph app and return a typed AgentState."""
    result_dict = app.invoke(initial_state.model_dump())
    return AgentState.model_validate(result_dict)


def _print_result(result: AgentState) -> None:
    """Print the result as formatted JSON, including Phase 5 fields if present."""
    output: dict = {}
    if result.report:
        output.update(result.report)
    if result.gear_recommendations is not None:
        output["gear_recommendations"] = result.gear_recommendations
    if result.alternatives is not None:
        output["alternatives"] = result.alternatives
    print(json.dumps(output, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mountain Safety Agent — evaluate trail conditions before you go.",
        usage="python main.py (--gpx PATH | --query TEXT | --voice)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpx", metavar="PATH", help="path to a GPX file")
    group.add_argument("--query", metavar="TEXT", help="text search query")
    group.add_argument("--voice", action="store_true", help="record audio via microphone")

    args = parser.parse_args()

    if args.gpx:
        # Validate that the file is readable before invoking the pipeline.
        try:
            with open(args.gpx, "rb") as f:
                f.read(1)  # confirm readability; actual parsing happens in parse_gpx node
        except OSError:
            print(f"GPX file not found or unreadable: {args.gpx}")
            sys.exit(1)

        initial_state = AgentState(gpx_file_path=args.gpx)
        result = _run_pipeline(initial_state)

        # If parse_gpx returned an ingestion error, fall back to text input.
        if result.error_code in ("INVALID_GPX", "EMPTY_TRACK", "NO_TRACKPOINTS"):
            print(f"GPX file is invalid or corrupt: {result.error}")
            print("Please describe the summit or track by text instead.")
            user_text = input("Enter summit or track description: ")
            initial_state = AgentState(text_query=user_text)
            result = _run_pipeline(initial_state)

    elif args.query:
        initial_state = AgentState(text_query=args.query)
        result = _run_pipeline(initial_state)

    else:  # --voice
        initial_state = AgentState(voice_mode=True)
        result = _run_pipeline(initial_state)

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    _print_result(result)


if __name__ == "__main__":
    main()

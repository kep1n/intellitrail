"""Temporary error-path verification script for plan 04-04. Delete after use."""
import os
os.environ['SERP_API_KEY'] = 'invalid'

from src.models.state import AgentState
from src.agent.nodes import resolve_text_query

state = AgentState(text_query='Mulhacen')
result = resolve_text_query(state)
print('Error message (expected):', result.get('error', 'NO ERROR')[:120])

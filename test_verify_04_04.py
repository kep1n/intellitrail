"""Temporary verification script for plan 04-04. Delete after use."""
from src.models.state import AgentState
from src.agent.nodes import resolve_text_query

state = AgentState(text_query='Mulhacen sierra nevada ruta')
print('Calling resolve_text_query...')
result = resolve_text_query(state)
print('Result keys:', list(result.keys()))

if 'error' in result:
    print('ERROR:', result['error'])
else:
    geom = result['geometry']
    print(f'Track: {geom.track_name}')
    print(f'Coordinates: {len(geom.coordinates)} points')
    print(f'Distance: {geom.distance_2d_km:.2f} km')
    print(f'Elevation (all None expected): max={geom.max_elevation_m}')
    print(f'URL: {result["selected_wikiloc_url"]}')

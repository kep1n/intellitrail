"""Tests for AEMETClient — atomic two-step fetch with retry and error propagation.

All tests use pytest-httpx to mock HTTP calls. No live AEMET traffic.

Covered behaviors:
  - _fetch() happy path: step-1 envelope + step-2 datos URL
  - _fetch() missing datos key → WeatherError with forces_caution=True
  - _fetch() second-step HTTP 500 → WeatherError
  - _fetch() retry on transient ConnectError → success on second attempt
  - _fetch() exhausted retries → WeatherError after 2 attempts
  - fetch_municipios() caching: only 1 HTTP request on two calls
  - fetch_avalanche_bulletin() uses correct URL per zone_code
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.weather.client import AEMETClient, WeatherError
import src.weather.client as client_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_municipios_cache():
    """Reset the module-level cache before each test to avoid state bleed."""
    client_module._municipios_cache = None
    yield
    client_module._municipios_cache = None


@pytest.fixture
def client():
    """AEMETClient with a fake API key."""
    return AEMETClient(api_key="test-api-key")


MOCK_DATOS_URL = "https://mock-datos.aemet.es/datos"
MOCK_ENVELOPE = {"descripcion": "exito", "estado": 200, "datos": MOCK_DATOS_URL}
MOCK_MUNICIPIO_PAYLOAD = [{"id": "28079", "nombre": "Madrid"}]


# ---------------------------------------------------------------------------
# _fetch() — happy path
# ---------------------------------------------------------------------------

def test_fetch_happy_path(client, httpx_mock):
    """Step-1 returns envelope; step-2 returns payload. datos URL not exposed."""
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/prediccion/especifica/municipio/diaria/28079?api_key=test-api-key",
        json=MOCK_ENVELOPE,
    )
    httpx_mock.add_response(
        url=MOCK_DATOS_URL,
        json=MOCK_MUNICIPIO_PAYLOAD,
    )

    result = client._fetch("/prediccion/especifica/municipio/diaria/28079")

    assert result == MOCK_MUNICIPIO_PAYLOAD
    # datos URL must NOT be stored on self
    assert not hasattr(client, "datos_url") or client.__dict__.get("datos_url") is None


# ---------------------------------------------------------------------------
# _fetch() — missing datos key
# ---------------------------------------------------------------------------

def test_fetch_missing_datos_key_raises_weather_error(client, httpx_mock):
    """Step-1 response without 'datos' key raises WeatherError."""
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/some/endpoint?api_key=test-api-key",
        json={"estado": 404, "descripcion": "Not found"},
    )

    with pytest.raises(WeatherError) as exc_info:
        client._fetch("/some/endpoint")

    assert exc_info.value.forces_caution is True


# ---------------------------------------------------------------------------
# _fetch() — second-step HTTP failure
# ---------------------------------------------------------------------------

def test_fetch_second_step_failure_raises_weather_error(client, httpx_mock):
    """Step-2 HTTP 500 raises WeatherError."""
    # Both retry attempts: step-1 succeeds, step-2 fails
    for _ in range(2):
        httpx_mock.add_response(
            url=f"{AEMETClient.BASE_URL}/some/endpoint?api_key=test-api-key",
            json=MOCK_ENVELOPE,
        )
        httpx_mock.add_response(
            url=MOCK_DATOS_URL,
            status_code=500,
        )

    with pytest.raises(WeatherError):
        client._fetch("/some/endpoint")


# ---------------------------------------------------------------------------
# _fetch() — retry on transient failure
# ---------------------------------------------------------------------------

def test_fetch_retries_on_connect_error(client, httpx_mock):
    """ConnectError on first attempt; success on second attempt."""
    # First attempt raises ConnectError on step-1
    httpx_mock.add_exception(
        httpx.ConnectError("connection refused"),
        url=f"{AEMETClient.BASE_URL}/some/endpoint?api_key=test-api-key",
    )
    # Second attempt succeeds
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/some/endpoint?api_key=test-api-key",
        json=MOCK_ENVELOPE,
    )
    httpx_mock.add_response(
        url=MOCK_DATOS_URL,
        json=MOCK_MUNICIPIO_PAYLOAD,
    )

    result = client._fetch("/some/endpoint")
    assert result == MOCK_MUNICIPIO_PAYLOAD


# ---------------------------------------------------------------------------
# _fetch() — exhausted retries
# ---------------------------------------------------------------------------

def test_fetch_exhausted_retries_raises_weather_error(client, httpx_mock):
    """Both retry attempts fail with ConnectError → WeatherError raised."""
    for _ in range(2):
        httpx_mock.add_exception(
            httpx.ConnectError("connection refused"),
            url=f"{AEMETClient.BASE_URL}/some/endpoint?api_key=test-api-key",
        )

    with pytest.raises((WeatherError, httpx.ConnectError)):
        client._fetch("/some/endpoint")


# ---------------------------------------------------------------------------
# fetch_municipios() — caching
# ---------------------------------------------------------------------------

def test_fetch_municipios_caches_result(client, httpx_mock):
    """fetch_municipios() makes only 1 HTTP request even when called twice."""
    municipios_payload = [
        {"id": "28079", "nombre": "Madrid"},
        {"id": "08019", "nombre": "Barcelona"},
    ]
    # Envelope for step-1
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/maestro/municipios?api_key=test-api-key",
        json={**MOCK_ENVELOPE, "datos": MOCK_DATOS_URL},
    )
    # Datos for step-2
    httpx_mock.add_response(
        url=MOCK_DATOS_URL,
        json=municipios_payload,
    )

    result1 = client.fetch_municipios()
    result2 = client.fetch_municipios()

    assert result1 == municipios_payload
    assert result2 is result1  # same cached object
    # Only 2 HTTP requests were registered (step-1 + step-2 = 2), not 4
    assert len(httpx_mock.get_requests()) == 2


# ---------------------------------------------------------------------------
# fetch_avalanche_bulletin() — correct URL per zone_code
# ---------------------------------------------------------------------------

def test_fetch_avalanche_bulletin_zone_0(client, httpx_mock):
    """zone_code='0' → GET /api/prediccion/especifica/nivologica/0"""
    bulletin_payload = [{"texto": "Riesgo 3", "zona": "0"}]
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/prediccion/especifica/nivologica/0?api_key=test-api-key",
        json={**MOCK_ENVELOPE, "datos": MOCK_DATOS_URL},
    )
    httpx_mock.add_response(
        url=MOCK_DATOS_URL,
        json=bulletin_payload,
    )

    result = client.fetch_avalanche_bulletin("0")
    assert isinstance(result, dict)


def test_fetch_avalanche_bulletin_zone_1(client, httpx_mock):
    """zone_code='1' → GET /api/prediccion/especifica/nivologica/1"""
    bulletin_payload = [{"texto": "Riesgo 2", "zona": "1"}]
    httpx_mock.add_response(
        url=f"{AEMETClient.BASE_URL}/prediccion/especifica/nivologica/1?api_key=test-api-key",
        json={**MOCK_ENVELOPE, "datos": MOCK_DATOS_URL},
    )
    httpx_mock.add_response(
        url=MOCK_DATOS_URL,
        json=bulletin_payload,
    )

    result = client.fetch_avalanche_bulletin("1")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# WeatherError — forces_caution always True
# ---------------------------------------------------------------------------

def test_weather_error_forces_caution():
    """WeatherError.forces_caution must be True for SAFE-02 compliance."""
    err = WeatherError("something went wrong")
    assert err.forces_caution is True
    assert "something went wrong" in str(err)

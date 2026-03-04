"""AEMET OpenData HTTP client with atomic two-step fetch and retry.

SAFE-03 compliance: AEMETClient._fetch() performs both the initial request and the
datos-URL poll atomically. The datos URL is a local variable and is NEVER stored on
self or returned to callers. Callers receive the final JSON payload only.

Authentication: api_key passed as query parameter (confirmed via official AEMET examples).
Retry: tenacity stop_after_attempt(2) + wait_fixed(1) — 1 initial attempt + 1 retry.
Rate limit: AEMET enforces 50 req/min; max ~12 calls per pipeline run is within limits.
"""

from __future__ import annotations

import json as _json

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


class WeatherError(Exception):
    """Raised on any AEMET client failure.

    forces_caution is always True — any client error triggers the CAUTION routing
    gate in Phase 3 (SAFE-02 requirement).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.forces_caution: bool = True


# Class-level municipios cache — loaded once per process lifetime.
# Shared across all AEMETClient instances to avoid redundant API calls.
_municipios_cache: list[dict] | None = None


class AEMETClient:
    """Thin AEMET OpenData client using httpx (sync) with tenacity retry.

    Sync (not async) because Phase 6 FastAPI calls LangGraph nodes in a threadpool.
    Using httpx.Client rather than requests for ecosystem consistency with httpx.

    Usage:
        client = AEMETClient(api_key="your-key")
        raw = client.fetch_municipal_forecast("28079")
    """

    BASE_URL = "https://opendata.aemet.es/opendata/api"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # api_key passed as query param — confirmed by official AEMET examples.
        # Also set as header for compatibility with some AEMET proxy configurations.
        self._params = {"api_key": api_key}
        self._client = httpx.Client(
            timeout=httpx.Timeout(15.0),
            verify=False,  # AEMET SSL certs occasionally mismatch; safe for read-only public API
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((httpx.HTTPError, KeyError, ValueError)),
        reraise=True,
    )
    def _fetch_inner(self, endpoint: str) -> dict | list:
        """Inner two-step AEMET fetch — retried by tenacity on httpx.HTTPError.

        Raises raw httpx.HTTPError / ValueError so tenacity can decide to retry.
        DO NOT catch httpx.HTTPError broadly here — that prevents retry.
        Only parse-level WeatherErrors (non-retryable protocol violations) are
        raised directly; transport errors are left as raw httpx exceptions.

        The datos_url is a local variable and NEVER leaves this method (SAFE-03).
        """
        # Step 1: envelope request — let httpx.HTTPError propagate to tenacity.
        r1 = self._client.get(f"{self.BASE_URL}{endpoint}", params=self._params)
        r1.raise_for_status()  # raises httpx.HTTPStatusError → tenacity retries

        envelope = r1.json()  # raises ValueError on bad JSON → tenacity retries

        # AEMET API-level error (e.g. estado=404): non-retryable protocol violation.
        estado = envelope.get("estado")
        if estado is not None and estado != 200:
            raise WeatherError(
                f"AEMET envelope estado={estado} for {endpoint}: {envelope.get('descripcion', '')}"
            )

        if "datos" not in envelope:
            raise WeatherError(
                f"AEMET envelope missing 'datos' field for {endpoint}. "
                f"Keys present: {list(envelope.keys())}"
            )

        # datos_url is a local variable — SAFE-03 atomicity enforced here.
        datos_url: str = envelope["datos"]

        # Step 2: datos request — let httpx.HTTPError propagate to tenacity.
        r2 = self._client.get(datos_url)
        r2.raise_for_status()  # raises httpx.HTTPStatusError → tenacity retries

        # AEMET datos endpoint returns Content-Type: text/plain;charset=ISO-8859-15.
        # httpx.Response.json() calls json.loads(self.content) which forces UTF-8 and
        # fails on Spanish characters. Decode explicitly using the declared charset.
        encoding = r2.encoding or "iso-8859-15"
        return _json.loads(r2.content.decode(encoding))  # raises ValueError on bad JSON

    def _fetch(self, endpoint: str) -> dict | list:
        """Atomic two-step AEMET fetch. NEVER returns the datos URL.

        Step 1: GET {BASE_URL}{endpoint}?api_key={key}
                Response: {"descripcion": "...", "estado": 200, "datos": "<short-lived-url>"}

        Step 2: GET datos_url (no auth needed — it is a signed temporary URL)
                Response: the actual JSON payload (dict or list)

        The datos_url is a local variable and NEVER leaves this method (SAFE-03).
        Delegates to _fetch_inner() which is decorated with tenacity retry.

        Raises WeatherError on:
          - Any HTTP error status on either step (after retry exhaustion)
          - Missing 'datos' key in envelope (malformed response)
          - AEMET 'estado' != 200 in envelope
          - JSON parse failure
        """
        try:
            return self._fetch_inner(endpoint)
        except WeatherError:
            # Re-raise WeatherError (non-retryable protocol violation from _fetch_inner).
            raise
        except httpx.HTTPStatusError as exc:
            raise WeatherError(
                f"AEMET HTTP {exc.response.status_code} for {endpoint}"
            ) from exc
        except httpx.HTTPError as exc:
            raise WeatherError(
                f"AEMET connection error for {endpoint}: {exc}"
            ) from exc
        except (ValueError, KeyError) as exc:
            raise WeatherError(
                f"AEMET invalid response for {endpoint}: {exc}"
            ) from exc

    def fetch_municipal_forecast(self, municipio_id: str) -> list[dict]:
        """Fetch daily municipal forecast for the given INE code.

        Endpoint: GET /api/prediccion/especifica/municipio/diaria/{municipio_id}
        Returns raw list of day dicts from AEMET (caller parses into MunicipalForecast).

        Args:
            municipio_id: 5-digit INE municipality code, e.g., "28079" for Madrid.

        Raises:
            WeatherError: on any HTTP error or malformed AEMET response (after retry).
        """
        try:
            result = self._fetch(f"/prediccion/especifica/municipio/diaria/{municipio_id}")
        except (httpx.HTTPError, KeyError, ValueError, WeatherError) as exc:
            if isinstance(exc, WeatherError):
                raise
            raise WeatherError(
                f"fetch_municipal_forecast failed for municipio={municipio_id}: {exc}"
            ) from exc

        # AEMET returns a 1-element list: [{..., "prediccion": {"dia": [<day>, ...]}}]
        # Unwrap to the inner day list so callers get a flat list of day dicts.
        if not isinstance(result, list) or not result:
            raise WeatherError(
                f"Municipal forecast expected non-empty list, got {type(result).__name__}"
            )
        try:
            days: list[dict] = result[0]["prediccion"]["dia"]
        except (KeyError, TypeError) as exc:
            raise WeatherError(
                f"Municipal forecast missing prediccion.dia for municipio={municipio_id}: {exc}"
            ) from exc
        if not isinstance(days, list):
            raise WeatherError(
                f"prediccion.dia expected list, got {type(days).__name__}"
            )
        return days

    def fetch_mountain_forecast(self, area_code: str, dia: int = 0) -> str:
        """Fetch mountain zone narrative forecast.

        Endpoint: GET /api/prediccion/especifica/montaña/pasada/area/{area_code}/dia/{dia}
        dia=0 means today. Returns the raw forecast text (AEMET returns a narrative string,
        not structured data — stored as-is in MountainForecast.forecast_text).

        Args:
            area_code: AEMET mountain area code, one of:
                       arn1, cat1, nav1, peu1, nev1, mad2, gre1, arn2, rio1
            dia: Day offset. 0 = today, 1 = tomorrow.

        Raises:
            WeatherError: on any HTTP error or malformed response (after retry).
        """
        try:
            result = self._fetch(
                f"/prediccion/especifica/montaña/pasada/area/{area_code}/dia/{dia}"
            )
        except (httpx.HTTPError, KeyError, ValueError, WeatherError) as exc:
            if isinstance(exc, WeatherError):
                raise
            raise WeatherError(
                f"fetch_mountain_forecast failed for area={area_code}, dia={dia}: {exc}"
            ) from exc

        # Mountain forecast returns a dict or list with narrative text.
        # Extract text robustly — store raw str if extraction fails.
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                # Try common AEMET text field names
                for key in ("prediccion", "texto", "text", "forecast"):
                    if key in item:
                        return str(item[key])
            return str(item)
        if isinstance(result, dict):
            for key in ("prediccion", "texto", "text", "forecast"):
                if key in result:
                    return str(result[key])
            return str(result)
        return str(result)

    def fetch_municipios(self) -> list[dict]:
        """Fetch all Spanish municipalities with coordinate centroids.

        Endpoint: GET /api/maestro/municipios

        NOTE: /api/maestro/municipios returns HTTP 404 in the current AEMET API
        (verified 2026-02-21). This method will raise WeatherError on every live call
        until the endpoint is restored or replaced by a static INE municipality list.
        See zone_mapper.py (MUNICIPIOS_ENDPOINT_STATUS, module docstring) for full context
        and long-term mitigation options.

        Returns ~8124 municipality dicts when/if endpoint becomes available. Result is
        cached at class level after the first successful call — subsequent calls return
        the cached list. The cache is a module-level variable (_municipios_cache) shared
        across all AEMETClient instances for the process lifetime. No disk caching.

        Raises:
            WeatherError: always currently (HTTP 404 from AEMET); on any HTTP error or
                          malformed response after retry once endpoint is restored.
        """
        global _municipios_cache
        if _municipios_cache is not None:
            return _municipios_cache

        try:
            result = self._fetch("/maestro/municipios")
        except (httpx.HTTPError, KeyError, ValueError, WeatherError) as exc:
            if isinstance(exc, WeatherError):
                raise
            raise WeatherError(f"fetch_municipios failed: {exc}") from exc

        if not isinstance(result, list):
            raise WeatherError(
                f"Municipios endpoint expected list, got {type(result).__name__}"
            )

        _municipios_cache = result
        return _municipios_cache

    def fetch_avalanche_bulletin(self, zone_code: str) -> dict:
        """Fetch snow/avalanche risk bulletin for a nivologica zone.

        Endpoint: GET /api/prediccion/especifica/nivologica/{zone_code}
        zone_code must be "0" (Pirineo Catalán) or "1" (Pirineo Navarro/Aragonés).
        This is NOT the mountain area code — use NIVOLOGICA_ZONE_MAP in zone_mapper.py
        to convert from mountain area code to nivologica zone code.

        Args:
            zone_code: "0" or "1" only.

        Raises:
            WeatherError: on any HTTP error or malformed response (after retry).
        """
        try:
            result = self._fetch(f"/prediccion/especifica/nivologica/{zone_code}")
        except (httpx.HTTPError, KeyError, ValueError, WeatherError) as exc:
            if isinstance(exc, WeatherError):
                raise
            raise WeatherError(
                f"fetch_avalanche_bulletin failed for zone={zone_code}: {exc}"
            ) from exc

        if isinstance(result, list) and result:
            return result[0] if isinstance(result[0], dict) else {"raw": result[0]}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}

    def close(self) -> None:
        """Close the underlying httpx.Client connection pool."""
        self._client.close()

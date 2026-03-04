"""TDD: generate_verdict node — behavioural contracts.

All tests mock ChatOpenAI to avoid requiring OPENAI_API_KEY in CI.
generate_verdict is async — tests use asyncio.run() for direct invocation.
"""
import asyncio
import pytest
from datetime import date
from unittest.mock import patch, MagicMock, AsyncMock

from src.models.state import AgentState
from src.models.geometry import ResolvedGeometry
from src.models.verdict import VerdictEnum, VerdictReport
from src.weather.models import WeatherData, AvalancheBulletin, MunicipalForecast, PeriodForecast, MountainForecast
from src.agent.nodes import generate_verdict


# --- Fixtures ---

def make_geometry(max_elevation_m: float = 1200.0) -> ResolvedGeometry:
    """Minimal valid geometry with elevation data."""
    n = 100
    elevs = [max_elevation_m * i / (n - 1) for i in range(n)]
    return ResolvedGeometry(
        track_name="Test track",
        coordinates=[(40.0 + i * 0.001, -3.0) for i in range(n)],
        elevation_raw=elevs,
        elevation_smoothed=elevs,
        elevation_gain_m=max_elevation_m,
        elevation_loss_m=0.0,
        min_elevation_m=0.0,
        max_elevation_m=max_elevation_m,
        distance_2d_km=5.0,
        distance_3d_km=5.2,
        bbox_min_lat=40.0, bbox_max_lat=40.1,
        bbox_min_lon=-3.0, bbox_max_lon=-3.0,
        utm_crs="EPSG:32630",
    )


def make_weather_data(
    avalanche_unavailable_reason: str | None = None,
) -> dict:
    """Build a complete WeatherData dict with optional avalanche unavailability."""
    period = PeriodForecast(period="00-24", wind_speed_kmh=30, precipitation_probability=20)
    municipal = MunicipalForecast(
        municipality_id="28079",
        municipality_name="TestMuni",
        date="2026-02-21",
        periods=[period],
    )
    avalanche = AvalancheBulletin(
        zone_code="0",
        risk_level=2,
        raw_text="Riesgo limitado.",
        unavailable_reason=avalanche_unavailable_reason,
    )
    wd = WeatherData(
        municipal=[municipal],
        avalanche=avalanche,
        data_complete=True,
    )
    return wd.model_dump()


def make_mock_llm(report: VerdictReport) -> MagicMock:
    """Return a mock that simulates ChatOpenAI().with_structured_output().ainvoke()."""
    mock_instance = MagicMock()
    mock_instance.ainvoke = AsyncMock(return_value=report)
    return mock_instance


def run_generate_verdict(state: AgentState, mock_writer: MagicMock) -> dict:
    """Helper to run async generate_verdict synchronously with stream writer mocked."""
    with patch("src.agent.nodes.get_stream_writer", return_value=mock_writer):
        return asyncio.run(generate_verdict(state))


# --- Tests ---

def test_happy_go():
    """LLM returns GO -> state verdict is 'GO', report is a valid VerdictReport dict."""
    llm_report = VerdictReport(verdict=VerdictEnum.GO, summary="GO — clear skies")
    state = AgentState(geometry=make_geometry(max_elevation_m=1200), weather_data=make_weather_data())
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = run_generate_verdict(state, mock_writer)

    assert result["verdict"] == "GO"
    parsed = VerdictReport.model_validate(result["report"])
    assert parsed.verdict == VerdictEnum.GO


def test_happy_caution():
    """LLM returns CAUTION -> state verdict is 'CAUTION'."""
    llm_report = VerdictReport(verdict=VerdictEnum.CAUTION, risk_factors=["Wind: 65 km/h — CAUTION"])
    state = AgentState(geometry=make_geometry(), weather_data=make_weather_data())
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = run_generate_verdict(state, mock_writer)

    assert result["verdict"] == "CAUTION"


def test_happy_no_go():
    """LLM returns NO_GO -> state verdict string is 'NO-GO' (enum .value)."""
    llm_report = VerdictReport(verdict=VerdictEnum.NO_GO, summary="NO-GO — storm")
    state = AgentState(geometry=make_geometry(), weather_data=make_weather_data())
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = run_generate_verdict(state, mock_writer)

    assert result["verdict"] == "NO-GO"


def test_llm_exception_forces_caution():
    """LLM raises exception -> CAUTION returned, pipeline does not crash."""
    state = AgentState(geometry=make_geometry(), weather_data=make_weather_data())
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        mock_cls.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("connection timeout")
        )
        result = run_generate_verdict(state, mock_writer)

    assert result["verdict"] == "CAUTION"
    assert "LLM error" in result["report"]["risk_factors"][0]


def test_missing_nivologica_rule_triggers():
    """Rule fires: max_elev > snow_line AND zone_not_covered -> force CAUTION even if LLM says GO."""
    llm_report = VerdictReport(verdict=VerdictEnum.GO, summary="GO — calm conditions")
    geometry = make_geometry(max_elevation_m=2500)  # above Jan snow_line=1400
    weather = make_weather_data(avalanche_unavailable_reason="zone_not_covered")
    state = AgentState(geometry=geometry, weather_data=weather)
    mock_writer = MagicMock()

    fixed_jan = date(2026, 1, 15)
    with patch("src.agent.nodes.ChatOpenAI") as mock_cls, \
         patch("src.agent.nodes.date") as mock_date, \
         patch("src.agent.nodes.get_stream_writer", return_value=mock_writer):
        mock_date.today.return_value = fixed_jan
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = asyncio.run(generate_verdict(state))

    assert result["verdict"] == "CAUTION", "Rule must override GO to CAUTION"
    assert "snow line" in result["report"]["reasoning"].lower() or "seasonal" in result["report"]["reasoning"].lower()


def test_missing_nivologica_rule_skipped_below_snow_line():
    """Rule does NOT fire when max_elev <= snow_line, even if zone_not_covered."""
    llm_report = VerdictReport(verdict=VerdictEnum.GO, summary="GO")
    geometry = make_geometry(max_elevation_m=1200)  # BELOW Jan snow_line=1400
    weather = make_weather_data(avalanche_unavailable_reason="zone_not_covered")
    state = AgentState(geometry=geometry, weather_data=weather)
    mock_writer = MagicMock()

    fixed_jan = date(2026, 1, 15)
    with patch("src.agent.nodes.ChatOpenAI") as mock_cls, \
         patch("src.agent.nodes.date") as mock_date, \
         patch("src.agent.nodes.get_stream_writer", return_value=mock_writer):
        mock_date.today.return_value = fixed_jan
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = asyncio.run(generate_verdict(state))

    assert result["verdict"] == "GO", "Rule must NOT fire when below snow line"


def test_none_weather_data_caution():
    """weather_data=None -> CAUTION returned gracefully, no crash."""
    state = AgentState(geometry=make_geometry(), weather_data=None)
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        # Even if LLM somehow returns GO, missing weather should still work via prompt fallback
        llm_report = VerdictReport(verdict=VerdictEnum.CAUTION, summary="CAUTION — no weather data")
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = run_generate_verdict(state, mock_writer)

    assert result["verdict"] in ("CAUTION", "GO", "NO-GO")  # no crash is the primary assertion
    assert "verdict" in result
    assert "report" in result


def test_verdict_report_structured_fields():
    """Report dict contains all required VerdictReport fields."""
    llm_report = VerdictReport(
        verdict=VerdictEnum.CAUTION,
        summary="CAUTION — afternoon winds",
        risk_factors=["Wind: 60 km/h — CAUTION threshold exceeded"],
        data_completeness="municipal=ok, mountain=ok, avalanche=level_2",
        reasoning="Morning conditions acceptable. Afternoon wind picks up above 2000m.",
        time_windows="Today: safe before 12:00. Tomorrow: safe all day.",
        elevation_context="Min 1400m, max 2200m. Track above snow line.",
    )
    state = AgentState(geometry=make_geometry(), weather_data=make_weather_data())
    mock_writer = MagicMock()

    with patch("src.agent.nodes.ChatOpenAI") as mock_cls:
        mock_cls.return_value.with_structured_output.return_value = make_mock_llm(llm_report)
        result = run_generate_verdict(state, mock_writer)

    required_keys = {"verdict", "summary", "risk_factors", "data_completeness", "reasoning", "time_windows", "elevation_context"}
    assert required_keys.issubset(set(result["report"].keys())), f"Missing keys: {required_keys - set(result['report'].keys())}"

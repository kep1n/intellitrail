import enum
from pydantic import BaseModel, Field


class VerdictEnum(str, enum.Enum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO-GO"


class VerdictReport(BaseModel):
    verdict: VerdictEnum
    summary: str = ""            # one-line label, e.g. "CAUTION — Strong winds above 2200m after 13:00"
    risk_factors: list[str] = Field(default_factory=list)  # e.g. ["Wind 85 km/h at summit — exceeds NO-GO threshold"]
    data_completeness: str = ""  # e.g. "municipal_forecast=ok, mountain=missing, avalanche=out_of_season"
    reasoning: str = ""          # bullets + narrative per factor; factor + value + implication
    time_windows: str = ""       # narrative: today and tomorrow safe/unsafe bands
    elevation_context: str = ""  # narrative: min/max/gain + altitude band conditions

"""TripPlan — output contract for the planning phase (phase 3).

The first artifact in the system that a user reads and uses directly,
and what phase 4 (critique) evaluates. Combines two different kinds of
decisions:
  1. LLM decisions (selection, day distribution, lodging/transport)
     -- qualitative judgment, but always over constraints already
     resolved in code (capacity, geographic clusters, closed_weekdays).
  2. Pure arithmetic (budget) -- only closes once a concrete itinerary
     exists to sum.

`brief_version`/`pack_id` follow the same invalidation discipline as
ResearchPack: a plan generated against an old brief is garbage once
the brief version changes.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import datetime

from pydantic import Field

from ..intake_agent.trip_brief import Base
from ..research_pipeline.research_pack import Price


class ScheduledPoi(Base):
    poi_id: str
    name: str
    block: str = Field(..., description="morning, afternoon, or evening")
    duration_min: int | None = None
    notes: str | None = Field(None, description="e.g. book ahead, closed-weekday swap explanation")


class DayPlan(Base):
    day_index: int
    date: _date | None = None
    weekday: int | None = Field(None, description="0=Monday ... 6=Sunday, null if exact dates aren't set")
    city: str
    items: list[ScheduledPoi] = Field(default_factory=list)
    notes: str | None = Field(None, description="Free-day suggestion, caveat, or other day-level note")


class LodgingChoice(Base):
    city: str
    area: str = Field(..., description="Neighborhood/region to stay in")
    nights: int
    rationale: str | None = None
    est_price_per_night: Price | None = None


class TransportLeg(Base):
    from_city: str
    to_city: str
    mode: str = Field(..., description="flight, train, bus, etc.")
    notes: str | None = None


class BudgetLine(Base):
    category: str = Field(..., description="flights, lodging, activities, food, local_transit")
    low: float
    high: float
    currency: str = "BRL"


class SelectionResult(Base):
    """LLM output_schema for the selection step."""
    selected_poi_ids: list[str] = Field(default_factory=list)
    dropped_poi_ids: list[str] = Field(default_factory=list)


class LodgingTransportResult(Base):
    """LLM output_schema for the lodging+transport step."""
    lodging: list[LodgingChoice] = Field(default_factory=list)
    transport: list[TransportLeg] = Field(default_factory=list)


class TripPlan(Base):
    plan_id: str
    brief_id: str
    brief_version: int
    pack_id: str
    generated_at: datetime | None = None

    days: list[DayPlan] = Field(default_factory=list)
    lodging: list[LodgingChoice] = Field(default_factory=list)
    transport: list[TransportLeg] = Field(default_factory=list)
    budget: list[BudgetLine] = Field(default_factory=list)
    budget_total: Price | None = None
    dropped_pois: list[str] = Field(default_factory=list, description="POI names cut during selection")
    notes: list[str] = Field(default_factory=list, description="Caveats/assumptions worth surfacing to the user")

    def stale(self, brief) -> bool:
        return brief.version != self.brief_version

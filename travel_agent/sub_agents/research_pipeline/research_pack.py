"""
ResearchPack — state contract for the research phase (minimal version).

What's left is what phase 3 can't function without:
  1. Stops -- which cities the trip actually bases itself in, and how many
     nights each gets. Resolved once, by the research phase itself, from
     the brief's (possibly country/region-level) destinations.
  2. POI validated by geocoding, with duration and closing day.
  3. Numeric cost anchors, so the budget can close.
  4. Text facts, with a source, for the planner to read.
  5. `brief_version`, which invalidates the pack when the brief changes.
"""

from __future__ import annotations

from datetime import date, datetime
from math import ceil

from pydantic import Field

from ..intake_agent.trip_brief import Base

MIN_POIS_PER_NIGHT = 3
"""Coverage bar used by `coverage_shortfall` -- below this, a stop doesn't
have enough validated POIs for phase 3 to build a real itinerary."""


class Stop(Base):
    """One city the trip actually bases itself in, with how many nights it
    gets. Resolved by `base_resolver` from the brief's `destinations` --
    which may be country/region-level ("Japan") rather than a city -- so
    every downstream step (POI search, geo_clustering) has a real city and
    a real night count to work with, not a country label."""
    city: str
    country: str | None = None
    nights: int
    rationale: str | None = Field(None, description="Why this city, and why this many nights")


class Price(Base):
    low: float
    high: float
    currency: str = "BRL"
    as_of: date | None = None


class POI(Base):
    poi_id: str
    name: str
    city: str
    category: str | None = Field(None, description="landmark, museum, food, nature, neighborhood")
    validated: bool = Field(False, description="Filled in by the geocoding tool, never by the LLM")
    lat: float | None = None
    lng: float | None = None
    duration_min: int | None = Field(None, description="Typical visit duration")
    ticket: Price | None = None
    closed_weekdays: list[int] = Field(default_factory=list, description="0=Monday ... 6=Sunday")
    brief_item: str | None = Field(None, description="Brief attraction that originated this POI")
    source_url: str | None = None


class Fact(Base):
    """Anything that isn't a POI or a number: climate, transport, visa, warning."""
    topic: str = Field(..., description="season, transport, entry, local, warning")
    content: str = Field(..., max_length=600)
    source_url: str | None = None


class CostAnchors(Base):
    """Structured output of cost_scout -- same shape as ResearchPack.costs."""
    costs: dict[str, Price] = Field(
        default_factory=dict,
        description="Anchors for budgeting: flight, lodging_night, meal, transit_day",
    )


def coverage_shortfall(stops: list[Stop], validated_pois: list[POI]) -> dict[str, int]:
    """How many more validated POIs each stop needs to clear the coverage
    bar (`MIN_POIS_PER_NIGHT` per night actually allocated to that city).

    Returns {city: shortfall} only for stops below the bar; empty dict when
    there are no stops yet or coverage is already sufficient everywhere.
    """
    counts: dict[str, int] = {}
    for poi in validated_pois:
        counts[poi.city] = counts.get(poi.city, 0) + 1
    return {
        stop.city: ceil(MIN_POIS_PER_NIGHT * stop.nights) - counts.get(stop.city, 0)
        for stop in stops
        if counts.get(stop.city, 0) < ceil(MIN_POIS_PER_NIGHT * stop.nights)
    }


class ResearchPack(Base):
    pack_id: str
    brief_id: str
    brief_version: int
    complete: bool = False
    generated_at: datetime | None = None

    stops: list[Stop] = Field(default_factory=list)
    pois: list[POI] = Field(default_factory=list)
    costs: dict[str, Price] = Field(
        default_factory=dict,
        description="Anchors for budgeting: flight, lodging_night, meal, transit_day",
    )
    facts: list[Fact] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="What wasn't found")

    # ---- deterministic helpers ----

    def validated_pois(self) -> list[POI]:
        return [p for p in self.pois if p.validated]

    def missing_musts(self, brief) -> list[str]:
        found = {p.brief_item for p in self.validated_pois()}
        return [a.name for a in brief.attractions if a.priority == "must" and a.name not in found]

    def undercovered_cities(self) -> list[str]:
        return list(coverage_shortfall(self.stops, self.validated_pois()))

    def ready_for_planning(self, brief) -> bool:
        return (
            brief.version == self.brief_version
            and bool(self.stops)
            and not self.missing_musts(brief)
            and not self.undercovered_cities()
            and len(self.validated_pois()) > 0
        )

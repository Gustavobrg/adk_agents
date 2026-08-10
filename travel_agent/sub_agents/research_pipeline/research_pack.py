"""
ResearchPack — state contract for the research phase (minimal version).

What's left is what phase 3 can't function without:
  1. POI validated by geocoding, with duration and closing day.
  2. Numeric cost anchors, so the budget can close.
  3. Text facts, with a source, for the planner to read.
  4. `brief_version`, which invalidates the pack when the brief changes.
"""

from __future__ import annotations

from datetime import date, datetime
from math import ceil

from pydantic import Field

from ..intake_agent.trip_brief import Base

MIN_POIS_PER_NIGHT = 3
"""Coverage bar used by `coverage_shortfall` -- below this, a destination
doesn't have enough validated POIs for phase 3 to build a real itinerary."""


class Price(Base):
    low: float
    high: float
    currency: str = "BRL"
    as_of: date | None = None


class POI(Base):
    poi_id: str
    name: str
    city: str
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


def coverage_shortfall(
    cities: list[str], nights: int | None, validated_pois: list[POI]
) -> dict[str, int]:
    """How many more validated POIs each city needs to clear the coverage
    bar. Nights are split evenly across `cities` (the brief has no
    per-city night breakdown) -- so 40 POIs skewed 35/5 across two cities
    still shows up here, even though the total alone looks fine.

    Returns {city: shortfall} only for cities below the bar; empty dict
    when nights/cities aren't known yet or coverage is already sufficient
    everywhere.
    """
    if not cities or not nights:
        return {}
    threshold = ceil(MIN_POIS_PER_NIGHT * nights / len(cities))
    counts: dict[str, int] = {}
    for poi in validated_pois:
        counts[poi.city] = counts.get(poi.city, 0) + 1
    return {
        city: threshold - counts.get(city, 0)
        for city in cities
        if counts.get(city, 0) < threshold
    }


class ResearchPack(Base):
    pack_id: str
    brief_id: str
    brief_version: int
    complete: bool = False
    generated_at: datetime | None = None

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

    def undercovered_cities(self, brief) -> list[str]:
        cities = [d.name for d in brief.destinations if d.decided]
        return list(coverage_shortfall(cities, brief.hard.nights, self.validated_pois()))

    def ready_for_planning(self, brief) -> bool:
        return (
            brief.version == self.brief_version
            and not self.missing_musts(brief)
            and not self.undercovered_cities(brief)
            and len(self.validated_pois()) > 0
        )
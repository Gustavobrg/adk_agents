"""TripBrief — state contract for the discovery phase.

Design rules:
  1. Every Field `description` is a prompt. The LLM reads it when filling the schema.
  2. Hard constraint (HardConstraints) goes to deterministic validator.
     Soft preference (SoftPreferences) goes to the planner's prompt.
  3. Field None = not yet collected. Assumption = filled by agent without
     user confirmation. Phase 1 ends only when no blocking assumptions pending.
  4. `extra="forbid"` because most structured output backends require
     additionalProperties: false.
  5. REQUIRED blocks `ready_for_research()`. RECOMMENDED does not --
     only appears in `missing_recommended()` until filled or listed in `declined`.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class Pace(str, Enum):
    relaxed = "relaxed"      # 1-2 activities/day, lots of free time
    balanced = "balanced"    # 3-4 activities/day
    packed = "packed"        # 5+ activities, long days


class AgeBand(str, Enum):
    infant = "infant"
    child = "child"
    teen = "teen"
    adult = "adult"
    senior = "senior"


class Priority(str, Enum):
    must = "must"            # dropping this invalidates the trip
    nice = "nice"            # include if it fits
    optional = "optional"    # cut first if needed


class Origin(str, Enum):
    user = "user"            # user said this explicitly
    agent = "agent"          # agent inferred or proposed


class BriefStatus(str, Enum):
    draft = "draft"                # collection in progress
    confirmed = "confirmed"        # ready for research phase


class Verdict(str, Enum):
    feasible = "feasible"
    tight = "tight"
    infeasible = "infeasible"


class ConflictKind(str, Enum):
    budget_vs_scope = "budget_vs_scope"
    distance_vs_days = "distance_vs_days"
    docs_vs_dates = "docs_vs_dates"


# --------------------------------------------------------------------------
# Shared validation
# --------------------------------------------------------------------------

def _normalize_country_code(value: str) -> str:
    """Country is always ISO 3166-1 alpha-2 -- never the full name.

    Without this, country comparison (e.g., domestic vs international trip
    in _docs_lead_time_conflict) breaks silently when one side is "BR"
    (schema default) and the other is "Brazil" (what LLM tends to write
    if not forced).
    """
    code = value.strip().upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError(
            f"country must be ISO 3166-1 alpha-2, 2 letters (e.g., BR, JP, PT), "
            f"not the full name. Received: {value!r}"
        )
    return code


_COUNTING_WORDS = frozenset({
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "couple", "adults", "adult", "people", "persons", "person", "family",
    "dois", "duas", "tres", "três", "quatro", "cinco", "seis",
    "casal", "pessoas", "pessoa", "adultos", "adulto", "familia", "família",
})


def _reject_counting_label(value: str) -> str:
    """One Traveler = one person. Counting labels ("Two adults") collapse
    N people into one entry and corrupt the budget arithmetic in tools.py,
    which uses len(brief.travelers) as the traveler count."""
    if any(ch.isdigit() for ch in value):
        raise ValueError(
            f"Traveler.label must name ONE person, not a count -- "
            f"create one entry per traveler. Received: {value!r}"
        )
    words = set(re.findall(r"\w+", value.lower()))
    if words & _COUNTING_WORDS:
        raise ValueError(
            f"Traveler.label looks like it's counting people instead of naming "
            f"a single person -- create one entry per traveler. Received: {value!r}"
        )
    return value


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

class Money(Base):
    amount: float
    currency: str = Field("BRL", description="ISO 4217, e.g., BRL, EUR, USD")


class DateWindow(Base):
    """Dates. If the user doesn't know, leave earliest/latest null and
    fill `season_hint` -- the seasonality agent resolves it later."""
    earliest: date | None = None
    latest: date | None = None
    flexible_days: int = Field(
        0, description="How many days the window can slide forward/backward"
    )
    season_hint: str | None = Field(
        None, description="E.g., 'sometime in Q3', 'July school holidays'"
    )
    blackout_dates: list[date] = Field(
        default_factory=list, description="Dates when the user cannot travel"
    )


class Traveler(Base):
    label: str = Field(..., description="How to refer to this person, e.g., 'me', 'spouse'")
    age_band: AgeBand = AgeBand.adult
    mobility_notes: str | None = Field(
        None, description="Mobility restrictions, wheelchair, bad knee"
    )
    dietary: list[str] = Field(
        default_factory=list, description="E.g., vegetarian, gluten-free, shellfish allergy"
    )
    interests: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(
        default_factory=list, description="Things this person refuses to do"
    )

    @field_validator("label")
    @classmethod
    def _validate_label(cls, v: str) -> str:
        return _reject_counting_label(v)


class OriginContext(Base):
    """Where the trip departs from. Affects flight, currency, visa, vaccination."""
    city: str
    country: str = Field("BR", description="ISO 3166-1 alpha-2, e.g., BR")
    departure_airports: list[str] = Field(
        default_factory=list, description="IATA codes, e.g., ['GIG', 'BSB']"
    )
    passport_countries: list[str] = Field(
        default_factory=list, description="Passport nationalities (ISO alpha-2) for visa rules"
    )

    @field_validator("country")
    @classmethod
    def _validate_country(cls, v: str) -> str:
        return _normalize_country_code(v)

    @field_validator("passport_countries")
    @classmethod
    def _validate_passport_countries(cls, v: list[str]) -> list[str]:
        return [_normalize_country_code(c) for c in v]


class DestinationIntent(Base):
    name: str
    country: str | None = Field(None, description="ISO 3166-1 alpha-2, e.g., JP")
    decided: bool = Field(
        False, description="False = still a candidate, brainstormer can replace it"
    )
    origin: Origin = Origin.user
    rationale: str | None = Field(None, description="Why this destination is on the list")

    @field_validator("country")
    @classmethod
    def _validate_country(cls, v: str | None) -> str | None:
        return _normalize_country_code(v) if v is not None else v


class AttractionIntent(Base):
    """Shortlist that the planning phase clusters and distributes.
    `priority` is what the planner uses to decide what to cut when the day overflows."""
    name: str
    city: str | None = None
    priority: Priority = Priority.nice
    origin: Origin = Origin.user
    est_duration_min: int | None = Field(None, description="Typical visit duration in minutes")
    notes: str | None = None


class HardConstraints(Base):
    """Nothing here is negotiable by LLM. Goes to deterministic validation."""
    budget_ceiling: Money | None = None
    budget_includes_flights: bool = True
    nights: int | None = Field(None, description="Number of nights away from home")
    date_window: DateWindow = Field(default_factory=DateWindow)
    accessibility_required: bool = False
    must_include: list[str] = Field(
        default_factory=list, description="Items without which the trip makes no sense"
    )
    must_avoid: list[str] = Field(default_factory=list)


class SoftPreferences(Base):
    """Goes into the planner's prompt, not the validator."""
    pace: Pace = Pace.balanced
    lodging_style: list[str] = Field(
        default_factory=list, description="E.g., central hotel, airbnb, hostel, guesthouse"
    )
    food_style: list[str] = Field(
        default_factory=list, description="E.g., street food, fine dining, markets"
    )
    trip_themes: list[str] = Field(
        default_factory=list, description="E.g., honeymoon, backpacking, family trip"
    )


class Assumption(Base):
    """The agent filled in something the user didn't say. Needs to become
    confirmed=True or be corrected before the brief leaves draft status."""
    field_path: str = Field(..., description="E.g., 'hard.nights', 'soft.pace'")
    value: str
    rationale: str
    blocking: bool = Field(
        True, description="If True, brief cannot be confirmed without checking with user"
    )
    confirmed: bool = False


class Conflict(Base):
    kind: ConflictKind
    detail: str = Field(..., description="Natural language explanation of the clash")
    evidence: str | None = Field(None, description="The number or fact supporting the conflict")
    suggested_relaxations: list[str] = Field(
        default_factory=list,
        description="Concrete adjustment options to offer the user",
    )


class Feasibility(Base):
    """Result of reality check. Runs before research, not after planning."""
    verdict: Verdict = Verdict.feasible
    conflicts: list[Conflict] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------

class TripBrief(Base):
    brief_id: str
    version: int = Field(1, description="Increments with each user-accepted revision")
    status: BriefStatus = BriefStatus.draft

    origin: OriginContext | None = None
    travelers: list[Traveler] = Field(default_factory=list)
    destinations: list[DestinationIntent] = Field(default_factory=list)
    attractions: list[AttractionIntent] = Field(default_factory=list)

    hard: HardConstraints = Field(default_factory=HardConstraints)
    soft: SoftPreferences = Field(default_factory=SoftPreferences)

    assumptions: list[Assumption] = Field(default_factory=list)
    declined: list[str] = Field(
        default_factory=list,
        description="Field paths of RECOMMENDED that the user explicitly declined "
        "(e.g., 'attractions') -- don't ask again.",
    )
    feasibility: Feasibility = Field(default_factory=Feasibility)

    # ---- helpers used by collection loop (not schema fields) ----

    REQUIRED: ClassVar[tuple[tuple[str, Callable[["TripBrief"], bool]], ...]] = (
        ("origin", lambda b: b.origin is not None),
        ("travelers", lambda b: len(b.travelers) > 0),
        ("destinations", lambda b: any(d.decided for d in b.destinations)),
        ("hard.nights", lambda b: b.hard.nights is not None),
        ("hard.date_window", lambda b: b.hard.date_window.earliest is not None
         or b.hard.date_window.season_hint is not None),
        ("hard.budget_ceiling", lambda b: b.hard.budget_ceiling is not None),
        ("origin.departure_airports", lambda b: bool(b.origin and b.origin.departure_airports)),
    )

    RECOMMENDED: ClassVar[tuple[tuple[str, Callable[["TripBrief"], bool]], ...]] = (
        ("travelers[].interests", lambda b: any(t.interests for t in b.travelers)),
        ("attractions", lambda b: len(b.attractions) > 0),
        ("soft.lodging_style", lambda b: bool(b.soft.lodging_style)),
        ("soft.food_style", lambda b: bool(b.soft.food_style)),
        ("soft.trip_themes", lambda b: bool(b.soft.trip_themes)),
    )

    def missing_required(self) -> list[str]:
        """Fields still blocking exit from discovery phase."""
        return [name for name, ok in self.REQUIRED if not ok(self)]

    def missing_recommended(self) -> list[str]:
        """Fields worth collecting but don't block ready_for_research --
        disappear from list when filled OR listed in `declined`."""
        return [
            name for name, ok in self.RECOMMENDED
            if not ok(self) and name not in self.declined
        ]

    def pending_assumptions(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.blocking and not a.confirmed]

    def ready_for_research(self) -> bool:
        return (
            not self.missing_required()
            and not self.pending_assumptions()
            and self.feasibility.verdict != Verdict.infeasible
        )

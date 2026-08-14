"""Research pipeline -- phase 2, runs after the intake_agent confirms the brief.

    research_pipeline (SequentialAgent)
      -> base_resolver (code decides HOW MANY, LLM decides WHICH) -> state["stops"]
           The brief's `destinations` may be country/region-level ("Japan",
           not "Tokyo") -- this step resolves real city-level stops (with
           nights each) before anything else runs, so every scout below
           gets real cities, not a country label.
      -> ParallelAgent
           poi_scout (code, PoiDiscovery) -> research_raw.pois
             one poi_discovery_agent call per resolved stop (not per
             city x category -- categories are a quota inside the prompt),
             each returning structured JSON facts directly; geocoding
             (lat/lng, for planning_pipeline's k-means) then runs in plain
             code, on every candidate, unconditionally -- no second LLM
             pass decides whether or when to confirm/geocode a candidate.
           season_scout     -> state["season_scout_output"]     (Fact: season, warning)
           logistics_scout  -> state["logistics_scout_output"]  (Fact: transport)
           entry_scout      -> state["entry_scout_output"]      (Fact: entry)
           cost_scout       -> state["cost_scout_output"]       (cost anchors)
      -> poi_coverage_retry (code, no LLM unless a stop is short) -> tops up under-covered stops
      -> aggregator (plain code, no LLM) -> state["research"] (ResearchPack, incl. `stops`)

Every scout is a background worker with no user in the loop: none of them
have a path back to the end user, so instructions below explicitly forbid
asking questions or making recommendations -- their only job is grounded
fact-finding, the planner (phase 3) is the one that decides and advises.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, AsyncGenerator

from pydantic import ValidationError

from google.adk.agents import Agent, BaseAgent, InvocationContext, ParallelAgent, SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ...callback_logging import log_model_response, log_query_to_model

from ..intake_agent.trip_brief import TripBrief
from ...prompts import NO_USER_CONTACT as _NO_USER_CONTACT
from .research_pack import Fact, POI, Price, ResearchPack, Stop, coverage_shortfall
from .tools import geocode, slugify

_FENCE_RE = re.compile(r"```\w*\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: Any) -> Any:
    """Best-effort JSON parse of a scout's raw text output.

    Scouts can't use `output_schema` here -- Vertex rejects combining
    controlled generation with the `google_search` tool in one request
    ("controlled generation is not supported with Search tool"), so they're
    instructed to just reply with JSON text instead, which can arrive
    wrapped in a ```json fence despite being told not to. Returns None on
    anything that isn't parseable JSON, so callers can treat that the same
    as "this scout found nothing."
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    stripped = raw.strip()
    match = _FENCE_RE.fullmatch(stripped)
    if match:
        stripped = match.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


_JSON_ONLY = """
            Vertex rejects combining structured output with the search
            tool, so you can't return a schema-enforced object here --
            reply with ONLY raw JSON text matching the shape below. No
            markdown code fence, no preamble, no explanation before or
            after it. If you found nothing at all, reply with the shape's
            empty form (`[]` or `{}`), never prose.
            """


def _search_scout(
    name: str, description: str, topic_instruction: str, output_key: str, json_shape: str
):
    from ...config import gemini_search_model

    return Agent(
        name=name,
        model=gemini_search_model(),
        description=description,
        instruction=(
            topic_instruction
            + f"\n\nReply shape:\n{json_shape}\n"
            + _JSON_ONLY
            + _NO_USER_CONTACT
        ),
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
        output_key=output_key,
    )


# --------------------------------------------------------------------------
# base_resolver -- resolves the brief's `destinations` (which may be
# country/region-level, e.g. "Japan") into real city-level `stops`, each
# with a night count. Code decides HOW MANY bases fit the trip (nights x
# pace); an LLM grounded via `google_search` decides WHICH real cities.
# Runs once, before the scouts -- everything downstream (poi_scout,
# season_scout, logistics_scout, cost_scout, geo_clustering in phase 3)
# reads `stops` instead of `brief.destinations`.
# --------------------------------------------------------------------------

_PACE_NIGHTS_PER_BASE = {"relaxed": 4, "balanced": 3, "packed": 2.5}
"""How many nights a base 'deserves' before it's worth splitting into a
second city, per pace -- a 4-night relaxed trip stays in one city; a
4-night packed trip can support two."""

_STOPS_JSON_SHAPE = (
    '{"stops": [{"city": "...", "country": "XX" or null, "nights": 0, '
    '"rationale": "..."}, ...], '
    '"attraction_cities": {"<attraction name>": "<city>", ...}}'
)


def _create_base_resolver_agent():
    """Gemini-only, google_search-only agent -- called once per trip.
    Code has already decided how many bases fit (`_PACE_NIGHTS_PER_BASE`);
    this agent only decides WHICH real cities, and maps the brief's own
    `attractions` onto them (an attraction with no city would otherwise
    never match any resolved stop downstream)."""
    from ...config import gemini_search_model

    return Agent(
        name="base_resolver_agent",
        model=gemini_search_model(),
        description="Picks which real cities a trip bases itself in, and how nights split across them.",
        instruction=(
            """
            You are called as a tool, once per trip -- never directly by a
            user. You'll get the trip's destination(s) below (which may
            already be cities, or may be a country/region like "Japan" or
            "Tuscany"), total nights, a hard cap on how many cities to
            base in, pace, traveler interests, trip themes, and the
            brief's own attraction list.

            Use `google_search` to pick AT MOST the given cap of real
            cities/towns worth basing the trip in -- if a destination is
            already a single city, just use that city (don't invent extra
            bases for a single-city trip). For each city, propose how many
            nights it deserves; exact rounding to match the trip's total
            is handled elsewhere, so focus on realistic proportions (a
            city you only day-trip from gets fewer nights than one you
            sleep in most of the trip).

            Then, for every attraction in the brief's attraction list,
            say which of your chosen cities it belongs to -- grounded in
            actual geography, look it up if you're not sure, never guess.
            """
            + f"\n\nReply shape:\n{_STOPS_JSON_SHAPE}\n"
            + _JSON_ONLY
            + _NO_USER_CONTACT
        ),
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
    )


def _normalize_stops(
    raw: dict, max_bases: int, total_nights: int, decided: list
) -> list[dict]:
    """Clamps the resolver's proposed cities to `max_bases` and rescales
    their `nights` (largest-remainder method) to sum to exactly
    `total_nights` -- arithmetic isn't the LLM's job, only *which* cities
    to use. Falls back to an even split of the brief's own destination
    names if the resolver's search returned nothing usable."""
    raw_stops = [s for s in (raw.get("stops") or []) if isinstance(s, dict) and s.get("city")]
    raw_stops = raw_stops[:max_bases]

    if not raw_stops:
        base = total_nights // len(decided)
        remainder = total_nights % len(decided)
        return [
            {
                "city": d.name,
                "country": d.country,
                "nights": max(1, base + (1 if i < remainder else 0)),
                "rationale": None,
            }
            for i, d in enumerate(decided)
        ]

    weights = [max(1, int(s.get("nights") or 1)) for s in raw_stops]
    weight_sum = sum(weights)
    exact = [total_nights * w / weight_sum for w in weights]
    nights = [max(1, int(e)) for e in exact]

    remainder = total_nights - sum(nights)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    i = 0
    while remainder > 0:
        nights[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    while remainder < 0:
        idx = max(range(len(nights)), key=lambda j: nights[j])
        if nights[idx] <= 1:
            break
        nights[idx] -= 1
        remainder += 1

    return [
        {
            "city": s["city"],
            "country": s.get("country"),
            "nights": n,
            "rationale": s.get("rationale"),
        }
        for s, n in zip(raw_stops, nights)
    ]


class BaseResolver(BaseAgent):
    """Resolves `brief.destinations` into real city-level `stops`, and
    assigns every brief `attraction` to one of them. No-ops to empty
    `stops`/`attraction_cities` if nothing is decided yet (shouldn't
    happen -- research only runs once the brief is confirmed, and
    `destinations` is a REQUIRED field). Writes state["stops"] and
    state["attraction_cities"]."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        decided = [d for d in brief.destinations if d.decided]

        if not decided:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                actions=EventActions(state_delta={"stops": [], "attraction_cities": {}}),
            )
            return

        total_nights = brief.hard.nights or 1
        per_base = _PACE_NIGHTS_PER_BASE.get(brief.soft.pace, _PACE_NIGHTS_PER_BASE["balanced"])
        max_bases = max(1, int(total_nights / per_base))

        interests = sorted({i for t in brief.travelers for i in t.interests})
        destinations_text = "\n".join(
            f"- {d.name}" + (f", {d.country}" if d.country else "") for d in decided
        )
        query = (
            f"Destination(s):\n{destinations_text}\n"
            f"Total nights: {total_nights}\n"
            f"Max cities to base in, total across all destinations: {max_bases}\n"
            f"Pace: {brief.soft.pace}\n"
            f"Traveler interests: {', '.join(interests) or 'none stated'}\n"
            f"Trip themes: {', '.join(brief.soft.trip_themes) or 'none stated'}\n"
            f"Brief attractions: {', '.join(a.name for a in brief.attractions) or 'none listed'}"
        )
        tool_context = ToolContext(ctx)
        agent_tool = AgentTool(agent=_create_base_resolver_agent())
        text = await agent_tool.run_async(args={"request": query}, tool_context=tool_context)
        raw = _parse_json(text) or {}

        stops = _normalize_stops(raw, max_bases, total_nights, decided)
        stop_cities = {s["city"] for s in stops}
        attraction_cities = {
            name: city
            for name, city in (raw.get("attraction_cities") or {}).items()
            if city in stop_cities
        }

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"stops": stops, "attraction_cities": attraction_cities}
            ),
        )


# --------------------------------------------------------------------------
# poi_scout -- code-driven discovery. The discovery loop is deliberately NOT
# left to an LLM's own judgment about when it has searched enough -- every
# resolved stop gets exactly one discovery call, in Python, every time.
# Categories are a quota inside that single call's prompt (not a separate
# call each) -- diversity is enforced by the quota + a category tag on
# every result, not by isolating each category into its own request. Each
# call also returns its facts (duration, price, closing day, source)
# directly as JSON -- no second LLM pass re-searches to confirm what the
# first call already found. Geocoding (lat/lng) then runs unconditionally
# in plain code (see `_build_pois` below), for every candidate, so
# planning_pipeline's k-means always has coordinates to work with.
# --------------------------------------------------------------------------

_POI_CATEGORIES = ("landmark", "museum", "food", "nature", "neighborhood")

_POI_JSON_SHAPE = (
    '[{"name": "...", "category": "landmark" | "museum" | "food" | '
    '"nature" | "neighborhood", "duration_min": 0 or null, '
    '"ticket_low": 0 or null, "ticket_high": 0 or null, '
    '"ticket_currency": "BRL", "closed_weekdays": [0..6] or [], '
    '"source_url": "..." or null}, ...]'
)


def _create_poi_discovery_agent():
    """Gemini-only, google_search-only agent -- called once per resolved
    stop (city), never once per (city, category) axis. Asks for a genuine
    spread across all 5 categories in the SAME call (a quota inside the
    prompt) instead of isolating each category into its own request --
    that's what keeps results varied now that it's one call per city."""
    from ...config import gemini_search_model

    return Agent(
        name="poi_discovery_agent",
        model=gemini_search_model(),
        description="Searches for a diverse spread of real attraction candidates in one city.",
        instruction=(
            """
            You are called as a tool, once per city -- never directly by a
            user. You'll get one city, trip context (interests, themes),
            and possibly a list of names already found for this trip
            (skip those, find different ones instead). Use `google_search`
            to find 12-15 real, currently-open, well-regarded options in
            that city.

            Cover a genuine SPREAD across these 5 categories -- at least 2
            of each, never all 12-15 clustered into just one or two:
            - landmark: iconic sights, monuments, historic sites.
            - museum: museums, galleries, cultural institutions.
            - food: specific restaurants, markets, food halls, culinary
              experiences (not "try local food" in the abstract).
            - nature: parks, gardens, viewpoints, outdoor activities.
            - neighborhood: districts/areas worth walking through for
              their own character, not a single landmark within one.

            The 2-per-category minimum is a target, not a license to
            invent: if this city genuinely has no real, well-regarded
            option in one category (e.g. no notable museum in a small
            beach town), fall below the minimum for that category instead
            of fabricating a candidate to hit the quota -- every candidate
            must be something you actually found via `google_search`.

            For each candidate, tag which of those 5 categories it belongs
            to, and also find its typical visit duration, ticket price
            range, weekly closing day(s), and a source URL -- you won't
            get a second chance to look these up, so gather them now, in
            this same search pass. Leave a field null/empty rather than
            guessing if you can't find it.
            """
            + f"\n\nReply shape:\n{_POI_JSON_SHAPE}\n"
            + _JSON_ONLY
            + _NO_USER_CONTACT
        ),
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
    )


def _parse_discovery_json(text: str, city: str) -> list[dict]:
    """Turns poi_discovery_agent's JSON reply into candidate dicts tagged
    with city. Anything that isn't a well-formed list entry (or a non-JSON
    reply entirely) is dropped -- see `_parse_json`'s docstring for why
    scouts can't use `output_schema` here. An unrecognized `category`
    (hallucinated tag, wrong casing) becomes None rather than dropping the
    candidate -- the POI is still real, it just won't count toward a
    specific category's coverage."""
    candidates = []
    for item in _parse_json(text) or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        category = item.get("category")
        candidates.append(
            {
                "name": item["name"],
                "city": city,
                "category": category if category in _POI_CATEGORIES else None,
                "duration_min": item.get("duration_min"),
                "ticket_low": item.get("ticket_low"),
                "ticket_high": item.get("ticket_high"),
                "ticket_currency": item.get("ticket_currency") or "BRL",
                "closed_weekdays": item.get("closed_weekdays") or [],
                "source_url": item.get("source_url"),
            }
        )
    return candidates


async def _discover_candidates(
    cities: list[str],
    brief: TripBrief,
    tool_context: ToolContext,
    exclude: set[str] | None = None,
) -> list[dict]:
    """Runs one poi_discovery_agent call per city in `cities`, concurrently
    -- the number of calls is decided entirely by the caller (a plain
    Python list), not by any LLM. Each call already covers all 5
    categories via a quota in its own prompt, so this is one call per
    city, not per city x category. `exclude` (already-known names) is
    passed so a coverage-retry round asks for NEW places instead of
    re-finding the same ones."""
    interests = sorted({i for t in brief.travelers for i in t.interests})
    themes = brief.soft.trip_themes
    exclude_text = ", ".join(sorted(exclude)) if exclude else "none"
    agent_tool = AgentTool(agent=_create_poi_discovery_agent())

    async def _discover_one(city: str) -> list[dict]:
        query = (
            f"City: {city}\n"
            f"Traveler interests: {', '.join(interests) or 'none stated'}\n"
            f"Trip themes: {', '.join(themes) or 'none stated'}\n"
            f"Already found, skip these: {exclude_text}"
        )
        text = await agent_tool.run_async(args={"request": query}, tool_context=tool_context)
        return _parse_discovery_json(text, city)

    results = await asyncio.gather(*(_discover_one(city) for city in cities))
    return [candidate for batch in results for candidate in batch]


def _poi_key(name: str, city: str) -> tuple[str, str]:
    return (name.strip().lower(), city.strip().lower())


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for c in candidates:
        key = _poi_key(c["name"], c["city"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


async def _build_pois(candidates: list[dict]) -> list[dict]:
    """Geocodes every candidate concurrently, in plain code -- no LLM
    decides whether or when to call this, every candidate handed in gets
    geocoded, unconditionally. `validated` is exactly "geocoding found a
    match"; lat/lng are what planning_pipeline's k-means clusters on.
    Returns finished POI dicts (`model_dump(mode="json")`), ready to write
    straight to state["research_raw"]["pois"]."""

    async def _build_one(c: dict) -> dict:
        coords = await asyncio.to_thread(geocode, f"{c['name']}, {c['city']}")
        ticket = None
        if c.get("ticket_low") is not None and c.get("ticket_high") is not None:
            ticket = Price(
                low=c["ticket_low"],
                high=c["ticket_high"],
                currency=c.get("ticket_currency") or "BRL",
            )
        poi = POI(
            poi_id=f"{slugify(c['name'])}-{slugify(c['city'])}",
            name=c["name"],
            city=c["city"],
            category=c.get("category"),
            validated=coords is not None,
            lat=coords[0] if coords else None,
            lng=coords[1] if coords else None,
            duration_min=c.get("duration_min"),
            ticket=ticket,
            closed_weekdays=c.get("closed_weekdays") or [],
            brief_item=c.get("brief_item"),
            source_url=c.get("source_url"),
        )
        return poi.model_dump(mode="json")

    return list(await asyncio.gather(*(_build_one(c) for c in candidates)))


class PoiDiscovery(BaseAgent):
    """Builds the full POI list in one code-driven pass: the brief's own
    `attractions` (assigned to a city by base_resolver, via
    state["attraction_cities"]) plus one poi_discovery_agent call per
    resolved stop -- always exactly len(stops) calls, each covering all 5
    categories via a quota in its prompt, regardless of what any LLM
    thinks is "enough". Every candidate is then geocoded in plain code
    (`_build_pois`), unconditionally. Writes state["research_raw"]["pois"].
    Requires base_resolver to have already run (reads state["stops"])."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        stops = state.get("stops") or []
        cities = [s["city"] for s in stops]
        attraction_cities = state.get("attraction_cities") or {}

        candidates = [
            {
                "name": a.name,
                "city": attraction_cities.get(a.name, cities[0] if cities else (a.city or "")),
                "brief_item": a.name,
            }
            for a in brief.attractions
        ]

        if cities:
            tool_context = ToolContext(ctx)
            candidates.extend(await _discover_candidates(cities, brief, tool_context))

        pois = await _build_pois(_dedupe_candidates(candidates))

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"research_raw": {"pois": pois}}),
        )


class PoiCoverageRetry(BaseAgent):
    """Runs after all scouts finish. Checks validated-POI coverage per
    stop (3 per night allocated to that city, see `coverage_shortfall`)
    and, if any stop is short, runs exactly one more discovery+geocode
    round scoped only to those cities, excluding names already found --
    the equivalent of a LoopAgent(max_iterations=2), but targeted instead
    of re-running scouts that are already fine. No-ops (zero events) when
    coverage is already sufficient everywhere."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        stops = [Stop.model_validate(s) for s in state.get("stops") or []]
        existing_raw = state.get("research_raw", {}).get("pois", [])
        validated = [POI.model_validate(p) for p in existing_raw if p.get("validated")]
        shortfall = coverage_shortfall(stops, validated)
        if not shortfall:
            return

        tool_context = ToolContext(ctx)
        already_found = {p.name for p in validated if p.city in shortfall}
        raw_candidates = await _discover_candidates(
            list(shortfall), brief, tool_context, exclude=already_found
        )

        already_seen = {_poi_key(p.name, p.city) for p in validated}
        fresh_candidates = [
            c
            for c in _dedupe_candidates(raw_candidates)
            if _poi_key(c["name"], c["city"]) not in already_seen
        ]
        if not fresh_candidates:
            return

        new_pois = await _build_pois(fresh_candidates)

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"research_raw": {"pois": existing_raw + new_pois}}
            ),
        )


def create_poi_scout():
    """Single code-driven step: one discovery call per resolved stop (plus
    the brief's own `attractions`) and geocodes every candidate,
    unconditionally, in plain code -- see PoiDiscovery."""
    return PoiDiscovery(name="poi_scout")


def create_season_scout():
    return _search_scout(
        name="season_scout",
        description="Researches season/climate fit and travel warnings for the trip's resolved stops.",
        topic_instruction="""
            You'll be given a confirmed trip brief and the trip's resolved
            stops (real cities + nights) below. Search for what the
            weather/season is actually like in each stop's city during
            the brief's date window (or `season_hint` if exact dates
            aren't set), and for any current travel warnings or advisories
            (safety, natural events, large disruptive events like strikes
            or major festivals that affect logistics).

            Return one Fact per distinct finding: use topic "season" for
            climate/weather fit, topic "warning" for anything advisory.
            Each `content` must be a grounded, specific statement (numbers,
            named events, dates) with its `source_url` -- not generic
            "weather varies" filler.

            Confirmed trip brief:
            {brief}
            Resolved stops:
            {stops}
            """,
        output_key="season_scout_output",
        json_shape='[{"topic": "season" | "warning", "content": "...", "source_url": "..." or null}, ...]',
    )


def create_logistics_scout():
    return _search_scout(
        name="logistics_scout",
        description="Researches inter/intra-city transport logistics for the trip's resolved stops.",
        topic_instruction="""
            You'll be given a confirmed trip brief and the trip's resolved
            stops (real cities + nights, in visiting order) below. Search
            for how travelers actually get between and around these
            cities: main airport(s)/stations, typical transfer options and
            travel times between consecutive stops, and the realistic
            local transit options in each city (metro, day passes,
            ride-hailing availability).

            Return one Fact per distinct finding, topic "transport", each
            `content` grounded with specific names/times/prices where
            available and its `source_url`.

            Confirmed trip brief:
            {brief}
            Resolved stops:
            {stops}
            """,
        output_key="logistics_scout_output",
        json_shape='[{"topic": "transport", "content": "...", "source_url": "..." or null}, ...]',
    )


def create_entry_scout():
    return _search_scout(
        name="entry_scout",
        description="Researches visa/entry requirements for the trip's travelers and destinations.",
        topic_instruction="""
            You'll be given a confirmed trip brief below. Using `origin`
            (city/country) and each traveler's `passport_countries`, search
            for the actual visa/entry requirement into each decided
            destination: visa-free, visa-on-arrival, e-visa, or full visa,
            plus any lead time, vaccination, or document requirement you
            find.

            Return one Fact per destination (or per passport country if
            requirements differ between travelers), topic "entry", grounded
            with the specific rule and its `source_url`.

            Confirmed trip brief:
            {brief}
            """,
        output_key="entry_scout_output",
        json_shape='[{"topic": "entry", "content": "...", "source_url": "..." or null}, ...]',
    )


def create_cost_scout():
    return _search_scout(
        name="cost_scout",
        description="Researches budget anchor prices (flight, lodging, meals, local transit) for the trip.",
        topic_instruction="""
            You'll be given a confirmed trip brief and the trip's resolved
            stops (real cities + nights) below. Search for current,
            realistic price anchors for this specific trip (origin,
            resolved stops, date window, number of travelers) and return
            them as a `low`/`high` range in the currency of
            `hard.budget_ceiling` (default BRL if unset):

            - "flight": round-trip per traveler from `origin` to the
              first stop.
            - "lodging_night": per night, for the travelers' group size
              and `soft.lodging_style` if set -- anchor it to the stops'
              actual cities, not the destination in general.
            - "meal": one typical meal per person.
            - "transit_day": one day of local transit/rides per person.

            Only include a key if you actually found grounded numbers for
            it -- don't guess to fill in all four.

            Confirmed trip brief:
            {brief}
            Resolved stops:
            {stops}
            """,
        output_key="cost_scout_output",
        json_shape=(
            '{"costs": {"<key>": {"low": 0, "high": 0, "currency": "BRL"}}} '
            "-- <key> is any of flight, lodging_night, meal, transit_day. "
            "Omit a key entirely if you found nothing for it, don't include "
            "it with null/zero values."
        ),
    )


class ResearchAggregator(BaseAgent):
    """Assembles the final ResearchPack from what the scouts wrote to state.
    Pure code, no LLM call -- deterministic merge + validation."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])

        stops = [Stop.model_validate(s) for s in state.get("stops") or []]

        raw_pois = state.get("research_raw", {}).get("pois", [])
        pois = [POI.model_validate(p) for p in raw_pois]

        facts: list[Fact] = []
        for key in (
            "season_scout_output",
            "logistics_scout_output",
            "entry_scout_output",
        ):
            for item in _parse_json(state.get(key)) or []:
                try:
                    facts.append(Fact.model_validate(item))
                except ValidationError:
                    continue  # malformed entry from a scout -- skip, don't fail the pack

        cost_output = _parse_json(state.get("cost_scout_output")) or {}
        costs = {}
        for key, value in (cost_output.get("costs") or {}).items():
            try:
                costs[key] = Price.model_validate(value)
            except ValidationError:
                continue

        pack = ResearchPack(
            pack_id=f"research-{brief.brief_id}-v{brief.version}",
            brief_id=brief.brief_id,
            brief_version=brief.version,
            generated_at=datetime.now(),
            stops=stops,
            pois=pois,
            costs=costs,
            facts=facts,
        )
        undercovered = pack.undercovered_cities()
        pack.gaps = pack.missing_musts(brief) + [
            f"low POI coverage in {city}" for city in undercovered
        ]
        pack.complete = pack.ready_for_planning(brief)

        summary = (
            f"Research pack assembled: {len(stops)} stop(s), "
            f"{len(pack.validated_pois())}/{len(pois)} "
            f"POI(s) validated, {len(facts)} fact(s), {len(costs)} cost anchor(s). "
            f"Gaps: {', '.join(pack.gaps) if pack.gaps else 'none'}."
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"research": pack.model_dump(mode="json")}
            ),
            content=types.Content(role="model", parts=[types.Part.from_text(text=summary)]),
        )


def create_research_pipeline():
    """Create and return the research_pipeline (phase 2) agent."""
    scouts = ParallelAgent(
        name="research_scouts",
        description="Runs all research scouts concurrently.",
        sub_agents=[
            create_poi_scout(),
            create_season_scout(),
            create_logistics_scout(),
            create_entry_scout(),
            create_cost_scout(),
        ],
    )

    return SequentialAgent(
        name="research_pipeline",
        description=(
            "Phase 2: runs after the trip brief is confirmed. Resolves the "
            "brief's destinations into real city-level stops, researches "
            "POIs, season/warnings, logistics, entry requirements, and "
            "cost anchors in parallel, tops up any stop that's still short "
            "on POI coverage, then assembles the ResearchPack into "
            "state['research']. Requires a confirmed state['brief']."
        ),
        sub_agents=[
            BaseResolver(name="base_resolver"),
            scouts,
            PoiCoverageRetry(name="poi_coverage_retry"),
            ResearchAggregator(name="aggregator"),
        ],
    )

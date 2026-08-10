"""Research pipeline -- phase 2, runs after the intake_agent confirms the brief.

    research_pipeline (SequentialAgent)
      -> ParallelAgent
           poi_scout (SequentialAgent)      -> research_raw.pois (via validate_poi tool)
             poi_discovery   -- code loop, one poi_discovery_agent call per (city, category)
             poi_validator   -- LLM, geocode-validates the candidate list poi_discovery built
           season_scout     -> state["season_scout_output"]     (Fact: season, warning)
           logistics_scout  -> state["logistics_scout_output"]  (Fact: transport)
           entry_scout      -> state["entry_scout_output"]      (Fact: entry)
           cost_scout       -> state["cost_scout_output"]       (cost anchors)
      -> poi_coverage_retry (code, no LLM unless a city is short) -> tops up under-covered cities
      -> aggregator (plain code, no LLM) -> state["research"] (ResearchPack)

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

from callback_logging import log_model_response, log_query_to_model

from ..intake_agent.trip_brief import TripBrief
from ...prompts import NO_USER_CONTACT as _NO_USER_CONTACT
from .research_pack import Fact, POI, Price, ResearchPack, coverage_shortfall
from .tools import validate_poi

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
# poi_scout -- code-driven discovery (poi_discovery) feeding an LLM
# validation pass (poi_validator). The discovery loop is deliberately NOT
# left to an LLM's own judgment about when it has searched enough -- every
# decided city gets the same fixed category sweep, in Python, every time.
# --------------------------------------------------------------------------

_POI_CATEGORIES = ("landmark", "museum", "food", "nature", "neighborhood")


def _create_poi_search_agent():
    """Gemini-only, google_search-only agent -- same shape as
    brainstorm_agent. google_search cannot share an agent with a custom
    function tool (validate_poi) or live behind a non-Gemini model, so
    poi_validator (Claude via OpenRouter) calls this as an AgentTool
    instead of holding google_search itself."""
    from ...config import gemini_search_model

    return Agent(
        name="poi_search_agent",
        model=gemini_search_model(),
        description="Searches Google for facts about a single candidate POI.",
        instruction="""
            You are called as a tool by poi_validator, never directly by a
            user. You'll get a short query naming one place (and its
            city). Use `google_search` to find whether it's real and
            currently operating, its typical visit duration, ticket price
            range, weekly closing day(s), and a source URL.

            Reply in plain text, one fact per line, ending with the best
            source URL you found. No preamble, no follow-up questions --
            this is read by another agent, not a user. If you can't
            confirm the place exists, say so plainly instead of guessing.
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
    )


def _create_poi_discovery_agent():
    """Gemini-only, google_search-only agent -- called once per (city,
    category) axis by code, never asked an open-ended "find attractions"
    question. Scoping each call to a single category keeps results varied
    instead of five overlapping "top 10 things to do" lists."""
    from ...config import gemini_search_model

    return Agent(
        name="poi_discovery_agent",
        model=gemini_search_model(),
        description="Searches for real attraction candidates in one destination, for one category.",
        instruction="""
            You are called as a tool, once per (destination, category)
            pair -- never directly by a user. You'll get one destination,
            ONE category to focus on, and trip context (interests,
            themes). Use `google_search` to find 3-6 real, currently-open,
            well-regarded options in that destination for that SPECIFIC
            category only -- don't drift into other categories, a
            separate call already covers those.

            Categories mean:
            - landmark: iconic sights, monuments, historic sites.
            - museum: museums, galleries, cultural institutions.
            - food: specific restaurants, markets, food halls, culinary
              experiences (not "try local food" in the abstract).
            - nature: parks, gardens, viewpoints, outdoor activities.
            - neighborhood: districts/areas worth walking through for
              their own character, not a single landmark within one.

            Reply with ONLY a plain list, one candidate per line:

            - <name> -- <one-sentence reason it fits>

            No preamble, no markdown headers, no follow-up questions --
            this is read by another agent, not a user.
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
    )


def _parse_candidate_lines(text: str, city: str, category: str) -> list[dict]:
    """Turns poi_discovery_agent's "- name -- reason" lines into candidate
    dicts. Ignores anything that isn't a list line instead of failing --
    a stray preamble sentence just gets skipped, not treated as a name."""
    candidates = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        body = line.lstrip("-").strip()
        name = body.split("--", 1)[0].strip().strip("*").strip()
        if name:
            candidates.append({"name": name, "city": city, "category": category, "priority": "nice"})
    return candidates


async def _discover_candidates(
    axis_pairs: list[tuple[str, str]], brief: TripBrief, tool_context: ToolContext
) -> list[dict]:
    """Runs one poi_discovery_agent call per (city, category) pair in
    `axis_pairs`, concurrently -- the number of calls is decided entirely by
    the caller (a plain Python list), not by any LLM."""
    interests = sorted({i for t in brief.travelers for i in t.interests})
    themes = brief.soft.trip_themes
    agent_tool = AgentTool(agent=_create_poi_discovery_agent())

    async def _discover_one(city: str, category: str) -> list[dict]:
        query = (
            f"Destination: {city}\n"
            f"Category to focus on: {category}\n"
            f"Traveler interests: {', '.join(interests) or 'none stated'}\n"
            f"Trip themes: {', '.join(themes) or 'none stated'}"
        )
        text = await agent_tool.run_async(args={"request": query}, tool_context=tool_context)
        return _parse_candidate_lines(text, city, category)

    results = await asyncio.gather(
        *(_discover_one(city, category) for city, category in axis_pairs)
    )
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


def _create_poi_validator_agent():
    """LLM step: given a pre-built candidate list (state["poi_candidates"],
    built entirely by code -- see PoiDiscovery/PoiCoverageRetry below),
    confirm and geocode-validate every one of them. This agent never
    decides what to search for, only how to process what it's handed."""
    from ...config import openrouter_model

    poi_search_tool = AgentTool(agent=_create_poi_search_agent())

    return Agent(
        name="poi_validator",
        model=openrouter_model(),
        description="Confirms and geocode-validates a pre-built list of POI candidates.",
        instruction="""
            You'll be given a pre-built candidate list below -- already
            assembled by a code-driven search sweep across this trip's
            destinations and categories, plus the brief's own
            `attractions`. Don't invent new candidates yourself and don't
            skip any for being numerous -- process every single one.

            For each candidate:
            1. Call the search tool with its name + city to confirm it's
               real and operating, and find its typical visit duration,
               ticket price range, weekly closing day(s), and a source URL.
            2. Call `validate_poi` with everything you found. The tool
               geocodes the name+city itself and tells you whether it
               validated -- you never decide or claim validation yourself.
            3. If `validate_poi` reports it could not validate, try once
               more with a corrected/more specific name+city (e.g. add the
               neighborhood or the country). If it still fails, leave it --
               don't call it validated and don't invent coordinates.

            Process in this order: candidates with `priority: must`, then
            `nice`, then `optional` last.

            Candidates:
            {poi_candidates}
            """ + _NO_USER_CONTACT,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[poi_search_tool, validate_poi],
    )


class PoiDiscovery(BaseAgent):
    """Builds the initial POI candidate list: the brief's own `attractions`
    plus one poi_discovery_agent call per (decided city, category) axis --
    5 fixed categories x N decided cities, always, regardless of what the
    LLM validation step downstream thinks is "enough". Writes
    state["poi_candidates"] for poi_validator to consume next."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        cities = [d.name for d in brief.destinations if d.decided]

        candidates = [
            {
                "name": a.name,
                "city": a.city or (cities[0] if cities else ""),
                "category": "brief",
                "priority": a.priority,
            }
            for a in brief.attractions
        ]

        if cities:
            tool_context = ToolContext(ctx)
            axis_pairs = [(city, category) for city in cities for category in _POI_CATEGORIES]
            candidates.extend(await _discover_candidates(axis_pairs, brief, tool_context))

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={"poi_candidates": _dedupe_candidates(candidates)}
            ),
        )


class PoiCoverageRetry(BaseAgent):
    """Runs after all scouts finish. Checks validated-POI coverage per
    city (2.5 per allocated night, see `coverage_shortfall`) and, if any
    decided city is short, runs exactly one more discovery+validate round
    scoped only to those cities -- the equivalent of a
    LoopAgent(max_iterations=2), but targeted instead of re-running scouts
    that are already fine. No-ops (zero events) when coverage is already
    sufficient everywhere."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        cities = [d.name for d in brief.destinations if d.decided]
        validated = [
            POI.model_validate(p)
            for p in state.get("research_raw", {}).get("pois", [])
            if p.get("validated")
        ]
        shortfall = coverage_shortfall(cities, brief.hard.nights, validated)
        if not shortfall:
            return

        tool_context = ToolContext(ctx)
        axis_pairs = [
            (city, category) for city in shortfall for category in _POI_CATEGORIES
        ]
        raw_candidates = await _discover_candidates(axis_pairs, brief, tool_context)

        already_seen = {_poi_key(p.name, p.city) for p in validated}
        fresh = [
            c
            for c in _dedupe_candidates(raw_candidates)
            if _poi_key(c["name"], c["city"]) not in already_seen
        ]
        if not fresh:
            return

        state["poi_candidates"] = fresh
        validator = _create_poi_validator_agent()
        async for event in validator.run_async(ctx):
            yield event


def create_poi_scout():
    """POI candidates get geocoded and saved straight to state by the
    `validate_poi` tool -- neither sub-step here has an output_schema
    because the tool calls ARE the persistence step, not any final text."""
    return SequentialAgent(
        name="poi_scout",
        description=(
            "Discovers (code-driven, per city x category) and "
            "geocode-validates POIs for the trip's destinations."
        ),
        sub_agents=[
            PoiDiscovery(name="poi_discovery"),
            _create_poi_validator_agent(),
        ],
    )


def create_season_scout():
    return _search_scout(
        name="season_scout",
        description="Researches season/climate fit and travel warnings for the trip's date window and destinations.",
        topic_instruction="""
            You'll be given a confirmed trip brief below. Search for what
            the weather/season is actually like in each destination during
            the brief's date window (or `season_hint` if exact dates aren't
            set), and for any current travel warnings or advisories
            (safety, natural events, large disruptive events like strikes
            or major festivals that affect logistics).

            Return one Fact per distinct finding: use topic "season" for
            climate/weather fit, topic "warning" for anything advisory.
            Each `content` must be a grounded, specific statement (numbers,
            named events, dates) with its `source_url` -- not generic
            "weather varies" filler.

            Confirmed trip brief:
            {brief}
            """,
        output_key="season_scout_output",
        json_shape='[{"topic": "season" | "warning", "content": "...", "source_url": "..." or null}, ...]',
    )


def create_logistics_scout():
    return _search_scout(
        name="logistics_scout",
        description="Researches inter/intra-city transport logistics for the trip's destinations.",
        topic_instruction="""
            You'll be given a confirmed trip brief below. Search for how
            travelers actually get between and around the brief's
            destinations: main airport(s)/stations, typical transfer
            options and travel times between decided destinations, and the
            realistic local transit options in each city (metro, day
            passes, ride-hailing availability).

            Return one Fact per distinct finding, topic "transport", each
            `content` grounded with specific names/times/prices where
            available and its `source_url`.

            Confirmed trip brief:
            {brief}
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
            You'll be given a confirmed trip brief below. Search for
            current, realistic price anchors for this specific trip
            (origin, destinations, date window, number of travelers) and
            return them as a `low`/`high` range in the currency of
            `hard.budget_ceiling` (default BRL if unset):

            - "flight": round-trip per traveler from `origin` to the first
              decided destination.
            - "lodging_night": per night, for the travelers' group size and
              `soft.lodging_style` if set.
            - "meal": one typical meal per person.
            - "transit_day": one day of local transit/rides per person.

            Only include a key if you actually found grounded numbers for
            it -- don't guess to fill in all four.

            Confirmed trip brief:
            {brief}
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
            pois=pois,
            costs=costs,
            facts=facts,
        )
        undercovered = pack.undercovered_cities(brief)
        pack.gaps = pack.missing_musts(brief) + [
            f"low POI coverage in {city}" for city in undercovered
        ]
        pack.complete = pack.ready_for_planning(brief)

        summary = (
            f"Research pack assembled: {len(pack.validated_pois())}/{len(pois)} "
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
            "Phase 2: runs after the trip brief is confirmed. Researches "
            "POIs, season/warnings, logistics, entry requirements, and "
            "cost anchors in parallel, tops up any city that's still short "
            "on POI coverage, then assembles the ResearchPack into "
            "state['research']. Requires a confirmed state['brief']."
        ),
        sub_agents=[
            scouts,
            PoiCoverageRetry(name="poi_coverage_retry"),
            ResearchAggregator(name="aggregator"),
        ],
    )

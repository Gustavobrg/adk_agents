"""Planning pipeline -- phase 3, runs after the research pipeline assembles
the ResearchPack. This is where the system finally decides: phases 1-2 only
collected (what the traveler wants, what exists and costs). Phase 3 turns
a loose list of validated POIs into "day 3: morning in Asakusa, afternoon
in Ueno, dinner in Yanaka".

    planning_pipeline (SequentialAgent)
      1. selection (SequentialAgent)
           selection_capacity  -- code: nights × pace → hard POI count
           selection_agent     -- LLM: which POIs fit that count, by priority
      2. geo_clustering        -- code: k-means on lat/lng, k = nights per city
      3. day_distribution_agent -- LLM: order each day's cluster into blocks
      4. lodging_transport_agent -- LLM: one lodging area per city + transport legs
      5. plan_assembler        -- code: budget arithmetic + final TripPlan
      6. plan_presenter        -- LLM: writes the TripPlan up as Markdown (the
                                   pipeline's actual user-facing message)
      7. plan_artifact         -- code: saves that Markdown as a downloadable
                                   .md artifact via ADK's artifact service

Steps 1, 3, 4 are LLM, but only ever see hard constraints already resolved
by code (a POI count, a geographic cluster, a weekday) -- they decide
*which* and *how*, never *how many fit* or *what's near what*. Steps 2 and
5 are pure code: spatial reasoning and arithmetic aren't an LLM's strong
suit, and getting the geography wrong is exactly what produces the
classic "criss-cross the city in one day" itinerary.

This pipeline is sequential end to end, on purpose: each step's input is
the previous step's committed decision, not just a convenience ordering
(can't cluster before knowing what's selected, can't pick a day before
knowing the cluster, can't pick lodging before knowing where the days are,
can't total a budget before there's a concrete itinerary). Running it in
parallel would let e.g., lodging and day-distribution disagree about which
city/area the trip centers on.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent, Context, InvocationContext, SequentialAgent
from google.adk.events import Event, EventActions
from google.genai import types

from ...callback_logging import log_model_response, log_query_to_model

from ..intake_agent.trip_brief import TripBrief
from ...prompts import NO_USER_CONTACT
from ..research_pipeline.research_pack import POI, Price, Stop
from .trip_plan import (
    BudgetLine,
    DayDistributionResult,
    DayPlan,
    LodgingChoice,
    LodgingTransportResult,
    SelectionResult,
    TransportLeg,
    TripPlan,
)

# --------------------------------------------------------------------------
# Step 1: selection -- code resolves *how many* fit, LLM decides *which*.
# --------------------------------------------------------------------------

_PACE_POIS_PER_DAY = {"relaxed": 1.5, "balanced": 3.5, "packed": 5.5}


class SelectionCapacity(BaseAgent):
    """Nights × pace → hard POI count. The selection LLM receives this
    number, it never decides it -- otherwise "how many POIs fit" becomes a
    guess that varies by how verbose the model feels."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        nights = brief.hard.nights or 1
        per_day = _PACE_POIS_PER_DAY.get(brief.soft.pace, _PACE_POIS_PER_DAY["balanced"])
        capacity = max(1, round(nights * per_day))

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(
                state_delta={
                    "selection_capacity": {
                        "max_pois": capacity,
                        "pace": brief.soft.pace,
                        "nights": nights,
                    }
                }
            ),
        )


def _create_selection_agent():
    from ...config import openrouter_model

    return Agent(
        name="selection_agent",
        model=openrouter_model(),
        description="Cuts the validated POI pool down to what fits the trip's nights and pace.",
        instruction="""
            You'll be given the confirmed trip brief, the research pack's
            validated POIs, and a hard capacity number below. Select at most
            `max_pois` POIs total from `research.pois` where `validated` is
            true -- that number already accounts for nights × pace, you don't
            recompute or override it. The one exception is tier 1 below:
            `must`-priority attractions are never dropped for capacity, even
            if including all of them pushes the total past `max_pois`.

            Priority order, highest first:
            1. Any validated POI whose `brief_item` matches a brief
               `attractions` entry with `priority: must` -- include ALL of
               these regardless of `max_pois`; a `must` means the trip
               makes no sense without it, so it overrides the cap rather
               than competing for a slot in it.
            2. Validated POIs whose `brief_item` matches a `priority: nice`
               attraction.
            3. Validated POIs with no `brief_item` (discovered candidates)
               that best fit the travelers' `interests` and
               `soft.trip_themes`.
            4. Validated POIs matching a `priority: optional` attraction.
            Fill tiers 2-4 only up to whatever room is left under
            `max_pois` after tier 1. Cut from the bottom of this order
            first when something has to give.

            Within that order, prefer a spread across the POIs' `city`
            and `category` (landmark, museum, food, nature, neighborhood)
            over picking every candidate from one bucket -- a day of five
            museums back to back is a worse itinerary than a mixed one,
            even if both fit the count.

            Capacity: {selection_capacity}
            Confirmed trip brief: {brief}
            Research pack: {research}
            """ + NO_USER_CONTACT,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        output_schema=SelectionResult,
        output_key="selection_output",
    )


# --------------------------------------------------------------------------
# Step 2: geo_clustering -- pure code. k-means over (lat, lng) per city,
# k = nights allocated to that city. This is what keeps a day from
# zigzagging across town -- an LLM has no reliable sense of "near".
# --------------------------------------------------------------------------


def _load_pois(research: dict) -> dict[str, POI]:
    return {p["poi_id"]: POI.model_validate(p) for p in research.get("pois", [])}


def _day_cluster(day_index: int, day_date: date | None, city: str, pois: list[POI]) -> dict:
    return {
        "day_index": day_index,
        "date": day_date.isoformat() if day_date else None,
        "weekday": day_date.weekday() if day_date else None,
        "city": city,
        "pois": [p.model_dump(mode="json") for p in pois],
    }


def _kmeans(points: list[tuple[float, float]], k: int, iterations: int = 25) -> list[int]:
    """Plain Euclidean k-means on (lat, lng), no numpy -- trip POI counts are
    always small (tens, not thousands), so this is accurate at city scale and
    keeps the dependency footprint at zero. Deterministic seeding (evenly
    spaced across sorted points) avoids the empty-cluster failure mode that
    naive random seeding hits on small n."""
    n = len(points)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    order = sorted(range(n), key=lambda i: points[i])
    if k == 1:
        seed_idx = [order[0]]
    else:
        seed_idx = [order[round(i * (n - 1) / (k - 1))] for i in range(k)]
    centroids = [points[i] for i in seed_idx]

    assignments = [0] * n
    for _ in range(iterations):
        changed = False
        for i, p in enumerate(points):
            best = min(
                range(k),
                key=lambda c: (p[0] - centroids[c][0]) ** 2 + (p[1] - centroids[c][1]) ** 2,
            )
            if assignments[i] != best:
                assignments[i] = best
                changed = True
        sums = [[0.0, 0.0, 0] for _ in range(k)]
        for i, p in enumerate(points):
            c = assignments[i]
            sums[c][0] += p[0]
            sums[c][1] += p[1]
            sums[c][2] += 1
        for c in range(k):
            if sums[c][2]:
                centroids[c] = (sums[c][0] / sums[c][2], sums[c][1] / sums[c][2])
        if not changed:
            break
    return assignments


class GeoClustering(BaseAgent):
    """k-means per stop on selected POIs' (lat, lng); assigns each cluster a
    day index and, if the brief has exact dates, a calendar date + weekday.
    Reads `research["stops"]` (real cities + nights, resolved once by
    research_pipeline's base_resolver) rather than `brief.destinations` --
    the brief's destination may be a country/region, not a city, so it's
    never itself a valid clustering key. A stop with more allocated nights
    than it has POI-clusters gets explicit empty "free day" entries instead
    of silently having fewer days than nights."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        research = state.get("research") or {}
        all_pois = _load_pois(research)
        stops = [Stop.model_validate(s) for s in research.get("stops") or []]

        selection = state.get("selection_output") or {}
        selected_ids = selection.get("selected_poi_ids") or []
        selected = [all_pois[pid] for pid in selected_ids if pid in all_pois]
        selected_by_city: dict[str, list[POI]] = defaultdict(list)
        for p in selected:
            selected_by_city[p.city].append(p)

        earliest = brief.hard.date_window.earliest

        clusters = []
        day_offset = 0
        for stop in stops:
            city_pois = selected_by_city.get(stop.city, [])
            requested_days = stop.nights
            k = min(requested_days, len(city_pois)) if city_pois else 0
            assignments = _kmeans([(p.lat, p.lng) for p in city_pois], k) if k else []

            for cluster_idx in range(k):
                day_offset += 1
                cluster_pois = [city_pois[i] for i, a in enumerate(assignments) if a == cluster_idx]
                day_date = earliest + timedelta(days=day_offset - 1) if earliest else None
                clusters.append(_day_cluster(day_offset, day_date, stop.city, cluster_pois))

            for _ in range(k, requested_days):
                day_offset += 1
                day_date = earliest + timedelta(days=day_offset - 1) if earliest else None
                clusters.append(_day_cluster(day_offset, day_date, stop.city, []))

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"day_clusters": clusters}),
        )


# --------------------------------------------------------------------------
# Step 3: day_distribution -- LLM orders each day's already-fixed cluster
# into time blocks, respecting closed_weekdays. Never chooses what's IN a day
# (step 2 did); only chooses block + order + how to react to closed-weekday
# conflicts.
# --------------------------------------------------------------------------


def _create_day_distribution_agent():
    from ...config import openrouter_model

    return Agent(
        name="day_distribution_agent",
        model=openrouter_model(),
        description="Writes the day-by-day itinerary from pre-clustered, pre-ordered POIs.",
        instruction="""
            You'll be given a list of day clusters below -- already split by
            city and grouped geographically by code, already in day order.
            Your job per day: decide which time block (morning, afternoon,
            or evening) each of its POIs goes in and in what order within
            the block, using each POI's `duration_min` as a guide to how
            much a block can hold.

            Closed-weekday handling: if a POI's `closed_weekdays` includes
            its day's `weekday` (0=Monday..6=Sunday) and `weekday` isn't
            null, try to swap it with a POI from a DIFFERENT day in the
            SAME city -- but only if BOTH sides land clean: the incoming
            POI's own `closed_weekdays` must not include the day it's
            moving TO, either. A swap that just relocates the conflict
            onto the other POI doesn't count as resolved. If no such clean
            swap exists, keep the POI on its assigned day but say so
            plainly in that day's `notes` -- never silently drop it and
            never invent a different `closed_weekdays` value than what's
            given. If `weekday` is null (exact dates aren't set yet),
            don't attempt swaps -- just note in `notes` that this hasn't
            been checked against real dates.

            A cluster with an empty POI list is a free/lighter day (more
            nights allocated to that city than POIs to fill) -- return it
            with an empty `items` list and put a concrete suggestion in
            `notes` (revisit a favorite spot, day trip, unstructured time),
            not a generic "relax" filler.

            Day clusters:
            {day_clusters}
            """ + NO_USER_CONTACT,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        output_schema=DayDistributionResult,
        output_key="day_distribution_output",
    )


# --------------------------------------------------------------------------
# Step 4: lodging_transport -- LLM, decided from the fixed clusters.
# --------------------------------------------------------------------------


def _create_lodging_transport_agent():
    from ...config import openrouter_model

    return Agent(
        name="lodging_transport_agent",
        model=openrouter_model(),
        description="Recommends a lodging area per city and the transport legs between them.",
        instruction="""
            You'll be given the day clusters, the confirmed trip brief,
            and the research pack below. For EACH city that appears in the
            day clusters:

            1. Recommend exactly one lodging area/neighborhood: pick
               whichever neighborhood sits closest to that city's POI
               clusters overall (favor proximity to the most days' worth
               of clusters, not just the single biggest one), so daily
               transit is minimized. Ground the choice in the research
               pack's transport-topic facts where relevant (e.g., near a
               metro line mentioned there). Respect `soft.lodging_style`
               if the brief set one.
            2. If another city follows this one in the day clusters' order,
               note the transport leg between them, grounded in the research
               pack's transport facts where available.

            Confirmed trip brief: {brief}
            Research pack: {research}
            Day clusters: {day_clusters}
            """ + NO_USER_CONTACT,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        output_schema=LodgingTransportResult,
        output_key="lodging_transport_output",
    )


# --------------------------------------------------------------------------
# Step 5: plan_assembler -- pure code. Budget arithmetic only makes sense
# once a concrete itinerary exists, and it's arithmetic, not judgment.
# --------------------------------------------------------------------------


class PlanAssembler(BaseAgent):
    """Computes the budget from the research pack's cost anchors plus the
    scheduled POIs' ticket prices, then merges every earlier step's output
    into the final TripPlan. No LLM involved."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = TripBrief.model_validate(state["brief"])
        research = state.get("research") or {}
        all_pois = _load_pois(research)
        costs = {k: Price.model_validate(v) for k, v in (research.get("costs") or {}).items()}

        day_distribution = state.get("day_distribution_output") or {}
        days = [DayPlan.model_validate(d) for d in day_distribution.get("days") or []]

        lt = state.get("lodging_transport_output") or {}
        lodging = [LodgingChoice.model_validate(l) for l in lt.get("lodging") or []]
        transport = [TransportLeg.model_validate(t) for t in lt.get("transport") or []]

        travelers = max(len(brief.travelers), 1)
        nights = brief.hard.nights or len(days) or 1
        currency = brief.hard.budget_ceiling.currency if brief.hard.budget_ceiling else "BRL"

        budget: list[BudgetLine] = []

        if "flight" in costs:
            f = costs["flight"]
            budget.append(BudgetLine(category="flights", low=f.low * travelers, high=f.high * travelers, currency=f.currency))

        if "lodging_night" in costs:
            # Assumes one lodging unit/room for the whole group -- flagged
            # in `notes` below since a larger group may need more than one.
            l = costs["lodging_night"]
            budget.append(BudgetLine(category="lodging", low=l.low * nights, high=l.high * nights, currency=l.currency))

        if "meal" in costs:
            # Assumes 2 paid meals/person/day (breakfast covered by lodging)
            # -- also flagged in `notes`.
            m = costs["meal"]
            budget.append(BudgetLine(
                category="food",
                low=m.low * travelers * nights * 2,
                high=m.high * travelers * nights * 2,
                currency=m.currency,
            ))

        if "transit_day" in costs:
            t = costs["transit_day"]
            budget.append(BudgetLine(category="local_transit", low=t.low * travelers * nights, high=t.high * travelers * nights, currency=t.currency))

        ticket_low = ticket_high = 0.0
        scheduled_ids = {item.poi_id for day in days for item in day.items}
        for poi_id in scheduled_ids:
            poi = all_pois.get(poi_id)
            if poi and poi.ticket:
                ticket_low += poi.ticket.low * travelers
                ticket_high += poi.ticket.high * travelers
        if ticket_low or ticket_high:
            budget.append(BudgetLine(category="activities", low=ticket_low, high=ticket_high, currency=currency))

        budget_total = (
            Price(low=sum(b.low for b in budget), high=sum(b.high for b in budget), currency=currency)
            if budget
            else None
        )

        selection = state.get("selection_output") or {}
        dropped_ids = selection.get("dropped_poi_ids") or []
        dropped_names = [all_pois[p].name for p in dropped_ids if p in all_pois]

        plan = TripPlan(
            plan_id=f"plan-{brief.brief_id}-v{brief.version}",
            brief_id=brief.brief_id,
            brief_version=brief.version,
            pack_id=research.get("pack_id", ""),
            generated_at=datetime.now(),
            days=days,
            lodging=lodging,
            transport=transport,
            budget=budget,
            budget_total=budget_total,
            dropped_pois=dropped_names,
            notes=[
                "Budget assumes 1 lodging unit for the whole group and 2 "
                "paid meals/person/day -- adjust if your group needs "
                "multiple rooms or has more meals covered elsewhere.",
            ],
        )

        if budget_total:
            summary = (
                f"Trip plan assembled: {len(days)} day(s), {len(lodging)} lodging "
                f"area(s), budget {budget_total.low:.0f}-{budget_total.high:.0f} "
                f"{budget_total.currency}."
            )
        else:
            summary = f"Trip plan assembled: {len(days)} day(s), no cost anchors available for a budget."

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"plan": plan.model_dump(mode="json")}),
            content=types.Content(role="model", parts=[types.Part.from_text(text=summary)]),
        )


# --------------------------------------------------------------------------
# Step 6: plan_presenter -- LLM. Writes the Markdown itinerary the user
# actually reads from the already-assembled TripPlan. Unlike every other
# LLM step in this pipeline, this one's output IS what the user sees, not
# an input to a later code step -- so it's the one place that gets to
# write for a reader instead of a parser, and the only step in this file
# that doesn't carry NO_USER_CONTACT.
# --------------------------------------------------------------------------


def _create_plan_presenter_agent():
    from ...config import openrouter_model

    return Agent(
        name="plan_presenter",
        model=openrouter_model(),
        description=(
            "Writes the final, user-facing itinerary from the assembled "
            "TripPlan, enriched with each attraction's real detail from "
            "the research pack."
        ),
        instruction="""
            The trip is fully planned -- `plan` below is final. Write the
            itinerary message the user actually reads: friendly, clear,
            well-organized Markdown, with REAL DETAIL about each place --
            not just its bare name. `plan` only carries what time block
            and order each activity got; for the facts to flesh it out
            with, look up its full record in `research`'s `pois` by
            `poi_id` -- its `category`, ticket price range if any
            (`ticket.low`-`ticket.high` `ticket.currency`), weekly closing
            day(s) if any (`closed_weekdays`, 0=Monday..6=Sunday), and
            cite `source_url` if given. Never invent a detail (a
            description, a price, hours, why it's worth visiting) that
            isn't actually in `research` or `plan` -- if a fact isn't
            there, just don't mention it, don't guess. Never second-guess
            the decisions behind the plan (why a POI was dropped, why
            this lodging area) -- just present what was decided.

            Structure it day by day (date/weekday if set, city, each
            activity's time block, duration, and the grounded details
            above), then lodging per city, transport between cities, the
            budget breakdown and total, and anything cut for not fitting.
            Fold in `plan.notes` wherever they're relevant instead of
            dumping them in a separate list. Skip a section entirely if
            its data is empty rather than writing "none" for it. This is
            the end of the pipeline -- don't mention next steps or offer
            to do more.

            Confirmed trip brief: {brief}
            Trip plan: {plan}
            Research pack (POI detail -- category, ticket, closed_weekdays,
            source_url -- keyed by poi_id): {research}
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        output_key="plan_presentation",
    )


# --------------------------------------------------------------------------
# Step 7: plan_artifact -- pure code. Saves plan_presenter's Markdown as a
# downloadable artifact via ADK's artifact service, so the final itinerary
# isn't only a chat message -- the user (or any client hitting the API,
# e.g. the ADK dev UI's Artifacts panel) can pull the .md file up again
# without re-reading the whole conversation.
# --------------------------------------------------------------------------


class PlanArtifactWriter(BaseAgent):
    """Persists state["plan_presentation"] (plan_presenter's Markdown
    text) as a `.md` artifact. No-ops if the presenter produced nothing --
    shouldn't happen, but there's no artifact to save either way."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        markdown = ctx.session.state.get("plan_presentation")
        if not markdown:
            return

        brief = ctx.session.state.get("brief") or {}
        filename = f"trip-plan-{brief.get('brief_id', 'trip')}-v{brief.get('version', 1)}.md"

        actions = EventActions()
        context = Context(ctx, event_actions=actions)
        await context.save_artifact(
            filename=filename,
            artifact=types.Part.from_text(text=markdown),
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=actions,
        )


def create_planning_pipeline():
    """Create and return the planning_pipeline (phase 3) agent."""
    selection = SequentialAgent(
        name="selection",
        description="Cuts the validated POI pool to what fits the trip's nights and pace.",
        sub_agents=[
            SelectionCapacity(name="selection_capacity"),
            _create_selection_agent(),
        ],
    )

    return SequentialAgent(
        name="planning_pipeline",
        description=(
            "Phase 3: turns the confirmed brief + research pack into a "
            "concrete day-by-day TripPlan in state['plan'] -- selection, "
            "geographic clustering, day distribution, lodging/transport, "
            "then budget. Requires state['research'] to already exist."
        ),
        sub_agents=[
            selection,
            GeoClustering(name="geo_clustering"),
            _create_day_distribution_agent(),
            _create_lodging_transport_agent(),
            PlanAssembler(name="plan_assembler"),
            _create_plan_presenter_agent(),
            PlanArtifactWriter(name="plan_artifact"),
        ],
    )

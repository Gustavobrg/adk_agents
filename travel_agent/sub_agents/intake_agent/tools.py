"""Tools for the travel agent."""
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from google.adk.tools.tool_context import ToolContext

from .trip_brief import Conflict, ConflictKind, TripBrief, Verdict


def _merge_patch(base: Any, patch: Any) -> Any:
    """Applies a JSON Merge Patch (RFC 7386): a dict is merged key by key
    recursively; any other type (including lists) replaces the value
    entirely. A `None` value in the patch deletes the key -- since every
    optional TripBrief field already defaults to None, this is equivalent
    to clearing the field.
    """
    if not isinstance(patch, dict):
        return patch
    if not isinstance(base, dict):
        base = {}
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_patch(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validated_brief(data: Dict[str, Any]) -> Tuple[Optional[TripBrief], Optional[Dict[str, Any]]]:
    """Validates a dict against TripBrief without letting ValidationError
    bubble up to the caller -- guards against a state["brief"] saved by an
    older version of the schema (extra="forbid" rejects a field that no
    longer exists). Returns (brief, None) on success or (None, error_dict)
    on failure.
    """
    try:
        return TripBrief.model_validate(data), None
    except ValidationError as e:
        return None, {
            "status": "error",
            "message": f'state["brief"] does not match the current TripBrief schema -- {e}',
        }


def update_brief(patch: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """Applies a PARTIAL patch to state["brief"] and merges it with what's
    already there.

    NEVER resend the whole TripBrief -- send only the fields that changed
    this turn. A key absent from the patch preserves the already-saved
    value; a key present with value `null` clears the field. To edit a
    nested field, replicate only the path down to it, e.g. {"hard":
    {"nights": 5}} sets hard.nights without touching the rest of `hard`.
    Lists (travelers, destinations, attractions, assumptions, declined) are
    replaced entirely when they appear in the patch -- include every item
    that should survive, not just the new ones.

    Args:
        patch: partial object in TripBrief's shape (sub_agents/intake_agent/trip_brief.py).
            Include only the fields you want to create or change right now.
            Field names are exact -- don't invent synonyms. Examples of the
            nested blocks that are easiest to get wrong:
              - hard.date_window: exact dates -> {"earliest": "2026-11-27",
                "latest": "2026-12-11", "flexible_days": 3}. Unknown dates
                -> {"season_hint": "second half of 2026"} instead of
                earliest/latest. Never "start"/"end"/"check_in"/
                "check_out"/"root"/"value" -- they don't exist in the schema.
              - hard.budget_ceiling: {"amount": 8000, "currency": "BRL"}.
              - origin: {"city": "Goiânia", "country": "BR",
                "departure_airports": ["GYN"], "passport_countries": ["BR"]}.
              - travelers: one object per person, never a count --
                [{"label": "me", "age_band": "adult", "interests": ["beach"]}].
              - destinations: [{"name": "Lisbon", "country": "PT",
                "decided": true, "origin": "user"}].
              - attractions: [{"name": "Belem Tower", "city": "Lisbon",
                "priority": "must"}].

    Returns:
        On success: status "success" plus the resulting brief snapshot
        (brief_status, missing_required, missing_recommended,
        pending_assumptions, ready_for_research) to decide the next
        question to ask the user.
        On validation failure: status "error" with the message -- state is
        NOT changed in that case.
    """
    current = tool_context.state.get("brief")
    if current is None:
        current = {"brief_id": f"trip-{tool_context.session.id}"}

    merged = _merge_patch(current, patch)

    brief, error = _validated_brief(merged)
    if error is not None:
        return error

    dumped = brief.model_dump(mode="json")
    tool_context.state["brief"] = dumped

    return {
        "status": "success",
        # read from the dump, not the model: enum default isn't coerced by
        # use_enum_values until it goes through validation (known pydantic
        # bug with defaults). model_dump(mode="json") always returns a plain str.
        "brief_status": dumped["status"],
        "missing_required": brief.missing_required(),
        "missing_recommended": brief.missing_recommended(),
        "pending_assumptions": [a.field_path for a in brief.pending_assumptions()],
        "ready_for_research": brief.ready_for_research(),
    }


def get_brief_status(tool_context: ToolContext) -> Dict[str, Any]:
    """Check this tool BEFORE deciding the next question to ask the user.

    Doesn't modify state -- only reads state["brief"] and reports what's
    still missing for the discovery phase to finish: required fields not
    yet filled, and blocking assumptions the agent filled in on its own
    that need user confirmation before the brief can advance to confirmed.

    Returns:
        brief_status: current status (draft/confirmed).
        missing_required: names of required fields still missing,
            e.g. "hard.nights", "hard.budget_ceiling" -- these block
            ready_for_research.
        missing_recommended: fields worth collecting but that don't block
            anything, e.g. "attractions", "soft.food_style". Ask about
            them OR, if the user declines, record the field_path in
            `declined` via update_brief -- otherwise they keep reappearing
            in this list.
        declined: field_paths from missing_recommended that the user has
            already explicitly declined.
        pending_assumptions: blocking assumptions not yet confirmed, each
            with field_path, value, and rationale -- use the rationale to
            phrase the confirmation question to the user.
        ready_for_research: True when nothing required is missing or
            pending and the discovery phase can be closed out
            (missing_recommended doesn't count toward this, but don't offer
            the final confirmation to the user while it's still non-empty).
    """
    current = tool_context.state.get("brief")
    if current is None:
        current = {"brief_id": f"trip-{tool_context.session.id}"}

    brief, error = _validated_brief(current)
    if error is not None:
        return error

    dumped = brief.model_dump(mode="json")

    return {
        "brief_status": dumped["status"],
        "missing_required": brief.missing_required(),
        "missing_recommended": brief.missing_recommended(),
        "declined": brief.declined,
        "pending_assumptions": [
            {
                "field_path": a.field_path,
                "value": a.value,
                "rationale": a.rationale,
            }
            for a in brief.pending_assumptions()
        ],
        "ready_for_research": brief.ready_for_research(),
    }


# --------------------------------------------------------------------------
# run_feasibility_check -- pure arithmetic, no LLM call.
#
# The constants below are rough heuristics to catch combinations that are
# obviously unworkable BEFORE spending the research phase on them (real
# cost/visa data per destination only arrives later, during research).
# They distinguish "obviously doesn't fit" (infeasible) from "fits, but
# tight" (tight) -- they aren't a definitive verdict.
# --------------------------------------------------------------------------

AVG_DAILY_COST_PER_TRAVELER = 150.0
MIN_NIGHTS_PER_CITY = 2
MIN_INTL_LEAD_DAYS = 21

_CheckResult = Optional[Tuple[Conflict, str]]  # (conflict, "tight" | "infeasible")


def _budget_conflict(brief: TripBrief) -> _CheckResult:
    """nights x average cost x travelers vs budget ceiling."""
    ceiling = brief.hard.budget_ceiling
    nights = brief.hard.nights
    if ceiling is None or not nights or not ceiling.amount:
        return None

    travelers = max(len(brief.travelers), 1)
    estimated = AVG_DAILY_COST_PER_TRAVELER * nights * travelers
    ratio = estimated / ceiling.amount
    if ratio <= 1.0:
        return None

    conflict = Conflict(
        kind=ConflictKind.budget_vs_scope,
        detail=(
            f"Rough estimate ({AVG_DAILY_COST_PER_TRAVELER:.0f} "
            f"{ceiling.currency}/person/night x {nights} nights x {travelers} "
            f"traveler(s) = {estimated:.0f} {ceiling.currency}) exceeds the "
            f"ceiling of {ceiling.amount:.0f} {ceiling.currency}."
        ),
        evidence=f"{estimated:.0f} vs {ceiling.amount:.0f} {ceiling.currency} ({ratio:.1f}x the ceiling)",
        suggested_relaxations=[
            "reduce number of nights",
            "raise the budget ceiling",
            "reduce number of travelers",
        ],
    )
    return conflict, ("infeasible" if ratio > 1.3 else "tight")


def _cities_vs_nights_conflict(brief: TripBrief) -> _CheckResult:
    """number of decided destinations vs available nights."""
    nights = brief.hard.nights
    if not nights:
        return None

    cities = {d.name for d in brief.destinations if d.decided}
    if not cities:
        return None

    nights_per_city = nights / len(cities)
    if nights_per_city >= MIN_NIGHTS_PER_CITY:
        return None

    conflict = Conflict(
        kind=ConflictKind.distance_vs_days,
        detail=(
            f"{len(cities)} destination(s) decided across {nights} nights "
            f"({nights_per_city:.1f} night(s)/destination, recommended "
            f"minimum {MIN_NIGHTS_PER_CITY})."
        ),
        evidence=f"{nights} nights / {len(cities)} destinations = {nights_per_city:.1f}",
        suggested_relaxations=[
            "cut one or more destinations",
            "increase the number of nights",
        ],
    )
    return conflict, ("infeasible" if nights_per_city < 1 else "tight")


def _docs_lead_time_conflict(brief: TripBrief) -> _CheckResult:
    """date window vs minimum lead time to sort out documents.

    Doesn't know the real visa rule for any destination -- only warns when
    the trip is international (destination country != origin country) and
    the earliest date is less than MIN_INTL_LEAD_DAYS days away, which
    doesn't leave time to even check the requirement, let alone meet it.
    """
    origin = brief.origin
    earliest = brief.hard.date_window.earliest
    if origin is None or earliest is None:
        return None

    is_international = any(
        d.decided and d.country and d.country != origin.country
        for d in brief.destinations
    )
    if not is_international:
        return None

    lead_days = (earliest - date.today()).days
    if lead_days >= MIN_INTL_LEAD_DAYS:
        return None

    conflict = Conflict(
        kind=ConflictKind.docs_vs_dates,
        detail=(
            f"International trip with the earliest date {lead_days} day(s) "
            f"away -- below the recommended minimum of {MIN_INTL_LEAD_DAYS} "
            "days to check/arrange passport and visa."
        ),
        evidence=f"{lead_days} days until {earliest.isoformat()}",
        suggested_relaxations=[
            "push the start date further out",
            "confirm with the user whether documents are already in order",
        ],
    )
    return conflict, ("infeasible" if lead_days < 7 else "tight")


_FEASIBILITY_CHECKS = (
    _budget_conflict,
    _cities_vs_nights_conflict,
    _docs_lead_time_conflict,
)


def run_feasibility_check(tool_context: ToolContext) -> Dict[str, Any]:
    """Deterministic reality check -- pure arithmetic, zero LLM.

    Runs 3 calculations against state["brief"]: nights x average cost vs
    budget ceiling, number of decided destinations vs available nights, and
    date window vs minimum document lead time for international travel.
    Each only fires when the brief has enough data (e.g. without
    budget_ceiling, the budget check is skipped). Persists the result to
    brief.feasibility via update_brief, so it survives for the next call to
    get_brief_status.

    Returns:
        On success: status "success", verdict (feasible/tight/infeasible),
        conflicts (list with kind, detail, evidence, suggested_relaxations),
        and an updated ready_for_research.
        If state["brief"] doesn't exist yet: status "error" asking to call
        update_brief first -- there's nothing to check.
    """
    current = tool_context.state.get("brief")
    if current is None:
        return {
            "status": "error",
            "message": 'state["brief"] does not exist yet -- call update_brief first.',
        }

    brief, error = _validated_brief(current)
    if error is not None:
        return error

    conflicts: list[Conflict] = []
    severities: list[str] = []
    for check in _FEASIBILITY_CHECKS:
        result = check(brief)
        if result is not None:
            conflict, severity = result
            conflicts.append(conflict)
            severities.append(severity)

    if "infeasible" in severities:
        verdict = Verdict.infeasible
    elif severities:
        verdict = Verdict.tight
    else:
        verdict = Verdict.feasible

    conflicts_dumped = [c.model_dump(mode="json") for c in conflicts]
    outcome = update_brief(
        {"feasibility": {"verdict": verdict.value, "conflicts": conflicts_dumped}},
        tool_context,
    )
    if outcome["status"] != "success":
        return outcome

    return {
        "status": "success",
        "verdict": verdict.value,
        "conflicts": conflicts_dumped,
        "ready_for_research": outcome["ready_for_research"],
    }


def confirm_assumptions(field_paths: List[str], tool_context: ToolContext) -> Dict[str, Any]:
    """Marks assumptions as confirmed by the user, by field_path.

    Call this after the user answers a confirmation question based on
    `pending_assumptions` (from get_brief_status or update_brief). Each
    field_path present in the brief has `confirmed` set to True -- this
    doesn't change `value` or `rationale`, it only lifts the block. If the
    user CORRECTS the value instead of confirming, use update_brief on the
    field itself (e.g. {"hard": {"nights": 6}}) instead of this tool; a
    corrected field no longer shows up as a pending assumption.

    Args:
        field_paths: the exact field_paths to confirm, e.g.
            ["hard.nights", "soft.pace"]. Come from `pending_assumptions`.

    Returns:
        On success: status "success", confirmed (field_paths that existed
        and got marked), not_found (field_paths that didn't match any
        assumption -- probably mistyped), plus updated
        pending_assumptions and ready_for_research.
        If state["brief"] doesn't exist yet: status "error".
    """
    if not field_paths:
        return {"status": "error", "message": "field_paths is empty -- nothing to confirm."}

    current = tool_context.state.get("brief")
    if current is None:
        return {
            "status": "error",
            "message": 'state["brief"] does not exist yet -- call update_brief first.',
        }

    brief, error = _validated_brief(current)
    if error is not None:
        return error

    requested = set(field_paths)
    matched: set[str] = set()
    updated_assumptions = []
    for a in brief.assumptions:
        dumped = a.model_dump(mode="json")
        if a.field_path in requested:
            matched.add(a.field_path)
            dumped["confirmed"] = True
        updated_assumptions.append(dumped)

    outcome = update_brief({"assumptions": updated_assumptions}, tool_context)
    if outcome["status"] != "success":
        return outcome

    return {
        "status": "success",
        "confirmed": sorted(matched),
        "not_found": sorted(requested - matched),
        "pending_assumptions": outcome["pending_assumptions"],
        "ready_for_research": outcome["ready_for_research"],
    }

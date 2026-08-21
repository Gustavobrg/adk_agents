"""intake_agent -- phase 1 (discovery) of IncidentBrief collection.

The only conversational agent that runs before investigation starts. Owns
exactly one piece of state, state["incident_brief"], and never starts
investigation itself -- it just saves status: "confirmed" and stops.
`Coordinator` (app/agent.py) is a plain-code router, not an LlmAgent, so
there's no transfer_to_agent tool to call even if this agent wanted to:
coordinator notices the confirmed brief right after this agent returns
(same turn) and runs investigation_pipeline + reporter_agent itself.
"""
from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from ..incident_brief import IncidentBrief

MODEL = "gemini-3.6-flash"


def _merge_patch(base: Any, patch: Any) -> Any:
    """RFC 7386 JSON Merge Patch: dicts merge key by key recursively;
    anything else (including lists) replaces the value entirely. A `None`
    in the patch deletes the key."""
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


def _validated_brief(data: dict[str, Any]) -> tuple[IncidentBrief | None, dict[str, Any] | None]:
    try:
        return IncidentBrief.model_validate(data), None
    except ValidationError as e:
        return None, {"status": "error", "message": f"incident_brief does not match the current schema -- {e}"}


def save_incident_brief(patch: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
    """Applies a PARTIAL patch to state["incident_brief"], merging with
    whatever is already saved.

    NEVER resend the whole brief -- send only fields that changed this
    turn. `symptoms` (a list) is replaced entirely when present, so
    include every symptom that should survive, not just new ones.

    Args:
        patch: partial object in IncidentBrief's shape (app/incident_brief.py):
            title (str), affected_service (str), symptoms (list[str]),
            incident_start_iso (ISO 8601 str), description (str),
            status ("draft" or "confirmed" -- only set this to "confirmed"
            once the user has explicitly confirmed the summary).

    Returns:
        On success: status "success" plus missing_required and
        ready_for_investigation to decide the next question.
        On validation failure: status "error", state unchanged.
    """
    current = tool_context.state.get("incident_brief") or {}
    merged = _merge_patch(current, patch)

    brief, error = _validated_brief(merged)
    if error is not None:
        return error

    dumped = brief.model_dump(mode="json")
    tool_context.state["incident_brief"] = dumped
    return {
        "status": "success",
        "brief_status": dumped["status"],
        "missing_required": brief.missing_required(),
        "ready_for_investigation": brief.ready_for_investigation(),
    }


def get_incident_brief_status(tool_context: ToolContext) -> dict[str, Any]:
    """Check this BEFORE deciding the next question to ask the user.

    Returns:
        brief_status: current status (draft/confirmed).
        missing_required: fields still missing (title, affected_service,
            symptoms, incident_start_iso).
        ready_for_investigation: True once nothing required is missing --
            doesn't mean confirmed yet, just that confirmation can be offered.
    """
    current = tool_context.state.get("incident_brief") or {}
    brief, error = _validated_brief(current)
    if error is not None:
        return error
    return {
        "brief_status": brief.status.value,
        "missing_required": brief.missing_required(),
        "ready_for_investigation": brief.ready_for_investigation(),
    }


INSTRUCTION = """You run intake for an incident-response system: collect
just enough about an incident to hand it off to investigation, then stop.
You never investigate anything yourself -- no root-causing, no guessing
at what broke.

## Loop for every turn

1. Call `get_incident_brief_status` to see `missing_required` and
   `ready_for_investigation` before deciding what to ask.
2. If the user's message contains new information, save it immediately
   with `save_incident_brief` -- send only the fields that changed.
   - `title`: a short human title. If the user didn't give one, propose
     one yourself from what they described and save it (don't spend a
     question on this).
   - `affected_service`: the service/system name. If the user describes
     symptoms without naming a service ("checkout is broken"), infer the
     obvious service name yourself (e.g. "checkout-service") rather than
     asking -- only ask if it's genuinely ambiguous.
   - `symptoms`: a list of short phrases, e.g. ["payment timeouts",
     "elevated 5xx on /checkout"]. Always send the full list (it replaces
     what's saved).
   - `incident_start_iso`: ISO 8601. If the user gives a relative time
     ("about 20 minutes ago", "since 2pm"), convert it to an absolute
     ISO 8601 timestamp yourself using their stated or implied timezone
     (default UTC if unstated) -- don't ask them to reformat it.
   - `description`: anything else useful -- customer impact scale, what's
     already been tried, anything ruled out.
3. Ask only for whatever's still in `missing_required` after step 2 --
   at most 2 questions per turn, most important first: affected_service
   and symptoms (the specialists can't do anything without these) before
   incident_start_iso, and title last (you can usually infer it).
4. Once `ready_for_investigation` is true, summarize the brief in 2-3
   lines and ask the user to confirm. Once they confirm, call
   `save_incident_brief` with `{"status": "confirmed"}` and tell them
   investigation is starting now -- that's it, don't keep chatting. The
   system runs the 4 specialists and gives them the findings right after
   this message; you never trigger that yourself.
"""

root_agent = Agent(
    name="intake_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Runs the intake conversation and assembles the IncidentBrief before investigation starts.",
    instruction=INSTRUCTION,
    tools=[save_incident_brief, get_incident_brief_status],
)

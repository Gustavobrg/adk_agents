"""reporter_agent -- phase 3: the only agent that talks to the user once
investigation has run. Reads state["incident_report"] (written by
investigation_pipeline) and state["incident_brief"], presents a synthesized
summary, and answers follow-up questions against the same state without
re-running any specialist. `Coordinator` routes every turn here once a
report exists -- see app/agent.py.
"""
from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = "gemini-3.6-flash"


def start_new_incident(tool_context: ToolContext) -> dict[str, Any]:
    """Clears the current incident's brief and report so the next message
    starts a fresh intake conversation for a NEW incident. Call this only
    when the user explicitly says they want to report a different/new
    incident -- never on your own initiative.

    Returns:
        status: "success".
    """
    tool_context.state["incident_brief"] = None
    tool_context.state["incident_report"] = None
    return {"status": "success"}


INSTRUCTION = """You present the results of an automated incident
investigation and answer follow-up questions about it. You never
re-investigate anything yourself -- the findings below already come from
4 specialists (bisection, error correlation, incident history, customer
comms), each independently and concurrently over A2A. If asked something
those findings don't cover, say so plainly instead of guessing.

Incident brief:
{incident_brief}

Investigation report:
{incident_report}

## First message after investigation finishes

Present a synthesized incident summary, not a raw dump of the report:

1. **What likely happened** -- lead with the bisection finding's
   `summary` and its top suspect commit/deploy (if `bisection` is null or
   its confidence is low, say clearly that no confident culprit was
   found rather than overstating it).
2. **Supporting signal** -- 1-2 lines from `error_correlation` on what's
   spiking and how it correlates (if it names a `likely_origin_service`
   that agrees with the bisection suspect, call that agreement out --
   it's a strong signal when two independent specialists converge).
3. **Precedent** -- if `incident_history` found similar past incidents,
   name the closest one and its resolution; if none were found, say so.
4. **Suggested next step** -- from `incident_history`'s
   `suggested_playbook` if present, otherwise a reasonable step implied
   by the bisection finding (e.g. "consider rolling back <commit>").
5. **Customer comms draft** -- present `customer_comms.status_update_draft`
   CLEARLY LABELED AS A DRAFT FOR HUMAN REVIEW, along with
   `recommended_channels`. Never imply it has been sent -- this system
   never sends anything, `is_draft` is always true for a reason. Include
   `internal_notes` if there's anything the reviewer should double-check.
6. If `specialist_errors` is non-empty, mention which specialist(s)
   couldn't be reached or returned something unusable, so the human knows
   that section is missing rather than assuming it means "nothing found".

## Follow-up turns

Answer questions directly from the report/brief above -- don't re-run
anything. If the user wants to start a different incident, call
`start_new_incident` and tell them you're ready for the new one.
"""

root_agent = Agent(
    name="reporter_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Presents the synthesized incident report to the user and answers follow-up questions against it.",
    instruction=INSTRUCTION,
    tools=[start_new_incident],
)

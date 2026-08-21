"""investigation.py -- phase 2: fans out to the 4 A2A specialists
concurrently and aggregates their findings into state["incident_report"].

    InvestigationPipeline (BaseAgent, plain code -- no LLM call of its own)
      wraps each specialist as an AgentTool(RemoteA2aAgent(...)) and calls
      all 4 via asyncio.gather -- genuine concurrent A2A network calls to
      4 separately deployed services, which is "coordinated in parallel"
      in the most literal sense: real concurrent HTTP requests, not just
      an ADK ParallelAgent branching construct. AgentTool.run_async also
      gives full control over exactly what's sent (a clean incident-brief
      payload, not raw conversation history) and hands back each
      specialist's final response text directly -- no separate step
      needed to fish results back out of session events.

Each specialist enforces its own JSON output shape server-side via
`output_schema`, but that's a controlled-generation contract on THEIR
side, not something AgentTool enforces across an A2A boundary (it only
applies output_schema validation to a local LlmAgent) -- so the response
here is still parsed defensively as free text (JSON, possibly fenced),
the same `_parse_json` pattern travel_agent's scouts use.

One specialist failing (unreachable, timeout, malformed response) is
recorded in `specialist_errors` and degrades that one section to `None`
-- it never sinks the whole investigation.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events import Event, EventActions
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, ValidationError

from ..config import specialist_agent_card_url
from ..incident_brief import IncidentBrief
from ..incident_report import (
    BisectionFinding,
    CustomerCommsDraft,
    ErrorCorrelationFinding,
    IncidentHistoryFinding,
    IncidentReport,
    SpecialistError,
)

_FENCE_RE = re.compile(r"```\w*\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: Any) -> Any:
    """Best-effort JSON parse of a specialist's raw text response. Returns
    None on anything that isn't parseable JSON."""
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


# (state key on IncidentReport, A2A agent name, description, env var
# holding its agent-card URL, pydantic model to validate its response
# against)
_SPECIALISTS: tuple[tuple[str, str, str, str, type[BaseModel]], ...] = (
    (
        "bisection",
        "bisection_specialist",
        "Finds the deploy/commit most likely to have caused the incident.",
        "BISECTION_AGENT_CARD_URL",
        BisectionFinding,
    ),
    (
        "error_correlation",
        "error_correlation_specialist",
        "Finds which error signals spike together across services.",
        "ERRORS_AGENT_CARD_URL",
        ErrorCorrelationFinding,
    ),
    (
        "incident_history",
        "incident_history_specialist",
        "Finds precedent from past incidents and a suggested playbook.",
        "HISTORY_AGENT_CARD_URL",
        IncidentHistoryFinding,
    ),
    (
        "customer_comms",
        "customer_comms_specialist",
        "Drafts a first customer-facing status update (draft only).",
        "COMMS_AGENT_CARD_URL",
        CustomerCommsDraft,
    ),
)


async def _call_specialist(
    state_key: str,
    name: str,
    description: str,
    env_var: str,
    model_cls: type[BaseModel],
    brief_text: str,
    tool_context: ToolContext,
) -> tuple[str, BaseModel | None, SpecialistError | None]:
    remote = RemoteA2aAgent(
        name=name,
        description=description,
        agent_card=specialist_agent_card_url(env_var),
    )
    tool = AgentTool(agent=remote, skip_summarization=True)
    try:
        raw = await tool.run_async(args={"request": brief_text}, tool_context=tool_context)
    except Exception as e:  # noqa: BLE001 -- an unreachable specialist must not sink the whole investigation
        return state_key, None, SpecialistError(specialist=name, error=f"{type(e).__name__}: {e}")

    parsed = _parse_json(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        preview = str(raw)[:300]
        return state_key, None, SpecialistError(specialist=name, error=f"non-JSON or empty response: {preview!r}")

    try:
        validated = model_cls.model_validate(parsed)
    except ValidationError as e:
        return state_key, None, SpecialistError(specialist=name, error=f"response didn't match expected shape: {e}")

    return state_key, validated, None


class InvestigationPipeline(BaseAgent):
    """Runs the 4 specialists concurrently and writes state["incident_report"].
    Pure code, no LLM call of its own -- only the specialists' own agents
    (on the other end of the A2A calls) call an LLM."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        brief = IncidentBrief.model_validate(state["incident_brief"])
        brief_text = brief.model_dump_json(exclude_none=True)

        tool_context = ToolContext(ctx)
        results = await asyncio.gather(
            *(
                _call_specialist(state_key, name, description, env_var, model_cls, brief_text, tool_context)
                for state_key, name, description, env_var, model_cls in _SPECIALISTS
            )
        )

        report_fields: dict[str, Any] = {"brief_title": brief.title or "Untitled incident"}
        errors: list[dict[str, Any]] = []
        for state_key, validated, error in results:
            if error is not None:
                errors.append(error.model_dump())
            else:
                report_fields[state_key] = validated.model_dump()
        report_fields["specialist_errors"] = errors

        report = IncidentReport.model_validate(report_fields)

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(state_delta={"incident_report": report.model_dump(mode="json")}),
        )


def create_investigation_pipeline() -> InvestigationPipeline:
    return InvestigationPipeline(
        name="investigation_pipeline",
        description=(
            "Fans out to the 4 A2A incident-response specialists concurrently "
            "and aggregates their findings into an IncidentReport."
        ),
    )

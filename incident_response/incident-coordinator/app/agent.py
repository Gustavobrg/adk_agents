"""Coordinator -- root_agent of incident-coordinator.

    coordinator (BaseAgent, plain code -- no LLM call of its own)
      |-- intake_agent            -- phase 1: conversational, produces state["incident_brief"]
      |-- investigation_pipeline  -- phase 2: headless, fans out to 4 A2A specialists
      |                              concurrently, produces state["incident_report"]
      +-- reporter_agent          -- phase 3: conversational, presents the report

Which phase runs next is never a judgment call -- it's fully determined by
`incident_brief.status` and whether `incident_report` already exists.
coordinator is a plain BaseAgent, not an LlmAgent: it has no
transfer_to_agent tool and nothing to prompt -- it just inspects
ctx.session.state and calls straight into whichever child agent(s) apply,
in the same turn, as many as are relevant. This mirrors travel_agent's
own root coordinator (adk_agents/travel_agent/agent.py) -- same reasoning
about why a plain BaseAgent is the right root here applies verbatim.
"""
from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.apps import App
from google.adk.events import Event

from .incident_brief import BriefStatus
from .sub_agents.intake_agent import root_agent as intake_agent
from .sub_agents.investigation import create_investigation_pipeline
from .sub_agents.reporter_agent import root_agent as reporter_agent


class Coordinator(BaseAgent):
    """Routes each turn to intake, investigation, and/or reporting by
    reading state alone. `self.sub_agents` is fixed at construction time
    (see `create_coordinator`) in the order (intake_agent,
    investigation_pipeline, reporter_agent) -- unpacked positionally below
    instead of adding extra pydantic fields for what's already the
    `sub_agents` list."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        intake, investigation, reporter = self.sub_agents
        state = ctx.session.state
        brief = state.get("incident_brief")

        if not brief or brief.get("status") != BriefStatus.confirmed.value:
            # Still in intake -- only intake_agent talks to the user.
            async for event in intake.run_async(ctx):
                yield event
            # Re-read: intake_agent may have just confirmed the brief this
            # very turn (the user's final "yes"). If so, fall through
            # below to start investigation immediately instead of waiting
            # for the next user message to notice.
            brief = ctx.session.state.get("incident_brief")
            if not brief or brief.get("status") != BriefStatus.confirmed.value:
                return

        if not ctx.session.state.get("incident_report"):
            async for event in investigation.run_async(ctx):
                yield event

        async for event in reporter.run_async(ctx):
            yield event


def create_coordinator() -> Coordinator:
    """Create and return the coordinator (root) agent."""
    return Coordinator(
        name="coordinator",
        description=(
            "Routes an incident end to end: intake (intake_agent), then "
            "investigation across 4 A2A specialists run concurrently "
            "(investigation_pipeline), then reporting (reporter_agent) -- "
            "purely from incident state, never an LLM decision."
        ),
        sub_agents=[
            intake_agent,
            create_investigation_pipeline(),
            reporter_agent,
        ],
    )


root_agent = create_coordinator()

app = App(
    root_agent=root_agent,
    name="incident_coordinator_app",
)

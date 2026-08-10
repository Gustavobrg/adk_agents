"""Coordinator -- root_agent of travel_agent.

    coordinator (BaseAgent, plain code -- no LLM call of its own)
      |-- intake_agent       -- phase 1: conversational, produces state["brief"]
      |-- research_pipeline  -- phase 2: headless, produces state["research"]
      +-- planning_pipeline  -- phase 3: headless, produces state["plan"]

Which phase runs next is never a judgment call -- it's fully determined by
`brief.status` and whether `research`/`plan` already exist for the CURRENT
brief version. That's exactly the kind of decision this codebase always
resolves in code instead of an LLM call (see SelectionCapacity,
GeoClustering, PlanAssembler in planning_pipeline for the same philosophy).
So coordinator is a plain BaseAgent, not an LlmAgent: it has no
`transfer_to_agent` tool and nothing to prompt -- it just inspects
ctx.session.state and calls straight into whichever child agent(s) apply,
in the same turn, as many as are relevant.

Being a plain BaseAgent (not an LlmAgent) also has a load-bearing side
effect on ADK's cross-turn agent resolution: the Runner only lets a
non-root agent "stay sticky" across turns (resume directly, bypassing the
root) when every ancestor up to the root supports transfer
(`disallow_transfer_to_parent`, an LlmAgent-only attribute). Since
coordinator doesn't have that attribute, that chain always breaks at
coordinator, so EVERY new user turn re-enters here first, guaranteeing the
state check below always runs before intake_agent (or anything else) gets
another turn.
"""
from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event

from .sub_agents.intake_agent import create_intake_agent
from .sub_agents.intake_agent.trip_brief import BriefStatus
from .sub_agents.planning_pipeline import create_planning_pipeline
from .sub_agents.research_pipeline import create_research_pipeline


class Coordinator(BaseAgent):
    """Routes each turn to intake, research, and/or planning by reading
    state alone. `self.sub_agents` is fixed at construction time (see
    `create_coordinator`) in the order (intake_agent, research_pipeline,
    planning_pipeline) -- unpacked positionally below instead of adding
    extra pydantic fields for what's already the `sub_agents` list.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        intake_agent, research_pipeline, planning_pipeline = self.sub_agents
        state = ctx.session.state
        brief = state.get("brief")

        if not brief or brief.get("status") != BriefStatus.confirmed.value:
            # Still in discovery -- only intake_agent talks to the user.
            async for event in intake_agent.run_async(ctx):
                yield event
            return

        version = brief.get("version")
        ran_a_pipeline = False

        research = state.get("research")
        if not research or research.get("brief_version") != version:
            ran_a_pipeline = True
            async for event in research_pipeline.run_async(ctx):
                yield event

        # Re-read: research_pipeline (if it ran above) just wrote this.
        research = ctx.session.state.get("research")
        plan = ctx.session.state.get("plan")
        if research and (not plan or plan.get("brief_version") != version):
            ran_a_pipeline = True
            async for event in planning_pipeline.run_async(ctx):
                yield event

        if not ran_a_pipeline:
            # Confirmed brief, research and plan both already current for
            # this version -- there's nothing left to compute. Hand the
            # turn to intake_agent so the user still has someone to talk
            # to (e.g. requesting a change updates the brief, and a higher
            # `version` there re-opens research/planning on the next turn).
            async for event in intake_agent.run_async(ctx):
                yield event


def create_coordinator():
    """Create and return the coordinator (root) agent."""
    return Coordinator(
        name="coordinator",
        description=(
            "Routes a trip end to end: discovery (intake_agent), then "
            "research (research_pipeline), then planning (planning_pipeline), "
            "purely from trip state -- never an LLM decision."
        ),
        sub_agents=[
            create_intake_agent(),
            create_research_pipeline(),
            create_planning_pipeline(),
        ],
    )


root_agent = create_coordinator()

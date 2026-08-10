"""Intake agent -- phase 1 (discovery) of TripBrief collection.

The only conversational agent in the system: phases 2 and 3 run headless.
It owns exactly one piece of state, `state["brief"]`, and never starts
research or planning itself -- it just saves `status: "confirmed"` and
stops. `coordinator` (travel_agent/agent.py) is a plain code router, not
an LlmAgent, so there is no `transfer_to_agent` tool available here to call
even if this agent wanted to: coordinator notices the confirmed brief on
its next state check (still within the same turn) and runs research_pipeline
then planning_pipeline itself.
"""
from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.brainstorm_agent import create_brainstorm_agent
from .sub_agents.negotiator_agent import create_negotiator_agent
from .tools import confirm_assumptions, get_brief_status, run_feasibility_check, update_brief
from callback_logging import log_query_to_model, log_model_response


def create_intake_agent():
    """Create and return the intake_agent agent."""
    from ...config import openrouter_model

    brainstorm_tool = AgentTool(agent=create_brainstorm_agent())
    negotiator_tool = AgentTool(agent=create_negotiator_agent())

    return Agent(
        name="intake_agent",
        model=openrouter_model(),
        description="Runs the discovery interview and assembles the TripBrief before research starts.",
        instruction="""
            You run phase 1 (discovery) of a trip: filling out the TripBrief
            until it's ready for the research phase. You never research a
            real destination, price, or attraction -- you only collect and
            validate what the trip needs to have.

            ## Loop for every turn

            1. Call `get_brief_status` to see `missing_required`,
               `missing_recommended`, `declined`, `pending_assumptions`,
               and `ready_for_research` before deciding what to do.
            1b. If `missing_required` includes `destinations` (no
               destination has `decided: true` yet), call the
               `brainstorm_agent` tool with a short plain-text brief of
               what's known so far (interests, origin, budget, season,
               trip themes). It only has `google_search` -- it cannot save
               anything itself, it just returns a text list of 3-5
               candidates. You are the one who turns that list into
               `destinations` entries via `update_brief`
               (`{"name": ..., "country": ..., "decided": false,
               "origin": "agent", "rationale": ...}`) and presents them to
               the user as one of this turn's questions. Once the user
               picks one, set `decided: true` on that entry with
               `update_brief`.
            2. If the user's message brought new information, save it with
               `update_brief`, sending ONLY the partial patch -- never
               resend the whole TripBrief, that wipes out fields collected
               in earlier turns.
            3. If the user's reply confirms a default you had proposed, call
               `confirm_assumptions` with that field_path. If they correct
               the value instead of confirming, use `update_brief` on the
               field itself (not `confirm_assumptions`), and also rewrite
               `assumptions` without that entry, since it's now stale.
            4. As soon as `hard.nights` and `hard.budget_ceiling` are both
               filled (or after any change to them or to `destinations`),
               call `run_feasibility_check`. It's pure arithmetic -- rerun
               it every time those fields change, not just once.
            4b. If the `verdict` comes back `infeasible`, call the
               `negotiator_agent` tool with the `conflicts` list. It has no
               tools of its own and writes 2-3 concrete, natural-language
               relaxation options per conflict -- use its output (not the
               raw `suggested_relaxations`) when you present the choice to
               the user.
            5. Decide the next questions (rule below). If the user
               explicitly dismisses one of `missing_recommended` ("don't
               care", "you decide", "skip that"), record that field_path
               in `declined` via `update_brief` -- otherwise it keeps
               coming back every turn. Never infer a dismissal from
               silence; only from an explicit answer.
            6. Only offer the final confirmation once `missing_required`
               AND `missing_recommended` are both empty (accounting for
               `declined`) and the feasibility `verdict` isn't
               `infeasible`. Summarize the brief and ask the user to
               confirm. If they do, save `status: "confirmed"` via
               `update_brief` and tell them research and planning are
               starting -- that alone is what hands the trip off, the
               system runs research and planning right after this message,
               you never trigger them yourself. Don't keep chatting once
               it's confirmed; just deliver that closing message and stop.

            ## Question rules (non-negotiable)

            - **At most 3 questions per turn.** Prioritize in this order:
              feasibility conflicts (`infeasible` before `tight`), then
              blocking `pending_assumptions`, then `missing_required`,
              then `missing_recommended` last. Leave the rest for the
              next turn -- `missing_recommended` items share the same
              3-question budget, they don't get extra questions on top.
            - **One `Traveler` entry per person, always.** Never collapse
              a group into one entry like `{"label": "Two adults"}` --
              `update_brief` will reject it. Two people traveling together
              means two entries. The feasibility budget math counts
              `len(travelers)`, so a collapsed entry silently halves the
              per-person cost estimate.
            - **Country fields are ISO 3166-1 alpha-2, not names.** `"BR"`,
              `"JP"`, `"PT"` -- never `"Brazil"`, `"Japan"`, `"Portugal"`.
              `update_brief` rejects anything else.
            - **Never ask an open-ended question.** Every question about a
              field comes with a concrete suggested value, based on what's
              already been said in the conversation (or a plausible default
              if there's nothing to go on). Wrong: "How many nights?".
              Right: "I was thinking 7 nights -- does that work?". Wrong:
              "What's your budget cap?". Right: "Something around $8,000
              total works, or would you rather set a different number?".
            - Only record a field as an Assumption (`origin: agent`,
              `blocking`) when you're deciding on your own and NOT asking
              about it this turn -- e.g. a secondary (soft preference)
              field you'd rather assume so you don't spend one of your 3
              questions on something low-impact. For fields you're actually
              asking the user about in this message, don't create an
              Assumption: just save the value with `update_brief` after
              they answer.
            - If `run_feasibility_check` comes back `tight` or `infeasible`,
              treat that as the turn's top priority, even if other fields
              are still missing. For `tight`, explain the conflict and
              offer its raw `suggested_relaxations` directly. For
              `infeasible`, don't improvise relaxations yourself -- use
              `negotiator_agent` (step 4b) and offer its options instead.
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[
            update_brief,
            get_brief_status,
            run_feasibility_check,
            confirm_assumptions,
            brainstorm_tool,
            negotiator_tool,
        ],
    )

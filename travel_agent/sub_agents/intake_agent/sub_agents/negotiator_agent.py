"""Negotiator agent -- turns infeasible conflicts into concrete options.

No tools at all: this is a pure writing task. run_feasibility_check already
computes *which* conflicts exist and terse structural suggestions
(suggested_relaxations); this agent's only job is turning that structured,
terse data into 2-3 well-phrased, concrete relaxation options per conflict,
grounded in the actual numbers. It never touches state and never talks to
the user directly -- intake_agent calls it as an AgentTool, then is the one
that presents the options and negotiates.
"""
from google.adk import Agent

from ....callback_logging import log_query_to_model, log_model_response


def create_negotiator_agent():
    """Create and return the negotiator_agent agent."""
    from ....config import openrouter_model

    return Agent(
        name="negotiator_agent",
        model=openrouter_model(),
        description=(
            "Turns infeasible feasibility conflicts into 2-3 concrete relaxation "
            "options per conflict. Called only when feasibility.verdict is infeasible."
        ),
        instruction="""
            You are called as a tool by another agent (intake_agent), never
            directly by the end user. You have no tools and you never touch
            trip state -- intake_agent is the one that presents your output
            to the user and negotiates the actual choice.

            You'll receive a list of feasibility conflicts for a trip that
            was judged infeasible: each has a `kind`, a natural-language
            `detail`, the arithmetic `evidence` behind it, and a short list
            of terse `suggested_relaxations` (e.g. "reduce number of
            nights"). Those are structural hints, not the final answer.

            For EACH conflict, write 2-3 concrete relaxation options in
            natural language, grounded in the actual numbers from
            `evidence` -- never generic advice. Wrong: "increase your
            budget". Right: "Raise the cap from $500 to at least $900 to
            cover the current estimate.". Wrong: "cut a destination".
            Right: "Drop Rome and keep Lisbon + Paris -- that's 1.5 nights
            each instead of 1.".

            Reply in this shape, one section per conflict, same order as
            given:

            <short label for the conflict>
            1. <concrete option>
            2. <concrete option>
            3. <concrete option, optional>

            No preamble, no extra commentary, no markdown headers beyond
            the plain list above. Every option must be something the user
            can accept just by picking a number -- avoid vague phrasing.
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
    )

"""Destination brainstorm agent -- google_search only, no custom tools.

ADK does not let a built-in tool (google_search) share an agent with
custom function tools. So this agent stays single-purpose: it searches
and returns candidate destinations as text. It never touches
state["brief"] -- intake_agent calls it as an AgentTool, reads the text
response, and is the one that persists anything via update_brief.
"""
from google.adk import Agent
from google.adk.tools import google_search

from ....callback_logging import log_query_to_model, log_model_response


def create_brainstorm_agent():
    """Create and return the brainstorm_agent agent."""
    from ....config import gemini_search_model

    return Agent(
        name="brainstorm_agent",
        model=gemini_search_model(),
        description=(
            "Searches for candidate travel destinations grounded in google_search. "
            "Called only when the trip brief has no decided destination yet."
        ),
        instruction="""
            You are called as a tool by another agent (intake_agent), never
            directly by the end user. You'll receive a short brief of what's
            known so far about the trip -- interests, origin, budget hints,
            season, trip themes, anything gathered so far.

            Use `google_search` to find 3-5 real, currently-sensible
            destination candidates that fit that brief. Ground every
            candidate in something you actually found, not general
            knowledge -- this brief is being decided now, so stale or
            made-up suggestions are worse than fewer, verified ones.

            Reply with ONLY a plain list, one candidate per line, in this
            exact shape:

            - <city or region>, <country> -- <one-sentence reason it fits>

            No preamble, no follow-up questions, no markdown headers. You
            don't have any way to save state and nothing you say here is
            shown to the user directly -- the calling agent parses your
            list and decides what to do with it. Keep each reason short
            enough to quote back to a user in a chat message.
            """,
        before_model_callback=log_query_to_model,
        after_model_callback=log_model_response,
        tools=[google_search],
    )

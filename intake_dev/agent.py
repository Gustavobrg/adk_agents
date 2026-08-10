"""Dev-only standalone runner for intake_agent.

Exists so `adk web intake_dev` can open intake_agent directly, without
going through travel_agent's coordinator (travel_agent/agent.py), which
always starts a turn by checking trip state and would otherwise pass
through the same code path anyway -- this just skips straight to it.
"""
from travel_agent.sub_agents.intake_agent import create_intake_agent

root_agent = create_intake_agent()

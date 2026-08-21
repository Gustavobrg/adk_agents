"""Config for incident-coordinator -- mainly where to find each specialist's
A2A agent card.

app/__init__.py imports agent.py (which needs these env vars, via
investigation.py) before fast_api_app.py's own load_dotenv() call runs, so
load here too.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_URLS = {
    "BISECTION_AGENT_CARD_URL": "http://localhost:8001/a2a/bisection_app/.well-known/agent-card.json",
    "ERRORS_AGENT_CARD_URL": "http://localhost:8002/a2a/errors_app/.well-known/agent-card.json",
    "HISTORY_AGENT_CARD_URL": "http://localhost:8003/a2a/history_app/.well-known/agent-card.json",
    "COMMS_AGENT_CARD_URL": "http://localhost:8004/a2a/comms_app/.well-known/agent-card.json",
}


def specialist_agent_card_url(env_var: str) -> str:
    """Resolves a specialist's agent-card URL, env var first, else the
    localhost default for that specialist (see the module-level defaults
    above and incident_response/README.md for the port assignments)."""
    return os.getenv(env_var, _DEFAULT_URLS[env_var])

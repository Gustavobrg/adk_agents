"""Integration test for the confirmed-brief -> investigation -> report path.

Not covered by tests/eval/: the jump from a confirmed IncidentBrief to
investigation_pipeline running is driven by plain-code state routing in
Coordinator (app/agent.py), not an LLM decision, and the eval framework's
multi-turn replay does not reconstruct session state built by real tool
calls (save_incident_brief) -- see tests/eval/eval_config.yaml. So this
path is tested directly here: seed a confirmed brief straight into
session state (bypassing the intake conversation) and run the real
Coordinator against the real specialists.

Requires the 4 specialist servers running locally (see
incident_response/README.md) -- skips with a clear reason if they're not
reachable, rather than failing the suite.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agent import root_agent
from app.config import specialist_agent_card_url

_SPECIALIST_ENV_VARS = (
    "BISECTION_AGENT_CARD_URL",
    "ERRORS_AGENT_CARD_URL",
    "HISTORY_AGENT_CARD_URL",
    "COMMS_AGENT_CARD_URL",
)


def _unreachable_specialists() -> list[str]:
    unreachable = []
    for env_var in _SPECIALIST_ENV_VARS:
        url = specialist_agent_card_url(env_var)
        try:
            urllib.request.urlopen(url, timeout=3)
        except (urllib.error.URLError, TimeoutError):
            unreachable.append(url)
    return unreachable


@pytest.fixture(scope="module")
def require_specialists():
    unreachable = _unreachable_specialists()
    if unreachable:
        pytest.skip(
            "Specialist(s) not reachable, start them first (see "
            "incident_response/README.md 'Run it locally'): " + ", ".join(unreachable)
        )


@pytest.mark.asyncio
async def test_confirmed_brief_produces_full_incident_report(require_specialists) -> None:
    session_service = InMemorySessionService()
    initial_state = {
        "incident_brief": {
            "status": "confirmed",
            "title": "Checkout failures spiking",
            "affected_service": "checkout-service",
            "symptoms": ["payment timeouts", "elevated 5xx on /checkout"],
            "incident_start_iso": "2026-08-20T14:32:00Z",
            "description": "Customers report checkout failing across the board since ~14:32 UTC.",
        }
    }
    session = await session_service.create_session(
        app_name="pytest", user_id="test_user", state=initial_state
    )
    runner = Runner(app_name="pytest", agent=root_agent, session_service=session_service)

    message = types.Content(role="user", parts=[types.Part.from_text(text="go")])
    got_reporter_text = False
    async for event in runner.run_async(user_id="test_user", session_id=session.id, new_message=message):
        if event.author == "reporter_agent" and event.content and event.content.parts:
            if any(p.text for p in event.content.parts):
                got_reporter_text = True

    final_session = await session_service.get_session(
        app_name="pytest", user_id="test_user", session_id=session.id
    )
    report = final_session.state.get("incident_report")

    assert got_reporter_text, "expected reporter_agent to produce a final text response"
    assert report is not None, "expected investigation_pipeline to write state['incident_report']"
    assert report.get("specialist_errors") == [], (
        f"expected all 4 specialists to succeed, got errors: {report.get('specialist_errors')}"
    )

    bisection = report.get("bisection")
    assert bisection is not None
    assert bisection.get("confidence") in ("low", "medium", "high")
    assert any(
        s.get("commit_sha") == "a3f9c21" for s in bisection.get("suspect_commits", [])
    ), f"expected commit a3f9c21 among suspects, got {bisection.get('suspect_commits')}"

    error_correlation = report.get("error_correlation")
    assert error_correlation is not None
    assert error_correlation.get("spiking_signals"), "expected at least one spiking signal"

    incident_history = report.get("incident_history")
    assert incident_history is not None

    customer_comms = report.get("customer_comms")
    assert customer_comms is not None
    assert customer_comms.get("is_draft") is True, "customer_comms must never claim to have sent anything"
    assert customer_comms.get("severity_estimate") in ("SEV1", "SEV2", "SEV3", "SEV4")

    await runner.close()

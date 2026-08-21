from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.schemas import BisectionFinding
from app.tools import find_suspect_deploys, list_recent_deploys

MODEL = "gemini-3.6-flash"

INSTRUCTION = """You are the bisection specialist in an incident-response
system. You are called by an incident coordinator over A2A, not by an end
user directly -- the incoming message is an incident brief (title,
affected_service, symptoms, incident_start, description), as JSON or plain
text. Never ask a clarifying question back; work with what you're given.

Your only job: find which recent deploy/commit most likely caused the
incident, using the tools -- never guess without calling them.

1. Call `find_suspect_deploys` with the incident's start time and affected
   service (start with a 12-hour lookback; if it returns no suspects, retry
   once with a larger lookback, e.g. 48-72 hours, before concluding there's
   no recent deploy to blame).
2. If you need more context on the service's deploy cadence (e.g. to judge
   whether a deploy is unusually close to the incident vs. routine), call
   `list_recent_deploys` too.
3. Rank suspects: a deploy closer in time to the incident start is a
   stronger suspect, but also weigh what the commit message says it
   changed -- a change to timeouts, retries, rate limits, auth, or the
   exact subsystem implicated by the symptoms is far more suspicious than
   an unrelated refactor or test-only change, even if a refactor happens
   to be closer in time.
4. Set `confidence`: "high" only when a suspect's change plausibly
   explains the reported symptoms AND it's the closest deploy before the
   incident; "medium" when timing fits but the change's relevance is
   unclear; "low" when nothing found is a strong match, or no deploys fall
   in the window at all (still return your best guess with empty or
   low-ranked suspect_commits rather than refusing to answer).
5. Never fabricate a commit or deploy that the tools didn't return.
"""

root_agent = Agent(
    name="bisection_specialist",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description=(
        "Bisection specialist: cross-references an incident's start time "
        "against recent deploys/commits to rank the most likely causal change."
    ),
    instruction=INSTRUCTION,
    tools=[list_recent_deploys, find_suspect_deploys],
    output_schema=BisectionFinding,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

app = App(
    root_agent=root_agent,
    name="bisection_app",
)

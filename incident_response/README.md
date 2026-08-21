# Incident Response -- A2A Multi-Agent System

An incident-response coordinator that fans out to 4 independent
specialists **concurrently, over the real A2A protocol** -- each one is
its own `agents-cli`-scaffolded project, independently runnable and
independently deployable.

```
incident-coordinator/         (port 8000) -- talks to the human, orchestrates
specialist-bisection/         (port 8001) -- which deploy/commit caused it
specialist-errors/            (port 8002) -- which error signals correlate
specialist-history/           (port 8003) -- has this happened before
specialist-customer-comms/    (port 8004) -- drafts a customer status update
```

## Architecture

```
incident-coordinator
  coordinator (BaseAgent, plain code)
    |-- intake_agent            conversational: builds IncidentBrief
    |-- investigation_pipeline  headless: asyncio.gather over 4
    |                           AgentTool(RemoteA2aAgent) calls -- real
    |                           concurrent HTTP requests to the 4
    |                           specialists below, aggregated into
    |                           an IncidentReport
    +-- reporter_agent          conversational: presents the report,
                                 answers follow-ups
```

Each specialist is a standalone Gemini `LlmAgent` with its own tools and
`output_schema`, served over A2A by the scaffold's built-in
`app/app_utils/a2a.py` wiring (agent-card + JSON-RPC endpoints). The
coordinator only knows each specialist's *documented* JSON contract
(`app/schemas.py` in each specialist vs. `app/incident_report.py` in the
coordinator) -- it does not import specialist code, matching how
independently-deployed A2A services actually integrate.

Concurrency is done with `asyncio.gather` over 4
`AgentTool(RemoteA2aAgent(...))` calls (see
`incident-coordinator/app/sub_agents/investigation.py`) rather than an
ADK `ParallelAgent` node, so each call's exact outgoing payload (a clean
incident-brief JSON, not raw chat history) and returned result are fully
under our control. One specialist failing (unreachable, timeout,
malformed response) is recorded in `incident_report.specialist_errors`
and degrades only that section -- it never sinks the whole investigation.

Data sources are synthetic and self-contained (`app/data/*.json` in each
specialist) so the whole system runs with zero external credentials
beyond Vertex AI/Gemini -- see the "Swap in real data" section below for
where to point each specialist at a real system.

## Run it locally

Each project has its own `.venv` (already installed). Open 5 terminals:

```bash
cd specialist-bisection      && .venv/Scripts/python -m uvicorn app.fast_api_app:app --port 8001
cd specialist-errors         && .venv/Scripts/python -m uvicorn app.fast_api_app:app --port 8002
cd specialist-history        && .venv/Scripts/python -m uvicorn app.fast_api_app:app --port 8003
cd specialist-customer-comms && .venv/Scripts/python -m uvicorn app.fast_api_app:app --port 8004
cd incident-coordinator      && agents-cli playground   # or: .venv/Scripts/python -m uvicorn app.fast_api_app:app --port 8000
```

The coordinator's `.env` already points at `http://localhost:8001-8004`
(`*_AGENT_CARD_URL`) -- start the 4 specialists first, then the
coordinator. In the playground, describe an incident, e.g.:

> Checkout is failing for customers -- payment timeouts and 5xx errors on
> /checkout, started around 14:32 UTC today, affecting checkout-service.

`intake_agent` will confirm the brief, then investigation runs against
all 4 specialists in parallel and `reporter_agent` presents the
synthesized report (bisection culprit, correlated errors, similar past
incident, and a customer-comms **draft** clearly marked for human
review -- this system never sends anything).

Verified end-to-end against this exact scenario during development: the
synthetic dataset has a matching deploy (`checkout-service`, timeout
reduced 5s -> 1.2s, 17 minutes before symptoms), a matching error spike
(`payments-service` retries exhausted -> `checkout-service` gateway
timeouts), and a matching past incident (`INC-2025-114`) so the demo
narrative is coherent out of the box.

## Evals

Every project has an `agents-cli` eval suite in `tests/eval/` -- run from
inside the project directory:

```bash
cd specialist-bisection && agents-cli eval run   # dataset from tests/eval/datasets/, results to artifacts/grade_results/
```

Each specialist runs 2 metrics per case: `output_schema_valid` (a local,
deterministic check that the response is valid JSON matching that
specialist's `app/schemas.py` shape) and `rubric_judge` (an LLM judge
graded against per-case `rubric_groups` on the dataset -- e.g.
"identifies commit a3f9c21 as the top suspect with high confidence", or
"never names an internal service or error type" for customer-comms).
Datasets deliberately include negative/honesty cases, not just clean
positives -- e.g. bisection's `unrelated_service_no_deploy` (no deploy
exists for the affected service -- must not fabricate one) and
`ambiguous_benign_change` (a timing match whose content doesn't actually
explain the symptoms -- must not overstate confidence).

`rubric_judge` (`tests/eval/rubric_judge.py`, identical across all 5
projects) exists instead of the eval skill's documented
`final_response_quality` + `metric_spec_parameters.rubric_group_key`
path -- verified locally that this `agents-cli` version's `eval_utils.py`
never implements `metric_spec_parameters` (an entry shaped that way
silently becomes an invalid `LLMMetric` and 400s with "Unsupported
metric type or invalid metric name"). `rubric_judge` reads a case's
`rubric_groups` directly and grades each criterion itself, which the
eval skill's own docs recommend as the workaround for exactly this case.

The coordinator has two datasets, since single-turn and multi-turn cases
can't share a metric run: `tests/eval/datasets/basic-dataset.json`
(single-turn intake behavior -- default `eval run` path) and
`tests/eval/datasets/multi-turn-followup.json` (does it avoid re-asking
for information already given), graded separately:

```bash
cd incident-coordinator
agents-cli eval run   # basic-dataset.json
agents-cli eval generate --dataset tests/eval/datasets/multi-turn-followup.json -o artifacts/traces-multiturn/
agents-cli eval grade --traces artifacts/traces-multiturn/ --config tests/eval/eval_config.yaml
```

The confirmed-brief -> investigation -> report path isn't covered by
these datasets at all: it's driven by plain-code state routing
(`Coordinator` in `app/agent.py`), not an LLM decision, and the eval
framework's multi-turn replay doesn't reconstruct session state built by
real tool calls. That path is covered instead by
`tests/integration/test_investigation_pipeline.py` -- a pytest that
seeds a confirmed brief directly into session state and runs the real
coordinator against the real specialists (skips cleanly with a clear
reason if the 4 specialist servers aren't running):

```bash
cd incident-coordinator && .venv/Scripts/python -m pytest tests/integration/test_investigation_pipeline.py -v
```

All of the above were run for real while building this suite (not just
authored) -- `agents-cli eval run` for every one of the 5 projects, the
multi-turn dataset via `generate`+`grade`, and the pytest integration
test, both with the 4 specialists up (pass) and down (clean skip). One
real bug surfaced this way and was fixed: `specialist-errors`'
`correlate_error_spikes` tool scans error logs across every service
un-scoped, and the agent's original instructions cited its top (global)
result even when it belonged to a different incident than the one it was
asked about -- caught by the `isolated_auth_signal_no_correlation` eval
case (score dropped to 0.83), fixed by adding an explicit
affected-service relevance check to the instructions (back to 1.0, other
cases unaffected).

## Deploying a specialist elsewhere

Each specialist is independently deployable (`agents-cli scaffold
enhance` was intentionally skipped -- these were scaffolded
`--prototype`, no deployment target/CI-CD, to keep local iteration fast).
To actually deploy one: `cd` into it and run `agents-cli scaffold
enhance .` to add a deployment target, then `agents-cli deploy`. Once
deployed, update the corresponding `*_AGENT_CARD_URL` in
`incident-coordinator/.env` to the deployed host's
`/a2a/<app-name>/.well-known/agent-card.json` URL -- the coordinator
doesn't care where a specialist runs, only that its agent card is
reachable.

## Swap in real data

Every specialist's `app/tools.py` reads a synthetic JSON file under
`app/data/`. Swap the `_load()` function for a real API call to point it
at production data -- keep the tool signatures and return shapes stable,
since that's the actual contract each specialist's agent reasons over:

- **specialist-bisection**: `app/data/{commits,deploys}.json` -> GitHub/
  GitLab API + your CD system's release log.
- **specialist-errors**: `app/data/error_logs.json` -> Datadog/Sentry/
  Elasticsearch query.
- **specialist-history**: `app/data/past_incidents.json` -> Jira,
  PagerDuty postmortems, or an internal wiki search API.
- **specialist-customer-comms**: no data file -- pure drafting. Add a
  send integration (Statuspage, Slack) only if you want to move past
  "draft for human review."

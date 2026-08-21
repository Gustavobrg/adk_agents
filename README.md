# adk_agents

Portfolio and testing ground for AI agents built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/). Each agent lives in its own top-level directory with its own README covering architecture, setup, and usage. This file is the entry point that ties them together.

## Agents

| Agent | Status | Description |
|---|---|---|
| [travel_agent](travel_agent/README.md) | Ready | Multi-agent trip planner. Interviews the traveler, then runs headless research and planning pipelines to produce a day-by-day, budgeted itinerary. Deployable to Vertex AI Agent Engine, with a Chainlit chat client included. |
| [incident_response](incident_response/README.md) | Ready | A2A multi-agent incident responder. A coordinator gathers an incident brief, then fans out concurrently over the real A2A protocol to 4 independently deployable specialists (bisection, error correlation, incident history, customer comms) and synthesizes a report. 5 separate `agents-cli`-scaffolded projects, each with its own eval suite. |
| [caveman-agent](caveman-agent/README.md) | Ready | "Grunk the caveman": compresses verbose, jargon-heavy text (emails, docs, tickets) into terse technical grunts, keeping facts/numbers/identifiers intact. Single-agent, `agents-cli`-scaffolded, runs on OpenRouter via `LiteLlm`, served over A2A. |

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `pip`
- Google Cloud project with Vertex AI / Google Cloud Logging enabled, if you plan to run or deploy the agents (see each agent's README for exactly which credentials it needs)

## Setup

```bash
# Install dependencies (uv, using the committed lockfile)
uv sync

# or with plain pip
pip install -r requirements.txt -e .

# Copy the env template and fill in your keys
cp .env.example .env
```

Environment variables are split between this root `.env.example` (shared across agents, e.g. Google Cloud Logging credentials) and each agent's own `.env.example` (agent-specific keys, e.g. `travel_agent/.env.example`). Check the agent's README for the full list it needs.

## Running an agent

ADK's CLI operates on whichever directory you point it at, run from the repository root:

```bash
# Web UI (chat + state/event inspector) - pick the agent from the dropdown
adk web

# Terminal chat with a specific agent
adk run travel_agent

# HTTP API server
adk api_server
```

## Optional: Chainlit client

`client/` is a standalone [Chainlit](https://docs.chainlit.io/) chat app for talking to an agent already deployed to Vertex AI Agent Engine (as opposed to running it locally via `adk web`/`adk run`). It's agent-agnostic at the transport level; point it at whichever deployed resource you want via its own `.env`. See [travel_agent's README](travel_agent/README.md#consuming-the-deployed-agent) for a full walkthrough of deploying and connecting to it.

```bash
uv sync --extra client
cp client/.env.example client/.env   # fill in project, location, and the deployed resource name
uv run chainlit run client/app.py -w
```

## Adding a new agent

1. Create a new top-level directory for it (mirroring `travel_agent/`'s layout: `agent.py` exposing `root_agent`, its own `.env.example`, a `README.md`).
2. Add any shared dependencies to the root `pyproject.toml`.
3. Add a row for it to the [Agents](#agents) table above, linking to its README.

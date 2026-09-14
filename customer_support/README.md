# customer_support

Single-agent e-commerce customer support bot. Demonstrates strategic tool
combination and error handling with a plain `LlmAgent`: checking order
status, processing refunds, and escalating to a human supervisor when a
tool can't resolve the issue.

Reference: https://google.github.io/adk-docs/tools-custom/

## Project Structure

```
customer_support/
├── agent.py       # root_agent + tools (check_order_status, process_refund, escalate_to_supervisor)
├── config.py      # OpenRouter/LiteLlm model wiring
└── .env.example   # OPENROUTER_API_KEY / MODEL template
```

> This agent runs on OpenRouter (via `LiteLlm`), not Vertex AI Gemini — see
> `config.py`. Set `OPENROUTER_API_KEY` and `MODEL` in `.env` to run it.

Order data is an in-memory `ORDERS_DB` dict (`ORD123`, `ORD456`, `ORD789`)
seeded in `agent.py` — there's no real backend, so restarting the process
resets it.

## Setup

```bash
cp customer_support/.env.example customer_support/.env
# fill in OPENROUTER_API_KEY and MODEL
```

## Running

From the repository root:

```bash
adk run customer_support   # terminal chat
adk web                    # web UI - pick customer_support from the dropdown
```
